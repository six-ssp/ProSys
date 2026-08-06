"""Run direct product-to-condition baselines on the maintained ProSys splits.

The canonical methods predict a reagent-solvent context from the target product
alone:

* ``product_naive_bayes`` fits independent Bernoulli Naive Bayes reagent and
  solvent heads on binary product Morgan fingerprints.
* ``product_gnn`` trains a family-specific product graph neural network with
  independent reagent and solvent multi-label heads.

Neither condition model receives a reactant route.  Stage 1 is used only after
condition inference to pair the top product-only contexts with frozen predicted
routes, yielding a comparable exact system-level candidate slate.
"""

from __future__ import annotations

import argparse
import tempfile
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.condition_modeling import (  # noqa: E402
    ConditionRow,
    ContextRecord,
    build_context_library,
    load_condition_rows,
    split_condition_tokens,
)
from prosys_shared.features import (  # noqa: E402
    canonicalize_reaction_side,
    canonicalize_smiles,
    product_morgan_fp,
    tanimoto_similarity_from_bitvect,
)
from prosys_shared.mainline import (  # noqa: E402
    build_candidate_training_table,
    display_family_name,
    evaluate_scored_frame_with_manifest,
    load_gold_condition_index,
    load_route_records,
    parse_families_arg,
    split_file_for_family,
)
from prosys_shared.nomenclature import direct_baseline_display_name  # noqa: E402
from stage3_XGBoost.reaction_gnn_features import (  # noqa: E402
    GraphEncoder,
    _batch_graphs,
    _graph_from_smiles,
)


METHODS = ('product_naive_bayes', 'product_gnn')
SUPPORTED_METHODS = METHODS
TOPKS = (1, 3, 5, 10)
CONDITION_WEIGHT = 1.0


@dataclass(frozen=True)
class ProductQuery:
    family: str
    sample_index: int
    reaction_id: str
    product: str


@dataclass(frozen=True)
class RouteProposal:
    sample_index: int
    reaction_id: str
    product: str
    reactants: str
    retro_rank: int
    retro_score: float
    retro_probability: float


@dataclass(frozen=True)
class ProductExample:
    product: str
    product_canonical: str
    reagent_tokens: tuple[str, ...]
    solvent_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ContextPrediction:
    reagent_norm: str
    solvent_norm: str
    score: float
    rank: int
    support: float = 0.0
    max_similarity: float = 0.0
    neighbor_count: int = 0


@dataclass(frozen=True)
class ProductGNNConfig:
    hidden_dim: int = 128
    message_passing_steps: int = 3
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 7
    device: str = 'cpu'
    random_state: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductNaiveBayesConfig:
    """Fixed, low-capacity product-fingerprint Naive Bayes configuration."""

    n_bits: int = 4096
    radius: int = 2
    alpha: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name != 'cpu' and torch.cuda.is_available():
        return torch.device(name)
    return torch.device('cpu')


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros(values.shape, dtype=np.float32)
    return ((values - float(np.mean(values))) / std).astype(np.float32)


def _cache_queries_and_routes(cache_file: Path, family: str) -> tuple[list[ProductQuery], dict[int, list[RouteProposal]]]:
    """Read only public Stage 1 predictions; intentionally ignore gold fields."""

    with cache_file.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)

    queries: list[ProductQuery] = []
    routes_by_sample: dict[int, list[RouteProposal]] = {}
    seen_indices: set[int] = set()
    for reaction in payload.get('reactions', []):
        sample_index = int(reaction['sample_index'])
        if sample_index in seen_indices:
            raise ValueError(f'Duplicate sample_index={sample_index} in {cache_file}.')
        seen_indices.add(sample_index)

        reaction_id = str(reaction['reaction_id'])
        product = str(reaction['product'])
        queries.append(
            ProductQuery(
                family=family,
                sample_index=sample_index,
                reaction_id=reaction_id,
                product=product,
            )
        )
        route_rows: list[RouteProposal] = []
        for route in reaction.get('routes', []):
            route_rows.append(
                RouteProposal(
                    sample_index=sample_index,
                    reaction_id=reaction_id,
                    product=product,
                    reactants=str(route.get('reactants', '')),
                    retro_rank=int(route.get('retro_rank', 1)),
                    retro_score=float(route.get('retro_score', 0.0)),
                    retro_probability=float(route.get('retro_probability', 0.0)),
                )
            )
        routes_by_sample[sample_index] = sorted(
            route_rows,
            key=lambda row: (row.retro_rank, -row.retro_score, row.reactants),
        )

    if not queries:
        raise ValueError(f'No test identities found in {cache_file}.')
    return sorted(queries, key=lambda row: row.sample_index), routes_by_sample


def _assert_cache_matches_split(
    *,
    queries: list[ProductQuery],
    split_file: Path,
    family: str,
) -> None:
    expected = load_route_records(split_file, family)
    observed_by_index = {
        query.sample_index: (query.reaction_id, canonicalize_smiles(query.product))
        for query in queries
    }
    expected_by_index = {
        int(record.sample_index): (str(record.reaction_id), canonicalize_smiles(record.product))
        for record in expected
    }
    if observed_by_index != expected_by_index:
        raise ValueError(
            'Stage 1 cache identity manifest does not match the formal split for '
            f'{family}: cache={len(observed_by_index)}, split={len(expected_by_index)}.'
        )


def condition_row_product_examples(rows: list[ConditionRow]) -> list[ProductExample]:
    """Keep one product-only supervised example per train-time condition record."""

    examples: list[ProductExample] = []
    for row in rows:
        product_canonical = canonicalize_smiles(row.product)
        if not product_canonical:
            continue
        examples.append(
            ProductExample(
                product=product_canonical,
                product_canonical=product_canonical,
                reagent_tokens=split_condition_tokens(row.reagent_norm),
                solvent_tokens=split_condition_tokens(row.solvent_norm),
            )
        )
    return examples


def _build_token_vocab(examples: list[ProductExample], field: str) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = {}
    for example in examples:
        for token in getattr(example, field):
            counts[token] = counts.get(token, 0) + 1
    vocab = sorted(counts, key=lambda token: (-counts[token], token))
    return vocab, {token: index for index, token in enumerate(vocab)}


def _multi_hot(tokens: tuple[str, ...], token_to_index: dict[str, int]) -> np.ndarray:
    values = np.zeros((len(token_to_index),), dtype=np.float32)
    for token in tokens:
        index = token_to_index.get(token)
        if index is not None:
            values[index] = 1.0
    return values


def _label_matrix(
    examples: list[ProductExample],
    reagent_to_index: dict[str, int],
    solvent_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    reagent = np.stack([_multi_hot(row.reagent_tokens, reagent_to_index) for row in examples], axis=0)
    solvent = np.stack([_multi_hot(row.solvent_tokens, solvent_to_index) for row in examples], axis=0)
    return reagent.astype(np.float32, copy=False), solvent.astype(np.float32, copy=False)


def _positive_weight(targets: np.ndarray) -> np.ndarray:
    positive = np.sum(targets, axis=0, dtype=np.float64)
    total = float(targets.shape[0])
    positive = np.maximum(positive, 1.0)
    negative = np.maximum(total - positive, 1.0)
    return np.clip(negative / positive, 1.0, 20.0).astype(np.float32)


class ContextScorer:
    """Map token probabilities to complete train-time condition contexts."""

    def __init__(
        self,
        contexts: list[ContextRecord],
        reagent_to_index: dict[str, int],
        solvent_to_index: dict[str, int],
        *,
        frequency_weight: float = 0.05,
    ):
        if not contexts:
            raise ValueError('A direct-condition model requires a non-empty training context library.')
        self.contexts = list(contexts)
        self.reagent_indices = [
            np.asarray(
                [reagent_to_index[token] for token in context.reagent_tokens if token in reagent_to_index],
                dtype=np.int64,
            )
            for context in self.contexts
        ]
        self.solvent_indices = [
            np.asarray(
                [solvent_to_index[token] for token in context.solvent_tokens if token in solvent_to_index],
                dtype=np.int64,
            )
            for context in self.contexts
        ]
        raw_frequency = np.log1p(np.asarray([context.context_count for context in self.contexts], dtype=np.float32))
        denominator = float(np.max(raw_frequency)) if raw_frequency.size else 1.0
        self.frequency_prior = raw_frequency / max(denominator, 1e-6)
        self.frequency_weight = float(frequency_weight)

    def score_logits(self, reagent_logits: np.ndarray, solvent_logits: np.ndarray) -> np.ndarray:
        reagent_log_prob = -np.logaddexp(0.0, -np.asarray(reagent_logits, dtype=np.float64))
        solvent_log_prob = -np.logaddexp(0.0, -np.asarray(solvent_logits, dtype=np.float64))
        scores = np.empty((len(self.contexts),), dtype=np.float32)
        for index, (reagent_idx, solvent_idx) in enumerate(zip(self.reagent_indices, self.solvent_indices)):
            token_scores: list[np.ndarray] = []
            if reagent_idx.size:
                token_scores.append(reagent_log_prob[reagent_idx])
            if solvent_idx.size:
                token_scores.append(solvent_log_prob[solvent_idx])
            mean_log_prob = float(np.mean(np.concatenate(token_scores))) if token_scores else -50.0
            scores[index] = mean_log_prob + self.frequency_weight * float(self.frequency_prior[index])
        return scores


def _rank_contexts(
    contexts: list[ContextRecord],
    scores: np.ndarray,
    *,
    top_contexts: int,
    support: np.ndarray | None = None,
    max_similarity: np.ndarray | None = None,
    neighbor_count: np.ndarray | None = None,
) -> list[ContextPrediction]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] != len(contexts):
        raise ValueError('Context score length does not match the training context library.')
    order = np.lexsort((np.arange(scores.shape[0]), -scores))[:top_contexts]
    predictions: list[ContextPrediction] = []
    for rank, index in enumerate(order, start=1):
        context = contexts[int(index)]
        predictions.append(
            ContextPrediction(
                reagent_norm=context.reagent_norm,
                solvent_norm=context.solvent_norm,
                score=float(scores[index]),
                rank=rank,
                support=(float(support[index]) if support is not None else 0.0),
                max_similarity=(float(max_similarity[index]) if max_similarity is not None else 0.0),
                neighbor_count=(int(neighbor_count[index]) if neighbor_count is not None else 0),
            )
        )
    return predictions


class ProductKNNConditionModel:
    """Same-family product-only KNN over complete training contexts."""

    def __init__(
        self,
        train_rows: list[ConditionRow],
        contexts: list[ContextRecord],
        *,
        n_bits: int = 4096,
        radius: int = 2,
        frequency_weight: float = 0.05,
    ):
        self.contexts = list(contexts)
        self.n_bits = int(n_bits)
        self.radius = int(radius)
        context_to_index = {
            (context.reagent_norm, context.solvent_norm): index
            for index, context in enumerate(self.contexts)
        }
        by_product: dict[str, dict[int, float]] = {}
        for row in train_rows:
            product_canonical = canonicalize_smiles(row.product)
            context_index = context_to_index.get((row.reagent_norm, row.solvent_norm))
            if not product_canonical or context_index is None:
                continue
            product_counts = by_product.setdefault(product_canonical, {})
            product_counts[context_index] = product_counts.get(context_index, 0.0) + 1.0
        if not by_product:
            raise ValueError('Product-KNN has no valid train products.')

        self.products = sorted(by_product)
        self.product_context_counts = [by_product[product] for product in self.products]
        self.fingerprints = np.stack(
            [product_morgan_fp(product, n_bits=self.n_bits, radius=self.radius) for product in self.products],
            axis=0,
        ).astype(np.uint8, copy=False)
        raw_frequency = np.log1p(np.asarray([context.context_count for context in self.contexts], dtype=np.float32))
        self.frequency_prior = raw_frequency / max(float(np.max(raw_frequency)), 1e-6)
        self.frequency_weight = float(frequency_weight)

    def predict(
        self,
        product: str,
        *,
        neighbors: int,
        top_contexts: int,
    ) -> list[ContextPrediction]:
        try:
            query_fp = product_morgan_fp(product, n_bits=self.n_bits, radius=self.radius)
            similarities = tanimoto_similarity_from_bitvect(query_fp, self.fingerprints)
        except ValueError:
            similarities = np.zeros((len(self.products),), dtype=np.float32)

        selected = np.lexsort((np.arange(similarities.shape[0]), -similarities))[:neighbors]
        support = np.zeros((len(self.contexts),), dtype=np.float32)
        max_similarity = np.zeros((len(self.contexts),), dtype=np.float32)
        neighbor_count = np.zeros((len(self.contexts),), dtype=np.int32)
        for product_index in selected:
            similarity = float(similarities[int(product_index)])
            if similarity <= 0.0:
                continue
            for context_index, count in self.product_context_counts[int(product_index)].items():
                support[context_index] += similarity * float(count)
                max_similarity[context_index] = max(max_similarity[context_index], similarity)
                neighbor_count[context_index] += 1

        scores = np.log1p(support) + self.frequency_weight * self.frequency_prior
        return _rank_contexts(
            self.contexts,
            scores,
            top_contexts=top_contexts,
            support=support,
            max_similarity=max_similarity,
            neighbor_count=neighbor_count,
        )


def _product_fingerprint_matrix(
    products: Iterable[str],
    *,
    n_bits: int,
    radius: int,
) -> np.ndarray:
    """Encode product SMILES as fixed binary fingerprints, with zero fallback."""

    fingerprints: list[np.ndarray] = []
    for product in products:
        try:
            fingerprint = product_morgan_fp(product, n_bits=n_bits, radius=radius)
        except ValueError:
            fingerprint = np.zeros((n_bits,), dtype=np.uint8)
        fingerprints.append(fingerprint.astype(np.uint8, copy=False))
    if not fingerprints:
        return np.zeros((0, n_bits), dtype=np.uint8)
    return np.stack(fingerprints, axis=0).astype(np.uint8, copy=False)


def _fit_bernoulli_nb_head(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit independent Bernoulli NB label heads in one vectorized operation."""

    if alpha <= 0.0:
        raise ValueError('Bernoulli Naive Bayes alpha must be positive.')
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError('Invalid product-fingerprint or label matrix for Bernoulli Naive Bayes.')

    num_examples = float(x.shape[0])
    positive_counts = np.sum(y, axis=0)
    negative_counts = num_examples - positive_counts
    all_feature_counts = np.sum(x, axis=0)
    positive_feature_counts = y.T @ x
    negative_feature_counts = all_feature_counts[None, :] - positive_feature_counts

    # Laplace smoothing keeps rare labels and absent fingerprint bits finite.
    positive_feature_prob = (positive_feature_counts + alpha) / (positive_counts[:, None] + 2.0 * alpha)
    negative_feature_prob = (negative_feature_counts + alpha) / (negative_counts[:, None] + 2.0 * alpha)
    positive_prior = (positive_counts + alpha) / (num_examples + 2.0 * alpha)
    negative_prior = 1.0 - positive_prior

    base_logits = np.log(positive_prior) - np.log(negative_prior)
    base_logits += np.sum(
        np.log1p(-positive_feature_prob) - np.log1p(-negative_feature_prob),
        axis=1,
    )
    active_bit_weights = (
        np.log(positive_feature_prob)
        - np.log1p(-positive_feature_prob)
        - np.log(negative_feature_prob)
        + np.log1p(-negative_feature_prob)
    )
    return base_logits.astype(np.float32), active_bit_weights.astype(np.float32)


def _predict_bernoulli_nb_logits(
    features: np.ndarray,
    *,
    base_logits: np.ndarray,
    active_bit_weights: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != active_bit_weights.shape[1]:
        raise ValueError('Product fingerprint width does not match the Bernoulli Naive Bayes artifact.')
    return (x @ active_bit_weights.T + base_logits[None, :]).astype(np.float32, copy=False)


def train_product_naive_bayes(
    *,
    train_rows: list[ConditionRow],
    artifact_dir: Path,
    config: ProductNaiveBayesConfig,
) -> dict[str, Any]:
    """Fit product-only independent reagent and solvent Bernoulli NB heads."""

    model_file = artifact_dir / 'product_naive_bayes.npz'
    metadata_file = artifact_dir / 'product_naive_bayes_metadata.json'
    if model_file.exists() and metadata_file.exists():
        return json.loads(metadata_file.read_text(encoding='utf-8'))

    examples = condition_row_product_examples(train_rows)
    if not examples:
        raise ValueError('Product-Naive-Bayes requires non-empty train-time product records.')
    reagent_vocab, reagent_to_index = _build_token_vocab(examples, 'reagent_tokens')
    solvent_vocab, solvent_to_index = _build_token_vocab(examples, 'solvent_tokens')
    if not reagent_vocab or not solvent_vocab:
        raise ValueError('Product-Naive-Bayes requires non-empty train-time reagent and solvent vocabularies.')

    features = _product_fingerprint_matrix(
        [example.product for example in examples],
        n_bits=config.n_bits,
        radius=config.radius,
    )
    reagent_targets, solvent_targets = _label_matrix(examples, reagent_to_index, solvent_to_index)
    reagent_base, reagent_weights = _fit_bernoulli_nb_head(features, reagent_targets, alpha=config.alpha)
    solvent_base, solvent_weights = _fit_bernoulli_nb_head(features, solvent_targets, alpha=config.alpha)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_file,
        reagent_vocab=np.asarray(reagent_vocab, dtype=np.str_),
        solvent_vocab=np.asarray(solvent_vocab, dtype=np.str_),
        reagent_base_logits=reagent_base,
        reagent_active_bit_weights=reagent_weights,
        solvent_base_logits=solvent_base,
        solvent_active_bit_weights=solvent_weights,
    )
    metadata = {
        'model_file': model_file.name,
        'model': 'Product Bernoulli Naive Bayes',
        'input': f'binary MorganFP(product), radius={config.radius}, n_bits={config.n_bits}',
        'num_train_records': len(examples),
        'num_train_products': len({example.product_canonical for example in examples}),
        'num_reagent_tokens': len(reagent_vocab),
        'num_solvent_tokens': len(solvent_vocab),
        'config': config.to_dict(),
    }
    _write_json(metadata_file, metadata)
    return metadata


class ProductNaiveBayesConditionModel:
    """Product-only dual multi-label Bernoulli Naive Bayes condition model."""

    def __init__(self, artifact_dir: Path):
        metadata = json.loads((artifact_dir / 'product_naive_bayes_metadata.json').read_text(encoding='utf-8'))
        self.config = ProductNaiveBayesConfig(**metadata['config'])
        with np.load(artifact_dir / 'product_naive_bayes.npz', allow_pickle=False) as payload:
            self.reagent_vocab = [str(value) for value in payload['reagent_vocab'].tolist()]
            self.solvent_vocab = [str(value) for value in payload['solvent_vocab'].tolist()]
            self.reagent_base_logits = payload['reagent_base_logits'].astype(np.float32, copy=False)
            self.reagent_active_bit_weights = payload['reagent_active_bit_weights'].astype(np.float32, copy=False)
            self.solvent_base_logits = payload['solvent_base_logits'].astype(np.float32, copy=False)
            self.solvent_active_bit_weights = payload['solvent_active_bit_weights'].astype(np.float32, copy=False)
        self.reagent_to_index = {token: index for index, token in enumerate(self.reagent_vocab)}
        self.solvent_to_index = {token: index for index, token in enumerate(self.solvent_vocab)}
        if self.reagent_active_bit_weights.shape[1] != self.config.n_bits:
            raise ValueError('Reagent Naive Bayes artifact fingerprint width disagrees with metadata.')
        if self.solvent_active_bit_weights.shape[1] != self.config.n_bits:
            raise ValueError('Solvent Naive Bayes artifact fingerprint width disagrees with metadata.')

    def predict_logits(self, products: list[str]) -> tuple[np.ndarray, np.ndarray]:
        features = _product_fingerprint_matrix(
            products,
            n_bits=self.config.n_bits,
            radius=self.config.radius,
        )
        return (
            _predict_bernoulli_nb_logits(
                features,
                base_logits=self.reagent_base_logits,
                active_bit_weights=self.reagent_active_bit_weights,
            ),
            _predict_bernoulli_nb_logits(
                features,
                base_logits=self.solvent_base_logits,
                active_bit_weights=self.solvent_active_bit_weights,
            ),
        )



class ProductGNN(nn.Module):
    """A lightweight product-only GNN with independent condition token heads."""

    def __init__(
        self,
        num_reagent_tokens: int,
        num_solvent_tokens: int,
        *,
        hidden_dim: int,
        message_passing_steps: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = GraphEncoder(hidden_dim, message_passing_steps, dropout)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reagent_head = nn.Linear(hidden_dim, num_reagent_tokens)
        self.solvent_head = nn.Linear(hidden_dim, num_solvent_tokens)

    def forward(
        self,
        product_graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.trunk(self.encoder(*product_graph))
        return self.reagent_head(embedding), self.solvent_head(embedding)


def _product_batches(examples: list[ProductExample], batch_size: int) -> Iterable[list[ProductExample]]:
    for start in range(0, len(examples), batch_size):
        yield examples[start:start + batch_size]


def _product_graph_batch(
    examples: list[ProductExample],
    device: torch.device,
    *,
    graph_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if graph_cache is None:
        graphs = [_graph_from_smiles(example.product, reaction_side=False) for example in examples]
    else:
        graphs = [graph_cache[example.product_canonical] for example in examples]
    return _batch_graphs(
        graphs,
        device,
    )


def _build_product_graph_cache(examples: Iterable[ProductExample]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse each unique product once; graph topology is independent of epoch."""

    return {
        example.product_canonical: _graph_from_smiles(example.product, reaction_side=False)
        for example in {row.product_canonical: row for row in examples}.values()
    }


def _product_loss(
    model: ProductGNN,
    examples: list[ProductExample],
    *,
    graph_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    reagent_to_index: dict[str, int],
    solvent_to_index: dict[str, int],
    reagent_loss: nn.Module,
    solvent_loss: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    reagent_targets, solvent_targets = _label_matrix(examples, reagent_to_index, solvent_to_index)
    reagent_logits, solvent_logits = model(_product_graph_batch(examples, device, graph_cache=graph_cache))
    return reagent_loss(
        reagent_logits,
        torch.as_tensor(reagent_targets, dtype=torch.float32, device=device),
    ) + solvent_loss(
        solvent_logits,
        torch.as_tensor(solvent_targets, dtype=torch.float32, device=device),
    )


def train_product_gnn(
    *,
    train_rows: list[ConditionRow],
    val_rows: list[ConditionRow],
    artifact_dir: Path,
    config: ProductGNNConfig,
    force_retrain: bool = False,
) -> dict[str, Any]:
    model_file = artifact_dir / 'product_gnn.pt'
    metadata_file = artifact_dir / 'product_gnn_metadata.json'
    if model_file.exists() and metadata_file.exists() and not force_retrain:
        return json.loads(metadata_file.read_text(encoding='utf-8'))

    _set_seed(config.random_state)
    train_examples = condition_row_product_examples(train_rows)
    val_examples = condition_row_product_examples(val_rows)
    if not train_examples or not val_examples:
        raise ValueError('Product-GNN requires non-empty train and validation product sets.')
    reagent_vocab, reagent_to_index = _build_token_vocab(train_examples, 'reagent_tokens')
    solvent_vocab, solvent_to_index = _build_token_vocab(train_examples, 'solvent_tokens')
    if not reagent_vocab or not solvent_vocab:
        raise ValueError('Product-GNN requires non-empty train-time reagent and solvent vocabularies.')
    graph_cache = _build_product_graph_cache([*train_examples, *val_examples])

    reagent_targets, solvent_targets = _label_matrix(train_examples, reagent_to_index, solvent_to_index)
    device = _device(config.device)
    model = ProductGNN(
        num_reagent_tokens=len(reagent_vocab),
        num_solvent_tokens=len(solvent_vocab),
        hidden_dim=config.hidden_dim,
        message_passing_steps=config.message_passing_steps,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    reagent_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(_positive_weight(reagent_targets), dtype=torch.float32, device=device)
    )
    solvent_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(_positive_weight(solvent_targets), dtype=torch.float32, device=device)
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float('inf')
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    train_order = list(range(len(train_examples)))
    for epoch in range(1, config.max_epochs + 1):
        random.shuffle(train_order)
        model.train()
        train_losses: list[float] = []
        for start in range(0, len(train_order), config.batch_size):
            batch = [train_examples[index] for index in train_order[start:start + config.batch_size]]
            loss = _product_loss(
                model,
                batch,
                graph_cache=graph_cache,
                reagent_to_index=reagent_to_index,
                solvent_to_index=solvent_to_index,
                reagent_loss=reagent_loss,
                solvent_loss=solvent_loss,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in _product_batches(val_examples, config.batch_size):
                loss = _product_loss(
                    model,
                    batch,
                    graph_cache=graph_cache,
                    reagent_to_index=reagent_to_index,
                    solvent_to_index=solvent_to_index,
                    reagent_loss=reagent_loss,
                    solvent_loss=solvent_loss,
                    device=device,
                )
                val_losses.append(float(loss.detach().cpu()))
        val_loss = float(np.mean(np.asarray(val_losses, dtype=np.float64))) if val_losses else float('inf')
        train_loss = float(np.mean(np.asarray(train_losses, dtype=np.float64))) if train_losses else float('inf')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'[product-gnn] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}', flush=True)

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state': best_state,
            'reagent_vocab': reagent_vocab,
            'solvent_vocab': solvent_vocab,
            'config': config.to_dict(),
        },
        model_file,
    )
    metadata = {
        'model_file': model_file.name,
        'num_train_records': len(train_examples),
        'num_val_records': len(val_examples),
        'num_train_products': len({example.product_canonical for example in train_examples}),
        'num_val_products': len({example.product_canonical for example in val_examples}),
        'graph_cache_entries': len(graph_cache),
        'num_reagent_tokens': len(reagent_vocab),
        'num_solvent_tokens': len(solvent_vocab),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'last_epoch': len(history),
        'config': config.to_dict(),
        'history': history,
    }
    _write_json(metadata_file, metadata)
    return metadata


class ProductGNNConditionModel:
    def __init__(self, artifact_dir: Path, *, device: str):
        payload = torch.load(artifact_dir / 'product_gnn.pt', map_location='cpu', weights_only=False)
        config = ProductGNNConfig(**payload['config'])
        self.config = ProductGNNConfig(**{**config.to_dict(), 'device': device})
        self.device = _device(device)
        self.reagent_vocab = [str(value) for value in payload['reagent_vocab']]
        self.solvent_vocab = [str(value) for value in payload['solvent_vocab']]
        self.reagent_to_index = {token: index for index, token in enumerate(self.reagent_vocab)}
        self.solvent_to_index = {token: index for index, token in enumerate(self.solvent_vocab)}
        self.model = ProductGNN(
            num_reagent_tokens=len(self.reagent_vocab),
            num_solvent_tokens=len(self.solvent_vocab),
            hidden_dim=self.config.hidden_dim,
            message_passing_steps=self.config.message_passing_steps,
            dropout=self.config.dropout,
        ).to(self.device)
        self.model.load_state_dict(payload['model_state'])
        self.model.eval()

    @torch.no_grad()
    def predict_logits(self, products: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not products:
            return (
                np.zeros((0, len(self.reagent_vocab)), dtype=np.float32),
                np.zeros((0, len(self.solvent_vocab)), dtype=np.float32),
            )
        reagent_rows: list[np.ndarray] = []
        solvent_rows: list[np.ndarray] = []
        for start in range(0, len(products), self.config.batch_size):
            batch_products = products[start:start + self.config.batch_size]
            examples = [
                ProductExample(
                    product=product,
                    product_canonical=canonicalize_smiles(product),
                    reagent_tokens=(),
                    solvent_tokens=(),
                )
                for product in batch_products
            ]
            reagent_logits, solvent_logits = self.model(_product_graph_batch(examples, self.device))
            reagent_rows.append(reagent_logits.detach().cpu().numpy().astype(np.float32, copy=False))
            solvent_rows.append(solvent_logits.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(reagent_rows, axis=0), np.concatenate(solvent_rows, axis=0)


def _prediction_json_rows(
    queries: list[ProductQuery],
    predictions: dict[int, list[ContextPrediction]],
) -> Iterable[dict[str, Any]]:
    for query in queries:
        contexts = predictions.get(query.sample_index, [])
        yield {
            'family': query.family,
            'sample_index': query.sample_index,
            'reaction_id': query.reaction_id,
            'product': query.product,
            'contexts': [
                {
                    'reagent_norm': context.reagent_norm,
                    'solvent_norm': context.solvent_norm,
                    'condition_score': context.score,
                    'context_rank': context.rank,
                    'knn_support': context.support,
                    'knn_max_similarity': context.max_similarity,
                    'knn_neighbor_count': context.neighbor_count,
                }
                for context in contexts
            ],
        }


def _predict_product_knn(
    model: ProductKNNConditionModel,
    queries: list[ProductQuery],
    *,
    neighbors: int,
    top_contexts: int,
) -> dict[int, list[ContextPrediction]]:
    cached_by_product: dict[str, list[ContextPrediction]] = {}
    predictions: dict[int, list[ContextPrediction]] = {}
    for query in queries:
        product_key = canonicalize_smiles(query.product) or query.product
        if product_key not in cached_by_product:
            cached_by_product[product_key] = model.predict(
                query.product,
                neighbors=neighbors,
                top_contexts=top_contexts,
            )
        predictions[query.sample_index] = cached_by_product[product_key]
    return predictions


def _predict_product_gnn(
    model: ProductGNNConditionModel,
    contexts: list[ContextRecord],
    queries: list[ProductQuery],
    *,
    top_contexts: int,
) -> dict[int, list[ContextPrediction]]:
    scorer = ContextScorer(contexts, model.reagent_to_index, model.solvent_to_index)
    unique_products: dict[str, str] = {}
    for query in queries:
        product_key = canonicalize_smiles(query.product) or query.product
        unique_products.setdefault(product_key, query.product)
    product_keys = list(unique_products)
    reagent_logits, solvent_logits = model.predict_logits([unique_products[key] for key in product_keys])
    predictions_by_product = {
        product_key: _rank_contexts(
            contexts,
            scorer.score_logits(reagent_logit, solvent_logit),
            top_contexts=top_contexts,
        )
        for product_key, reagent_logit, solvent_logit in zip(product_keys, reagent_logits, solvent_logits)
    }

    return {
        query.sample_index: predictions_by_product[canonicalize_smiles(query.product) or query.product]
        for query in queries
    }


def _predict_product_naive_bayes(
    model: ProductNaiveBayesConditionModel,
    contexts: list[ContextRecord],
    queries: list[ProductQuery],
    *,
    top_contexts: int,
) -> dict[int, list[ContextPrediction]]:
    """Score historical complete contexts from independent NB token logits."""

    scorer = ContextScorer(contexts, model.reagent_to_index, model.solvent_to_index)
    unique_products: dict[str, str] = {}
    for query in queries:
        product_key = canonicalize_smiles(query.product) or query.product
        unique_products.setdefault(product_key, query.product)
    product_keys = list(unique_products)
    reagent_logits, solvent_logits = model.predict_logits([unique_products[key] for key in product_keys])
    predictions_by_product = {
        product_key: _rank_contexts(
            contexts,
            scorer.score_logits(reagent_logit, solvent_logit),
            top_contexts=top_contexts,
        )
        for product_key, reagent_logit, solvent_logit in zip(product_keys, reagent_logits, solvent_logits)
    }
    return {
        query.sample_index: predictions_by_product[canonicalize_smiles(query.product) or query.product]
        for query in queries
    }


def evaluate_direct_context_predictions(
    queries: list[ProductQuery],
    predictions: dict[int, list[ContextPrediction]],
    gold_split_file: Path,
    *,
    topks: tuple[int, ...] = TOPKS,
) -> dict[str, Any]:
    gold_index = load_gold_condition_index(gold_split_file)
    hits = {k: 0 for k in topks}
    pool_hits = 0
    for query in queries:
        product_key = canonicalize_smiles(query.product)
        bucket = gold_index.get((query.reaction_id, product_key))
        gold_contexts = set(bucket.context_keys) if bucket is not None else set()
        predicted = predictions.get(query.sample_index, [])
        if gold_contexts and any((row.reagent_norm, row.solvent_norm) in gold_contexts for row in predicted):
            pool_hits += 1
        for k in topks:
            if gold_contexts and any((row.reagent_norm, row.solvent_norm) in gold_contexts for row in predicted[:k]):
                hits[k] += 1
    denominator = len(queries)
    metrics: dict[str, Any] = {
        'num_product_queries': denominator,
        'condition_pool_coverage': (pool_hits / denominator if denominator else 0.0),
    }
    for k in topks:
        metrics[f'condition_top{k}_all'] = hits[k] / denominator if denominator else 0.0
    return metrics


def _unique_routes(routes: list[RouteProposal]) -> list[RouteProposal]:
    selected: dict[str, RouteProposal] = {}
    for route in routes:
        route_key = canonicalize_reaction_side(route.reactants) or route.reactants
        previous = selected.get(route_key)
        if previous is None or (route.retro_rank, -route.retro_score, route.reactants) < (
            previous.retro_rank,
            -previous.retro_score,
            previous.reactants,
        ):
            selected[route_key] = route
    return sorted(
        selected.values(),
        key=lambda row: (row.retro_rank, -row.retro_score, row.reactants),
    )

def build_joint_candidate_frame(
    queries: list[ProductQuery],
    routes_by_sample: dict[int, list[RouteProposal]],
    predictions: dict[int, list[ContextPrediction]],
    *,
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query in queries:
        contexts = predictions.get(query.sample_index, [])
        if not contexts:
            continue
        for route in _unique_routes(routes_by_sample.get(query.sample_index, [])):
            route_canonical = canonicalize_reaction_side(route.reactants)
            for context in contexts:
                rows.append(
                    {
                        'family': query.family,
                        'sample_index': query.sample_index,
                        'reaction_id': query.reaction_id,
                        'product': query.product,
                        'reactants': route.reactants,
                        'route_canonical': route_canonical,
                        'retro_rank': route.retro_rank,
                        'retro_score': route.retro_score,
                        'retro_probability': route.retro_probability,
                        'reagent_norm': context.reagent_norm,
                        'solvent_norm': context.solvent_norm,
                        'condition_score': context.score,
                        'condition_rank': context.rank,
                        'direct_condition_method': method,
                        'product_knn_support': context.support,
                        'product_knn_max_similarity': context.max_similarity,
                        'product_knn_neighbor_count': context.neighbor_count,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=['family', 'sample_index', 'reaction_id', 'product', 'reactants', 'reagent_norm', 'solvent_norm'])
    return pd.DataFrame(rows).sort_values(
        ['sample_index', 'retro_rank', 'condition_rank', 'reactants', 'reagent_norm', 'solvent_norm'],
        kind='mergesort',
    ).reset_index(drop=True)


def fuse_candidate_scores(
    frame: pd.DataFrame,
    *,
    route_weight: float,
    condition_weight: float = CONDITION_WEIGHT,
) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        out['system_score'] = pd.Series(dtype=np.float64)
        return out
    out = frame.copy()
    out['route_score_z'] = 0.0
    out['condition_score_z'] = 0.0
    for _, group in out.groupby('sample_index', sort=False):
        indices = group.index
        route_z = _zscore(group['retro_score'].to_numpy(dtype=np.float64))
        condition_z = _zscore(group['condition_score'].to_numpy(dtype=np.float64))
        out.loc[indices, 'route_score_z'] = route_z
        out.loc[indices, 'condition_score_z'] = condition_z
        tie_break = -1e-9 * group['retro_rank'].to_numpy(dtype=np.float64) - 1e-11 * group['condition_rank'].to_numpy(dtype=np.float64)
        out.loc[indices, 'system_score'] = route_weight * route_z + condition_weight * condition_z + tie_break
    return out.sort_values(
        ['sample_index', 'system_score', 'retro_rank', 'condition_rank', 'reactants', 'reagent_norm', 'solvent_norm'],
        ascending=[True, False, True, True, True, True, True],
        kind='mergesort',
    ).reset_index(drop=True)

def select_route_weight(
    validation_frame: pd.DataFrame,
    expected_sample_indices: list[int],
    *,
    route_weights: tuple[float, ...],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for route_weight in route_weights:
        fused = fuse_candidate_scores(validation_frame, route_weight=float(route_weight))
        metrics = evaluate_scored_frame_with_manifest(
            fused,
            expected_sample_indices=expected_sample_indices,
            score_column='system_score',
            topks=TOPKS,
        )
        records.append(
            {
                'route_weight': float(route_weight),
                'condition_weight': CONDITION_WEIGHT,
                'metrics': metrics,
            }
        )
    selected = sorted(
        records,
        key=lambda row: (
            -float(row['metrics']['system_top10_all']),
            -float(row['metrics']['system_top1_all']),
            float(row['route_weight']),
        ),
    )[0]
    return {
        'selection_rule': 'max validation Sys@10, then Sys@1, then smaller route weight',
        'candidates': records,
        'selected': selected,
    }

def product_overlap_stats(train_rows: list[ConditionRow], queries: list[ProductQuery]) -> dict[str, int]:
    train_products = {canonicalize_smiles(row.product) for row in train_rows}
    train_products.discard('')
    query_products = {canonicalize_smiles(query.product) for query in queries}
    query_products.discard('')
    return {
        'train_unique_products': len(train_products),
        'query_unique_products': len(query_products),
        'train_query_product_overlap': len(train_products & query_products),
    }


def _ensure_new_run_directory(path: Path, *, resume: bool) -> bool:
    if (path / 'run_metadata.json').exists() and resume:
        return False

    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty run directory: {path}')


    path.mkdir(parents=True, exist_ok=True)
    return True

def _write_top10_audit(path: Path, frame: pd.DataFrame) -> None:
    """Persist only the ranked candidate slice needed for manual auditing."""

    audit = frame.groupby('sample_index', sort=False, group_keys=False).head(max(TOPKS))
    audit.to_csv(path, index=False, compression='gzip')


def run_family(
    *,
    repo_root: Path,
    family: str,
    method: str,
    output_root: Path,
    route_root: Path,
    validation_route_root: Path,
    device: str,
    top_contexts: int,
    knn_neighbors: int,
    naive_bayes_alpha: float,
    hidden_dim: int,
    message_passing_steps: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    route_weights: tuple[float, ...],
    resume: bool,
) -> dict[str, Any]:
    family_root = output_root / method / family
    if not _ensure_new_run_directory(family_root, resume=resume):
        return json.loads((family_root / 'run_metadata.json').read_text(encoding='utf-8'))
    if method not in SUPPORTED_METHODS:
        raise ValueError(f'Unsupported direct-condition method: {method}')

    train_split = split_file_for_family(repo_root, family, 'train')
    val_split = split_file_for_family(repo_root, family, 'val')
    test_split = split_file_for_family(repo_root, family, 'test')
    test_cache = route_root / family / 'route_cache.json'
    val_cache = validation_route_root / family / 'route_cache.json'
    test_queries, test_routes = _cache_queries_and_routes(test_cache, family)
    val_queries, val_routes = _cache_queries_and_routes(val_cache, family)
    _assert_cache_matches_split(queries=test_queries, split_file=test_split, family=family)
    _assert_cache_matches_split(queries=val_queries, split_file=val_split, family=family)
    train_rows = load_condition_rows(train_split)
    val_rows = load_condition_rows(val_split)
    contexts = build_context_library(train_rows)
    if not contexts:
        raise ValueError(f'No train-time condition contexts for {family}.')

    artifact_dir = family_root / 'artifacts'
    if method == 'product_naive_bayes':
        config = ProductNaiveBayesConfig(alpha=naive_bayes_alpha)
        model_metadata = train_product_naive_bayes(
            train_rows=train_rows,
            artifact_dir=artifact_dir,
            config=config,
        )
        model_metadata = {**model_metadata, 'num_train_contexts': len(contexts)}
        model = ProductNaiveBayesConditionModel(artifact_dir)
        val_predictions = _predict_product_naive_bayes(
            model,
            contexts,
            val_queries,
            top_contexts=top_contexts,
        )
        test_predictions = _predict_product_naive_bayes(
            model,
            contexts,
            test_queries,
            top_contexts=top_contexts,
        )
    elif method == 'product_knn':
        model = ProductKNNConditionModel(
            train_rows,
            contexts,
            n_bits=4096,
            radius=2,
        )
        model_metadata: dict[str, Any] = {
            'model': 'Product-KNN Condition Retrieval',
            'input': 'MorganFP(product), radius=2, n_bits=4096',
            'num_train_products': len(model.products),
            'num_train_contexts': len(contexts),
            'knn_neighbors': knn_neighbors,
        }
        val_predictions = _predict_product_knn(
            model,
            val_queries,
            neighbors=knn_neighbors,
            top_contexts=top_contexts,
        )
        test_predictions = _predict_product_knn(
            model,
            test_queries,
            neighbors=knn_neighbors,
            top_contexts=top_contexts,
        )
    elif method == 'product_gnn':
        config = ProductGNNConfig(
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            random_state=seed,
        )
        model_metadata = train_product_gnn(
            train_rows=train_rows,
            val_rows=val_rows,
            artifact_dir=artifact_dir,
            config=config,
        )
        model = ProductGNNConditionModel(artifact_dir, device=device)
        val_predictions = _predict_product_gnn(
            model,
            contexts,
            val_queries,
            top_contexts=top_contexts,
        )
        test_predictions = _predict_product_gnn(
            model,
            contexts,
            test_queries,
            top_contexts=top_contexts,
        )
    else:
        raise AssertionError(f'Unhandled direct-condition method: {method}')

    _write_json(artifact_dir / 'model_metadata.json', model_metadata)
    _write_jsonl(family_root / 'validation_condition_predictions.jsonl', _prediction_json_rows(val_queries, val_predictions))
    _write_jsonl(family_root / 'test_condition_predictions.jsonl', _prediction_json_rows(test_queries, test_predictions))

    validation_condition_metrics = evaluate_direct_context_predictions(
        val_queries,
        val_predictions,
        val_split,
    )
    test_condition_metrics = evaluate_direct_context_predictions(
        test_queries,
        test_predictions,
        test_split,
    )
    with tempfile.TemporaryDirectory(prefix='candidate_work_', dir=family_root) as temporary_dir:
        work_dir = Path(temporary_dir)
        validation_raw = build_joint_candidate_frame(
            val_queries,
            val_routes,
            val_predictions,
            method=method,
        )
        validation_raw_file = work_dir / 'validation_raw.csv'
        validation_raw.to_csv(validation_raw_file, index=False)
        validation_labeled = build_candidate_training_table(validation_raw_file, val_split)
        fusion = select_route_weight(
            validation_labeled,
            [query.sample_index for query in val_queries],
            route_weights=route_weights,
        )
        _write_json(family_root / 'fusion_selection.json', fusion)

        selected_route_weight = float(fusion['selected']['route_weight'])
        validation_scored = fuse_candidate_scores(validation_labeled, route_weight=selected_route_weight)
        _write_top10_audit(family_root / 'validation_top10_candidates.csv.gz', validation_scored)
        del validation_raw, validation_labeled, validation_scored

        test_raw = build_joint_candidate_frame(
            test_queries,
            test_routes,
            test_predictions,
            method=method,
        )
        test_scored = fuse_candidate_scores(test_raw, route_weight=selected_route_weight)
        test_candidate_file = work_dir / 'test_scored.csv'
        test_scored.to_csv(test_candidate_file, index=False)
        test_labeled = build_candidate_training_table(test_candidate_file, test_split)
        test_labeled = fuse_candidate_scores(test_labeled, route_weight=selected_route_weight)
        _write_top10_audit(family_root / 'test_top10_candidates.csv.gz', test_labeled)
        test_candidate_rows = int(len(test_labeled))
        test_metrics = evaluate_scored_frame_with_manifest(
            test_labeled,
            expected_sample_indices=[query.sample_index for query in test_queries],
            score_column='system_score',
            topks=TOPKS,
        )

    route_counts = {
        'validation_stage1_routes': sum(len(rows) for rows in val_routes.values()),
        'test_stage1_routes': sum(len(rows) for rows in test_routes.values()),
        'validation_samples_without_routes': sum(not val_routes.get(query.sample_index) for query in val_queries),
        'test_samples_without_routes': sum(not test_routes.get(query.sample_index) for query in test_queries),
    }
    result = {
        'family': family,
        'method': method,
        'condition_model_input': 'target product only',
        'condition_model_uses_stage1_route': False,
        'system_composition': 'top product-only contexts paired with frozen Stage 1 routes',
        'train_split': str(train_split),
        'validation_split': str(val_split),
        'test_split': str(test_split),
        'test_route_cache': str(test_cache),
        'validation_route_cache': str(val_cache),
        'model_metadata': model_metadata,
        'train_test_product_overlap': product_overlap_stats(train_rows, test_queries),
        'route_cache_counts': route_counts,
        'validation_condition_metrics': validation_condition_metrics,
        'test_condition_metrics': test_condition_metrics,
        'fusion': fusion,
        'test_candidate_rows': test_candidate_rows,
        'candidate_artifacts': {
            'validation_top10': 'validation_top10_candidates.csv.gz',
            'test_top10': 'test_top10_candidates.csv.gz',
            'full_candidate_tables': 'temporary during evaluation; removed automatically',
        },
        'test_metrics': test_metrics,
        'temperature_column': None,
    }
    _write_json(family_root / 'run_metadata.json', result)
    return result


def _result_row(record: dict[str, Any]) -> dict[str, Any]:
    system = record['test_metrics']
    condition = record['test_condition_metrics']
    row = {
        'method': record['method'],
        'family': record['family'],
        'display_family': display_family_name(record['family']),
        'condition_pool': condition.get('condition_pool_coverage'),
        'cover': system.get('pool_coverage'),
        'mrr': system.get('system_mrr'),
        'ndcg10': system.get('system_ndcg10'),
        'candidate_slates': system.get('candidate_slates'),
        'missing_candidate_slates': system.get('missing_candidate_slates'),
        'test_candidate_rows': record.get('test_candidate_rows'),
        'route_weight': record['fusion']['selected']['route_weight'],
        'condition_weight': record['fusion']['selected']['condition_weight'],
    }
    for k in TOPKS:
        row[f'condition{k}'] = condition.get(f'condition_top{k}_all')
        row[f'sys{k}'] = system.get(f'system_top{k}_all')
    return row


def _macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows = list(rows)
    for method in SUPPORTED_METHODS:
        method_rows = [row for row in rows if row['method'] == method]
        if not method_rows:
            continue
        macro = {'method': method, 'family': 'MACRO-AVG', 'display_family': 'Macro average'}
        for field in method_rows[0]:
            if field in {'method', 'family', 'display_family'}:

                continue
            values = [float(row[field]) for row in method_rows if row.get(field) is not None]
            macro[field] = float(np.mean(values)) if values else None
        all_rows.append(macro)
    return all_rows

def write_summary(output_root: Path, records: list[dict[str, Any]]) -> None:
    rows = _macro_rows([_result_row(record) for record in records])
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / 'summary.csv', index=False)

    lines = ['# Direct Product-to-Condition Baselines', '']
    lines.append('Each condition model sees only the target product; Stage 1 routes are used only to form final system candidates.')
    reported_methods = [
        (direct_baseline_display_name(method), method)
        for method in SUPPORTED_METHODS
        if any(row['method'] == method for row in rows)
    ]
    display_names = ', '.join(name for name, _ in reported_methods) or 'none'
    stored_keys = ', '.join(method for _, method in reported_methods) or 'none'
    lines.append(f'Reported direct method(s): {display_names} (stored key(s): {stored_keys}).')
    lines.append('')
    lines.append('| Method | Family | Condition@1 | Condition@10 | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy | MRR | nDCG@10 |')
    lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for method in SUPPORTED_METHODS:
        method_rows = [row for row in rows if row['method'] == method]
        for row in method_rows:
            lines.append(
                '| {method} | {family} | {condition1:.2f} | {condition10:.2f} | {cover:.2f} | {sys1:.2f} | {sys3:.2f} | {sys5:.2f} | {sys10:.2f} | {mrr:.2f} | {ndcg:.2f} |'.format(
                    method=direct_baseline_display_name(str(row['method'])),
                    family=row['display_family'],
                    condition1=100.0 * float(row.get('condition1') or 0.0),
                    condition10=100.0 * float(row.get('condition10') or 0.0),
                    cover=100.0 * float(row.get('cover') or 0.0),
                    sys1=100.0 * float(row.get('sys1') or 0.0),
                    sys3=100.0 * float(row.get('sys3') or 0.0),
                    sys5=100.0 * float(row.get('sys5') or 0.0),
                    sys10=100.0 * float(row.get('sys10') or 0.0),
                    mrr=100.0 * float(row.get('mrr') or 0.0),
                    ndcg=100.0 * float(row.get('ndcg10') or 0.0),
                )
            )
    (output_root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')



def _load_completed_records(output_root: Path) -> list[dict[str, Any]]:
    """Rebuild the aggregate table from all completed method-family runs."""

    records: list[dict[str, Any]] = []
    for method in SUPPORTED_METHODS:
        method_root = output_root / method
        if not method_root.exists():
            continue
        for metadata_file in sorted(method_root.glob('*/run_metadata.json')):
            record = json.loads(metadata_file.read_text(encoding='utf-8'))
            if record.get('method') != method:
                raise ValueError(f'Method metadata mismatch in {metadata_file}.')
            records.append(record)
    return records


def _parse_methods(value: str) -> list[str]:
    if value.strip().lower() == 'all':
        return list(METHODS)
    methods = [item.strip() for item in value.replace(',', ' ').split() if item.strip()]
    unknown = [method for method in methods if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f'Unknown direct-condition method(s): {unknown}')
    return methods


def _parse_route_weights(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(',') if item.strip())
    if not parsed:
        raise ValueError('At least one route weight is required.')
    return parsed


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', default='all')
    parser.add_argument('--methods', default='all')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--route-root', default='outputs/stage1_routes')
    parser.add_argument('--validation-route-root', default='outputs/stage1_routes_validation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--top-contexts', type=int, default=20)
    parser.add_argument('--knn-neighbors', type=int, default=64)
    parser.add_argument('--naive-bayes-alpha', type=float, default=1.0)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--message-passing-steps', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.10)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--route-weights', default='0,0.25,0.5,1,1.5,2')
    parser.add_argument('--resume', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.top_contexts <= 0:
        raise ValueError('--top-contexts must be positive.')
    if args.knn_neighbors <= 0:
        raise ValueError('--knn-neighbors must be positive.')
    if args.naive_bayes_alpha <= 0.0:
        raise ValueError('--naive-bayes-alpha must be positive.')
    repo_root = REPO_ROOT
    output_root = _resolve(repo_root, args.output_root)
    route_root = _resolve(repo_root, args.route_root)
    validation_route_root = _resolve(repo_root, args.validation_route_root)
    families = parse_families_arg(args.families)
    methods = _parse_methods(args.methods)
    route_weights = _parse_route_weights(args.route_weights)
    records: list[dict[str, Any]] = []
    for method in methods:
        for family in families:
            print(f'[direct-product-baseline] method={method} family={family}', flush=True)
            records.append(
                run_family(
                    repo_root=repo_root,
                    family=family,
                    method=method,
                    output_root=output_root,
                    route_root=route_root,
                    validation_route_root=validation_route_root,
                    device=args.device,
                    top_contexts=args.top_contexts,
                    knn_neighbors=args.knn_neighbors,
                    naive_bayes_alpha=args.naive_bayes_alpha,
                    hidden_dim=args.hidden_dim,
                    message_passing_steps=args.message_passing_steps,
                    dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    seed=args.seed,
                    route_weights=route_weights,
                    resume=args.resume,
                )
            )
    all_records = _load_completed_records(output_root)
    write_summary(output_root, all_records)
    _write_json(
        output_root / 'run_config.json',
        {
            'arguments': vars(args),
            'families': families,
            'methods': methods,
            'num_completed_runs': len(all_records),
            'num_runs_this_invocation': len(records),
        },
    )
    print(f'[direct-product-baseline] completed {len(records)} runs; summary contains {len(all_records)} runs in {output_root}', flush=True)


if __name__ == '__main__':
    main()
