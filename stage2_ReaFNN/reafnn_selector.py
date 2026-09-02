"""ReaFNN selector used after a widened KNN pool in Stage 2.

The model predicts reagent-token and solvent-token probabilities from route
features (`product_fp + delta_fp` plus light descriptors), then re-scores
historically seen contexts and optionally proposes extra historical
combinations consistent with the top predicted tokens.
"""

from __future__ import annotations

from itertools import combinations
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
import torch.nn as nn

from prosys_shared.condition_modeling import (
    AggregatedReactionExample,
    ContextRecord,
    aggregate_reaction_examples,
    build_context_library,
    build_token_vocab,
    load_condition_rows,
    multi_hot_from_tokens,
    route_feature_vector,
    split_condition_tokens,
)
MODEL_FILE_NAME = 'reafnn_model.pt'
METADATA_FILE_NAME = 'reafnn_meta.json'


@dataclass(frozen=True)
class ReaFNNConfig:
    fpsize: int = 4096
    radius: int = 2
    hidden_dim: int = 384
    hidden_layers: int = 2
    dropout: float = 0.10
    activation: str = 'relu'
    use_layer_norm: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 5
    top_reagent_tokens: int = 12
    top_solvent_tokens: int = 8
    max_generated_contexts: int = 48
    max_novel_contexts: int = 8
    max_selected_novel_contexts: int = 1
    max_reagent_tokens_per_context: int = 3
    max_solvent_tokens_per_context: int = 2
    max_reagent_combinations: int = 16
    max_solvent_combinations: int = 8
    generation_min_context_support: float = 0.0
    min_token_probability: float = 0.05
    token_prior_weight: float = 0.05
    combination_size_penalty: float = 0.02
    historical_context_bonus: float = 0.40
    historical_support_weight: float = 0.15
    novel_context_penalty: float = 0.60
    # Keep the production path KNN-anchored: ReaFNN calibrates retrieved
    # contexts by default and expansion remains an explicit experiment.
    knn_anchor_contexts: int = 0
    correction_weight: float = 0.20
    correction_clip: float = 0.10
    enable_context_augmentation: bool = False
    enable_knn_wide_refinement: bool = False
    # Maintained policy: score the full train-only historical context library
    # independently, then fuse its top contexts with KNN.
    enable_independent_post_fusion: bool = True
    independent_contexts: int = 64
    post_fusion_weight_grid: str = '0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0'
    post_fusion_validation_source: str = 'reference_split_routes'
    device: str = 'cpu'
    random_state: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureStandardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    @classmethod
    def fit(cls, features: np.ndarray) -> 'FeatureStandardizer':
        mean = np.mean(features, axis=0).astype(np.float32)
        std = np.std(features, axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean, std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32, copy=False)

    def state_dict(self) -> dict:
        return {
            'mean': self.mean,
            'std': self.std,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> 'FeatureStandardizer':
        return cls(
            mean=np.asarray(state['mean'], dtype=np.float32),
            std=np.asarray(state['std'], dtype=np.float32),
        )


def _activation_layer(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == 'relu':
        return nn.ReLU()
    if normalized == 'gelu':
        return nn.GELU()
    raise ValueError(f'Unsupported ReaFNN activation: {name!r}')


class ReaFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_reagent_tokens: int,
        num_solvent_tokens: int,
        *,
        hidden_dim: int,
        hidden_layers: int,
        dropout: float,
        activation: str,
        use_layer_norm: bool,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(1, hidden_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.extend([
                _activation_layer(activation),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.reagent_head = nn.Linear(hidden_dim, num_reagent_tokens)
        self.solvent_head = nn.Linear(hidden_dim, num_solvent_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.reagent_head(hidden), self.solvent_head(hidden)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _vectorize_examples(
    examples: list[AggregatedReactionExample],
    *,
    fpsize: int,
    radius: int,
) -> np.ndarray:
    cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[np.ndarray] = []
    for example in examples:
        key = (example.reactants, example.product)
        vec = cache.get(key)
        if vec is None:
            vec = route_feature_vector(example.reactants, example.product, fpsize=fpsize, radius=radius)
            cache[key] = vec
        rows.append(vec)
    return np.stack(rows).astype(np.float32, copy=False)


def _label_matrix(
    examples: list[AggregatedReactionExample],
    *,
    token_to_index: dict[str, int],
    field: str,
) -> np.ndarray:
    return np.stack(
        [multi_hot_from_tokens(getattr(example, field), token_to_index) for example in examples],
        axis=0,
    ).astype(np.float32, copy=False)


def _pos_weight(targets: np.ndarray) -> np.ndarray:
    positive = np.sum(targets, axis=0, dtype=np.float64)
    total = float(targets.shape[0])
    negative = np.maximum(total - positive, 1.0)
    positive = np.maximum(positive, 1.0)
    weights = negative / positive
    weights = np.clip(weights, 1.0, 20.0)
    return weights.astype(np.float32)


@torch.no_grad()
def _evaluate_loss(
    model: ReaFNN,
    x: torch.Tensor,
    y_reagent: torch.Tensor,
    y_solvent: torch.Tensor,
    *,
    reagent_loss_fn: nn.Module,
    solvent_loss_fn: nn.Module,
) -> float:
    model.eval()
    reagent_logits, solvent_logits = model(x)
    loss = reagent_loss_fn(reagent_logits, y_reagent) + solvent_loss_fn(solvent_logits, y_solvent)
    return float(loss.detach().cpu())


def train_reafnn_selector(
    train_split_file: str | Path,
    val_split_file: str | Path,
    output_dir: str | Path,
    *,
    config: ReaFNNConfig | None = None,
    force_retrain: bool = False,
) -> dict:
    config = config or ReaFNNConfig()
    output_dir = Path(output_dir)
    model_file = output_dir / MODEL_FILE_NAME
    metadata_file = output_dir / METADATA_FILE_NAME
    if model_file.exists() and metadata_file.exists() and not force_retrain:
        return json.loads(metadata_file.read_text(encoding='utf-8'))

    _set_seed(config.random_state)

    train_rows = load_condition_rows(train_split_file)
    val_rows = load_condition_rows(val_split_file)
    train_examples = aggregate_reaction_examples(train_rows)
    val_examples = aggregate_reaction_examples(val_rows)

    reagent_vocab, reagent_to_index = build_token_vocab(train_examples, field='reagent_tokens')
    solvent_vocab, solvent_to_index = build_token_vocab(train_examples, field='solvent_tokens')
    if not reagent_vocab or not solvent_vocab:
        raise ValueError('ReaFNN requires non-empty reagent and solvent vocabularies.')

    x_train = _vectorize_examples(train_examples, fpsize=config.fpsize, radius=config.radius)
    x_val = _vectorize_examples(val_examples, fpsize=config.fpsize, radius=config.radius)
    scaler = FeatureStandardizer.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    y_reagent_train = _label_matrix(train_examples, token_to_index=reagent_to_index, field='reagent_tokens')
    y_solvent_train = _label_matrix(train_examples, token_to_index=solvent_to_index, field='solvent_tokens')
    y_reagent_val = _label_matrix(val_examples, token_to_index=reagent_to_index, field='reagent_tokens')
    y_solvent_val = _label_matrix(val_examples, token_to_index=solvent_to_index, field='solvent_tokens')

    device = torch.device(config.device if config.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = ReaFNN(
        input_dim=int(x_train.shape[1]),
        num_reagent_tokens=len(reagent_vocab),
        num_solvent_tokens=len(solvent_vocab),
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    reagent_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(_pos_weight(y_reagent_train), device=device)
    )
    solvent_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(_pos_weight(y_solvent_train), device=device)
    )

    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_reagent_train_t = torch.as_tensor(y_reagent_train, dtype=torch.float32, device=device)
    y_solvent_train_t = torch.as_tensor(y_solvent_train, dtype=torch.float32, device=device)
    y_reagent_val_t = torch.as_tensor(y_reagent_val, dtype=torch.float32, device=device)
    y_solvent_val_t = torch.as_tensor(y_solvent_val, dtype=torch.float32, device=device)

    best_val = float('inf')
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0

    for _epoch in range(config.max_epochs):
        model.train()
        permutation = torch.randperm(x_train_t.shape[0], device=device)
        for start_index in range(0, int(permutation.shape[0]), config.batch_size):
            batch_index = permutation[start_index:start_index + config.batch_size]
            reagent_logits, solvent_logits = model(x_train_t[batch_index])
            loss = (
                reagent_loss_fn(reagent_logits, y_reagent_train_t[batch_index])
                + solvent_loss_fn(solvent_logits, y_solvent_train_t[batch_index])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_loss = _evaluate_loss(
            model,
            x_val_t,
            y_reagent_val_t,
            y_solvent_val_t,
            reagent_loss_fn=reagent_loss_fn,
            solvent_loss_fn=solvent_loss_fn,
        )
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    payload = {
        'model_state': best_state,
        'input_dim': int(x_train.shape[1]),
        'reagent_vocab': reagent_vocab,
        'solvent_vocab': solvent_vocab,
        'scaler': scaler.state_dict(),
        'config': config.to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_file)

    metadata = {
        'model_file': model_file.name,
        'input_dim': int(x_train.shape[1]),
        'num_train_examples': int(len(train_examples)),
        'num_val_examples': int(len(val_examples)),
        'num_reagent_tokens': int(len(reagent_vocab)),
        'num_solvent_tokens': int(len(solvent_vocab)),
        'best_val_loss': float(best_val),
        'config': config.to_dict(),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return metadata


class ReaFNNSelector:
    def __init__(
        self,
        artifact_dir: str | Path,
        context_library: list[ContextRecord],
        *,
        device: str = 'cpu',
    ):
        artifact_dir = Path(artifact_dir)
        payload = torch.load(artifact_dir / MODEL_FILE_NAME, map_location='cpu', weights_only=False)
        config_payload = dict(payload['config'])
        for legacy_field in (
            'enable_route_validity_head',
            'route_negative_per_sample',
            'route_validity_loss_weight',
        ):
            config_payload.pop(legacy_field, None)
        config = ReaFNNConfig(**config_payload)
        self.config = ReaFNNConfig(**{**config.to_dict(), 'device': device})
        self.scaler = FeatureStandardizer.from_state_dict(payload['scaler'])
        self.reagent_vocab = list(payload['reagent_vocab'])
        self.solvent_vocab = list(payload['solvent_vocab'])
        self.reagent_to_index = {token: idx for idx, token in enumerate(self.reagent_vocab)}
        self.solvent_to_index = {token: idx for idx, token in enumerate(self.solvent_vocab)}
        self.context_library = list(context_library)
        self.context_by_key = {
            (context.reagent_norm, context.solvent_norm): context
            for context in self.context_library
        }
        self._context_library_index = {
            (context.reagent_norm, context.solvent_norm): index
            for index, context in enumerate(self.context_library)
        }

        self.device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.model = ReaFNN(
            input_dim=int(payload['input_dim']),
            num_reagent_tokens=len(self.reagent_vocab),
            num_solvent_tokens=len(self.solvent_vocab),
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
            dropout=self.config.dropout,
            activation=self.config.activation,
            use_layer_norm=self.config.use_layer_norm,
        ).to(self.device)
        model_state = dict(payload['model_state'])
        model_state.pop('route_validity_head.weight', None)
        model_state.pop('route_validity_head.bias', None)
        self.model.load_state_dict(model_state)
        self.model.eval()
        self._feature_cache: dict[tuple[str, str], np.ndarray] = {}
        self.reagent_token_prior = self._build_token_prior('reagent_tokens')
        self.solvent_token_prior = self._build_token_prior('solvent_tokens')
        (
            self._library_reagent_weights,
            self._library_solvent_weights,
            self._library_prior_scores,
            self._library_size_penalties,
        ) = self._build_library_score_matrices()
        self._library_reagent_labels = np.asarray(
            [context.reagent_norm for context in self.context_library],
            dtype=str,
        )
        self._library_solvent_labels = np.asarray(
            [context.solvent_norm for context in self.context_library],
            dtype=str,
        )

    def _build_token_prior(self, field: str) -> dict[str, float]:
        weights: dict[str, float] = {}
        for context in self.context_library:
            for token in getattr(context, field):
                weights[token] = weights.get(token, 0.0) + float(context.context_count)
        if not weights:
            return {}
        scale = max(weights.values()) or 1.0
        return {
            token: float(weight) / float(scale)
            for token, weight in weights.items()
        }

    def _build_library_score_matrices(
        self,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
        """Precompute sparse train-library token averages for bulk rescoring."""

        library_size = len(self.context_library)
        reagent_rows: list[int] = []
        reagent_columns: list[int] = []
        reagent_values: list[float] = []
        solvent_rows: list[int] = []
        solvent_columns: list[int] = []
        solvent_values: list[float] = []
        prior_scores = np.zeros((library_size,), dtype=np.float32)
        size_penalties = np.zeros((library_size,), dtype=np.float32)

        for row_index, context in enumerate(self.context_library):
            reagent_indices = [
                self.reagent_to_index[token]
                for token in context.reagent_tokens
                if token in self.reagent_to_index
            ]
            solvent_indices = [
                self.solvent_to_index[token]
                for token in context.solvent_tokens
                if token in self.solvent_to_index
            ]
            if reagent_indices:
                reagent_rows.extend([row_index] * len(reagent_indices))
                reagent_columns.extend(reagent_indices)
                reagent_values.extend([1.0 / float(len(reagent_indices))] * len(reagent_indices))
            if solvent_indices:
                solvent_rows.extend([row_index] * len(solvent_indices))
                solvent_columns.extend(solvent_indices)
                solvent_values.extend([1.0 / float(len(solvent_indices))] * len(solvent_indices))

            reagent_prior = self._token_prior_score(
                context.reagent_tokens,
                self.reagent_token_prior,
            )
            solvent_prior = self._token_prior_score(
                context.solvent_tokens,
                self.solvent_token_prior,
            )
            prior_scores[row_index] = np.float32((reagent_prior + solvent_prior) / 2.0)
            size_penalties[row_index] = np.float32(
                self.config.combination_size_penalty
                * float(
                    max(len(context.reagent_tokens) - 1, 0)
                    + max(len(context.solvent_tokens) - 1, 0)
                )
            )

        reagent_weights = sparse.csr_matrix(
            (
                np.asarray(reagent_values, dtype=np.float32),
                (
                    np.asarray(reagent_rows, dtype=np.int32),
                    np.asarray(reagent_columns, dtype=np.int32),
                ),
            ),
            shape=(library_size, len(self.reagent_vocab)),
            dtype=np.float32,
        )
        solvent_weights = sparse.csr_matrix(
            (
                np.asarray(solvent_values, dtype=np.float32),
                (
                    np.asarray(solvent_rows, dtype=np.int32),
                    np.asarray(solvent_columns, dtype=np.int32),
                ),
            ),
            shape=(library_size, len(self.solvent_vocab)),
            dtype=np.float32,
        )
        return reagent_weights, solvent_weights, prior_scores, size_penalties

    def _route_feature(self, reactants: str, product: str) -> np.ndarray:
        key = (reactants, product)
        vec = self._feature_cache.get(key)
        if vec is None:
            vec = route_feature_vector(reactants, product, fpsize=self.config.fpsize, radius=self.config.radius)
            self._feature_cache[key] = vec
        return vec

    @torch.no_grad()
    def predict_token_probabilities(self, reactants: str, product: str) -> tuple[np.ndarray, np.ndarray]:
        vec = self.scaler.transform(self._route_feature(reactants, product)[None, :])
        x = torch.as_tensor(vec, dtype=torch.float32, device=self.device)
        reagent_logits, solvent_logits = self.model(x)
        reagent_probs = torch.sigmoid(reagent_logits).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        solvent_probs = torch.sigmoid(solvent_logits).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        return reagent_probs, solvent_probs

    def _token_score(
        self,
        tokens: tuple[str, ...],
        probs: np.ndarray,
        token_to_index: dict[str, int],
    ) -> float:
        scores: list[float] = []
        for token in tokens:
            index = token_to_index.get(token)
            if index is None:
                continue
            scores.append(float(probs[index]))
        if not scores:
            return 0.0
        return float(np.mean(np.asarray(scores, dtype=np.float32)))

    def _token_prior_score(
        self,
        tokens: tuple[str, ...],
        token_prior: dict[str, float],
    ) -> float:
        if not tokens:
            return 0.0
        values = [float(token_prior.get(token, 0.0)) for token in tokens]
        if not values:
            return 0.0
        return float(np.mean(np.asarray(values, dtype=np.float32)))

    def _compose_context_score(
        self,
        *,
        reagent_score: float,
        solvent_score: float,
        reagent_tokens: tuple[str, ...],
        solvent_tokens: tuple[str, ...],
        context_support: float,
        is_historical: bool,
    ) -> dict[str, float]:
        token_score = float(np.mean(np.asarray([reagent_score, solvent_score], dtype=np.float32)))
        prior_score = float(
            np.mean(
                np.asarray(
                    [
                        self._token_prior_score(reagent_tokens, self.reagent_token_prior),
                        self._token_prior_score(solvent_tokens, self.solvent_token_prior),
                    ],
                    dtype=np.float32,
                )
            )
        )
        size_penalty = self.config.combination_size_penalty * float(
            max(len(reagent_tokens) - 1, 0) + max(len(solvent_tokens) - 1, 0)
        )
        historical_bonus = self.config.historical_context_bonus if is_historical else 0.0
        support_bonus = self.config.historical_support_weight * float(context_support) if is_historical else 0.0
        novelty_penalty = 0.0 if is_historical else self.config.novel_context_penalty
        context_score = (
            token_score
            + (self.config.token_prior_weight * prior_score)
            + historical_bonus
            + support_bonus
            - novelty_penalty
            - size_penalty
        )
        return {
            'reafnn_token_score': token_score,
            'reafnn_prior_score': prior_score,
            'reafnn_historical_bonus': historical_bonus + support_bonus,
            'reafnn_novelty_penalty': novelty_penalty,
            'reafnn_context_score': context_score,
        }

    def score_context(
        self,
        reagent_norm: str,
        solvent_norm: str,
        reagent_probs: np.ndarray,
        solvent_probs: np.ndarray,
        *,
        is_historical: bool = True,
        context_support: float = 0.0,
    ) -> dict[str, float]:
        reagent_tokens = split_condition_tokens(reagent_norm)
        solvent_tokens = split_condition_tokens(solvent_norm)
        reagent_score = self._token_score(
            reagent_tokens,
            reagent_probs,
            self.reagent_to_index,
        )
        solvent_score = self._token_score(
            solvent_tokens,
            solvent_probs,
            self.solvent_to_index,
        )
        score_fields = self._compose_context_score(
            reagent_score=reagent_score,
            solvent_score=solvent_score,
            reagent_tokens=reagent_tokens,
            solvent_tokens=solvent_tokens,
            context_support=context_support,
            is_historical=is_historical,
        )
        return {
            'reafnn_reagent_score': reagent_score,
            'reafnn_solvent_score': solvent_score,
            **score_fields,
        }

    def _selected_tokens(
        self,
        probs: np.ndarray,
        vocab: list[str],
        *,
        top_k: int,
    ) -> list[str]:
        if probs.size == 0:
            return []
        order = np.argsort(probs)[::-1]
        chosen: list[str] = []
        for index in order[:top_k]:
            if float(probs[index]) < self.config.min_token_probability and chosen:
                continue
            token = vocab[int(index)]
            if token in chosen:
                continue
            chosen.append(token)
        return chosen

    def _rank_token_combinations(
        self,
        selected_tokens: list[str],
        probs: np.ndarray,
        token_to_index: dict[str, int],
        token_prior: dict[str, float],
        *,
        max_tokens_per_context: int,
        max_combinations: int,
    ) -> list[dict]:
        rows: list[dict] = []
        if not selected_tokens:
            return rows

        capped_tokens = list(selected_tokens)
        upper = min(len(capped_tokens), max(1, int(max_tokens_per_context)))
        for size in range(1, upper + 1):
            for combo in combinations(capped_tokens, size):
                tokens = tuple(sorted(combo))
                token_score = self._token_score(tokens, probs, token_to_index)
                prior_score = self._token_prior_score(tokens, token_prior)
                size_penalty = self.config.combination_size_penalty * float(max(len(tokens) - 1, 0))
                combo_score = token_score + (self.config.token_prior_weight * prior_score) - size_penalty
                rows.append(
                    {
                        'tokens': tokens,
                        'token_score': token_score,
                        'prior_score': prior_score,
                        'combo_score': combo_score,
                    }
                )

        rows.sort(
            key=lambda row: (
                -float(row['combo_score']),
                -float(row['token_score']),
                len(row['tokens']),
                row['tokens'],
            )
        )
        return rows[:max_combinations]

    def generate_historical_contexts(
        self,
        reactants: str,
        product: str,
        *,
        existing_contexts: set[tuple[str, str]],
    ) -> list[dict]:
        reagent_probs, solvent_probs = self.predict_token_probabilities(reactants, product)
        reagent_tokens = self._selected_tokens(
            reagent_probs,
            self.reagent_vocab,
            top_k=self.config.top_reagent_tokens,
        )
        solvent_tokens = self._selected_tokens(
            solvent_probs,
            self.solvent_vocab,
            top_k=self.config.top_solvent_tokens,
        )
        reagent_set = set(reagent_tokens)
        solvent_set = set(solvent_tokens)

        rows: list[dict] = []
        for context in self.context_library:
            key = (context.reagent_norm, context.solvent_norm)
            if key in existing_contexts:
                continue
            if context.context_support < self.config.generation_min_context_support:
                continue
            if context.reagent_tokens and not set(context.reagent_tokens).issubset(reagent_set):
                continue
            if context.solvent_tokens and not set(context.solvent_tokens).issubset(solvent_set):
                continue

            score_fields = self.score_context(
                context.reagent_norm,
                context.solvent_norm,
                reagent_probs,
                solvent_probs,
                is_historical=True,
                context_support=float(context.context_support),
            )
            rows.append(
                {
                    'reagent_norm': context.reagent_norm,
                    'solvent_norm': context.solvent_norm,
                    **score_fields,
                    'reafnn_context_count': float(context.context_count),
                    'reafnn_context_support': float(context.context_support),
                    'reafnn_mean_yield': float(context.mean_yield),
                    'from_reafnn_generated': 1,
                    'from_reafnn_novel': 0,
                    'reafnn_is_historical': 1,
                }
            )

        rows.sort(
            key=lambda row: (
                -float(row['reafnn_context_score']),
                -float(row['reafnn_context_support']),
                -float(row['reafnn_context_count']),
                row['reagent_norm'],
                row['solvent_norm'],
            )
        )
        return rows[: self.config.max_generated_contexts]

    def generate_novel_contexts(
        self,
        reactants: str,
        product: str,
        *,
        existing_contexts: set[tuple[str, str]],
    ) -> list[dict]:
        reagent_probs, solvent_probs = self.predict_token_probabilities(reactants, product)
        reagent_selected = self._selected_tokens(
            reagent_probs,
            self.reagent_vocab,
            top_k=self.config.top_reagent_tokens,
        )
        solvent_selected = self._selected_tokens(
            solvent_probs,
            self.solvent_vocab,
            top_k=self.config.top_solvent_tokens,
        )
        reagent_groups = self._rank_token_combinations(
            reagent_selected,
            reagent_probs,
            self.reagent_to_index,
            self.reagent_token_prior,
            max_tokens_per_context=self.config.max_reagent_tokens_per_context,
            max_combinations=self.config.max_reagent_combinations,
        )
        solvent_groups = self._rank_token_combinations(
            solvent_selected,
            solvent_probs,
            self.solvent_to_index,
            self.solvent_token_prior,
            max_tokens_per_context=self.config.max_solvent_tokens_per_context,
            max_combinations=self.config.max_solvent_combinations,
        )

        rows: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for reagent_group in reagent_groups:
            for solvent_group in solvent_groups:
                reagent_norm = split_condition_tokens('; '.join(reagent_group['tokens']))
                solvent_norm = split_condition_tokens('; '.join(solvent_group['tokens']))
                if not reagent_norm or not solvent_norm:
                    continue
                reagent_label = '; '.join(reagent_norm)
                solvent_label = '; '.join(solvent_norm)
                key = (reagent_label, solvent_label)
                if key in seen or key in existing_contexts or key in self.context_by_key:
                    continue
                seen.add(key)

                score_fields = self.score_context(
                    reagent_label,
                    solvent_label,
                    reagent_probs,
                    solvent_probs,
                    is_historical=False,
                    context_support=0.0,
                )
                rows.append(
                    {
                        'reagent_norm': reagent_label,
                        'solvent_norm': solvent_label,
                        **score_fields,
                        'reafnn_context_count': 0.0,
                        'reafnn_context_support': 0.0,
                        'reafnn_mean_yield': 0.0,
                        'from_reafnn_generated': 1,
                        'from_reafnn_novel': 1,
                        'reafnn_is_historical': 0,
                    }
                )

        rows.sort(
            key=lambda row: (
                -float(row['reafnn_context_score']),
                -float(row['reafnn_token_score']),
                row['reagent_norm'],
                row['solvent_norm'],
            )
        )
        return rows[: self.config.max_novel_contexts]

    def generate_contexts(
        self,
        reactants: str,
        product: str,
        *,
        existing_contexts: set[tuple[str, str]],
    ) -> list[dict]:
        historical = self.generate_historical_contexts(
            reactants,
            product,
            existing_contexts=existing_contexts,
        )
        merged_keys = set(existing_contexts)
        merged_keys.update((str(row['reagent_norm']), str(row['solvent_norm'])) for row in historical)
        novel = self.generate_novel_contexts(
            reactants,
            product,
            existing_contexts=merged_keys,
        )
        rows = historical + novel
        rows.sort(
            key=lambda row: (
                -float(row['reafnn_context_score']),
                -int(row.get('reafnn_is_historical', 0)),
                -float(row.get('reafnn_context_support', 0.0)),
                row['reagent_norm'],
                row['solvent_norm'],
            )
        )
        return rows[: self.config.max_generated_contexts]

    def _rescore_contexts_scalar(
        self,
        reactants: str,
        product: str,
        contexts: list[dict],
    ) -> list[dict]:
        reagent_probs, solvent_probs = self.predict_token_probabilities(reactants, product)
        scored: list[dict] = []
        for context in contexts:
            is_historical = int(context.get('reafnn_is_historical', 1 if context.get('from_reafnn_novel', 0) == 0 else 0)) == 1
            context_support = float(context.get('reafnn_context_support', context.get('context_support', 0.0)))
            score_fields = self.score_context(
                str(context['reagent_norm']),
                str(context['solvent_norm']),
                reagent_probs,
                solvent_probs,
                is_historical=is_historical,
                context_support=context_support,
            )
            row = dict(context)
            row.update(score_fields)
            row['reafnn_context_count'] = float(context.get('reafnn_context_count', context.get('context_count', 0.0)))
            row['reafnn_context_support'] = context_support
            row['reafnn_mean_yield'] = float(context.get('reafnn_mean_yield', context.get('mean_yield', 0.0)))
            row['from_reafnn_generated'] = int(context.get('from_reafnn_generated', 0))
            row['from_reafnn_novel'] = int(context.get('from_reafnn_novel', 0))
            row['reafnn_is_historical'] = 0 if row['from_reafnn_novel'] else 1
            scored.append(row)
        return scored

    def _library_indices_for_contexts(self, contexts: list[dict]) -> np.ndarray | None:
        indices: list[int] = []
        for context in contexts:
            context_index = self._context_library_index.get(
                (str(context['reagent_norm']), str(context['solvent_norm']))
            )
            if context_index is None:
                return None
            indices.append(context_index)
        return np.asarray(indices, dtype=np.int64)

    def _known_library_score_arrays(
        self,
        *,
        contexts: list[dict],
        library_indices: np.ndarray,
        reagent_probs: np.ndarray,
        solvent_probs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        reagent_scores = np.asarray(
            self._library_reagent_weights[library_indices] @ reagent_probs,
            dtype=np.float32,
        ).reshape(-1)
        solvent_scores = np.asarray(
            self._library_solvent_weights[library_indices] @ solvent_probs,
            dtype=np.float32,
        ).reshape(-1)
        token_scores = np.mean(
            np.stack((reagent_scores, solvent_scores), axis=1),
            axis=1,
            dtype=np.float32,
        ).astype(np.float32, copy=False)
        context_supports = np.asarray(
            [
                float(context.get('reafnn_context_support', context.get('context_support', 0.0)))
                for context in contexts
            ],
            dtype=np.float32,
        )
        is_historical = np.asarray(
            [
                int(
                    context.get(
                        'reafnn_is_historical',
                        1 if context.get('from_reafnn_novel', 0) == 0 else 0,
                    )
                ) == 1
                for context in contexts
            ],
            dtype=bool,
        )
        config = self.config
        historical_bonus = np.where(
            is_historical,
            np.float32(config.historical_context_bonus)
            + (np.float32(config.historical_support_weight) * context_supports),
            np.float32(0.0),
        ).astype(np.float32, copy=False)
        novelty_penalty = np.where(
            is_historical,
            np.float32(0.0),
            np.float32(config.novel_context_penalty),
        ).astype(np.float32, copy=False)
        prior_scores = self._library_prior_scores[library_indices]
        size_penalties = self._library_size_penalties[library_indices]
        context_scores = (
            token_scores
            + (np.float32(config.token_prior_weight) * prior_scores)
            + historical_bonus
            - novelty_penalty
            - size_penalties
        ).astype(np.float32, copy=False)
        return {
            'reagent': reagent_scores,
            'solvent': solvent_scores,
            'token': token_scores,
            'prior': prior_scores,
            'historical_bonus': historical_bonus,
            'novelty_penalty': novelty_penalty,
            'context': context_scores,
            'support': context_supports,
        }

    def _materialize_scored_context_row(
        self,
        context: dict,
        *,
        index: int,
        score_arrays: dict[str, np.ndarray],
    ) -> dict:
        row = dict(context)
        row.update({
            'reafnn_reagent_score': float(score_arrays['reagent'][index]),
            'reafnn_solvent_score': float(score_arrays['solvent'][index]),
            'reafnn_token_score': float(score_arrays['token'][index]),
            'reafnn_prior_score': float(score_arrays['prior'][index]),
            'reafnn_historical_bonus': float(score_arrays['historical_bonus'][index]),
            'reafnn_novelty_penalty': float(score_arrays['novelty_penalty'][index]),
            'reafnn_context_score': float(score_arrays['context'][index]),
        })
        row['reafnn_context_count'] = float(
            context.get('reafnn_context_count', context.get('context_count', 0.0))
        )
        row['reafnn_context_support'] = float(score_arrays['support'][index])
        row['reafnn_mean_yield'] = float(
            context.get('reafnn_mean_yield', context.get('mean_yield', 0.0))
        )
        row['from_reafnn_generated'] = int(context.get('from_reafnn_generated', 0))
        row['from_reafnn_novel'] = int(context.get('from_reafnn_novel', 0))
        row['reafnn_is_historical'] = 0 if row['from_reafnn_novel'] else 1
        return row

    def _rescore_known_library_contexts(
        self,
        *,
        contexts: list[dict],
        library_indices: np.ndarray,
        reagent_probs: np.ndarray,
        solvent_probs: np.ndarray,
    ) -> list[dict]:
        """Score known train-library contexts with vectorized token averages."""

        score_arrays = self._known_library_score_arrays(
            contexts=contexts,
            library_indices=library_indices,
            reagent_probs=reagent_probs,
            solvent_probs=solvent_probs,
        )
        return [
            self._materialize_scored_context_row(
                context,
                index=index,
                score_arrays=score_arrays,
            )
            for index, context in enumerate(contexts)
        ]

    def select_historical_contexts(
        self,
        reactants: str,
        product: str,
        contexts: list[dict],
        *,
        lookup_keys: set[tuple[str, str]],
        top_k: int,
    ) -> tuple[dict[tuple[str, str], dict], list[dict]]:
        """Return ReaFNN's top contexts plus scores needed for KNN overlap."""

        if not contexts:
            return {}, []
        library_indices = self._library_indices_for_contexts(contexts)
        if library_indices is None:
            scored = self._rescore_contexts_scalar(reactants, product, contexts)
            scored.sort(
                key=lambda row: (
                    -float(row.get('reafnn_context_score', 0.0)),
                    -float(row.get('reafnn_token_score', 0.0)),
                    -float(row.get('reafnn_context_support', 0.0)),
                    str(row.get('reagent_norm', '')),
                    str(row.get('solvent_norm', '')),
                )
            )
            lookup = {
                (str(row['reagent_norm']), str(row['solvent_norm'])): row
                for row in scored
                if (str(row['reagent_norm']), str(row['solvent_norm'])) in lookup_keys
            }
            return lookup, scored[:max(0, int(top_k))]

        reagent_probs, solvent_probs = self.predict_token_probabilities(reactants, product)
        score_arrays = self._known_library_score_arrays(
            contexts=contexts,
            library_indices=library_indices,
            reagent_probs=reagent_probs,
            solvent_probs=solvent_probs,
        )
        order = np.lexsort((
            self._library_solvent_labels[library_indices],
            self._library_reagent_labels[library_indices],
            -score_arrays['support'],
            -score_arrays['token'],
            -score_arrays['context'],
        ))
        top_positions = [int(index) for index in order[:max(0, int(top_k))]]
        rows_by_position: dict[int, dict] = {}

        def row_at(position: int) -> dict:
            row = rows_by_position.get(position)
            if row is None:
                row = self._materialize_scored_context_row(
                    contexts[position],
                    index=position,
                    score_arrays=score_arrays,
                )
                rows_by_position[position] = row
            return row

        selected = [row_at(position) for position in top_positions]
        lookup: dict[tuple[str, str], dict] = {}
        for position, context in enumerate(contexts):
            key = (str(context['reagent_norm']), str(context['solvent_norm']))
            if key in lookup_keys:
                lookup[key] = row_at(position)
        return lookup, selected

    def rescore_contexts(
        self,
        reactants: str,
        product: str,
        contexts: list[dict],
    ) -> list[dict]:
        """Rescore a context list, using the vector path for known history."""

        if not contexts:
            return []
        library_indices = self._library_indices_for_contexts(contexts)
        if library_indices is None:
            return self._rescore_contexts_scalar(reactants, product, contexts)
        reagent_probs, solvent_probs = self.predict_token_probabilities(reactants, product)
        return self._rescore_known_library_contexts(
            contexts=contexts,
            library_indices=library_indices,
            reagent_probs=reagent_probs,
            solvent_probs=solvent_probs,
        )



def build_default_context_library(split_file: str | Path) -> list[ContextRecord]:
    return build_context_library(load_condition_rows(split_file))
