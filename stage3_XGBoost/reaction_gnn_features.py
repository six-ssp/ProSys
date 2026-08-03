"""Reaction-GNN feature encoder used to augment Stage 3 XGBoost tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from prosys_shared.condition_modeling import (
    aggregate_reaction_examples,
    build_token_vocab,
    load_condition_rows,
    multi_hot_from_tokens,
    split_condition_tokens,
)
from prosys_shared.features import canonicalize_reaction_side, canonicalize_smiles


MODEL_FILE_NAME = 'reaction_gnn.pt'
METADATA_FILE_NAME = 'reaction_gnn_meta.json'


@dataclass(frozen=True)
class ReactionGNNConfig:
    hidden_dim: int = 64
    embedding_dim: int = 64
    message_passing_steps: int = 3
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 48
    max_epochs: int = 20
    patience: int = 5
    device: str = 'cpu'
    random_state: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _one_hot(value: int, choices: list[int]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _hybridization_code(atom: Chem.Atom) -> int:
    mapping = {
        Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2,
        Chem.HybridizationType.SP3D: 3,
        Chem.HybridizationType.SP3D2: 4,
    }
    return mapping.get(atom.GetHybridization(), 5)


def _atom_features(atom: Chem.Atom) -> list[float]:
    atomic_choices = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
    degree_choices = [0, 1, 2, 3, 4, 5]
    charge = float(atom.GetFormalCharge())
    return [
        *_one_hot(atom.GetAtomicNum(), atomic_choices),
        float(atom.GetAtomicNum() not in atomic_choices),
        *_one_hot(int(atom.GetDegree()), degree_choices),
        float(charge),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetTotalNumHs(includeNeighbors=True)) / 4.0,
        float(_hybridization_code(atom)) / 5.0,
        float(atom.GetMass()) / 200.0,
    ]


ATOM_FEATURE_DIM = len(_atom_features(Chem.MolFromSmiles('CC').GetAtomWithIdx(0)))


def _mol_from_smiles(smiles: str, *, reaction_side: bool) -> Chem.Mol | None:
    normalized = canonicalize_reaction_side(smiles) if reaction_side else canonicalize_smiles(smiles)
    if not normalized:
        return None
    return Chem.MolFromSmiles(normalized)


def _graph_from_smiles(smiles: str, *, reaction_side: bool) -> tuple[np.ndarray, np.ndarray]:
    mol = _mol_from_smiles(smiles, reaction_side=reaction_side)
    if mol is None or mol.GetNumAtoms() == 0:
        return (
            np.zeros((1, ATOM_FEATURE_DIM), dtype=np.float32),
            np.zeros((2, 0), dtype=np.int64),
        )

    node_features = np.asarray([_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        edges.append((begin, end))
        edges.append((end, begin))
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return node_features, edge_index


def _batch_graphs(graphs: list[tuple[np.ndarray, np.ndarray]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_blocks: list[np.ndarray] = []
    edge_blocks: list[np.ndarray] = []
    graph_index: list[np.ndarray] = []
    node_offset = 0
    for graph_id, (node_features, edge_index) in enumerate(graphs):
        node_blocks.append(node_features)
        graph_index.append(np.full((node_features.shape[0],), graph_id, dtype=np.int64))
        if edge_index.size > 0:
            edge_blocks.append(edge_index + node_offset)
        node_offset += int(node_features.shape[0])

    nodes = torch.as_tensor(np.concatenate(node_blocks, axis=0), dtype=torch.float32, device=device)
    if edge_blocks:
        edge_index = torch.as_tensor(np.concatenate(edge_blocks, axis=1), dtype=torch.long, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    graph_ids = torch.as_tensor(np.concatenate(graph_index, axis=0), dtype=torch.long, device=device)
    return nodes, edge_index, graph_ids


class GraphConv(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.relu(self.self_linear(x))
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        return torch.relu(self.self_linear(x) + self.neigh_linear(agg))


class GraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, message_passing_steps: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.layers = nn.ModuleList([GraphConv(hidden_dim) for _ in range(message_passing_steps)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, graph_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_proj(node_features))
        for layer in self.layers:
            hidden = self.dropout(layer(hidden, edge_index))

        num_graphs = int(graph_ids.max().item()) + 1 if graph_ids.numel() else 0
        pooled = torch.zeros((num_graphs, hidden.shape[1]), dtype=hidden.dtype, device=hidden.device)
        pooled.index_add_(0, graph_ids, hidden)
        counts = torch.zeros((num_graphs,), dtype=hidden.dtype, device=hidden.device)
        counts.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=hidden.dtype))
        return pooled / counts.clamp_min(1.0).unsqueeze(-1)


class ReactionGNN(nn.Module):
    def __init__(
        self,
        num_reagent_tokens: int,
        num_solvent_tokens: int,
        *,
        hidden_dim: int,
        embedding_dim: int,
        message_passing_steps: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = GraphEncoder(hidden_dim, message_passing_steps, dropout)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reagent_head = nn.Linear(embedding_dim, num_reagent_tokens)
        self.solvent_head = nn.Linear(embedding_dim, num_solvent_tokens)

    def forward(
        self,
        reactant_graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        product_graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_reactant = self.encoder(*reactant_graph)
        h_product = self.encoder(*product_graph)
        reaction_embedding = self.embedding(torch.cat((h_reactant, h_product, h_product - h_reactant), dim=1))
        return reaction_embedding, self.reagent_head(reaction_embedding), self.solvent_head(reaction_embedding)


def _route_batches(
    examples,
    *,
    batch_size: int,
):
    for start in range(0, len(examples), batch_size):
        yield examples[start:start + batch_size]


def _build_labels(examples, reagent_to_index: dict[str, int], solvent_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    reagent = np.stack([multi_hot_from_tokens(example.reagent_tokens, reagent_to_index) for example in examples], axis=0)
    solvent = np.stack([multi_hot_from_tokens(example.solvent_tokens, solvent_to_index) for example in examples], axis=0)
    return reagent.astype(np.float32, copy=False), solvent.astype(np.float32, copy=False)


def _pos_weight(targets: np.ndarray) -> np.ndarray:
    positive = np.sum(targets, axis=0, dtype=np.float64)
    total = float(targets.shape[0])
    negative = np.maximum(total - positive, 1.0)
    positive = np.maximum(positive, 1.0)
    return np.clip(negative / positive, 1.0, 20.0).astype(np.float32)


def train_reaction_gnn_feature_model(
    train_split_file: str | Path,
    val_split_file: str | Path,
    output_dir: str | Path,
    *,
    config: ReactionGNNConfig | None = None,
    force_retrain: bool = False,
) -> dict:
    config = config or ReactionGNNConfig()
    output_dir = Path(output_dir)
    model_file = output_dir / MODEL_FILE_NAME
    metadata_file = output_dir / METADATA_FILE_NAME
    if model_file.exists() and metadata_file.exists() and not force_retrain:
        return json.loads(metadata_file.read_text(encoding='utf-8'))

    _set_seed(config.random_state)
    train_examples = aggregate_reaction_examples(load_condition_rows(train_split_file))
    val_examples = aggregate_reaction_examples(load_condition_rows(val_split_file))
    reagent_vocab, reagent_to_index = build_token_vocab(train_examples, field='reagent_tokens')
    solvent_vocab, solvent_to_index = build_token_vocab(train_examples, field='solvent_tokens')
    if not reagent_vocab or not solvent_vocab:
        raise ValueError('Reaction-GNN feature model requires non-empty reagent and solvent vocabularies.')

    y_reagent_train, y_solvent_train = _build_labels(train_examples, reagent_to_index, solvent_to_index)
    y_reagent_val, y_solvent_val = _build_labels(val_examples, reagent_to_index, solvent_to_index)

    device = torch.device(config.device if config.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = ReactionGNN(
        num_reagent_tokens=len(reagent_vocab),
        num_solvent_tokens=len(solvent_vocab),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        message_passing_steps=config.message_passing_steps,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    reagent_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(_pos_weight(y_reagent_train), device=device))
    solvent_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(_pos_weight(y_solvent_train), device=device))

    best_val = float('inf')
    best_state = None
    patience = 0

    train_order = list(range(len(train_examples)))
    for _epoch in range(config.max_epochs):
        np.random.shuffle(train_order)
        model.train()
        for start in range(0, len(train_order), config.batch_size):
            batch_examples = [train_examples[index] for index in train_order[start:start + config.batch_size]]
            batch_reagent, batch_solvent = _build_labels(batch_examples, reagent_to_index, solvent_to_index)
            reactant_graph = _batch_graphs(
                [_graph_from_smiles(example.reactants, reaction_side=True) for example in batch_examples],
                device,
            )
            product_graph = _batch_graphs(
                [_graph_from_smiles(example.product, reaction_side=False) for example in batch_examples],
                device,
            )
            _, reagent_logits, solvent_logits = model(reactant_graph, product_graph)
            loss = reagent_loss_fn(
                reagent_logits,
                torch.as_tensor(batch_reagent, dtype=torch.float32, device=device),
            ) + solvent_loss_fn(
                solvent_logits,
                torch.as_tensor(batch_solvent, dtype=torch.float32, device=device),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_examples in _route_batches(val_examples, batch_size=config.batch_size):
                batch_reagent, batch_solvent = _build_labels(batch_examples, reagent_to_index, solvent_to_index)
                reactant_graph = _batch_graphs(
                    [_graph_from_smiles(example.reactants, reaction_side=True) for example in batch_examples],
                    device,
                )
                product_graph = _batch_graphs(
                    [_graph_from_smiles(example.product, reaction_side=False) for example in batch_examples],
                    device,
                )
                _, reagent_logits, solvent_logits = model(reactant_graph, product_graph)
                loss = reagent_loss_fn(
                    reagent_logits,
                    torch.as_tensor(batch_reagent, dtype=torch.float32, device=device),
                ) + solvent_loss_fn(
                    solvent_logits,
                    torch.as_tensor(batch_solvent, dtype=torch.float32, device=device),
                )
                val_losses.append(float(loss.detach().cpu()))

        val_loss = float(np.mean(np.asarray(val_losses, dtype=np.float32))) if val_losses else float('inf')
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
        'reagent_vocab': reagent_vocab,
        'solvent_vocab': solvent_vocab,
        'config': config.to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_file)

    metadata = {
        'model_file': model_file.name,
        'embedding_dim': int(config.embedding_dim),
        'num_train_examples': int(len(train_examples)),
        'num_val_examples': int(len(val_examples)),
        'num_reagent_tokens': int(len(reagent_vocab)),
        'num_solvent_tokens': int(len(solvent_vocab)),
        'best_val_loss': float(best_val),
        'config': config.to_dict(),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return metadata


class ReactionGNNFeatureEncoder:
    def __init__(self, artifact_dir: str | Path, *, device: str = 'cpu'):
        artifact_dir = Path(artifact_dir)
        payload = torch.load(artifact_dir / MODEL_FILE_NAME, map_location='cpu', weights_only=False)
        config = ReactionGNNConfig(**payload['config'])
        self.config = ReactionGNNConfig(**{**config.to_dict(), 'device': device})
        self.device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.model = ReactionGNN(
            num_reagent_tokens=len(payload['reagent_vocab']),
            num_solvent_tokens=len(payload['solvent_vocab']),
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.embedding_dim,
            message_passing_steps=self.config.message_passing_steps,
            dropout=self.config.dropout,
        ).to(self.device)
        self.model.load_state_dict(payload['model_state'])
        self.model.eval()
        self.reagent_vocab = tuple(str(token) for token in payload['reagent_vocab'])
        self.solvent_vocab = tuple(str(token) for token in payload['solvent_vocab'])
        self.reagent_to_index = {token: index for index, token in enumerate(self.reagent_vocab)}
        self.solvent_to_index = {token: index for index, token in enumerate(self.solvent_vocab)}

    @torch.no_grad()
    def predict_routes(self, routes: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return route embeddings and the auxiliary condition-head logits.

        The logits are retained separately from the route embedding so they can
        be used as candidate-specific reagent/solvent compatibility signals.
        """
        if not routes:
            return (
                np.zeros((0, self.config.embedding_dim), dtype=np.float32),
                np.zeros((0, len(self.reagent_vocab)), dtype=np.float32),
                np.zeros((0, len(self.solvent_vocab)), dtype=np.float32),
            )

        embedding_rows: list[np.ndarray] = []
        reagent_rows: list[np.ndarray] = []
        solvent_rows: list[np.ndarray] = []
        for start in range(0, len(routes), self.config.batch_size):
            batch_routes = routes[start:start + self.config.batch_size]
            reactant_graph = _batch_graphs(
                [_graph_from_smiles(reactants, reaction_side=True) for reactants, _product in batch_routes],
                self.device,
            )
            product_graph = _batch_graphs(
                [_graph_from_smiles(product, reaction_side=False) for _reactants, product in batch_routes],
                self.device,
            )
            embedding, reagent_logits, solvent_logits = self.model(reactant_graph, product_graph)
            embedding_rows.append(embedding.detach().cpu().numpy().astype(np.float32, copy=False))
            reagent_rows.append(reagent_logits.detach().cpu().numpy().astype(np.float32, copy=False))
            solvent_rows.append(solvent_logits.detach().cpu().numpy().astype(np.float32, copy=False))
        return (
            np.concatenate(embedding_rows, axis=0),
            np.concatenate(reagent_rows, axis=0),
            np.concatenate(solvent_rows, axis=0),
        )

    @torch.no_grad()
    def encode_routes(self, routes: list[tuple[str, str]]) -> np.ndarray:
        return self.predict_routes(routes)[0]

    def score_condition_candidates(
        self,
        frame: pd.DataFrame,
        *,
        unknown_token_logit: float = -12.0,
    ) -> pd.DataFrame:
        """Attach candidate-specific GNN reagent and solvent compatibility.

        Each score is the mean predicted logit over the normalized tokens in a
        proposed condition. Unknown tokens receive a fixed low score. These
        values are intentionally returned uncalibrated: the Stage 3 validation
        split must select any downstream fusion weight.
        """
        required = {'reactants', 'product', 'reagent_norm', 'solvent_norm'}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f'Candidate table is missing GNN compatibility columns: {missing}')

        work = frame.copy()
        if work.empty:
            work['route_gnn_reagent_compatibility'] = np.asarray([], dtype=np.float32)
            work['route_gnn_solvent_compatibility'] = np.asarray([], dtype=np.float32)
            return work

        unique_routes = work[['reactants', 'product']].drop_duplicates().reset_index(drop=True)
        routes = [(str(row.reactants), str(row.product)) for row in unique_routes.itertuples(index=False)]
        _embedding, reagent_logits, solvent_logits = self.predict_routes(routes)
        route_lookup = {
            (str(row['reactants']), str(row['product'])): index
            for index, row in unique_routes.iterrows()
        }

        def mean_candidate_logit(labels: object, logits: np.ndarray, token_to_index: dict[str, int]) -> float:
            tokens = split_condition_tokens('' if pd.isna(labels) else str(labels))
            if not tokens:
                return 0.0
            values = [
                float(logits[token_to_index[token]]) if token in token_to_index else float(unknown_token_logit)
                for token in tokens
            ]
            return float(np.mean(np.asarray(values, dtype=np.float32)))

        reagent_scores: list[float] = []
        solvent_scores: list[float] = []
        for row in work.itertuples(index=False):
            route_index = route_lookup[(str(row.reactants), str(row.product))]
            reagent_scores.append(
                mean_candidate_logit(row.reagent_norm, reagent_logits[route_index], self.reagent_to_index)
            )
            solvent_scores.append(
                mean_candidate_logit(row.solvent_norm, solvent_logits[route_index], self.solvent_to_index)
            )

        work['route_gnn_reagent_compatibility'] = np.asarray(reagent_scores, dtype=np.float32)
        work['route_gnn_solvent_compatibility'] = np.asarray(solvent_scores, dtype=np.float32)
        return work


def route_gnn_feature_columns(embedding_dim: int) -> list[str]:
    return [f'route_gnn_feat_{idx}' for idx in range(embedding_dim)]


def add_reaction_gnn_features_to_frame(
    frame: pd.DataFrame,
    *,
    artifact_dir: str | Path,
    device: str = 'cpu',
) -> pd.DataFrame:
    work = frame.copy()
    if work.empty:
        return work

    encoder = ReactionGNNFeatureEncoder(artifact_dir, device=device)
    feature_columns = route_gnn_feature_columns(encoder.config.embedding_dim)

    unique_routes = work[['reactants', 'product']].drop_duplicates().reset_index(drop=True)
    route_pairs = [(str(row.reactants), str(row.product)) for row in unique_routes.itertuples(index=False)]
    embeddings = encoder.encode_routes(route_pairs)
    for idx, column in enumerate(feature_columns):
        unique_routes[column] = embeddings[:, idx]

    work = work.merge(unique_routes, on=['reactants', 'product'], how='left', sort=False)
    for column in feature_columns:
        work[column] = pd.to_numeric(work[column], errors='coerce').fillna(0.0).astype(np.float32)
    return work


def augment_table_with_reaction_gnn_features(
    table_file: str | Path,
    *,
    artifact_dir: str | Path,
    output_file: str | Path | None = None,
    device: str = 'cpu',
) -> Path:
    table_path = Path(table_file)
    work = pd.read_csv(table_path)
    augmented = add_reaction_gnn_features_to_frame(
        work,
        artifact_dir=artifact_dir,
        device=device,
    )
    destination = Path(output_file) if output_file is not None else table_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(destination, index=False)
    return destination
