"""Candidate-aware residual ranker built on frozen reaction-GNN embeddings.

The original Stage 3 GNN emits one structural vector per route, so every
condition candidate attached to that route receives the same feature. This
module keeps that learned graph representation but adds a lightweight
condition-token interaction head, producing one score per route--context pair.
It is designed for score-level fusion with a separately trained XGBoost ranker;
therefore no in-sample GNN score is fed back as an XGBoost training feature.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from prosys_shared.condition_modeling import split_condition_tokens


MODEL_FILE_NAME = 'condition_aware_gnn.pt'
METADATA_FILE_NAME = 'condition_aware_gnn_meta.json'


@dataclass(frozen=True)
class ConditionAwareGNNConfig:
    token_dim: int = 32
    hidden_dim: int = 96
    dropout: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 12
    max_positive_weight: float = 12.0
    device: str = 'cpu'
    random_state: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    return torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')


def _route_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith('route_gnn_feat_')]
    columns.sort(key=lambda column: int(column.rsplit('_', 1)[1]))
    if not columns:
        raise ValueError('Candidate table does not contain frozen route-GNN features.')
    return columns


def _tokens(value: object) -> tuple[str, ...]:
    return split_condition_tokens('' if pd.isna(value) else str(value))


def _build_vocab(labels: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in labels:
        for token in _tokens(value):
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    # Zero is padding and one is a train-time unknown-token slot.
    return {token: index + 2 for index, token in enumerate(ordered)}


def _max_tokens(labels: pd.Series) -> int:
    return max(1, max((len(_tokens(value)) for value in labels), default=0))


def _token_matrix(labels: pd.Series, token_to_index: dict[str, int], width: int) -> np.ndarray:
    matrix = np.zeros((len(labels), width), dtype=np.int64)
    for row_index, value in enumerate(labels):
        for column_index, token in enumerate(_tokens(value)[:width]):
            matrix[row_index, column_index] = token_to_index.get(token, 1)
    return matrix


def _masked_embedding_mean(embedding: nn.Embedding, token_ids: torch.Tensor) -> torch.Tensor:
    token_vectors = embedding(token_ids)
    mask = (token_ids != 0).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp_min(1)
    return (token_vectors * mask).sum(dim=1) / counts


class ConditionAwareGNNRanker(nn.Module):
    def __init__(
        self,
        *,
        route_dim: int,
        reagent_vocab_size: int,
        solvent_vocab_size: int,
        config: ConditionAwareGNNConfig,
    ):
        super().__init__()
        self.route_projection = nn.Sequential(
            nn.Linear(route_dim, route_dim),
            nn.ReLU(),
            nn.LayerNorm(route_dim),
        )
        self.reagent_embedding = nn.Embedding(reagent_vocab_size, config.token_dim, padding_idx=0)
        self.solvent_embedding = nn.Embedding(solvent_vocab_size, config.token_dim, padding_idx=0)
        self.condition_projection = nn.Sequential(
            nn.Linear(config.token_dim * 2 + 2, route_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(route_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        route_features: torch.Tensor,
        reagent_ids: torch.Tensor,
        solvent_ids: torch.Tensor,
    ) -> torch.Tensor:
        route = self.route_projection(route_features)
        reagent = _masked_embedding_mean(self.reagent_embedding, reagent_ids)
        solvent = _masked_embedding_mean(self.solvent_embedding, solvent_ids)
        counts = torch.stack(
            (
                (reagent_ids != 0).sum(dim=1).to(torch.float32) / max(1, reagent_ids.shape[1]),
                (solvent_ids != 0).sum(dim=1).to(torch.float32) / max(1, solvent_ids.shape[1]),
            ),
            dim=1,
        )
        condition = self.condition_projection(torch.cat((reagent, solvent, counts), dim=1))
        interaction = torch.cat((route, condition, route * condition, torch.abs(route - condition)), dim=1)
        return self.score_head(interaction).squeeze(1)


def _tensor_inputs(
    route_features: np.ndarray,
    reagent_ids: np.ndarray,
    solvent_ids: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(route_features, dtype=torch.float32, device=device),
        torch.as_tensor(reagent_ids, dtype=torch.long, device=device),
        torch.as_tensor(solvent_ids, dtype=torch.long, device=device),
    )


def _score_model(
    model: ConditionAwareGNNRanker,
    route_features: np.ndarray,
    reagent_ids: np.ndarray,
    solvent_ids: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(route_features), batch_size):
            batch = _tensor_inputs(
                route_features[start:start + batch_size],
                reagent_ids[start:start + batch_size],
                solvent_ids[start:start + batch_size],
                device,
            )
            scores.append(model(*batch).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def train_condition_aware_gnn_ranker(
    train_table_file: str | Path,
    output_dir: str | Path,
    *,
    config: ConditionAwareGNNConfig | None = None,
    force_retrain: bool = False,
) -> dict:
    """Train only on the Stage 3 training table with fixed hyperparameters."""
    config = config or ConditionAwareGNNConfig()
    output_dir = Path(output_dir)
    model_file = output_dir / MODEL_FILE_NAME
    metadata_file = output_dir / METADATA_FILE_NAME
    if model_file.exists() and metadata_file.exists() and not force_retrain:
        return json.loads(metadata_file.read_text(encoding='utf-8'))

    frame = pd.read_csv(train_table_file)
    if frame.empty or 'label' not in frame.columns:
        raise ValueError('Condition-aware GNN ranker requires a non-empty labeled training table.')
    route_columns = _route_feature_columns(frame)
    reagent_to_index = _build_vocab(frame['reagent_norm'])
    solvent_to_index = _build_vocab(frame['solvent_norm'])
    reagent_width = _max_tokens(frame['reagent_norm'])
    solvent_width = _max_tokens(frame['solvent_norm'])
    route_features = frame.loc[:, route_columns].fillna(0.0).to_numpy(dtype=np.float32)
    route_mean = route_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    route_std = route_features.std(axis=0, dtype=np.float64).astype(np.float32)
    route_std = np.where(route_std > 1e-6, route_std, 1.0).astype(np.float32)
    route_features = ((route_features - route_mean) / route_std).astype(np.float32)
    reagent_ids = _token_matrix(frame['reagent_norm'], reagent_to_index, reagent_width)
    solvent_ids = _token_matrix(frame['solvent_norm'], solvent_to_index, solvent_width)
    labels = (pd.to_numeric(frame['label'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32) > 0.5).astype(np.float32)

    _set_seed(config.random_state)
    device = _resolve_device(config.device)
    model = ConditionAwareGNNRanker(
        route_dim=len(route_columns),
        reagent_vocab_size=len(reagent_to_index) + 2,
        solvent_vocab_size=len(solvent_to_index) + 2,
        config=config,
    ).to(device)
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels)) - positives, 1.0)
    positive_weight = min(max(negatives / positives, 1.0), config.max_positive_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    losses: list[float] = []
    order = np.arange(len(labels), dtype=np.int64)
    for _epoch in range(config.max_epochs):
        np.random.shuffle(order)
        model.train()
        epoch_losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            batch = _tensor_inputs(route_features[indices], reagent_ids[indices], solvent_ids[indices], device)
            target = torch.as_tensor(labels[indices], dtype=torch.float32, device=device)
            loss = loss_fn(model(*batch), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(np.asarray(epoch_losses, dtype=np.float32))))

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        'route_columns': route_columns,
        'route_mean': route_mean,
        'route_std': route_std,
        'reagent_to_index': reagent_to_index,
        'solvent_to_index': solvent_to_index,
        'reagent_width': reagent_width,
        'solvent_width': solvent_width,
        'config': config.to_dict(),
    }
    torch.save(payload, model_file)
    metadata = {
        'model_file': model_file.name,
        'num_train_rows': int(len(frame)),
        'num_train_positive_rows': int(labels.sum()),
        'route_dim': int(len(route_columns)),
        'num_reagent_tokens': int(len(reagent_to_index)),
        'num_solvent_tokens': int(len(solvent_to_index)),
        'reagent_width': int(reagent_width),
        'solvent_width': int(solvent_width),
        'positive_weight': float(positive_weight),
        'train_loss_by_epoch': losses,
        'config': config.to_dict(),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return {
        **metadata,
        'model_file': str(model_file),
        'metadata_file': str(metadata_file),
    }


def score_condition_aware_gnn_ranker(
    table_file: str | Path,
    model_file: str | Path,
    *,
    device: str = 'cpu',
) -> pd.DataFrame:
    """Score a candidate table without inspecting its labels."""
    payload = torch.load(Path(model_file), map_location='cpu', weights_only=False)
    config = ConditionAwareGNNConfig(**payload['config'])
    run_config = ConditionAwareGNNConfig(**{**config.to_dict(), 'device': device})
    frame = pd.read_csv(table_file)
    route_columns = list(payload['route_columns'])
    missing = [column for column in route_columns if column not in frame.columns]
    if missing:
        raise ValueError(f'Candidate table is missing frozen route-GNN columns: {missing[:3]}')
    route_features = frame.loc[:, route_columns].fillna(0.0).to_numpy(dtype=np.float32)
    route_features = ((route_features - payload['route_mean']) / payload['route_std']).astype(np.float32)
    reagent_ids = _token_matrix(frame['reagent_norm'], payload['reagent_to_index'], int(payload['reagent_width']))
    solvent_ids = _token_matrix(frame['solvent_norm'], payload['solvent_to_index'], int(payload['solvent_width']))
    run_device = _resolve_device(run_config.device)
    model = ConditionAwareGNNRanker(
        route_dim=len(route_columns),
        reagent_vocab_size=len(payload['reagent_to_index']) + 2,
        solvent_vocab_size=len(payload['solvent_to_index']) + 2,
        config=run_config,
    ).to(run_device)
    model.load_state_dict(payload['model_state'])
    work = frame.copy()
    work['condition_aware_gnn_score_raw'] = _score_model(
        model,
        route_features,
        reagent_ids,
        solvent_ids,
        device=run_device,
        batch_size=run_config.batch_size,
    )
    return work
