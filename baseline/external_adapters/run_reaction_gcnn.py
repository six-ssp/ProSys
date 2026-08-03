"""Train and infer the ProSys-adapted Reaction-GCNN baseline.

The baseline is a graph encoder with reagent and solvent multi-label heads. It
does not use KNN evidence, ReaFNN scores, or XGBoost reranking.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from stage3_XGBoost.reaction_gnn_features import ReactionGNN, _batch_graphs, _graph_from_smiles

from .model_common import (
    ContextLibrary,
    load_jsonl_rows,
    load_vocabulary,
    multi_hot,
    positive_class_weight,
    token_id_index,
    write_prediction_rows,
)


MODEL_FILE = 'reaction_gcnn.pt'
METADATA_FILE = 'reaction_gcnn_meta.json'


@dataclass(frozen=True)
class ReactionGCNNConfig:
    hidden_dim: int = 64
    embedding_dim: int = 64
    message_passing_steps: int = 3
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 48
    max_epochs: int = 50
    patience: int = 7
    device: str = 'cpu'
    random_state: int = 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    return torch.device(value if value == 'cpu' or torch.cuda.is_available() else 'cpu')


def _graphs(rows: list[dict[str, Any]], device: torch.device):
    reactant_graph = _batch_graphs(
        [_graph_from_smiles(str(row['reactants']), reaction_side=True) for row in rows],
        device,
    )
    product_graph = _batch_graphs(
        [_graph_from_smiles(str(row['product']), reaction_side=False) for row in rows],
        device,
    )
    return reactant_graph, product_graph


@torch.no_grad()
def _evaluate(
    model: ReactionGNN,
    rows: list[dict[str, Any]],
    reagent_targets: np.ndarray,
    solvent_targets: np.ndarray,
    reagent_loss: nn.Module,
    solvent_loss: nn.Module,
    config: ReactionGCNNConfig,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for start in range(0, len(rows), config.batch_size):
        batch_rows = rows[start:start + config.batch_size]
        reactant_graph, product_graph = _graphs(batch_rows, device)
        _embedding, reagent_logits, solvent_logits = model(reactant_graph, product_graph)
        reagent_target = torch.as_tensor(reagent_targets[start:start + len(batch_rows)], dtype=torch.float32, device=device)
        solvent_target = torch.as_tensor(solvent_targets[start:start + len(batch_rows)], dtype=torch.float32, device=device)
        losses.append(float((reagent_loss(reagent_logits, reagent_target) + solvent_loss(solvent_logits, solvent_target)).cpu()))
    return float(np.mean(losses)) if losses else float('inf')


def train(input_dir: Path, artifact_dir: Path, config: ReactionGCNNConfig) -> dict[str, Any]:
    _set_seed(config.random_state)
    vocabulary = load_vocabulary(input_dir / 'label_vocabulary.json')
    reagent_ids, reagent_index = token_id_index(vocabulary, 'reagent')
    solvent_ids, solvent_index = token_id_index(vocabulary, 'solvent')
    train_rows = load_jsonl_rows(input_dir / 'train_routes.jsonl')
    val_rows = load_jsonl_rows(input_dir / 'val_routes.jsonl')
    if not train_rows or not val_rows:
        raise ValueError('Reaction-GCNN requires non-empty train and validation route files.')
    y_reagent_train = multi_hot(train_rows, 'reagent_ids', reagent_index)
    y_solvent_train = multi_hot(train_rows, 'solvent_ids', solvent_index)
    y_reagent_val = multi_hot(val_rows, 'reagent_ids', reagent_index)
    y_solvent_val = multi_hot(val_rows, 'solvent_ids', solvent_index)

    device = _device(config.device)
    model = ReactionGNN(
        num_reagent_tokens=len(reagent_ids),
        num_solvent_tokens=len(solvent_ids),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        message_passing_steps=config.message_passing_steps,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    reagent_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(positive_class_weight(y_reagent_train), dtype=torch.float32, device=device)
    )
    solvent_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(positive_class_weight(y_solvent_train), dtype=torch.float32, device=device)
    )

    best_loss = float('inf')
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    last_epoch = 0
    order = np.arange(len(train_rows))
    for epoch in range(1, config.max_epochs + 1):
        np.random.shuffle(order)
        model.train()
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            batch_rows = [train_rows[int(index)] for index in indices]
            reactant_graph, product_graph = _graphs(batch_rows, device)
            _embedding, reagent_logits, solvent_logits = model(reactant_graph, product_graph)
            reagent_target = torch.as_tensor(y_reagent_train[indices], dtype=torch.float32, device=device)
            solvent_target = torch.as_tensor(y_solvent_train[indices], dtype=torch.float32, device=device)
            loss = reagent_loss(reagent_logits, reagent_target) + solvent_loss(solvent_logits, solvent_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_loss = _evaluate(
            model,
            val_rows,
            y_reagent_val,
            y_solvent_val,
            reagent_loss,
            solvent_loss,
            config,
            device,
        )
        last_epoch = epoch
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break

    if best_state is None:
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state': best_state,
        'config': asdict(config),
        'reagent_ids': reagent_ids,
        'solvent_ids': solvent_ids,
    }
    torch.save(checkpoint, artifact_dir / MODEL_FILE)
    metadata = {
        'model_file': MODEL_FILE,
        'best_val_loss': best_loss,
        'last_epoch': last_epoch,
        'num_train_routes': len(train_rows),
        'num_val_routes': len(val_rows),
        'num_reagent_tokens': len(reagent_ids),
        'num_solvent_tokens': len(solvent_ids),
        'temperature_head': False,
        'config': asdict(config),
    }
    (artifact_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata


def predict(
    input_dir: Path,
    artifact_dir: Path,
    output_file: Path,
    *,
    top_contexts: int,
    device_name: str,
    routes_file: Path | None = None,
) -> int:
    vocabulary = load_vocabulary(input_dir / 'label_vocabulary.json')
    checkpoint = torch.load(artifact_dir / MODEL_FILE, map_location='cpu', weights_only=False)
    config = ReactionGCNNConfig(**{**checkpoint['config'], 'device': device_name})
    device = _device(config.device)
    model = ReactionGNN(
        num_reagent_tokens=len(checkpoint['reagent_ids']),
        num_solvent_tokens=len(checkpoint['solvent_ids']),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        message_passing_steps=config.message_passing_steps,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    routes = load_jsonl_rows(routes_file or input_dir / 'test_stage1_routes.jsonl')
    library = ContextLibrary.from_file(input_dir / 'train_context_library.jsonl', vocabulary)

    prediction_rows = []
    with torch.no_grad():
        for start in range(0, len(routes), config.batch_size):
            batch_rows = routes[start:start + config.batch_size]
            reactant_graph, product_graph = _graphs(batch_rows, device)
            _embedding, reagent_logits, solvent_logits = model(reactant_graph, product_graph)
            reagent_probabilities = torch.sigmoid(reagent_logits).cpu().numpy()
            solvent_probabilities = torch.sigmoid(solvent_logits).cpu().numpy()
            for row, reagent_prob, solvent_prob in zip(batch_rows, reagent_probabilities, solvent_probabilities):
                prediction_rows.append(
                    {
                        'sample_index': int(row['sample_index']),
                        'retro_rank': int(row['retro_rank']),
                        'candidates': library.top_candidates(reagent_prob, solvent_prob, limit=top_contexts),
                    }
                )
    write_prediction_rows(output_file, prediction_rows)
    return len(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--input-dir', type=Path, required=True)
    train_parser.add_argument('--artifact-dir', type=Path, required=True)
    train_parser.add_argument('--device', default='cpu')
    train_parser.add_argument('--max-epochs', type=int, default=50)
    train_parser.add_argument('--patience', type=int, default=7)
    train_parser.add_argument('--seed', type=int, default=0)
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--input-dir', type=Path, required=True)
    predict_parser.add_argument('--artifact-dir', type=Path, required=True)
    predict_parser.add_argument('--output', type=Path, required=True)
    predict_parser.add_argument('--top-contexts', type=int, default=20)
    predict_parser.add_argument('--routes-file', type=Path, default=None)
    predict_parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if args.command == 'train':
        config = ReactionGCNNConfig(
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            random_state=args.seed,
        )
        print(json.dumps(train(args.input_dir, args.artifact_dir, config), indent=2))
    else:
        count = predict(
            args.input_dir,
            args.artifact_dir,
            args.output,
            top_contexts=args.top_contexts,
            device_name=args.device,
            routes_file=args.routes_file,
        )
        print(f'Wrote predictions for {count} Stage 1 routes to {args.output}')


if __name__ == '__main__':
    main()
