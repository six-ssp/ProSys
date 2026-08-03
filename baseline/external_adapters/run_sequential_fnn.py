"""Train and infer the ProSys-adapted Sequential FNN baseline.

This keeps the hierarchical idea of the reference implementation while using
the official ProSys label contract: reagent set -> solvent set -> temperature.
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

from .model_common import (
    ContextLibrary,
    FeatureStandardizer,
    load_jsonl_rows,
    load_vocabulary,
    multi_hot,
    positive_class_weight,
    route_feature_matrix,
    temperature_targets,
    token_id_index,
    write_prediction_rows,
)


MODEL_FILE = 'sequential_fnn.pt'
METADATA_FILE = 'sequential_fnn_meta.json'


@dataclass(frozen=True)
class SequentialFNNConfig:
    fp_size: int = 4096
    radius: int = 2
    hidden_dim: int = 512
    bottleneck_dim: int = 256
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 7
    temperature_loss_weight: float = 0.2
    device: str = 'cpu'
    random_state: int = 0


class SequentialFNN(nn.Module):
    def __init__(self, input_dim: int, num_reagents: int, num_solvents: int, config: SequentialFNNConfig):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.reagent_head = nn.Linear(config.bottleneck_dim, num_reagents)
        self.solvent_trunk = nn.Sequential(
            nn.Linear(config.bottleneck_dim + num_reagents, config.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.solvent_head = nn.Linear(config.bottleneck_dim, num_solvents)
        self.temperature_head = nn.Sequential(
            nn.Linear(config.bottleneck_dim + num_reagents + num_solvents, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        reagent_logits = self.reagent_head(hidden)
        reagent_probabilities = torch.sigmoid(reagent_logits)
        solvent_hidden = self.solvent_trunk(torch.cat((hidden, reagent_probabilities), dim=1))
        solvent_logits = self.solvent_head(solvent_hidden)
        solvent_probabilities = torch.sigmoid(solvent_logits)
        temperature = self.temperature_head(
            torch.cat((hidden, reagent_probabilities, solvent_probabilities), dim=1)
        ).squeeze(1)
        return reagent_logits, solvent_logits, temperature


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    return torch.device(value if value == 'cpu' or torch.cuda.is_available() else 'cpu')


def _loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reagent_target: torch.Tensor,
    solvent_target: torch.Tensor,
    temperature_target: torch.Tensor,
    temperature_mask: torch.Tensor,
    reagent_loss: nn.Module,
    solvent_loss: nn.Module,
    config: SequentialFNNConfig,
) -> torch.Tensor:
    reagent_logits, solvent_logits, temperature = outputs
    total = reagent_loss(reagent_logits, reagent_target) + solvent_loss(solvent_logits, solvent_target)
    if bool(temperature_mask.any()):
        total = total + config.temperature_loss_weight * nn.functional.smooth_l1_loss(
            temperature[temperature_mask], temperature_target[temperature_mask]
        )
    return total


@torch.no_grad()
def _validation_loss(
    model: SequentialFNN,
    features: torch.Tensor,
    reagent_target: torch.Tensor,
    solvent_target: torch.Tensor,
    temperature_target: torch.Tensor,
    temperature_mask: torch.Tensor,
    reagent_loss: nn.Module,
    solvent_loss: nn.Module,
    config: SequentialFNNConfig,
) -> float:
    model.eval()
    return float(
        _loss(
            model(features),
            reagent_target,
            solvent_target,
            temperature_target,
            temperature_mask,
            reagent_loss,
            solvent_loss,
            config,
        ).detach().cpu()
    )


def train(input_dir: Path, artifact_dir: Path, config: SequentialFNNConfig) -> dict[str, Any]:
    _set_seed(config.random_state)
    vocabulary = load_vocabulary(input_dir / 'label_vocabulary.json')
    reagent_ids, reagent_index = token_id_index(vocabulary, 'reagent')
    solvent_ids, solvent_index = token_id_index(vocabulary, 'solvent')
    train_rows = load_jsonl_rows(input_dir / 'train_routes.jsonl')
    val_rows = load_jsonl_rows(input_dir / 'val_routes.jsonl')
    if not train_rows or not val_rows:
        raise ValueError('Sequential FNN requires non-empty train and validation route files.')

    x_train_raw = route_feature_matrix(train_rows, fp_size=config.fp_size, radius=config.radius)
    x_val_raw = route_feature_matrix(val_rows, fp_size=config.fp_size, radius=config.radius)
    standardizer = FeatureStandardizer.fit(x_train_raw)
    x_train = standardizer.transform(x_train_raw)
    x_val = standardizer.transform(x_val_raw)
    y_reagent_train = multi_hot(train_rows, 'reagent_ids', reagent_index)
    y_solvent_train = multi_hot(train_rows, 'solvent_ids', solvent_index)
    y_reagent_val = multi_hot(val_rows, 'reagent_ids', reagent_index)
    y_solvent_val = multi_hot(val_rows, 'solvent_ids', solvent_index)
    y_temp_train, mask_temp_train = temperature_targets(train_rows)
    y_temp_val, mask_temp_val = temperature_targets(val_rows)

    device = _device(config.device)
    model = SequentialFNN(x_train.shape[1], len(reagent_ids), len(solvent_ids), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    reagent_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(positive_class_weight(y_reagent_train), dtype=torch.float32, device=device)
    )
    solvent_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(positive_class_weight(y_solvent_train), dtype=torch.float32, device=device)
    )
    tensors = {
        'x_train': torch.as_tensor(x_train, dtype=torch.float32, device=device),
        'x_val': torch.as_tensor(x_val, dtype=torch.float32, device=device),
        'yr_train': torch.as_tensor(y_reagent_train, dtype=torch.float32, device=device),
        'ys_train': torch.as_tensor(y_solvent_train, dtype=torch.float32, device=device),
        'yr_val': torch.as_tensor(y_reagent_val, dtype=torch.float32, device=device),
        'ys_val': torch.as_tensor(y_solvent_val, dtype=torch.float32, device=device),
        'yt_train': torch.as_tensor(y_temp_train, dtype=torch.float32, device=device),
        'yt_val': torch.as_tensor(y_temp_val, dtype=torch.float32, device=device),
        'mt_train': torch.as_tensor(mask_temp_train, dtype=torch.bool, device=device),
        'mt_val': torch.as_tensor(mask_temp_val, dtype=torch.bool, device=device),
    }

    best_loss = float('inf')
    best_state: dict[str, torch.Tensor] | None = None
    stopped_after = 0
    patience = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(tensors['x_train'].shape[0], device=device)
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            loss = _loss(
                model(tensors['x_train'][batch]),
                tensors['yr_train'][batch],
                tensors['ys_train'][batch],
                tensors['yt_train'][batch],
                tensors['mt_train'][batch],
                reagent_loss,
                solvent_loss,
                config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_loss = _validation_loss(
            model,
            tensors['x_val'],
            tensors['yr_val'],
            tensors['ys_val'],
            tensors['yt_val'],
            tensors['mt_val'],
            reagent_loss,
            solvent_loss,
            config,
        )
        stopped_after = epoch
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
        'standardizer': standardizer.to_dict(),
        'reagent_ids': reagent_ids,
        'solvent_ids': solvent_ids,
    }
    torch.save(checkpoint, artifact_dir / MODEL_FILE)
    metadata = {
        'model_file': MODEL_FILE,
        'best_val_loss': best_loss,
        'last_epoch': stopped_after,
        'num_train_routes': len(train_rows),
        'num_val_routes': len(val_rows),
        'num_reagent_tokens': len(reagent_ids),
        'num_solvent_tokens': len(solvent_ids),
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
    config = SequentialFNNConfig(**{**checkpoint['config'], 'device': device_name})
    device = _device(config.device)
    model = SequentialFNN(
        len(checkpoint['standardizer']['mean']),
        len(checkpoint['reagent_ids']),
        len(checkpoint['solvent_ids']),
        config,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    standardizer = FeatureStandardizer.from_dict(checkpoint['standardizer'])
    routes = load_jsonl_rows(routes_file or input_dir / 'test_stage1_routes.jsonl')
    library = ContextLibrary.from_file(input_dir / 'train_context_library.jsonl', vocabulary)
    features = standardizer.transform(route_feature_matrix(routes, fp_size=config.fp_size, radius=config.radius))

    prediction_rows = []
    with torch.no_grad():
        for start in range(0, len(routes), config.batch_size):
            batch_rows = routes[start:start + config.batch_size]
            batch_features = torch.as_tensor(features[start:start + config.batch_size], dtype=torch.float32, device=device)
            reagent_logits, solvent_logits, temperatures = model(batch_features)
            reagent_probs = torch.sigmoid(reagent_logits).cpu().numpy()
            solvent_probs = torch.sigmoid(solvent_logits).cpu().numpy()
            temperature_values = temperatures.cpu().numpy()
            for row, reagent_prob, solvent_prob, temperature in zip(
                batch_rows, reagent_probs, solvent_probs, temperature_values
            ):
                prediction_rows.append(
                    {
                        'sample_index': int(row['sample_index']),
                        'retro_rank': int(row['retro_rank']),
                        'candidates': library.top_candidates(
                            reagent_prob,
                            solvent_prob,
                            limit=top_contexts,
                            temperature_pred=float(temperature),
                        ),
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
        config = SequentialFNNConfig(
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            random_state=args.seed,
        )
        metadata = train(args.input_dir, args.artifact_dir, config)
        print(json.dumps(metadata, indent=2))
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
