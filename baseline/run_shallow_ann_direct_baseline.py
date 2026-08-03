"""Run a shallow ANN baseline that directly replaces Stage 2 and Stage 3.

Workflow:

1. Keep Stage 1 fixed and load Non-Oracle route caches.
2. Train a family-specific shallow neural network on condition-modeling splits.
   Input: route-conditioned reaction features from reactants + product.
   Outputs:
     - context-class logits over train-time reagent-solvent contexts
     - a scalar temperature prediction
3. For each Stage 1 predicted route, enumerate the ANN top-k contexts directly,
   without KNN filtering or XGBoost reranking.
4. Label and evaluate with the maintained Non-Oracle metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.features import reaction_graph_descriptors, reaction_morgan_fp  # noqa: E402
from prosys_shared.mainline import (  # noqa: E402
    display_family_name,
    evaluate_scored_frame,
    label_candidate_table,
    parse_families_arg,
    split_file_for_family,
)
from prosys_shared.product_memory import normalize_condition_labels, safe_float  # noqa: E402
from prosys_shared.route_cache import load_route_records_from_cache  # noqa: E402
from baseline.common import base_candidate_row  # noqa: E402
from baseline.run_non_oracle_baselines import stage1_route_recall  # noqa: E402

TOPKS = (1, 3, 5, 10)


@dataclass(frozen=True)
class ConditionExample:
    reaction_id: str
    reactants: str
    product: str
    reagent_norm: str
    solvent_norm: str
    temperature: float


class ShallowANNDirect(nn.Module):
    def __init__(self, input_dim: int, num_contexts: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.context_head = nn.Linear(hidden_dim, num_contexts)
        self.temperature_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.context_head(h)
        temperature = self.temperature_head(h).squeeze(-1)
        return logits, temperature


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _context_key(reagent_norm: str, solvent_norm: str) -> str:
    return reagent_norm + ' ||| ' + solvent_norm


def _parse_context_key(context_key: str) -> tuple[str, str]:
    reagent_norm, solvent_norm = context_key.split(' ||| ', 1)
    return reagent_norm, solvent_norm


def load_condition_examples(split_file: str | Path) -> list[ConditionExample]:
    rows: list[ConditionExample] = []
    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 7:
                continue
            rows.append(
                ConditionExample(
                    reaction_id=str(parts[0]),
                    reactants=str(parts[1]),
                    product=str(parts[2]),
                    reagent_norm=normalize_condition_labels(parts[4]),
                    solvent_norm=normalize_condition_labels(parts[5]),
                    temperature=safe_float(parts[6]),
                )
            )
    return rows


def build_context_vocab(train_rows: list[ConditionExample]) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = {}
    for row in train_rows:
        key = _context_key(row.reagent_norm, row.solvent_norm)
        counts[key] = counts.get(key, 0) + 1
    vocab = sorted(counts, key=lambda key: (-counts[key], key))
    return vocab, {key: idx for idx, key in enumerate(vocab)}


def filter_rows_to_known_contexts(
    rows: list[ConditionExample],
    *,
    context_to_index: dict[str, int],
) -> list[ConditionExample]:
    kept: list[ConditionExample] = []
    for row in rows:
        if _context_key(row.reagent_norm, row.solvent_norm) in context_to_index:
            kept.append(row)
    return kept


def _route_feature_vector(
    reactants: str,
    product: str,
    *,
    fpsize: int,
    radius: int,
) -> np.ndarray:
    route_fp = reaction_morgan_fp(reactants, product, fpsize=fpsize, radius=radius).astype(np.float32, copy=False)
    route_graph = reaction_graph_descriptors(reactants, product).astype(np.float32, copy=False)
    return np.concatenate((route_fp, route_graph), axis=0).astype(np.float32, copy=False)


def _feature_matrix(
    rows: list[ConditionExample],
    *,
    fpsize: int,
    radius: int,
    cache: dict[tuple[str, str], np.ndarray] | None = None,
) -> np.ndarray:
    feature_rows: list[np.ndarray] = []
    for row in rows:
        key = (row.reactants, row.product)
        if cache is not None and key in cache:
            feature_rows.append(cache[key])
            continue
        vec = _route_feature_vector(row.reactants, row.product, fpsize=fpsize, radius=radius)
        if cache is not None:
            cache[key] = vec
        feature_rows.append(vec)
    return np.stack(feature_rows).astype(np.float32, copy=False)


def _label_arrays(
    rows: list[ConditionExample],
    *,
    context_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_context = np.asarray(
        [context_to_index[_context_key(row.reagent_norm, row.solvent_norm)] for row in rows],
        dtype=np.int64,
    )
    y_temp = np.asarray([row.temperature for row in rows], dtype=np.float32)
    temp_mask = np.isfinite(y_temp)
    return y_context, y_temp, temp_mask


def _class_weights(y_context: np.ndarray, num_contexts: int) -> np.ndarray:
    counts = np.bincount(y_context, minlength=num_contexts).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (float(num_contexts) * counts)
    weights = np.clip(weights, 0.25, 8.0)
    return weights.astype(np.float32)


def _train_epoch(
    model: ShallowANNDirect,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y_context: torch.Tensor,
    y_temp: torch.Tensor,
    temp_mask: torch.Tensor,
    *,
    batch_size: int,
    class_weight: torch.Tensor,
    temperature_loss_weight: float,
    device: torch.device,
) -> float:
    model.train()
    indices = torch.randperm(x.shape[0], device=device)
    losses: list[float] = []
    ce_loss = nn.CrossEntropyLoss(weight=class_weight)
    mse_loss = nn.MSELoss()

    for start in range(0, int(indices.shape[0]), batch_size):
        batch_index = indices[start:start + batch_size]
        batch_x = x[batch_index]
        batch_context = y_context[batch_index]
        batch_temp = y_temp[batch_index]
        batch_mask = temp_mask[batch_index]

        logits, temp_pred = model(batch_x)
        loss = ce_loss(logits, batch_context)
        if bool(batch_mask.any()):
            loss = loss + temperature_loss_weight * mse_loss(temp_pred[batch_mask], batch_temp[batch_mask])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(np.asarray(losses, dtype=np.float32))) if losses else float('nan')


@torch.no_grad()
def _eval_epoch(
    model: ShallowANNDirect,
    x: torch.Tensor,
    y_context: torch.Tensor,
    y_temp: torch.Tensor,
    temp_mask: torch.Tensor,
    *,
    batch_size: int,
    class_weight: torch.Tensor,
    temperature_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    ce_loss = nn.CrossEntropyLoss(weight=class_weight)
    mse_loss = nn.MSELoss()
    losses: list[float] = []
    context_correct = 0
    context_total = 0
    temp_errors: list[np.ndarray] = []

    for start in range(0, int(x.shape[0]), batch_size):
        batch_x = x[start:start + batch_size]
        batch_context = y_context[start:start + batch_size]
        batch_temp = y_temp[start:start + batch_size]
        batch_mask = temp_mask[start:start + batch_size]

        logits, temp_pred = model(batch_x)
        loss = ce_loss(logits, batch_context)
        if bool(batch_mask.any()):
            loss = loss + temperature_loss_weight * mse_loss(temp_pred[batch_mask], batch_temp[batch_mask])
            errors = torch.abs(temp_pred[batch_mask] - batch_temp[batch_mask]).detach().cpu().numpy()
            temp_errors.append(errors)
        losses.append(float(loss.detach().cpu()))

        pred_context = torch.argmax(logits, dim=1)
        context_correct += int((pred_context == batch_context).sum().item())
        context_total += int(batch_context.shape[0])

    mae = None
    if temp_errors:
        all_errors = np.concatenate(temp_errors).astype(np.float32, copy=False)
        mae = float(np.mean(all_errors))
    return {
        'loss': float(np.mean(np.asarray(losses, dtype=np.float32))) if losses else float('nan'),
        'context_acc': (context_correct / context_total if context_total else 0.0),
        'temp_mae': mae,
    }


def train_shallow_ann_direct(
    *,
    train_rows: list[ConditionExample],
    val_rows: list[ConditionExample],
    output_dir: Path,
    fpsize: int,
    radius: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    temperature_loss_weight: float,
    seed: int,
    device: torch.device,
) -> dict:
    _set_seed(seed)
    context_vocab, context_to_index = build_context_vocab(train_rows)
    num_contexts = len(context_vocab)
    if num_contexts < 2:
        raise ValueError('Need at least two train-time contexts for ANN baseline.')
    val_rows = filter_rows_to_known_contexts(val_rows, context_to_index=context_to_index)
    if not val_rows:
        raise ValueError('Validation split has no contexts overlapping with the train-time context vocabulary.')

    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    x_train_np = _feature_matrix(train_rows, fpsize=fpsize, radius=radius, cache=feature_cache)
    x_val_np = _feature_matrix(val_rows, fpsize=fpsize, radius=radius, cache=feature_cache)
    y_train_np, y_train_temp_np, train_temp_mask_np = _label_arrays(train_rows, context_to_index=context_to_index)
    y_val_np, y_val_temp_np, val_temp_mask_np = _label_arrays(val_rows, context_to_index=context_to_index)

    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(x_train_np).astype(np.float32, copy=False)
    x_val_np = scaler.transform(x_val_np).astype(np.float32, copy=False)

    x_train = torch.from_numpy(x_train_np).to(device=device, dtype=torch.float32)
    x_val = torch.from_numpy(x_val_np).to(device=device, dtype=torch.float32)
    y_train = torch.from_numpy(y_train_np).to(device=device, dtype=torch.long)
    y_val = torch.from_numpy(y_val_np).to(device=device, dtype=torch.long)
    y_train_temp = torch.from_numpy(y_train_temp_np).to(device=device, dtype=torch.float32)
    y_val_temp = torch.from_numpy(y_val_temp_np).to(device=device, dtype=torch.float32)
    train_temp_mask = torch.from_numpy(train_temp_mask_np).to(device=device, dtype=torch.bool)
    val_temp_mask = torch.from_numpy(val_temp_mask_np).to(device=device, dtype=torch.bool)

    class_weight = torch.from_numpy(_class_weights(y_train_np, num_contexts)).to(device=device, dtype=torch.float32)

    model = ShallowANNDirect(
        input_dim=int(x_train.shape[1]),
        num_contexts=num_contexts,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float('inf')
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(
            model,
            optimizer,
            x_train,
            y_train,
            y_train_temp,
            train_temp_mask,
            batch_size=batch_size,
            class_weight=class_weight,
            temperature_loss_weight=temperature_loss_weight,
            device=device,
        )
        val_metrics = _eval_epoch(
            model,
            x_val,
            y_val,
            y_val_temp,
            val_temp_mask,
            batch_size=batch_size,
            class_weight=class_weight,
            temperature_loss_weight=temperature_loss_weight,
        )
        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_context_acc': val_metrics['context_acc'],
                'val_temp_mae': val_metrics['temp_mae'],
            }
        )
        print(
            f"[ann-direct] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_context_acc={val_metrics['context_acc']:.4f} "
            f"val_temp_mae={val_metrics['temp_mae'] if val_metrics['temp_mae'] is not None else 'NA'}",
            flush=True,
        )

        if val_metrics['loss'] < best_val_loss - 1e-5:
            best_val_loss = float(val_metrics['loss'])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / 'ann_direct.pt'
    metadata_file = output_dir / 'ann_direct_meta.json'
    torch.save(best_state, model_file)
    metadata = {
        'model_file': model_file.name,
        'feature_spec': {
            'reaction_morgan_fpsize': fpsize,
            'reaction_morgan_radius': radius,
            'reaction_graph_dim': int(reaction_graph_descriptors('CC', 'CC').shape[0]),
        },
        'input_dim': int(x_train.shape[1]),
        'num_contexts': num_contexts,
        'context_vocab': context_vocab,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'patience': patience,
        'temperature_loss_weight': temperature_loss_weight,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_rows': int(len(train_rows)),
        'val_rows': int(len(val_rows)),
        'scaler_mean': scaler.mean_.astype(np.float32).tolist(),
        'scaler_scale': scaler.scale_.astype(np.float32).tolist(),
        'history': history,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {
        'model_file': str(model_file),
        'metadata_file': str(metadata_file),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_rows': len(train_rows),
        'val_rows': len(val_rows),
    }


def load_shallow_ann_direct(
    model_file: str | Path,
    metadata_file: str | Path,
    *,
    device: torch.device,
) -> tuple[ShallowANNDirect, dict]:
    metadata = json.loads(Path(metadata_file).read_text(encoding='utf-8'))
    model = ShallowANNDirect(
        input_dim=int(metadata['input_dim']),
        num_contexts=int(metadata['num_contexts']),
        hidden_dim=int(metadata['hidden_dim']),
        dropout=float(metadata['dropout']),
    ).to(device)
    state_dict = torch.load(str(model_file), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def _scale_features(features: np.ndarray, metadata: dict) -> np.ndarray:
    mean = np.asarray(metadata['scaler_mean'], dtype=np.float32)
    scale = np.asarray(metadata['scaler_scale'], dtype=np.float32)
    scale[scale == 0] = 1.0
    return ((features.astype(np.float32, copy=False) - mean) / scale).astype(np.float32, copy=False)


@torch.no_grad()
def build_non_oracle_candidate_table(
    *,
    family: str,
    route_cache: Path,
    model_file: Path,
    metadata_file: Path,
    output_file: Path,
    fpsize: int,
    radius: int,
    max_contexts: int,
    device: torch.device,
) -> Path:
    routes = load_route_records_from_cache(route_cache, family=family)
    model, metadata = load_shallow_ann_direct(model_file, metadata_file, device=device)
    context_vocab = list(metadata['context_vocab'])
    feature_cache: dict[tuple[str, str], np.ndarray] = {}

    rows: list[dict] = []
    total_routes = len(routes)
    for route_idx, record in enumerate(routes, start=1):
        feature_key = (record.reactants, record.product)
        feature = feature_cache.get(feature_key)
        if feature is None:
            feature = _route_feature_vector(record.reactants, record.product, fpsize=fpsize, radius=radius)
            feature_cache[feature_key] = feature
        feature = _scale_features(feature.reshape(1, -1), metadata)
        x = torch.from_numpy(feature).to(device=device, dtype=torch.float32)
        logits, temp_pred = model(x)
        score_values = logits.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        temperature_value = float(temp_pred.squeeze(0).detach().cpu().item())
        top_indices = np.argsort(score_values)[::-1][:max_contexts]

        base = base_candidate_row(record)
        for rank, context_index in enumerate(top_indices, start=1):
            reagent_norm, solvent_norm = _parse_context_key(context_vocab[int(context_index)])
            rows.append(
                {
                    **base,
                    'reagent_norm': reagent_norm,
                    'solvent_norm': solvent_norm,
                    'ann_rank': rank,
                    'ann_score': float(score_values[int(context_index)]),
                    'ann_temperature_pred': temperature_value,
                }
            )
        if route_idx % 100 == 0 or route_idx == total_routes:
            print(f'[ann-direct] {family} non_oracle {route_idx}/{total_routes}', flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file, index=False)
    return output_file


def _write_json(data: dict | list, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output_file


def _render_overview(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    flat_rows: list[dict] = []
    for row in rows:
        metrics = row['metrics']
        temp = metrics.get('temperature', {})
        flat_rows.append(
            {
                'family': row['family'],
                'display_family': display_family_name(row['family']),
                'rr10': row['stage1_route_recall'].get('route_recall_top10'),
                'pool_coverage': metrics.get('pool_coverage'),
                'context_top10': metrics.get('context_top10_all'),
                'system_top1': metrics.get('system_top1_all'),
                'system_top3': metrics.get('system_top3_all'),
                'system_top5': metrics.get('system_top5_all'),
                'system_top10': metrics.get('system_top10_all'),
                'temp_mae': temp.get('mae'),
                'temp_within_5c': temp.get('within_5c'),
                'temp_within_10c': temp.get('within_10c'),
                'temp_within_20c': temp.get('within_20c'),
            }
        )
    frame = pd.DataFrame(flat_rows)
    csv_file = output_root / 'results_flat.csv'
    frame.to_csv(csv_file, index=False)

    lines = ['# Shallow ANN direct baseline', '']
    lines.append('## Route + System')
    lines.append('')
    lines.append('| Family | rr@10 | cover | context@10 | sys@1 | sys@3 | sys@5 | sys@10 |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in flat_rows:
        lines.append(
            '| {family} | {rr10:.1f} | {cover:.1f} | {ctx10:.1f} | {sys1:.1f} | {sys3:.1f} | {sys5:.1f} | {sys10:.1f} |'.format(
                family=row['display_family'],
                rr10=100.0 * float(row['rr10'] or 0.0),
                cover=100.0 * float(row['pool_coverage'] or 0.0),
                ctx10=100.0 * float(row['context_top10'] or 0.0),
                sys1=100.0 * float(row['system_top1'] or 0.0),
                sys3=100.0 * float(row['system_top3'] or 0.0),
                sys5=100.0 * float(row['system_top5'] or 0.0),
                sys10=100.0 * float(row['system_top10'] or 0.0),
            )
        )
    if flat_rows:
        macro = frame.mean(numeric_only=True)
        lines.append(
            '| MACRO-AVG | {rr10:.1f} | {cover:.1f} | {ctx10:.1f} | {sys1:.1f} | {sys3:.1f} | {sys5:.1f} | {sys10:.1f} |'.format(
                rr10=100.0 * float(macro.get('rr10', 0.0)),
                cover=100.0 * float(macro.get('pool_coverage', 0.0)),
                ctx10=100.0 * float(macro.get('context_top10', 0.0)),
                sys1=100.0 * float(macro.get('system_top1', 0.0)),
                sys3=100.0 * float(macro.get('system_top3', 0.0)),
                sys5=100.0 * float(macro.get('system_top5', 0.0)),
                sys10=100.0 * float(macro.get('system_top10', 0.0)),
            )
        )

    lines.extend(['', '## Temperature', ''])
    lines.append('| Family | Temp MAE | Temp±5C | Temp±10C | Temp±20C |')
    lines.append('| --- | ---: | ---: | ---: | ---: |')
    for row in flat_rows:
        mae = row['temp_mae']
        lines.append(
            '| {family} | {mae} | {t5} | {t10} | {t20} |'.format(
                family=row['display_family'],
                mae='NA' if pd.isna(mae) else f'{float(mae):.2f}',
                t5='NA' if pd.isna(row['temp_within_5c']) else f'{100.0 * float(row["temp_within_5c"]):.1f}',
                t10='NA' if pd.isna(row['temp_within_10c']) else f'{100.0 * float(row["temp_within_10c"]):.1f}',
                t20='NA' if pd.isna(row['temp_within_20c']) else f'{100.0 * float(row["temp_within_20c"]):.1f}',
            )
        )
    md_file = output_root / 'overview.md'
    md_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return csv_file, md_file


def _write_compare_to_mainline(rows: list[dict], repo_root: Path, output_root: Path) -> tuple[Path, Path]:
    compare_rows: list[dict] = []
    for row in rows:
        family = row['family']
        mainline_file = repo_root / 'outputs' / 'stage23_mainline' / family / 'knn_xgb' / 'non_oracle' / 'result.json'
        if not mainline_file.exists():
            continue
        mainline = json.loads(mainline_file.read_text(encoding='utf-8'))
        ann_metrics = row['metrics']
        main_metrics = mainline['metrics']
        compare_rows.append(
            {
                'family': family,
                'display_family': display_family_name(family),
                'rr10': row['stage1_route_recall'].get('route_recall_top10'),
                'ann_cover': ann_metrics.get('pool_coverage'),
                'main_cover': main_metrics.get('pool_coverage'),
                'ann_context10': ann_metrics.get('context_top10_all'),
                'main_context10': main_metrics.get('context_top10_all'),
                'ann_sys1': ann_metrics.get('system_top1_all'),
                'main_sys1': main_metrics.get('system_top1_all'),
                'ann_sys5': ann_metrics.get('system_top5_all'),
                'main_sys5': main_metrics.get('system_top5_all'),
                'ann_sys10': ann_metrics.get('system_top10_all'),
                'main_sys10': main_metrics.get('system_top10_all'),
            }
        )
    frame = pd.DataFrame(compare_rows)
    csv_file = output_root / 'compare_to_mainline.csv'
    frame.to_csv(csv_file, index=False)

    lines = ['# Shallow ANN direct baseline vs mainline', '']
    lines.append('| Family | rr@10 | ann cover | main cover | ann context@10 | main context@10 | ann sys@1 | main sys@1 | ann sys@5 | main sys@5 | ann sys@10 | main sys@10 |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in compare_rows:
        lines.append(
            '| {family} | {rr10:.1f} | {ac:.1f} | {mc:.1f} | {actx:.1f} | {mctx:.1f} | {as1:.1f} | {ms1:.1f} | {as5:.1f} | {ms5:.1f} | {as10:.1f} | {ms10:.1f} |'.format(
                family=row['display_family'],
                rr10=100.0 * float(row['rr10'] or 0.0),
                ac=100.0 * float(row['ann_cover'] or 0.0),
                mc=100.0 * float(row['main_cover'] or 0.0),
                actx=100.0 * float(row['ann_context10'] or 0.0),
                mctx=100.0 * float(row['main_context10'] or 0.0),
                as1=100.0 * float(row['ann_sys1'] or 0.0),
                ms1=100.0 * float(row['main_sys1'] or 0.0),
                as5=100.0 * float(row['ann_sys5'] or 0.0),
                ms5=100.0 * float(row['main_sys5'] or 0.0),
                as10=100.0 * float(row['ann_sys10'] or 0.0),
                ms10=100.0 * float(row['main_sys10'] or 0.0),
            )
        )
    if compare_rows:
        macro = frame.mean(numeric_only=True)
        lines.append(
            '| MACRO-AVG | {rr10:.1f} | {ac:.1f} | {mc:.1f} | {actx:.1f} | {mctx:.1f} | {as1:.1f} | {ms1:.1f} | {as5:.1f} | {ms5:.1f} | {as10:.1f} | {ms10:.1f} |'.format(
                rr10=100.0 * float(macro.get('rr10', 0.0)),
                ac=100.0 * float(macro.get('ann_cover', 0.0)),
                mc=100.0 * float(macro.get('main_cover', 0.0)),
                actx=100.0 * float(macro.get('ann_context10', 0.0)),
                mctx=100.0 * float(macro.get('main_context10', 0.0)),
                as1=100.0 * float(macro.get('ann_sys1', 0.0)),
                ms1=100.0 * float(macro.get('main_sys1', 0.0)),
                as5=100.0 * float(macro.get('ann_sys5', 0.0)),
                ms5=100.0 * float(macro.get('main_sys5', 0.0)),
                as10=100.0 * float(macro.get('ann_sys10', 0.0)),
                ms10=100.0 * float(macro.get('main_sys10', 0.0)),
            )
        )
    md_file = output_root / 'compare_to_mainline.md'
    md_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return csv_file, md_file


def run_family(
    *,
    repo_root: Path,
    family: str,
    output_root: Path,
    route_root: Path,
    fpsize: int,
    radius: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    temperature_loss_weight: float,
    max_contexts: int,
    seed: int,
    device: torch.device,
    force: bool,
) -> dict:
    family_root = output_root / family / 'ann_direct'
    model_dir = family_root / 'model'
    result_file = family_root / 'non_oracle' / 'result.json'
    model_file = model_dir / 'ann_direct.pt'
    metadata_file = model_dir / 'ann_direct_meta.json'
    route_cache = route_root / family / 'route_cache.json'
    train_split = split_file_for_family(repo_root, family, 'train')
    val_split = split_file_for_family(repo_root, family, 'val')
    test_split = split_file_for_family(repo_root, family, 'test')

    if not route_cache.exists():
        raise FileNotFoundError(f'Missing Stage 1 route cache: {route_cache}')

    if force or not (model_file.exists() and metadata_file.exists()):
        train_rows = load_condition_examples(train_split)
        val_rows = load_condition_examples(val_split)
        print(f'[ann-direct] training {family}: train={len(train_rows)} val={len(val_rows)}', flush=True)
        train_shallow_ann_direct(
            train_rows=train_rows,
            val_rows=val_rows,
            output_dir=model_dir,
            fpsize=fpsize,
            radius=radius,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            temperature_loss_weight=temperature_loss_weight,
            seed=seed,
            device=device,
        )

    if result_file.exists() and not force:
        return json.loads(result_file.read_text(encoding='utf-8'))

    candidate_file = family_root / 'non_oracle' / 'candidate_pool_test.csv'
    labeled_file = family_root / 'non_oracle' / 'test_labeled.csv'
    build_non_oracle_candidate_table(
        family=family,
        route_cache=route_cache,
        model_file=model_file,
        metadata_file=metadata_file,
        output_file=candidate_file,
        fpsize=fpsize,
        radius=radius,
        max_contexts=max_contexts,
        device=device,
    )
    label_candidate_table(candidate_file, test_split, labeled_file)
    frame = pd.read_csv(labeled_file)
    model_meta = json.loads(metadata_file.read_text(encoding='utf-8'))
    result = {
        'baseline': 'ann_direct',
        'family': family,
        'candidate_table': str(labeled_file),
        'raw_candidate_file': str(candidate_file),
        'metrics': evaluate_scored_frame(frame, score_column='ann_score', temperature_column='ann_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
        'model': {
            'model_file': str(model_file),
            'metadata_file': str(metadata_file),
            'best_epoch': model_meta.get('best_epoch'),
            'best_val_loss': model_meta.get('best_val_loss'),
            'train_rows': model_meta.get('train_rows'),
            'val_rows': model_meta.get('val_rows'),
            'num_contexts': model_meta.get('num_contexts'),
            'hidden_dim': model_meta.get('hidden_dim'),
            'dropout': model_meta.get('dropout'),
            'feature_spec': model_meta.get('feature_spec'),
        },
    }
    _write_json(result, result_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the shallow ANN direct Non-Oracle baseline for ProSys.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--output_root', type=str, default='outputs/baselines/non_oracle_ann_direct')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--fpsize', type=int, default=1024)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--temperature_loss_weight', type=float, default=0.05)
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    families = parse_families_arg(args.families)
    device = torch.device(args.device)

    results: list[dict] = []
    for family in families:
        print(f'[ann-direct] running {family}', flush=True)
        result = run_family(
            repo_root=repo_root,
            family=family,
            output_root=output_root,
            route_root=route_root,
            fpsize=args.fpsize,
            radius=args.radius,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            temperature_loss_weight=args.temperature_loss_weight,
            max_contexts=args.max_contexts,
            seed=args.seed,
            device=device,
            force=args.force,
        )
        metrics = result['metrics']
        print(
            f"[ann-direct] {family}: cover={metrics.get('pool_coverage', 0.0) * 100:.1f} "
            f"context@10={metrics.get('context_top10_all', 0.0) * 100:.1f} "
            f"sys@10={metrics.get('system_top10_all', 0.0) * 100:.1f}",
            flush=True,
        )
        results.append(result)

    _write_json(results, output_root / 'all_results.json')
    csv_file, md_file = _render_overview(results, output_root)
    compare_csv, compare_md = _write_compare_to_mainline(results, repo_root, output_root)
    print(f'[ann-direct] wrote {csv_file}', flush=True)
    print(f'[ann-direct] wrote {md_file}', flush=True)
    print(f'[ann-direct] wrote {compare_csv}', flush=True)
    print(f'[ann-direct] wrote {compare_md}', flush=True)


if __name__ == '__main__':
    main()
