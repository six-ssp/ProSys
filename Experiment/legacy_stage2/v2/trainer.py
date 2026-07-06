"""Training utilities for ProSys Stage 2 V2."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .losses import binary_candidate_loss, pairwise_margin_loss, weighted_listMLE
from .model import Stage2ModelConfigV2, Stage2NeuralRankerV2


@dataclass
class Stage2TrainConfigV2:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_rank: float = 1.0
    lambda_temp: float = 0.2
    lambda_listmle: float = 1.0
    lambda_bce: float = 0.5
    lambda_margin: float = 0.5
    margin: float = 0.5
    max_negatives: int = 8
    temperature_zscore_by_family: bool = True
    temp_positive_only: bool = True
    device: str = 'cpu'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemperatureStats:
    mean: float = 0.0
    std: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_temperature_stats(frame: pd.DataFrame) -> TemperatureStats:
    positives = frame[(frame['label'] > 0.5) & np.isfinite(frame['temperature_gold'])]
    if positives.empty:
        return TemperatureStats()

    mean = float(positives['temperature_gold'].mean())
    std = float(positives['temperature_gold'].std(ddof=0))
    if std <= 0:
        std = 1.0
    return TemperatureStats(mean=mean, std=std)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _topk_hits(logits: torch.Tensor, labels: torch.Tensor, sample_index: torch.Tensor) -> dict[str, float]:
    top1_hits = 0.0
    top3_hits = 0.0
    num_slates = 0.0
    for group_id in torch.unique(sample_index):
        group_mask = sample_index == group_id
        group_logits = logits[group_mask]
        group_labels = labels[group_mask]
        if group_logits.numel() == 0:
            continue
        sorted_indices = torch.argsort(group_logits, descending=True)
        top1_hits += float(torch.any(group_labels[sorted_indices[:1]] > 0.5))
        top3_hits += float(torch.any(group_labels[sorted_indices[:3]] > 0.5))
        num_slates += 1.0

    if num_slates == 0:
        return {'top1_hit': 0.0, 'top3_hit': 0.0, 'num_slates': 0.0}
    return {
        'top1_hit': top1_hits / num_slates,
        'top3_hit': top3_hits / num_slates,
        'num_slates': num_slates,
    }


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    train_config: Stage2TrainConfigV2,
    temperature_stats: TemperatureStats,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs['score_logit']
    labels = batch['label']
    sample_index = batch['sample_index']
    sample_weight = batch['sample_weight']

    listmle_loss = weighted_listMLE(
        logits=logits,
        relevance=batch['rank_relevance'],
        sample_index=sample_index,
        sample_weight=sample_weight,
    )
    bce_loss = binary_candidate_loss(logits, labels, sample_weight=sample_weight)
    margin_loss = pairwise_margin_loss(
        logits=logits,
        labels=labels,
        route_match=batch['route_match'],
        context_match=batch['context_match'],
        sample_index=sample_index,
        margin=train_config.margin,
        max_negatives=train_config.max_negatives,
    )
    rank_loss = (
        train_config.lambda_listmle * listmle_loss
        + train_config.lambda_bce * bce_loss
        + train_config.lambda_margin * margin_loss
    )

    temperature_pred = outputs['temperature_pred']
    temperature_gold = batch['temperature_gold']
    temp_mask = torch.isfinite(temperature_gold)
    if train_config.temp_positive_only:
        temp_mask = temp_mask & (labels > 0.5)

    if torch.any(temp_mask):
        targets = temperature_gold[temp_mask]
        preds = temperature_pred[temp_mask]
        if train_config.temperature_zscore_by_family:
            targets = (targets - temperature_stats.mean) / temperature_stats.std
        temp_loss = nn.functional.smooth_l1_loss(preds, targets)
    else:
        temp_loss = torch.zeros((), device=logits.device)

    total_loss = train_config.lambda_rank * rank_loss + train_config.lambda_temp * temp_loss

    metrics = {
        'loss_total': float(total_loss.detach().cpu()),
        'loss_rank': float(rank_loss.detach().cpu()),
        'loss_listmle': float(listmle_loss.detach().cpu()),
        'loss_bce': float(bce_loss.detach().cpu()),
        'loss_margin': float(margin_loss.detach().cpu()),
        'loss_temp': float(temp_loss.detach().cpu()),
    }
    metrics.update(_topk_hits(logits.detach(), labels.detach(), sample_index.detach()))
    return total_loss, metrics


def run_epoch(
    model: Stage2NeuralRankerV2,
    loader,
    optimizer: torch.optim.Optimizer | None,
    train_config: Stage2TrainConfigV2,
    temperature_stats: TemperatureStats,
) -> dict[str, float]:
    device = torch.device(train_config.device)
    training = optimizer is not None
    model.train(training)

    aggregate = defaultdict(float)
    num_batches = 0.0
    total_slates = 0.0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(
                route_fp=batch['route_fp'],
                route_graph_features=batch['route_graph_features'],
                route_dense_features=batch['route_dense_features'],
                context_dense_features=batch['context_dense_features'],
                product_fp=batch['product_fp'],
                product_features=batch['product_features'],
                reagent_features=batch['reagent_features'],
                solvent_features=batch['solvent_features'],
                support_features=batch['support_features'],
            )
            total_loss, metrics = compute_losses(outputs, batch, train_config, temperature_stats)

            if training:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        batch_num_slates = metrics.get('num_slates', 0.0)
        total_slates += batch_num_slates
        for key, value in metrics.items():
            if key in {'top1_hit', 'top3_hit'}:
                aggregate[key] += value * batch_num_slates
            elif key != 'num_slates':
                aggregate[key] += value
        num_batches += 1.0

    if num_batches == 0:
        return {}

    result = {key: value / num_batches for key, value in aggregate.items() if key not in {'top1_hit', 'top3_hit'}}
    if total_slates > 0:
        result['top1_hit'] = aggregate['top1_hit'] / total_slates
        result['top3_hit'] = aggregate['top3_hit'] / total_slates
        result['num_slates'] = total_slates
    else:
        result['top1_hit'] = 0.0
        result['top3_hit'] = 0.0
        result['num_slates'] = 0.0
    return result


def save_checkpoint(
    output_dir: str | Path,
    model: Stage2NeuralRankerV2,
    model_config: Stage2ModelConfigV2,
    train_config: Stage2TrainConfigV2,
    temperature_stats: TemperatureStats,
    epoch: int,
    metrics: dict[str, float],
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / 'best_model.pt'
    torch.save(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'model_config': model_config.to_dict(),
            'train_config': train_config.to_dict(),
            'temperature_stats': temperature_stats.to_dict(),
            'metrics': metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path


def train_stage2_model_v2(
    *,
    train_loader,
    val_loader,
    output_dir: str | Path,
    model_config: Stage2ModelConfigV2,
    train_config: Stage2TrainConfigV2,
    train_frame: pd.DataFrame,
) -> dict:
    device = torch.device(train_config.device)
    model = Stage2NeuralRankerV2(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    temperature_stats = compute_temperature_stats(train_frame)

    history = []
    best_val_loss = float('inf')
    best_checkpoint = None

    for epoch in range(1, train_config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, train_config, temperature_stats)
        val_metrics = run_epoch(model, val_loader, None, train_config, temperature_stats)

        epoch_record = {
            'epoch': epoch,
            'train': train_metrics,
            'validate': val_metrics,
        }
        history.append(epoch_record)

        val_loss = val_metrics.get('loss_total', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = save_checkpoint(
                output_dir=output_dir,
                model=model,
                model_config=model_config,
                train_config=train_config,
                temperature_stats=temperature_stats,
                epoch=epoch,
                metrics=val_metrics,
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history_path = output_path / 'train_history.json'
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    result = {
        'best_checkpoint': str(best_checkpoint) if best_checkpoint is not None else '',
        'best_val_loss': best_val_loss,
        'temperature_stats': temperature_stats.to_dict(),
        'history_file': str(history_path),
        'history': history,
    }
    (output_path / 'train_summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return result
