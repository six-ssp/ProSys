"""Evaluation for the ProSys Stage 2 V2 neural ranker.

A single evaluator serves both settings from `stage2_detail.md`:

- **Oracle** (section 8, stage 1): the candidate table is built from the gold
  route, so every candidate has ``route_match == 1`` and the task reduces to
  ranking the correct *context*.
- **Non-Oracle** (section 8, stage 2): the candidate table is built from a
  Stage 1 route cache, so ``route_match`` varies and top-k measures the full
  route+context system.

The metric code is identical for both; the mode only labels the output and
selects which candidate table is passed in. Reported metrics follow
``ProSys_goal.md`` section 4 (route hit, context hit, end-to-end/system hit,
candidate-pool coverage, temperature)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .dataset import Stage2CandidateDataLoaderV2, Stage2CandidateDatasetV2
from .model import Stage2ModelConfigV2, Stage2NeuralRankerV2
from .trainer import Stage2TrainConfigV2, TemperatureStats

DEFAULT_TOPKS = (1, 3, 5, 10)


@dataclass
class LoadedModel:
    model: Stage2NeuralRankerV2
    model_config: Stage2ModelConfigV2
    train_config: Stage2TrainConfigV2
    temperature_stats: TemperatureStats


def load_stage2_model_v2(checkpoint_path: str | Path, device: str = 'cpu') -> LoadedModel:
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_config = Stage2ModelConfigV2(**state['model_config'])
    train_config = Stage2TrainConfigV2(**state['train_config'])
    temperature_stats = TemperatureStats(**state['temperature_stats'])

    model = Stage2NeuralRankerV2(model_config).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return LoadedModel(model, model_config, train_config, temperature_stats)


@torch.no_grad()
def collect_predictions(model: Stage2NeuralRankerV2, loader, device: str) -> dict[str, np.ndarray]:
    """Run the model over a candidate loader and gather per-candidate arrays."""
    torch_device = torch.device(device)
    keys = ('sample_index', 'label', 'route_match', 'context_match', 'temperature_gold')
    collected: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    collected['score_logit'] = []
    collected['temperature_pred'] = []

    for batch in loader:
        outputs = model(
            route_fp=batch['route_fp'].to(torch_device),
            route_graph_features=batch['route_graph_features'].to(torch_device),
            route_dense_features=batch['route_dense_features'].to(torch_device),
            context_dense_features=batch['context_dense_features'].to(torch_device),
            product_fp=batch['product_fp'].to(torch_device),
            product_features=batch['product_features'].to(torch_device),
            reagent_features=batch['reagent_features'].to(torch_device),
            solvent_features=batch['solvent_features'].to(torch_device),
            support_features=batch['support_features'].to(torch_device),
        )
        collected['score_logit'].append(outputs['score_logit'].detach().cpu().numpy())
        collected['temperature_pred'].append(outputs['temperature_pred'].detach().cpu().numpy())
        for key in keys:
            collected[key].append(batch[key].detach().cpu().numpy())

    return {key: np.concatenate(values) if values else np.array([]) for key, values in collected.items()}


def _denormalize_temperature(pred: np.ndarray, stats: TemperatureStats, zscore: bool) -> np.ndarray:
    if zscore:
        return pred * stats.std + stats.mean
    return pred


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    temperature_stats: TemperatureStats,
    *,
    temperature_zscore: bool,
    topks: tuple[int, ...] = DEFAULT_TOPKS,
) -> dict:
    """Compute slate-level ranking metrics + temperature error.

    ``system`` hit uses ``label`` (route_match AND context_match); ``context``
    uses ``context_match``; ``route`` uses ``route_match``.
    """
    sample_index = predictions['sample_index'].astype(np.int64)
    score = predictions['score_logit'].astype(np.float64)
    label = predictions['label'].astype(np.float64)
    route_match = predictions['route_match'].astype(np.float64)
    context_match = predictions['context_match'].astype(np.float64)
    temp_gold = predictions['temperature_gold'].astype(np.float64)
    temp_pred = _denormalize_temperature(
        predictions['temperature_pred'].astype(np.float64), temperature_stats, temperature_zscore
    )

    hit_counters = {f'system_top{k}': 0 for k in topks}
    hit_counters.update({f'context_top{k}': 0 for k in topks})
    hit_counters.update({f'route_top{k}': 0 for k in topks})

    num_slates = 0
    covered_slates = 0  # slate has at least one gold positive candidate in the pool
    covered_hit = {f'system_top{k}': 0 for k in topks}

    temp_abs_errors: list[float] = []

    for group_id in np.unique(sample_index):
        mask = sample_index == group_id
        group_score = score[mask]
        if group_score.size == 0:
            continue
        num_slates += 1

        order = np.argsort(-group_score, kind='stable')
        group_label = label[mask][order]
        group_context = context_match[mask][order]
        group_route = route_match[mask][order]

        has_positive = bool(np.any(group_label > 0.5))
        if has_positive:
            covered_slates += 1

        for k in topks:
            sys_hit = bool(np.any(group_label[:k] > 0.5))
            hit_counters[f'system_top{k}'] += int(sys_hit)
            hit_counters[f'context_top{k}'] += int(np.any(group_context[:k] > 0.5))
            hit_counters[f'route_top{k}'] += int(np.any(group_route[:k] > 0.5))
            if has_positive:
                covered_hit[f'system_top{k}'] += int(sys_hit)

        # Temperature: evaluate on the top-ranked gold-positive candidate (the
        # temperature head is only trained on positives).
        group_gold_t = temp_gold[mask][order]
        group_pred_t = temp_pred[mask][order]
        positive_rows = np.where((group_label > 0.5) & np.isfinite(group_gold_t))[0]
        if positive_rows.size > 0:
            row = positive_rows[0]
            temp_abs_errors.append(abs(group_pred_t[row] - group_gold_t[row]))

    def _rate(counter: dict[str, int], denom: int) -> dict[str, float]:
        return {key: (value / denom if denom else 0.0) for key, value in counter.items()}

    metrics: dict = {
        'num_slates': num_slates,
        'pool_coverage': (covered_slates / num_slates if num_slates else 0.0),
        'covered_slates': covered_slates,
    }
    metrics.update({f'{name}_all': rate for name, rate in _rate(hit_counters, num_slates).items()})
    # accuracy restricted to slates whose pool actually contains the gold system
    for k in topks:
        key = f'system_top{k}'
        metrics[f'{key}_covered'] = covered_hit[key] / covered_slates if covered_slates else 0.0

    if temp_abs_errors:
        errors = np.asarray(temp_abs_errors, dtype=np.float64)
        metrics['temperature'] = {
            'n': int(errors.size),
            'mae': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'within_10c': float(np.mean(errors <= 10.0)),
            'within_20c': float(np.mean(errors <= 20.0)),
        }
    else:
        metrics['temperature'] = {'n': 0, 'mae': None, 'rmse': None, 'within_10c': None, 'within_20c': None}

    return metrics


def run_stage2_v2_eval(
    *,
    family_dir: str | Path,
    candidate_table: str | Path,
    checkpoint_path: str | Path,
    device: str = 'cpu',
    mode: str = 'oracle',
    slates_per_batch: int = 8,
    num_workers: int = 0,
    route_fpsize: int = 4096,
    product_fpsize: int = 2048,
    radius: int = 2,
    topks: tuple[int, ...] = DEFAULT_TOPKS,
    output_file: str | Path | None = None,
) -> dict:
    loaded = load_stage2_model_v2(checkpoint_path, device=device)

    dataset = Stage2CandidateDatasetV2(
        candidate_table_file=candidate_table,
        family_dir=family_dir,
        route_fpsize=route_fpsize,
        product_fpsize=product_fpsize,
        radius=radius,
    )
    loader = Stage2CandidateDataLoaderV2(
        dataset,
        slates_per_batch=slates_per_batch,
        shuffle=False,
        num_workers=num_workers,
    )

    predictions = collect_predictions(loaded.model, loader, device=device)
    metrics = evaluate_predictions(
        predictions,
        loaded.temperature_stats,
        temperature_zscore=loaded.train_config.temperature_zscore_by_family,
        topks=topks,
    )

    result = {
        'mode': mode,
        'family': dataset.family,
        'candidate_table': str(candidate_table),
        'checkpoint': str(checkpoint_path),
        'num_candidates': int(len(dataset)),
        'metrics': metrics,
    }

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    return result
