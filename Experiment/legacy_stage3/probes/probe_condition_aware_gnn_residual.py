"""Probe a validation-gated condition-aware GNN residual for Stage 3.

The GNN residual model trains on each family's Stage 3 training table only.
Its score is fused with a fixed no-GNN XGBoost ranker. All fusion choices are
made on validation metrics; test data is read only after selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.mainline import FAMILY_ORDER, evaluate_scored_frame_with_manifest, parse_families_arg
from prosys_shared.route_cache import load_route_cache_sample_indices
from stage3_XGBoost import score_table_with_xgb
from Experiment.legacy_stage3.condition_aware_gnn import (
    ConditionAwareGNNConfig,
    score_condition_aware_gnn_ranker,
    train_condition_aware_gnn_ranker,
)


DEFAULT_MAINLINE_ROOT = REPO_ROOT / 'outputs' / 'stage23_mainline_reafnn_gnn_fused_20260723'
DEFAULT_ABLATION_ROOT = REPO_ROOT / 'outputs' / 'ablation_reafnn_gnn_20260726'
DEFAULT_VAL_ROUTE_ROOT = REPO_ROOT / 'outputs' / 'stage1_routes_validation'
DEFAULT_TEST_ROUTE_ROOT = REPO_ROOT / 'outputs' / 'stage1_routes'


def _zscore_within_slate(values: pd.Series, sample_indices: pd.Series) -> np.ndarray:
    work = pd.DataFrame({'sample_index': sample_indices.to_numpy(), 'value': values.to_numpy(dtype=np.float32)})
    groups = work.groupby('sample_index', sort=False)['value']
    std = groups.transform('std').fillna(0.0).mask(lambda value: value <= 1e-8, 1.0)
    return ((work['value'] - groups.transform('mean')) / std).to_numpy(dtype=np.float32)


def _metrics(frame: pd.DataFrame, route_cache_file: Path, score_column: str) -> dict:
    return evaluate_scored_frame_with_manifest(
        frame,
        expected_sample_indices=load_route_cache_sample_indices(route_cache_file),
        score_column=score_column,
    )


def _compact(metrics: dict) -> dict[str, float]:
    return {
        'sys1': float(metrics['system_top1_all']),
        'sys3': float(metrics['system_top3_all']),
        'sys5': float(metrics['system_top5_all']),
        'sys10': float(metrics['system_top10_all']),
        'mrr': float(metrics['system_mrr']),
        'ndcg10': float(metrics['system_ndcg10']),
    }


def _better(candidate: dict, incumbent: dict) -> bool:
    candidate_key = (
        round(float(candidate['metrics']['system_top10_all']), 12),
        round(float(candidate['metrics']['system_top1_all']), 12),
        round(float(candidate['metrics']['system_mrr']), 12),
        -float(candidate['alpha']),
    )
    incumbent_key = (
        round(float(incumbent['metrics']['system_top10_all']), 12),
        round(float(incumbent['metrics']['system_top1_all']), 12),
        round(float(incumbent['metrics']['system_mrr']), 12),
        -float(incumbent['alpha']),
    )
    return candidate_key > incumbent_key


def _score_xgb(table_file: Path, model_dir: Path) -> pd.DataFrame:
    return score_table_with_xgb(
        table_file=table_file,
        model_file=model_dir / 'xgb_ranker.json',
        metadata_file=model_dir / 'xgb_ranker_meta.json',
    )


def _fused_frame(xgb_frame: pd.DataFrame, gnn_frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    key_columns = ['sample_index', 'reaction_id', 'reactants', 'product', 'reagent_norm', 'solvent_norm']
    gnn_scores = gnn_frame.loc[:, key_columns + ['condition_aware_gnn_score_raw']]
    work = xgb_frame.merge(gnn_scores, on=key_columns, how='left', validate='one_to_one', sort=False)
    if work['condition_aware_gnn_score_raw'].isna().any():
        raise ValueError('Condition-aware GNN scores did not align with the XGBoost candidate table.')
    residual = _zscore_within_slate(work['condition_aware_gnn_score_raw'], work['sample_index'])
    work['condition_aware_gnn_score_z'] = residual
    work['condition_aware_gnn_fused_score'] = (
        work['xgb_score'].to_numpy(dtype=np.float32) + alpha * residual
    ).astype(np.float32)
    return work


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False, float_format='%.8f')


def run(args: argparse.Namespace) -> dict:
    mainline_root = Path(args.mainline_root).resolve()
    ablation_root = Path(args.ablation_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    val_root = Path(args.val_route_root).resolve()
    test_root = Path(args.test_route_root).resolve()
    alphas = [float(value) for value in args.alphas.split(',')]
    families = [family for family in FAMILY_ORDER if family in parse_families_arg(args.families)]
    rows: list[dict] = []
    grid_rows: list[dict] = []

    for family in families:
        family_root = mainline_root / family
        table_root = family_root / '_shared_reaction_gnn' / 'training_tables'
        train_table = table_root / 'train.csv'
        val_table = table_root / 'val.csv'
        test_table = table_root / 'test.csv'
        xgb_dir = ablation_root / family / 'no_gnn_xgb' / 'non_oracle' / 'model'
        val_cache = val_root / family / 'route_cache.json'
        test_cache = test_root / family / 'route_cache.json'
        needed = [
            train_table,
            val_table,
            test_table,
            xgb_dir / 'xgb_ranker.json',
            xgb_dir / 'xgb_ranker_meta.json',
            val_cache,
            test_cache,
        ]
        missing = [str(path) for path in needed if not path.exists()]
        if missing:
            raise FileNotFoundError(f'{family} missing inputs: {missing}')

        model_dir = output_root / family / 'model'
        artifacts = train_condition_aware_gnn_ranker(
            train_table,
            model_dir,
            config=ConditionAwareGNNConfig(
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                device=args.device,
                random_state=args.seed,
            ),
            force_retrain=args.force_retrain,
        )
        val_xgb = _score_xgb(val_table, xgb_dir)
        test_xgb = _score_xgb(test_table, xgb_dir)
        val_gnn = score_condition_aware_gnn_ranker(val_table, artifacts['model_file'], device=args.device)
        test_gnn = score_condition_aware_gnn_ranker(test_table, artifacts['model_file'], device=args.device)
        val_base = _metrics(val_xgb, val_cache, 'xgb_score')
        test_base = _metrics(test_xgb, test_cache, 'xgb_score')
        val_direct = _metrics(val_gnn, val_cache, 'condition_aware_gnn_score_raw')
        test_direct = _metrics(test_gnn, test_cache, 'condition_aware_gnn_score_raw')

        candidates: list[dict] = []
        for alpha in alphas:
            fused = _fused_frame(val_xgb, val_gnn, alpha)
            metrics = _metrics(fused, val_cache, 'condition_aware_gnn_fused_score')
            candidate = {'alpha': alpha, 'metrics': metrics}
            candidates.append(candidate)
            grid_rows.append({
                'family': family,
                'alpha': alpha,
                **{f'val_{key}': value for key, value in _compact(metrics).items()},
            })
        best = candidates[0]
        for candidate in candidates[1:]:
            if _better(candidate, best):
                best = candidate
        accepted = bool(
            best['alpha'] > 0.0
            and float(best['metrics']['system_top10_all'])
            >= float(val_base['system_top10_all']) + args.min_val_sys10_gain
        )
        selected = best if accepted else {'alpha': 0.0, 'metrics': val_base}
        selected_test = _fused_frame(test_xgb, test_gnn, selected['alpha'])
        selected_test_metrics = _metrics(selected_test, test_cache, 'condition_aware_gnn_fused_score')

        rows.append({
            'family': family,
            'selected_alpha': selected['alpha'],
            'validation_gain_requirement_sys10': args.min_val_sys10_gain,
            'gnn_residual_accepted': accepted,
            'model_file': artifacts['model_file'],
            **{f'val_base_{key}': value for key, value in _compact(val_base).items()},
            **{f'val_direct_gnn_{key}': value for key, value in _compact(val_direct).items()},
            **{f'val_selected_{key}': value for key, value in _compact(selected['metrics']).items()},
            **{f'test_base_{key}': value for key, value in _compact(test_base).items()},
            **{f'test_direct_gnn_{key}': value for key, value in _compact(test_direct).items()},
            **{f'test_selected_{key}': value for key, value in _compact(selected_test_metrics).items()},
        })

    per_family = pd.DataFrame(rows)
    _write_tsv(per_family, output_root / 'per_family.tsv')
    _write_tsv(pd.DataFrame(grid_rows), output_root / 'validation_grid.tsv')
    summary = {
        'num_families': len(per_family),
        'mean_test_base_sys10': float(per_family['test_base_sys10'].mean()),
        'mean_test_selected_sys10': float(per_family['test_selected_sys10'].mean()),
        'mean_test_base_sys1': float(per_family['test_base_sys1'].mean()),
        'mean_test_selected_sys1': float(per_family['test_selected_sys1'].mean()),
        'families_with_nonzero_gnn': int(per_family['gnn_residual_accepted'].sum()),
    }
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (output_root / 'README.md').write_text(
        '# Condition-aware GNN residual probe\n\n'
        'The frozen 64-dimensional route-GNN embedding is combined with learned '
        'reagent and solvent token embeddings in a candidate-specific interaction '
        'network. The interaction head is trained on the Stage 3 training table '
        'only. A score-level residual is selected on validation data only and '
        'requires a pre-specified validation Sys@10 gain. Formal mainline and '
        'ablation outputs remain unchanged.\n',
        encoding='utf-8',
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--mainline_root', type=Path, default=DEFAULT_MAINLINE_ROOT)
    parser.add_argument('--ablation_root', type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument('--val_route_root', type=Path, default=DEFAULT_VAL_ROUTE_ROOT)
    parser.add_argument('--test_route_root', type=Path, default=DEFAULT_TEST_ROUTE_ROOT)
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--alphas', type=str, default='0,0.025,0.05,0.10,0.15,0.20,0.30')
    parser.add_argument('--min_val_sys10_gain', type=float, default=0.01)
    parser.add_argument('--force_retrain', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
