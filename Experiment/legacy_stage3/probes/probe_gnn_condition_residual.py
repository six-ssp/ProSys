"""Validation-only probe for a candidate-specific GNN residual in Stage 3.

This script never trains on, calibrates against, or selects parameters using
the test set. It reuses the saved GNN auxiliary heads to score each proposed
reagent/solvent context, then tunes a small residual-fusion grid on validation
metrics only. It writes a standalone experimental directory and does not
replace any formal mainline or ablation artifact.
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

from prosys_shared.mainline import (
    FAMILY_ORDER,
    evaluate_scored_frame_with_manifest,
    parse_families_arg,
)
from prosys_shared.route_cache import load_route_cache_sample_indices
from stage3_XGBoost import score_table_with_xgb
from stage3_XGBoost.reaction_gnn_features import ReactionGNNFeatureEncoder


DEFAULT_MAINLINE_ROOT = REPO_ROOT / 'outputs' / 'stage23_mainline_reafnn_gnn_fused_20260723'
DEFAULT_ABLATION_ROOT = REPO_ROOT / 'outputs' / 'ablation_reafnn_gnn_20260726'
DEFAULT_VAL_ROUTE_ROOT = REPO_ROOT / 'outputs' / 'stage1_routes_validation'
DEFAULT_TEST_ROUTE_ROOT = REPO_ROOT / 'outputs' / 'stage1_routes'


def _zscore_within_slate(values: pd.Series, sample_indices: pd.Series) -> np.ndarray:
    work = pd.DataFrame({'sample_index': sample_indices.to_numpy(), 'value': values.to_numpy(dtype=np.float32)})
    grouped = work.groupby('sample_index', sort=False)['value']
    mean = grouped.transform('mean')
    std = grouped.transform('std').fillna(0.0)
    return ((work['value'] - mean) / std.mask(std <= 1e-8, 1.0)).to_numpy(dtype=np.float32)


def _metrics(frame: pd.DataFrame, route_cache_file: Path, score_column: str) -> dict:
    return evaluate_scored_frame_with_manifest(
        frame,
        expected_sample_indices=load_route_cache_sample_indices(route_cache_file),
        score_column=score_column,
    )


def _summary_metrics(metrics: dict) -> dict[str, float]:
    return {
        'sys1': float(metrics['system_top1_all']),
        'sys3': float(metrics['system_top3_all']),
        'sys5': float(metrics['system_top5_all']),
        'sys10': float(metrics['system_top10_all']),
        'mrr': float(metrics['system_mrr']),
        'ndcg10': float(metrics['system_ndcg10']),
    }


def _better(candidate: dict, incumbent: dict) -> bool:
    """Use fixed, validation-only tie breaks and prefer less GNN influence."""
    candidate_key = (
        round(float(candidate['metrics']['system_top10_all']), 12),
        round(float(candidate['metrics']['system_top1_all']), 12),
        round(float(candidate['metrics']['system_mrr']), 12),
        -float(candidate['alpha']),
        -abs(float(candidate['reagent_weight']) - 0.5),
    )
    incumbent_key = (
        round(float(incumbent['metrics']['system_top10_all']), 12),
        round(float(incumbent['metrics']['system_top1_all']), 12),
        round(float(incumbent['metrics']['system_mrr']), 12),
        -float(incumbent['alpha']),
        -abs(float(incumbent['reagent_weight']) - 0.5),
    )
    return candidate_key > incumbent_key


def _candidate_table(mainline_root: Path, family: str, split: str) -> Path:
    return mainline_root / family / '_shared_reaction_gnn' / 'training_tables' / f'{split}.csv'


def _no_gnn_model_dir(ablation_root: Path, family: str) -> Path:
    return ablation_root / family / 'no_gnn_xgb' / 'non_oracle' / 'model'


def _score_no_gnn(table_file: Path, model_dir: Path) -> pd.DataFrame:
    return score_table_with_xgb(
        table_file=table_file,
        model_file=model_dir / 'xgb_ranker.json',
        metadata_file=model_dir / 'xgb_ranker_meta.json',
    )


def _attach_gnn_residual_scores(frame: pd.DataFrame, encoder: ReactionGNNFeatureEncoder) -> pd.DataFrame:
    work = encoder.score_condition_candidates(frame)
    work['route_gnn_reagent_compatibility_z'] = _zscore_within_slate(
        work['route_gnn_reagent_compatibility'], work['sample_index']
    )
    work['route_gnn_solvent_compatibility_z'] = _zscore_within_slate(
        work['route_gnn_solvent_compatibility'], work['sample_index']
    )
    return work


def _fused_frame(frame: pd.DataFrame, *, alpha: float, reagent_weight: float) -> pd.DataFrame:
    work = frame.copy()
    gnn_score = (
        reagent_weight * work['route_gnn_reagent_compatibility_z'].to_numpy(dtype=np.float32)
        + (1.0 - reagent_weight) * work['route_gnn_solvent_compatibility_z'].to_numpy(dtype=np.float32)
    )
    work['gnn_condition_residual'] = gnn_score.astype(np.float32)
    work['gnn_condition_fused_score'] = (
        work['xgb_score'].to_numpy(dtype=np.float32) + alpha * gnn_score
    ).astype(np.float32)
    return work


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False, float_format='%.8f')


def run(args: argparse.Namespace) -> dict:
    mainline_root = Path(args.mainline_root).resolve()
    ablation_root = Path(args.ablation_root).resolve()
    val_route_root = Path(args.val_route_root).resolve()
    test_route_root = Path(args.test_route_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    families = parse_families_arg(args.families)
    families = [family for family in FAMILY_ORDER if family in families]
    alphas = [float(value) for value in args.alphas.split(',')]
    reagent_weights = [float(value) for value in args.reagent_weights.split(',')]
    min_val_gain = float(args.min_val_sys10_gain)

    per_family: list[dict] = []
    grid_rows: list[dict] = []

    for family in families:
        val_table = _candidate_table(mainline_root, family, 'val')
        test_table = _candidate_table(mainline_root, family, 'test')
        model_dir = _no_gnn_model_dir(ablation_root, family)
        gnn_dir = mainline_root / family / '_shared_reaction_gnn' / 'model'
        val_route_cache = val_route_root / family / 'route_cache.json'
        test_route_cache = test_route_root / family / 'route_cache.json'
        required = [
            val_table,
            test_table,
            model_dir / 'xgb_ranker.json',
            model_dir / 'xgb_ranker_meta.json',
            gnn_dir / 'reaction_gnn.pt',
            val_route_cache,
            test_route_cache,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f'{family} is missing probe input(s): {missing}')

        encoder = ReactionGNNFeatureEncoder(gnn_dir, device=args.device)
        val = _attach_gnn_residual_scores(_score_no_gnn(val_table, model_dir), encoder)
        test = _attach_gnn_residual_scores(_score_no_gnn(test_table, model_dir), encoder)
        base_metrics = _metrics(val, val_route_cache, 'xgb_score')
        base_test_metrics = _metrics(test, test_route_cache, 'xgb_score')

        candidates: list[dict] = []
        for reagent_weight in reagent_weights:
            for alpha in alphas:
                candidate_val = _fused_frame(val, alpha=alpha, reagent_weight=reagent_weight)
                metrics = _metrics(candidate_val, val_route_cache, 'gnn_condition_fused_score')
                candidate = {
                    'alpha': alpha,
                    'reagent_weight': reagent_weight,
                    'metrics': metrics,
                }
                candidates.append(candidate)
                grid_rows.append({
                    'family': family,
                    'alpha': alpha,
                    'reagent_weight': reagent_weight,
                    **{f'val_{key}': value for key, value in _summary_metrics(metrics).items()},
                })

        best = candidates[0]
        for candidate in candidates[1:]:
            if _better(candidate, best):
                best = candidate

        # Require a meaningful primary-metric validation gain before allowing a
        # nonzero GNN residual. Otherwise the exact no-GNN score is retained.
        accepted = bool(
            best['alpha'] > 0.0
            and float(best['metrics']['system_top10_all'])
            >= float(base_metrics['system_top10_all']) + min_val_gain
        )
        selected = best if accepted else {
            'alpha': 0.0,
            'reagent_weight': 0.5,
            'metrics': base_metrics,
        }

        selected_test = _fused_frame(test, alpha=selected['alpha'], reagent_weight=selected['reagent_weight'])
        selected_test_metrics = _metrics(selected_test, test_route_cache, 'gnn_condition_fused_score')

        per_family.append({
            'family': family,
            'selected_alpha': selected['alpha'],
            'selected_reagent_weight': selected['reagent_weight'],
            'validation_gain_requirement_sys10': min_val_gain,
            'gnn_residual_accepted': accepted,
            **{f'val_base_{key}': value for key, value in _summary_metrics(base_metrics).items()},
            **{f'val_selected_{key}': value for key, value in _summary_metrics(selected['metrics']).items()},
            **{f'test_base_{key}': value for key, value in _summary_metrics(base_test_metrics).items()},
            **{f'test_selected_{key}': value for key, value in _summary_metrics(selected_test_metrics).items()},
        })

    per_family_frame = pd.DataFrame(per_family)
    _write_tsv(per_family_frame, output_root / 'per_family.tsv')
    _write_tsv(pd.DataFrame(grid_rows), output_root / 'validation_grid.tsv')

    macro = {
        'num_families': len(per_family),
        'mean_test_base_sys1': float(per_family_frame['test_base_sys1'].mean()),
        'mean_test_selected_sys1': float(per_family_frame['test_selected_sys1'].mean()),
        'mean_test_base_sys10': float(per_family_frame['test_base_sys10'].mean()),
        'mean_test_selected_sys10': float(per_family_frame['test_selected_sys10'].mean()),
        'mean_test_base_mrr': float(per_family_frame['test_base_mrr'].mean()),
        'mean_test_selected_mrr': float(per_family_frame['test_selected_mrr'].mean()),
        'families_with_nonzero_gnn': int(per_family_frame['gnn_residual_accepted'].sum()),
    }
    (output_root / 'summary.json').write_text(json.dumps(macro, indent=2) + '\n', encoding='utf-8')
    (output_root / 'README.md').write_text(
        '# Validation-gated candidate-specific GNN residual probe\n\n'
        'This is an isolated Stage 3 experiment. The existing no-GNN XGBoost '
        'ranker is kept fixed. The GNN auxiliary reagent and solvent heads score '
        'each candidate context; their within-slate standardized score is added '
        'as a small residual. `alpha` and the reagent/solvent balance are chosen '
        'using validation data only. A nonzero residual is accepted only when it '
        'improves validation Sys@10 by the pre-specified threshold. Test labels '
        'are not used for model, grid, or gate selection.\n',
        encoding='utf-8',
    )
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mainline_root', type=Path, default=DEFAULT_MAINLINE_ROOT)
    parser.add_argument('--ablation_root', type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument('--val_route_root', type=Path, default=DEFAULT_VAL_ROUTE_ROOT)
    parser.add_argument('--test_route_root', type=Path, default=DEFAULT_TEST_ROUTE_ROOT)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--alphas', type=str, default='0,0.05,0.10,0.20,0.30')
    parser.add_argument('--reagent_weights', type=str, default='0,0.5,1')
    parser.add_argument('--min_val_sys10_gain', type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
