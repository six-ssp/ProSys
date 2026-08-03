"""Evaluate a validation-gated GNN temperature branch with fixed ranking.

The no-GNN XGBoost ranker determines the route--condition order. A separate
temperature regressor trained with route-GNN features is accepted per family
only when validation MAE improves by a pre-specified margin. Consequently, the
GNN cannot lower any ranking or Sys@k metric in this branch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.mainline import FAMILY_ORDER, evaluate_scored_frame_with_manifest, parse_families_arg
from prosys_shared.route_cache import load_route_cache_sample_indices
from stage3_XGBoost import score_table_with_xgb


DEFAULT_MAINLINE_ROOT = REPO_ROOT / 'outputs' / 'stage23_mainline_reafnn_gnn_fused_20260723'
DEFAULT_ABLATION_ROOT = REPO_ROOT / 'outputs' / 'ablation_reafnn_gnn_20260726'
DEFAULT_VAL_ROUTE_ROOT = REPO_ROOT / 'outputs' / 'stage1_routes_validation'
DEFAULT_TEST_ROUTE_ROOT = REPO_ROOT / 'outputs' / 'stage1_routes'


def _score(
    table_file: Path,
    rank_model_dir: Path,
    temperature_model_dir: Path,
) -> pd.DataFrame:
    return score_table_with_xgb(
        table_file=table_file,
        model_file=rank_model_dir / 'xgb_ranker.json',
        metadata_file=rank_model_dir / 'xgb_ranker_meta.json',
        temperature_model_file=temperature_model_dir / 'xgb_temperature.json',
        temperature_metadata_file=temperature_model_dir / 'xgb_temperature_meta.json',
    )


def _metrics(frame: pd.DataFrame, route_cache: Path) -> dict:
    return evaluate_scored_frame_with_manifest(
        frame,
        expected_sample_indices=load_route_cache_sample_indices(route_cache),
        score_column='xgb_score',
        temperature_column='xgb_temperature_pred',
    )


def _temperature(metrics: dict) -> dict[str, float | int | None]:
    values = metrics.get('temperature') or {}
    return {
        'n': int(values.get('n') or 0),
        'mae': values.get('mae'),
        'mse': values.get('mse'),
        'rmse': values.get('rmse'),
        'within_5c': values.get('within_5c'),
        'within_10c': values.get('within_10c'),
        'within_20c': values.get('within_20c'),
    }


def _ranking(metrics: dict) -> dict[str, float]:
    return {
        'sys1': float(metrics['system_top1_all']),
        'sys3': float(metrics['system_top3_all']),
        'sys5': float(metrics['system_top5_all']),
        'sys10': float(metrics['system_top10_all']),
        'mrr': float(metrics['system_mrr']),
        'ndcg10': float(metrics['system_ndcg10']),
    }


def _flat(prefix: str, metrics: dict) -> dict:
    output = {f'{prefix}_{key}': value for key, value in _ranking(metrics).items()}
    output.update({f'{prefix}_temp_{key}': value for key, value in _temperature(metrics).items()})
    return output


def _valid_mae(metrics: dict) -> float | None:
    value = (metrics.get('temperature') or {}).get('mae')
    return float(value) if value is not None and np.isfinite(value) else None


def _write_tsv(frame: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, sep='\t', index=False, float_format='%.8f')


def run(args: argparse.Namespace) -> dict:
    mainline_root = Path(args.mainline_root).resolve()
    ablation_root = Path(args.ablation_root).resolve()
    val_route_root = Path(args.val_route_root).resolve()
    test_route_root = Path(args.test_route_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    families = [family for family in FAMILY_ORDER if family in parse_families_arg(args.families)]
    rows: list[dict] = []

    for family in families:
        table_root = mainline_root / family / '_shared_reaction_gnn' / 'training_tables'
        no_gnn_model_dir = ablation_root / family / 'no_gnn_xgb' / 'non_oracle' / 'model'
        gnn_model_dir = ablation_root / family / 'full_mainline' / 'non_oracle' / 'model'
        val_cache = val_route_root / family / 'route_cache.json'
        test_cache = test_route_root / family / 'route_cache.json'
        needed = [
            table_root / 'val.csv',
            table_root / 'test.csv',
            no_gnn_model_dir / 'xgb_ranker.json',
            no_gnn_model_dir / 'xgb_ranker_meta.json',
            no_gnn_model_dir / 'xgb_temperature.json',
            no_gnn_model_dir / 'xgb_temperature_meta.json',
            gnn_model_dir / 'xgb_temperature.json',
            gnn_model_dir / 'xgb_temperature_meta.json',
            val_cache,
            test_cache,
        ]
        missing = [str(path) for path in needed if not path.exists()]
        if missing:
            raise FileNotFoundError(f'{family} missing inputs: {missing}')

        val_base = _metrics(_score(table_root / 'val.csv', no_gnn_model_dir, no_gnn_model_dir), val_cache)
        val_gnn_temp = _metrics(_score(table_root / 'val.csv', no_gnn_model_dir, gnn_model_dir), val_cache)
        test_base = _metrics(_score(table_root / 'test.csv', no_gnn_model_dir, no_gnn_model_dir), test_cache)
        test_gnn_temp = _metrics(_score(table_root / 'test.csv', no_gnn_model_dir, gnn_model_dir), test_cache)
        base_mae = _valid_mae(val_base)
        gnn_mae = _valid_mae(val_gnn_temp)
        accepted = bool(
            base_mae is not None
            and gnn_mae is not None
            and gnn_mae <= base_mae - float(args.min_val_mae_improvement)
        )
        selected_val = val_gnn_temp if accepted else val_base
        selected_test = test_gnn_temp if accepted else test_base

        if _ranking(selected_test) != _ranking(test_base):
            raise AssertionError('Temperature-branch selection unexpectedly changed ranking metrics.')
        rows.append({
            'family': family,
            'gnn_temperature_accepted': accepted,
            'validation_mae_improvement_requirement_c': args.min_val_mae_improvement,
            **_flat('val_base', val_base),
            **_flat('val_gnn_temp', val_gnn_temp),
            **_flat('val_selected', selected_val),
            **_flat('test_base', test_base),
            **_flat('test_gnn_temp', test_gnn_temp),
            **_flat('test_selected', selected_test),
        })

    per_family = pd.DataFrame(rows)
    _write_tsv(per_family, output_root / 'per_family.tsv')
    accepted = per_family['gnn_temperature_accepted'].astype(bool)
    selected_mae = pd.to_numeric(per_family['test_selected_temp_mae'], errors='coerce')
    base_mae = pd.to_numeric(per_family['test_base_temp_mae'], errors='coerce')
    summary = {
        'num_families': int(len(per_family)),
        'families_with_gnn_temperature_branch': int(accepted.sum()),
        'macro_test_base_temperature_mae': float(base_mae.mean()),
        'macro_test_selected_temperature_mae': float(selected_mae.mean()),
        'macro_test_base_temperature_within_10c': float(pd.to_numeric(per_family['test_base_temp_within_10c']).mean()),
        'macro_test_selected_temperature_within_10c': float(pd.to_numeric(per_family['test_selected_temp_within_10c']).mean()),
        'ranking_unchanged_by_design': True,
    }
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (output_root / 'README.md').write_text(
        '# Validation-gated GNN temperature branch\n\n'
        'The Stage 3 no-GNN ranker fixes candidate order. The GNN is retained '
        'only in an independently trained temperature regressor. For each family, '
        'the GNN temperature model is used only if its validation MAE improves '
        'over the no-GNN temperature model by the pre-specified threshold. Thus '
        'all Sys@k metrics are exactly unchanged by design; test labels are not '
        'used to select the branch.\n',
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
    parser.add_argument('--min_val_mae_improvement', type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
