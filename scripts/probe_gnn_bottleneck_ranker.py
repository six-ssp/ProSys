"""Validation-only Stage 3 probe with a low-dimensional GNN feature bottleneck.

The original full GNN branch gives XGBoost 64 route-level features. This probe
fits PCA on training-route embeddings only, retains a small fixed number of
components, and retrains the Stage 3 ranker. The component count and any
nonzero GNN branch are selected on validation data only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.mainline import FAMILY_ORDER, evaluate_scored_frame_with_manifest, parse_families_arg
from prosys_shared.route_cache import load_route_cache_sample_indices
from stage3_XGBoost import score_table_with_xgb, train_xgb_ranker_and_temperature


DEFAULT_MAINLINE_ROOT = REPO_ROOT / 'outputs' / 'stage23_mainline_reafnn_gnn_fused_20260723'
DEFAULT_ABLATION_ROOT = REPO_ROOT / 'outputs' / 'ablation_reafnn_gnn_20260726'
DEFAULT_VAL_ROUTE_ROOT = REPO_ROOT / 'outputs' / 'stage1_routes_validation'
DEFAULT_TEST_ROUTE_ROOT = REPO_ROOT / 'outputs' / 'stage1_routes'


def _route_gnn_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith('route_gnn_feat_')]
    columns.sort(key=lambda column: int(column.rsplit('_', 1)[1]))
    if not columns:
        raise ValueError('Candidate table does not contain route-GNN features.')
    return columns


def _metrics(frame: pd.DataFrame, route_cache: Path, score_column: str) -> dict:
    return evaluate_scored_frame_with_manifest(
        frame,
        expected_sample_indices=load_route_cache_sample_indices(route_cache),
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
        round(float(candidate['val_metrics']['system_top10_all']), 12),
        round(float(candidate['val_metrics']['system_top1_all']), 12),
        round(float(candidate['val_metrics']['system_mrr']), 12),
        -int(candidate['components']),
    )
    incumbent_key = (
        round(float(incumbent['val_metrics']['system_top10_all']), 12),
        round(float(incumbent['val_metrics']['system_top1_all']), 12),
        round(float(incumbent['val_metrics']['system_mrr']), 12),
        -int(incumbent['components']),
    )
    return candidate_key > incumbent_key


def _compress_tables(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    components: int,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    feature_columns = _route_gnn_columns(train)
    route_columns = [column for column in ('reactants', 'product') if column in train.columns]
    fit_frame = train.drop_duplicates(route_columns) if route_columns else train
    fit_values = fit_frame.loc[:, feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    pca = PCA(n_components=components, svd_solver='full', random_state=seed)
    pca.fit(fit_values)

    outputs: list[pd.DataFrame] = []
    for frame in (train, val, test):
        work = frame.drop(columns=feature_columns).copy()
        transformed = pca.transform(frame.loc[:, feature_columns].fillna(0.0).to_numpy(dtype=np.float32))
        for index in range(components):
            work[f'route_gnn_bottleneck_{index}'] = transformed[:, index].astype(np.float32)
        outputs.append(work)
    return (*outputs, float(np.sum(pca.explained_variance_ratio_)))


def _score_no_gnn(table: Path, model_dir: Path) -> pd.DataFrame:
    return score_table_with_xgb(
        table_file=table,
        model_file=model_dir / 'xgb_ranker.json',
        metadata_file=model_dir / 'xgb_ranker_meta.json',
    )


def _write_tsv(frame: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, sep='\t', index=False, float_format='%.8f')


def run(args: argparse.Namespace) -> dict:
    mainline_root = Path(args.mainline_root).resolve()
    ablation_root = Path(args.ablation_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    val_root = Path(args.val_route_root).resolve()
    test_root = Path(args.test_route_root).resolve()
    components_grid = [int(value) for value in args.components.split(',')]
    if any(value <= 0 for value in components_grid):
        raise ValueError('GNN bottleneck component counts must be positive.')
    families = [family for family in FAMILY_ORDER if family in parse_families_arg(args.families)]

    per_family_rows: list[dict] = []
    grid_rows: list[dict] = []
    for family in families:
        table_root = mainline_root / family / '_shared_reaction_gnn' / 'training_tables'
        train_file = table_root / 'train.csv'
        val_file = table_root / 'val.csv'
        test_file = table_root / 'test.csv'
        no_gnn_model_dir = ablation_root / family / 'no_gnn_xgb' / 'non_oracle' / 'model'
        val_cache = val_root / family / 'route_cache.json'
        test_cache = test_root / family / 'route_cache.json'
        needed = [
            train_file,
            val_file,
            test_file,
            no_gnn_model_dir / 'xgb_ranker.json',
            no_gnn_model_dir / 'xgb_ranker_meta.json',
            val_cache,
            test_cache,
        ]
        missing = [str(path) for path in needed if not path.exists()]
        if missing:
            raise FileNotFoundError(f'{family} missing inputs: {missing}')

        train = pd.read_csv(train_file)
        val = pd.read_csv(val_file)
        test = pd.read_csv(test_file)
        base_val_metrics = _metrics(_score_no_gnn(val_file, no_gnn_model_dir), val_cache, 'xgb_score')
        base_test_metrics = _metrics(_score_no_gnn(test_file, no_gnn_model_dir), test_cache, 'xgb_score')
        candidates: list[dict] = []

        with tempfile.TemporaryDirectory(prefix=f'prosys_gnn_bottleneck_{family}_') as temporary_dir:
            temporary_root = Path(temporary_dir)
            for components in components_grid:
                train_comp, val_comp, test_comp, variance = _compress_tables(
                    train, val, test, components, seed=args.seed
                )
                config_root = temporary_root / f'pca{components}'
                train_path = config_root / 'train.csv'
                val_path = config_root / 'val.csv'
                test_path = config_root / 'test.csv'
                config_root.mkdir(parents=True, exist_ok=True)
                train_comp.to_csv(train_path, index=False)
                val_comp.to_csv(val_path, index=False)
                test_comp.to_csv(test_path, index=False)
                artifacts = train_xgb_ranker_and_temperature(
                    train_table_file=train_path,
                    val_table_file=val_path,
                    output_dir=config_root / 'model',
                    random_state=args.seed,
                )
                val_scored = score_table_with_xgb(
                    table_file=val_path,
                    model_file=artifacts['model_file'],
                    metadata_file=artifacts['metadata_file'],
                )
                test_scored = score_table_with_xgb(
                    table_file=test_path,
                    model_file=artifacts['model_file'],
                    metadata_file=artifacts['metadata_file'],
                )
                candidate = {
                    'components': components,
                    'variance_explained': variance,
                    'val_metrics': _metrics(val_scored, val_cache, 'xgb_score'),
                    'test_metrics': _metrics(test_scored, test_cache, 'xgb_score'),
                }
                candidates.append(candidate)
                grid_rows.append({
                    'family': family,
                    'components': components,
                    'variance_explained': variance,
                    **{f'val_{key}': value for key, value in _compact(candidate['val_metrics']).items()},
                    **{f'test_{key}': value for key, value in _compact(candidate['test_metrics']).items()},
                })

        best = candidates[0]
        for candidate in candidates[1:]:
            if _better(candidate, best):
                best = candidate
        accepted = bool(
            float(best['val_metrics']['system_top10_all'])
            >= float(base_val_metrics['system_top10_all']) + args.min_val_sys10_gain
        )
        selected = best if accepted else None
        per_family_rows.append({
            'family': family,
            'selected_components': int(selected['components']) if selected else 0,
            'selected_variance_explained': float(selected['variance_explained']) if selected else 0.0,
            'validation_gain_requirement_sys10': args.min_val_sys10_gain,
            'gnn_bottleneck_accepted': accepted,
            **{f'val_base_{key}': value for key, value in _compact(base_val_metrics).items()},
            **{f'val_selected_{key}': value for key, value in _compact(selected['val_metrics'] if selected else base_val_metrics).items()},
            **{f'test_base_{key}': value for key, value in _compact(base_test_metrics).items()},
            **{f'test_selected_{key}': value for key, value in _compact(selected['test_metrics'] if selected else base_test_metrics).items()},
        })

    per_family = pd.DataFrame(per_family_rows)
    _write_tsv(per_family, output_root / 'per_family.tsv')
    _write_tsv(pd.DataFrame(grid_rows), output_root / 'validation_grid.tsv')
    summary = {
        'num_families': len(per_family),
        'mean_test_base_sys10': float(per_family['test_base_sys10'].mean()),
        'mean_test_selected_sys10': float(per_family['test_selected_sys10'].mean()),
        'mean_test_base_sys1': float(per_family['test_base_sys1'].mean()),
        'mean_test_selected_sys1': float(per_family['test_selected_sys1'].mean()),
        'families_with_accepted_gnn_bottleneck': int(per_family['gnn_bottleneck_accepted'].sum()),
    }
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (output_root / 'README.md').write_text(
        '# Validation-gated GNN bottleneck probe\n\n'
        'PCA is fitted on unique training routes only. Raw 64-dimensional GNN '
        'features are removed before XGBoost training and replaced by the stated '
        'number of bottleneck components. Component count is selected on '
        'validation Sys@10 with a fixed minimum gain requirement. Test labels '
        'are never used for selection, and formal outputs are unchanged.\n',
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
    parser.add_argument('--components', type=str, default='2,4,8,16')
    parser.add_argument('--min_val_sys10_gain', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
