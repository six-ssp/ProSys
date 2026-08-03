"""Collect, compact, and summarize controlled Stage-2/3 seed experiments.

The experiment deliberately freezes the persisted Stage-1 route caches. It
therefore measures variability from the learned Stage-2/3 modules and the B3
Sequential-FNN baseline without conflating it with a costly Stage-1 retrain.
Raw candidate tables are large, so this utility retains the metric records and
small configuration metadata before optionally pruning completed work roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.mainline import FAMILY_ORDER


SYSTEM_PROSYS = 'ProSys (Stage 2/3 fixed-Stage-1 robustness)'
SYSTEM_B3 = 'B3 Sequential FNN (fixed-Stage-1 robustness)'

METRIC_COLUMNS = [
    'cover',
    'sys1',
    'sys3',
    'sys5',
    'sys10',
    'mrr',
    'ndcg10',
    'temp_mae',
    'temp_within_5c',
    'temp_within_10c',
    'temp_within_20c',
]

ROW_COLUMNS = [
    'system',
    'seed',
    'family',
    'candidate_slates',
    'missing_candidate_slates',
    'temp_n',
    'rr1',
    'rr3',
    'rr5',
    'rr10',
    *METRIC_COLUMNS,
    'source_root',
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_run_spec(value: str) -> tuple[int, Path]:
    if '=' not in value:
        raise argparse.ArgumentTypeError('Run specification must be SEED=PATH.')
    seed_text, path_text = value.split('=', 1)
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'Invalid seed in {value!r}.') from exc
    if seed < 0:
        raise argparse.ArgumentTypeError('Seed must be non-negative.')
    if not path_text.strip():
        raise argparse.ArgumentTypeError('Run path cannot be empty.')
    return seed, Path(path_text)


def _resolve_run_specs(repo_root: Path, values: list[str], label: str) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for value in values:
        seed, raw_path = _parse_run_spec(value)
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        path = path.resolve()
        if seed in runs:
            raise ValueError(f'Duplicate {label} seed: {seed}.')
        if not path.exists():
            raise FileNotFoundError(f'Missing {label} run root: {path}')
        runs[seed] = path
    return runs


def _copy_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _compact_prosys_metadata(
    *,
    output_root: Path,
    seed: int,
    family: str,
    source_root: Path,
    result: dict[str, Any],
) -> None:
    target = output_root / 'compact' / 'prosys' / f'seed_{seed}' / family
    _write_json(result, target / 'result.json')
    metadata_paths = [
        (source_root / family / '_shared_knn' / 'reafnn' / 'reafnn_meta.json', 'reafnn_meta.json'),
        (source_root / family / '_shared_reaction_gnn' / 'model' / 'reaction_gnn_meta.json', 'reaction_gnn_meta.json'),
        (
            source_root / family / 'knn_xgb' / 'non_oracle' / 'model' / 'xgb_ranker_meta.json',
            'xgb_ranker_meta.json',
        ),
        (
            source_root / family / 'knn_xgb' / 'non_oracle' / 'model' / 'xgb_temperature_meta.json',
            'xgb_temperature_meta.json',
        ),
    ]
    for source, name in metadata_paths:
        _copy_if_present(source, target / name)


def _collect_prosys_run(
    *,
    output_root: Path,
    seed: int,
    source_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        result_file = source_root / family / 'knn_xgb' / 'non_oracle' / 'result.json'
        if not result_file.exists():
            raise FileNotFoundError(f'Missing ProSys result for {family}: {result_file}')
        result = _read_json(result_file)
        recorded_seed = result.get('seed')
        if recorded_seed is not None and int(recorded_seed) != seed:
            raise ValueError(
                f'ProSys result seed mismatch for {family}: expected {seed}, found {recorded_seed}.'
            )

        metrics = result['metrics']
        temperature = metrics.get('temperature', {})
        recall = result.get('stage1_route_recall', {})
        rows.append(
            {
                'system': SYSTEM_PROSYS,
                'seed': seed,
                'family': family,
                'candidate_slates': metrics.get('candidate_slates'),
                'missing_candidate_slates': metrics.get('missing_candidate_slates'),
                'temp_n': temperature.get('n'),
                'rr1': recall.get('route_recall_top1'),
                'rr3': recall.get('route_recall_top3'),
                'rr5': recall.get('route_recall_top5'),
                'rr10': recall.get('route_recall_top10'),
                'cover': metrics.get('pool_coverage'),
                'sys1': metrics.get('system_top1_all'),
                'sys3': metrics.get('system_top3_all'),
                'sys5': metrics.get('system_top5_all'),
                'sys10': metrics.get('system_top10_all'),
                'mrr': metrics.get('system_mrr'),
                'ndcg10': metrics.get('system_ndcg10'),
                'temp_mae': temperature.get('mae'),
                'temp_within_5c': temperature.get('within_5c'),
                'temp_within_10c': temperature.get('within_10c'),
                'temp_within_20c': temperature.get('within_20c'),
                'source_root': str(source_root),
            }
        )
        _compact_prosys_metadata(
            output_root=output_root,
            seed=seed,
            family=family,
            source_root=source_root,
            result=result,
        )
    return rows


def _compact_b3_metadata(
    *,
    output_root: Path,
    seed: int,
    family: str,
    source_root: Path,
) -> None:
    source_family = source_root / 'sequential_fnn' / family
    target = output_root / 'compact' / 'b3_sequential_fnn' / f'seed_{seed}' / family
    metadata_paths = [
        (source_family / 'run_metadata.json', 'run_metadata.json'),
        (source_family / 'fusion_selection.json', 'fusion_selection.json'),
        (source_family / 'artifacts' / 'sequential_fnn_meta.json', 'sequential_fnn_meta.json'),
    ]
    for source, name in metadata_paths:
        _copy_if_present(source, target / name)


def _collect_b3_run(
    *,
    output_root: Path,
    seed: int,
    source_root: Path,
) -> list[dict[str, Any]]:
    summary_file = source_root / 'summary.csv'
    if not summary_file.exists():
        raise FileNotFoundError(f'Missing B3 summary: {summary_file}')
    summary = pd.read_csv(summary_file)
    summary = summary.loc[
        (summary['method'] == 'sequential_fnn') & (summary['family'] != 'MACRO-AVG')
    ].copy()
    expected = set(FAMILY_ORDER)
    observed = set(summary['family'].astype(str))
    if observed != expected:
        raise ValueError(
            f'B3 summary families differ from the fixed family set. Missing={sorted(expected - observed)}, '
            f'unexpected={sorted(observed - expected)}.'
        )

    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        item = summary.loc[summary['family'] == family].iloc[0]
        rows.append(
            {
                'system': SYSTEM_B3,
                'seed': seed,
                'family': family,
                'candidate_slates': item.get('candidate_slates'),
                'missing_candidate_slates': item.get('missing_candidate_slates'),
                'temp_n': item.get('temperature_n'),
                'rr1': np.nan,
                'rr3': np.nan,
                'rr5': np.nan,
                'rr10': np.nan,
                'cover': item.get('cover'),
                'sys1': item.get('sys1'),
                'sys3': item.get('sys3'),
                'sys5': item.get('sys5'),
                'sys10': item.get('sys10'),
                'mrr': item.get('mrr'),
                'ndcg10': item.get('ndcg10'),
                'temp_mae': item.get('temperature_mae'),
                'temp_within_5c': item.get('temperature_within_5c'),
                'temp_within_10c': item.get('temperature_within_10c'),
                'temp_within_20c': item.get('temperature_within_20c'),
                'source_root': str(source_root),
            }
        )
        _compact_b3_metadata(
            output_root=output_root,
            seed=seed,
            family=family,
            source_root=source_root,
        )
    _copy_if_present(
        source_root / 'run_config.json',
        output_root / 'compact' / 'b3_sequential_fnn' / f'seed_{seed}' / 'run_config.json',
    )
    return rows


def _load_existing_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ROW_COLUMNS)
    frame = pd.read_csv(path)
    for column in ROW_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, ROW_COLUMNS]


def _update_rows(existing: pd.DataFrame, incoming: list[dict[str, Any]]) -> pd.DataFrame:
    if not incoming:
        return existing
    incoming_frame = pd.DataFrame(incoming).loc[:, ROW_COLUMNS]
    updated_pairs = set(zip(incoming_frame['system'], incoming_frame['seed'].astype(int)))
    if not existing.empty:
        current_pairs = list(zip(existing['system'], pd.to_numeric(existing['seed'], errors='coerce')))
        keep = [pair not in updated_pairs for pair in current_pairs]
        existing = existing.loc[keep].copy()
    merged = pd.concat([existing, incoming_frame], ignore_index=True)
    for column in ['seed', 'candidate_slates', 'missing_candidate_slates', 'temp_n']:
        merged[column] = pd.to_numeric(merged[column], errors='coerce')
    for column in [*METRIC_COLUMNS, 'rr1', 'rr3', 'rr5', 'rr10']:
        merged[column] = pd.to_numeric(merged[column], errors='coerce')
    family_order = {family: index for index, family in enumerate(FAMILY_ORDER)}
    merged['_family_order'] = merged['family'].map(family_order).fillna(len(family_order))
    merged = merged.sort_values(['system', 'seed', '_family_order'], kind='mergesort').drop(columns='_family_order')
    return merged.reset_index(drop=True)


def _macro_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (system, seed), group in frame.groupby(['system', 'seed'], sort=True):
        row: dict[str, Any] = {
            'system': system,
            'seed': int(seed),
            'n_families': int(group['family'].nunique()),
            'n_test_manifest': int((group['candidate_slates'] + group['missing_candidate_slates']).sum()),
            'n_candidate_slates': int(group['candidate_slates'].sum()),
            'n_missing_candidate_slates': int(group['missing_candidate_slates'].sum()),
            'temperature_support': int(group['temp_n'].sum(skipna=True)),
        }
        for metric in [*METRIC_COLUMNS, 'rr1', 'rr3', 'rr5', 'rr10']:
            row[metric] = float(group[metric].mean()) if group[metric].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_std(macro: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system, group in macro.groupby('system', sort=True):
        row: dict[str, Any] = {
            'system': system,
            'seeds': ','.join(str(int(seed)) for seed in sorted(group['seed'].tolist())),
            'n_seeds': int(len(group)),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(np.mean(values)) if len(values) else np.nan
            row[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _per_family_mean_std(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (system, family), group in frame.groupby(['system', 'family'], sort=True):
        row: dict[str, Any] = {
            'system': system,
            'family': family,
            'seeds': ','.join(str(int(seed)) for seed in sorted(group['seed'].tolist())),
            'n_seeds': int(len(group)),
            'n_test_manifest': int((group['candidate_slates'] + group['missing_candidate_slates']).iloc[0]),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(np.mean(values)) if len(values) else np.nan
            row[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _format_rate(value: float) -> str:
    return 'NA' if pd.isna(value) else f'{100.0 * value:.2f}'


def _format_number(value: float) -> str:
    return 'NA' if pd.isna(value) else f'{value:.2f}'


def _format_mean_std(mean: float, std: float, *, rate: bool) -> str:
    formatter = _format_rate if rate else _format_number
    if pd.isna(mean):
        return 'NA'
    if pd.isna(std):
        return formatter(mean)
    return f'{formatter(mean)} +/- {formatter(std)}'


def _write_report(output_root: Path, macro: pd.DataFrame, summary: pd.DataFrame) -> None:
    report = [
        '# Fixed-Stage-1 Multi-Seed Robustness',
        '',
        '## Scope',
        '',
        'This is a conditional Stage-2/3 robustness experiment. The persisted fixed split, '
        'the six test manifests, and the Stage-1 route caches are held fixed across seeds. '
        'It does not claim variability from retraining EditRetro Stage 1.',
        '',
        'Randomized learned components are ReaFNN, Reaction-GNN, XGBoost (subsample and '
        'column-subsample), and B3 Sequential FNN. KNN retrieval, route caches, data split, '
        'vocabularies derived from the fixed training split, candidate evaluation, and '
        'validation-only score-fusion selection are fixed by protocol.',
        '',
        'Values below are unweighted macro averages over the six families. Rates are percentages; '
        'standard deviations use the sample definition (`ddof=1`). Temperature MAE is the '
        'unweighted mean of family-level conditional MAEs, and its support can vary by seed.',
        '',
        '## Mean +/- Std',
        '',
        '| System | Seeds | Sys@1 | Sys@10 | MRR | nDCG@10 | Temp MAE (C) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for _, row in summary.iterrows():
        report.append(
            '| '
            + ' | '.join(
                [
                    str(row['system']),
                    str(row['seeds']),
                    _format_mean_std(row['sys1_mean'], row['sys1_std'], rate=True),
                    _format_mean_std(row['sys10_mean'], row['sys10_std'], rate=True),
                    _format_mean_std(row['mrr_mean'], row['mrr_std'], rate=True),
                    _format_mean_std(row['ndcg10_mean'], row['ndcg10_std'], rate=True),
                    _format_mean_std(row['temp_mae_mean'], row['temp_mae_std'], rate=False),
                ]
            )
            + ' |'
        )
    report.extend(['', '## Per-Seed Macro Results', ''])
    report.append('| System | Seed | Families | Test manifest | Candidate slates | Sys@1 | Sys@10 | MRR | nDCG@10 | Temp support | Temp MAE (C) |')
    report.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for _, row in macro.iterrows():
        report.append(
            '| '
            + ' | '.join(
                [
                    str(row['system']),
                    str(int(row['seed'])),
                    str(int(row['n_families'])),
                    str(int(row['n_test_manifest'])),
                    str(int(row['n_candidate_slates'])),
                    _format_rate(row['sys1']),
                    _format_rate(row['sys10']),
                    _format_rate(row['mrr']),
                    _format_rate(row['ndcg10']),
                    str(int(row['temperature_support'])),
                    _format_number(row['temp_mae']),
                ]
            )
            + ' |'
        )
    report.extend(
        [
            '',
            '## Retention policy',
            '',
            'The `compact/` directory keeps per-family metric records and model metadata. '
            'When `--prune-run` is used, large candidate tables, scored tables, and checkpoint '
            'files under that explicitly supplied `*_work` directory are removed only after this '
            'collector has successfully written the compact record.',
            '',
        ]
    )
    (output_root / 'README.md').write_text('\n'.join(report), encoding='utf-8')


def _route_cache_hashes(route_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for family in FAMILY_ORDER:
        path = route_root / family / 'route_cache.json'
        if path.exists():
            hashes[family] = _sha256(path)
    return hashes


def _prune_completed_runs(
    *,
    output_root: Path,
    paths: list[Path],
    processed_roots: set[Path],
) -> list[str]:
    removed: list[str] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in processed_roots:
            raise ValueError(f'Refusing to prune an uncollected run root: {resolved}')
        if output_root not in resolved.parents or not resolved.name.endswith('_work'):
            raise ValueError(
                'Refusing to prune outside this experiment output root or a path not named *_work: '
                f'{resolved}'
            )
        shutil.rmtree(resolved)
        removed.append(str(resolved))
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=str, default='.')
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--prosys-run', action='append', default=[], metavar='SEED=PATH')
    parser.add_argument('--b3-run', action='append', default=[], metavar='SEED=PATH')
    parser.add_argument('--route-root', type=str, default='outputs/stage1_routes')
    parser.add_argument(
        '--prune-run',
        action='append',
        default=[],
        metavar='PATH',
        help='Delete a collected temporary *_work run root after compacting its metadata.',
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    route_root = (repo_root / args.route_root).resolve()
    prosys_runs = _resolve_run_specs(repo_root, args.prosys_run, 'ProSys')
    b3_runs = _resolve_run_specs(repo_root, args.b3_run, 'B3')
    if not prosys_runs and not b3_runs:
        raise ValueError('Provide at least one --prosys-run or --b3-run.')

    incoming: list[dict[str, Any]] = []
    processed_roots: set[Path] = set()
    for seed, source_root in sorted(prosys_runs.items()):
        incoming.extend(_collect_prosys_run(output_root=output_root, seed=seed, source_root=source_root))
        processed_roots.add(source_root)
    for seed, source_root in sorted(b3_runs.items()):
        incoming.extend(_collect_b3_run(output_root=output_root, seed=seed, source_root=source_root))
        processed_roots.add(source_root)

    rows_file = output_root / 'per_family_seed_metrics.csv'
    frame = _update_rows(_load_existing_rows(rows_file), incoming)
    frame.to_csv(rows_file, index=False)
    macro = _macro_by_seed(frame)
    macro.to_csv(output_root / 'macro_by_seed.csv', index=False)
    summary = _mean_std(macro)
    summary.to_csv(output_root / 'macro_mean_std.csv', index=False)
    _per_family_mean_std(frame).to_csv(output_root / 'per_family_mean_std.csv', index=False)
    _write_report(output_root, macro, summary)

    prune_paths = [
        (Path(value) if Path(value).is_absolute() else repo_root / value).resolve()
        for value in args.prune_run
    ]
    removed = _prune_completed_runs(
        output_root=output_root,
        paths=prune_paths,
        processed_roots=processed_roots,
    )
    provenance = {
        'protocol': 'fixed_stage1_route_cache_multiseed_stage23_and_b3',
        'family_order': FAMILY_ORDER,
        'prosys_runs_updated': {str(seed): str(path) for seed, path in sorted(prosys_runs.items())},
        'b3_runs_updated': {str(seed): str(path) for seed, path in sorted(b3_runs.items())},
        'stage1_route_cache_sha256': _route_cache_hashes(route_root),
        'pruned_work_roots': removed,
        'retained_files': [
            'per_family_seed_metrics.csv',
            'macro_by_seed.csv',
            'macro_mean_std.csv',
            'per_family_mean_std.csv',
            'README.md',
            'compact/',
        ],
    }
    _write_json(provenance, output_root / 'provenance.json')


if __name__ == '__main__':
    main()
