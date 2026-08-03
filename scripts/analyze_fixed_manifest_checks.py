"""Recompute checklist uncertainty and error analyses from fixed test scores.

This utility is read-only with respect to model artifacts. It mirrors the
formal full-manifest ranking definition: all cached test identities remain in
the denominator, including identities with no candidate slate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.features import canonicalize_smiles
from prosys_shared.mainline import FAMILY_ORDER, display_family_name


@dataclass(frozen=True)
class SourceSpec:
    name: str
    family_file_template: str
    score_column: str

    def file_for(self, root: Path, family: str) -> Path:
        return root / self.family_file_template.format(family=family)


SOURCE_SPECS = (
    SourceSpec(
        name='ProSys',
        family_file_template='{family}/knn_xgb/non_oracle/test_scored.csv',
        score_column='xgb_score',
    ),
    SourceSpec(
        name='B3 Sequential FNN',
        family_file_template='sequential_fnn/{family}/test_labeled_candidates.csv',
        score_column='system_score',
    ),
    SourceSpec(
        name='Matched full mainline ablation',
        family_file_template='{family}/full_mainline/non_oracle/test_scored.csv',
        score_column='xgb_score',
    ),
    SourceSpec(
        name='Frequency-top20 pool + XGBoost',
        family_file_template='{family}/frequency_top20_xgb/non_oracle/test_scored.csv',
        score_column='xgb_score',
    ),
    SourceSpec(
        name='No Stage 3 XGBoost',
        family_file_template='{family}/no_stage3/non_oracle/test_scored.csv',
        score_column='stage2_prior_score',
    ),
    SourceSpec(
        name='No Reaction-GNN',
        family_file_template='{family}/no_gnn_xgb/non_oracle/test_scored.csv',
        score_column='xgb_score',
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _cache_manifest(cache_file: Path) -> pd.DataFrame:
    cache = _read_json(cache_file)
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for reaction in cache.get('reactions', []):
        sample_index = int(reaction['sample_index'])
        if sample_index in seen:
            raise ValueError(f'Duplicate sample_index={sample_index} in {cache_file}.')
        seen.add(sample_index)
        product = str(reaction.get('product', ''))
        product_key = canonicalize_smiles(product)
        records.append(
            {
                'sample_index': sample_index,
                'product_canonical': product_key or f'__unparsed_sample_{sample_index}',
                'stage1_route_count': int(len(reaction.get('routes', []))),
            }
        )
    if not records:
        raise ValueError(f'No cached reaction identities in {cache_file}.')
    return pd.DataFrame(records).sort_values('sample_index').reset_index(drop=True)


def _read_score_columns(path: Path, score_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing scored table: {path}')
    needed = {'sample_index', 'label', 'route_match', 'context_match', score_column}
    frame = pd.read_csv(path, usecols=lambda name: name in needed)
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f'{path} is missing scored columns: {sorted(missing)}')
    frame['sample_index'] = pd.to_numeric(frame['sample_index'], errors='raise').astype(int)
    for column in ['label', 'route_match', 'context_match', score_column]:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    if frame[score_column].isna().any():
        raise ValueError(f'{path} contains NaN {score_column} values.')
    return frame


def _per_sample_metrics(
    *,
    manifest: pd.DataFrame,
    scored: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    expected = set(manifest['sample_index'].astype(int))
    observed = set(scored['sample_index'].astype(int))
    unknown = sorted(observed - expected)
    if unknown:
        raise ValueError(f'Scored table includes unknown sample indices: {unknown[:5]}')

    grouped = {int(index): group for index, group in scored.groupby('sample_index', sort=False)}
    rows: list[dict[str, Any]] = []
    for item in manifest.to_dict(orient='records'):
        sample_index = int(item['sample_index'])
        group = grouped.get(sample_index)
        if group is None:
            rows.append(
                {
                    **item,
                    'has_candidate_slate': False,
                    'has_gold_route': False,
                    'has_exact_system': False,
                    'exact_system_rank': np.nan,
                    'sys1': 0.0,
                    'sys10': 0.0,
                }
            )
            continue

        ranked = group.sort_values(score_column, ascending=False, kind='mergesort').reset_index(drop=True)
        label = ranked['label'].to_numpy(dtype=float) > 0.5
        route_match = ranked['route_match'].to_numpy(dtype=float) > 0.5
        positive = np.flatnonzero(label)
        rank = int(positive[0] + 1) if positive.size else np.nan
        rows.append(
            {
                **item,
                'has_candidate_slate': True,
                'has_gold_route': bool(np.any(route_match)),
                'has_exact_system': bool(positive.size),
                'exact_system_rank': rank,
                'sys1': float(np.any(label[:1])),
                'sys10': float(np.any(label[:10])),
            }
        )
    return pd.DataFrame(rows).sort_values('sample_index').reset_index(drop=True)


def _build_sources(
    *,
    mainline_root: Path,
    b3_root: Path,
    ablation_root: Path,
    route_root: Path,
) -> dict[str, dict[str, pd.DataFrame]]:
    roots = {
        'ProSys': mainline_root,
        'B3 Sequential FNN': b3_root,
        'Matched full mainline ablation': ablation_root,
        'Frequency-top20 pool + XGBoost': ablation_root,
        'No Stage 3 XGBoost': ablation_root,
        'No Reaction-GNN': ablation_root,
    }
    result: dict[str, dict[str, pd.DataFrame]] = {spec.name: {} for spec in SOURCE_SPECS}
    manifests: dict[str, pd.DataFrame] = {}
    for family in FAMILY_ORDER:
        manifest = _cache_manifest(route_root / family / 'route_cache.json')
        manifests[family] = manifest
        for spec in SOURCE_SPECS:
            scored = _read_score_columns(spec.file_for(roots[spec.name], family), spec.score_column)
            result[spec.name][family] = _per_sample_metrics(
                manifest=manifest,
                scored=scored,
                score_column=spec.score_column,
            )
    return result


def _macro_point(source: dict[str, pd.DataFrame], metric: str) -> float:
    return float(np.mean([frame[metric].mean() for frame in source.values()]))


def _product_cluster_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    indices: list[np.ndarray] = []
    for _, group in frame.groupby('product_canonical', sort=True):
        indices.append(group.index.to_numpy(dtype=int))
    if not indices:
        raise ValueError('No product groups available for bootstrap.')
    return indices


def _bootstrap_macro(
    *,
    sources: dict[str, dict[str, pd.DataFrame]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    source_names = list(sources)
    metrics = ('sys1', 'sys10')
    draws = {
        metric: {source_name: np.empty(n_bootstrap, dtype=float) for source_name in source_names}
        for metric in metrics
    }
    family_groups = {
        family: _product_cluster_indices(sources[source_names[0]][family])
        for family in FAMILY_ORDER
    }
    for family in FAMILY_ORDER:
        expected_indices = sources[source_names[0]][family]['sample_index'].to_numpy(dtype=int)
        for source_name in source_names[1:]:
            observed_indices = sources[source_name][family]['sample_index'].to_numpy(dtype=int)
            if not np.array_equal(expected_indices, observed_indices):
                raise ValueError(f'Sample ordering differs for {source_name}, {family}.')

    for draw_index in range(n_bootstrap):
        family_values = {
            metric: {source_name: [] for source_name in source_names}
            for metric in metrics
        }
        for family in FAMILY_ORDER:
            groups = family_groups[family]
            sampled_group_indices = rng.integers(0, len(groups), size=len(groups))
            sampled_rows = np.concatenate([groups[index] for index in sampled_group_indices])
            for source_name in source_names:
                frame = sources[source_name][family]
                for metric in metrics:
                    family_values[metric][source_name].append(
                        float(frame[metric].to_numpy(dtype=float)[sampled_rows].mean())
                    )
        for metric in metrics:
            for source_name in source_names:
                draws[metric][source_name][draw_index] = float(
                    np.mean(family_values[metric][source_name])
                )
    return draws


def _bootstrap_summary(
    *,
    sources: dict[str, dict[str, pd.DataFrame]],
    draws: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int,
) -> pd.DataFrame:
    comparisons = (
        ('ProSys', 'B3 Sequential FNN', 'ProSys vs B3 Sequential FNN'),
        (
            'Matched full mainline ablation',
            'Frequency-top20 pool + XGBoost',
            'Stage 2 KNN+ReaFNN pool vs global frequency pool',
        ),
        (
            'Matched full mainline ablation',
            'No Stage 3 XGBoost',
            'Stage 3 XGBoost retained vs removed',
        ),
        (
            'Matched full mainline ablation',
            'No Reaction-GNN',
            'Reaction-GNN retained vs removed',
        ),
    )
    rows: list[dict[str, Any]] = []
    for metric, source_draws in draws.items():
        for reference, comparator, label in comparisons:
            ref_draw = source_draws[reference]
            cmp_draw = source_draws[comparator]
            delta = ref_draw - cmp_draw
            rows.append(
                {
                    'comparison': label,
                    'metric': metric,
                    'n_bootstrap': n_bootstrap,
                    'reference': reference,
                    'comparator': comparator,
                    'reference_point': _macro_point(sources[reference], metric),
                    'comparator_point': _macro_point(sources[comparator], metric),
                    'delta_point': _macro_point(sources[reference], metric)
                    - _macro_point(sources[comparator], metric),
                    'reference_ci_low': float(np.quantile(ref_draw, 0.025)),
                    'reference_ci_high': float(np.quantile(ref_draw, 0.975)),
                    'comparator_ci_low': float(np.quantile(cmp_draw, 0.025)),
                    'comparator_ci_high': float(np.quantile(cmp_draw, 0.975)),
                    'delta_ci_low': float(np.quantile(delta, 0.025)),
                    'delta_ci_high': float(np.quantile(delta, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def _error_category(item: dict[str, Any]) -> str:
    if not bool(item['has_candidate_slate']):
        if int(item['stage1_route_count']) == 0:
            return 'No valid Stage 1 route'
        return 'No Stage 2 candidate slate despite Stage 1 routes'
    if not bool(item['has_gold_route']):
        return 'Gold route absent from Stage 1 top-10'
    if not bool(item['has_exact_system']):
        return 'Gold route present but exact context absent'
    rank = int(item['exact_system_rank'])
    if rank == 1:
        return 'Exact system ranked 1'
    if rank <= 10:
        return 'Exact system ranked 2-10'
    return 'Exact system in pool but ranked >10'


def _error_decomposition(prosys: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    for family, frame in prosys.items():
        work = frame.copy()
        work['family'] = family
        work['error_category'] = [_error_category(item) for item in work.to_dict(orient='records')]
        records.append(work)
    per_sample = pd.concat(records, ignore_index=True)
    categories = [
        'No valid Stage 1 route',
        'No Stage 2 candidate slate despite Stage 1 routes',
        'Gold route absent from Stage 1 top-10',
        'Gold route present but exact context absent',
        'Exact system in pool but ranked >10',
        'Exact system ranked 2-10',
        'Exact system ranked 1',
    ]
    rows: list[dict[str, Any]] = []
    for family, group in per_sample.groupby('family', sort=False):
        for category in categories:
            count = int((group['error_category'] == category).sum())
            rows.append(
                {
                    'family': family,
                    'category': category,
                    'count': count,
                    'rate': count / len(group),
                    'n_test': len(group),
                }
            )
    for category in categories:
        count = int((per_sample['error_category'] == category).sum())
        rows.append(
            {
                'family': 'ALL-FAMILIES',
                'category': category,
                'count': count,
                'rate': count / len(per_sample),
                'n_test': len(per_sample),
            }
        )
    return per_sample, pd.DataFrame(rows)


def _pct(value: float) -> str:
    return f'{100.0 * value:.2f}'


def _write_report(output_root: Path, bootstrap: pd.DataFrame, errors: pd.DataFrame) -> None:
    lines = [
        '# Fixed-Manifest Uncertainty and Error Checks',
        '',
        '## Bootstrap protocol',
        '',
        'For every family, canonical target-product clusters were sampled with replacement; '
        'all reaction records associated with a selected product were retained together. Each '
        'bootstrap replicate computes a family-level sample mean and then the unweighted mean '
        'across the six families. The reported intervals are percentile 95% confidence intervals.',
        '',
        'The input is the saved final scored candidate table. All fixed-manifest identities, '
        'including missing candidate slates, contribute zeros. Therefore the bootstrap matches '
        'the formal end-to-end denominator rather than a candidate-only denominator.',
        '',
        '## Sys@10 bootstrap comparisons',
        '',
        '| Comparison | Reference | Comparator | Reference point | Comparator point | Difference (pp) | 95% CI for difference (pp) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    sys10 = bootstrap.loc[bootstrap['metric'] == 'sys10']
    for _, row in sys10.iterrows():
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(row['comparison']),
                    str(row['reference']),
                    str(row['comparator']),
                    _pct(row['reference_point']),
                    _pct(row['comparator_point']),
                    _pct(row['delta_point']),
                    f"{_pct(row['delta_ci_low'])}, {_pct(row['delta_ci_high'])}",
                ]
            )
            + ' |'
        )
    lines.extend(['', '## Mutually exclusive end-to-end error decomposition', ''])
    lines.append('| Category | Count | Rate (%) |')
    lines.append('| --- | ---: | ---: |')
    for _, row in errors.loc[errors['family'] == 'ALL-FAMILIES'].iterrows():
        lines.append(f"| {row['category']} | {int(row['count'])} | {_pct(row['rate'])} |")
    lines.extend(
        [
            '',
            'The categories are exhaustive and mutually exclusive. Rank 1 is separated from '
            'ranks 2-10 so it is not double counted with the final-top-10 success category.',
            '',
        ]
    )
    (output_root / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=str, default='.')
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--mainline-root', type=str, required=True)
    parser.add_argument('--b3-root', type=str, required=True)
    parser.add_argument('--ablation-root', type=str, required=True)
    parser.add_argument('--route-root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260730)
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        raise ValueError('--n-bootstrap must be positive.')

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resolve = lambda value: (Path(value) if Path(value).is_absolute() else repo_root / value).resolve()
    sources = _build_sources(
        mainline_root=resolve(args.mainline_root),
        b3_root=resolve(args.b3_root),
        ablation_root=resolve(args.ablation_root),
        route_root=resolve(args.route_root),
    )

    per_sample_records: list[pd.DataFrame] = []
    for source_name, family_frames in sources.items():
        for family, frame in family_frames.items():
            work = frame.loc[:, ['sample_index', 'product_canonical', 'sys1', 'sys10']].copy()
            work.insert(0, 'family', family)
            work.insert(0, 'source', source_name)
            per_sample_records.append(work)
    pd.concat(per_sample_records, ignore_index=True).to_csv(
        output_root / 'per_sample_system_metrics.csv', index=False
    )

    draws = _bootstrap_macro(sources=sources, n_bootstrap=args.n_bootstrap, seed=args.seed)
    bootstrap = _bootstrap_summary(sources=sources, draws=draws, n_bootstrap=args.n_bootstrap)
    bootstrap.to_csv(output_root / 'bootstrap_sys_metrics.csv', index=False)
    per_sample_errors, errors = _error_decomposition(sources['ProSys'])
    per_sample_errors.to_csv(output_root / 'mainline_error_decomposition_per_sample.csv', index=False)
    errors.to_csv(output_root / 'mainline_error_decomposition.csv', index=False)
    _write_report(output_root, bootstrap, errors)
    _write_json(
        {
            'n_bootstrap': args.n_bootstrap,
            'random_seed': args.seed,
            'bootstrap_unit': 'canonical target-product cluster within each family',
            'macro_aggregation': 'unweighted mean of six family-level metrics',
            'sources': {spec.name: spec.score_column for spec in SOURCE_SPECS},
        },
        output_root / 'provenance.json',
    )


if __name__ == '__main__':
    main()
