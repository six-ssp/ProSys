"""Re-evaluate direct product-to-condition baselines on unseen-product subsets.

The formal split is canonical-reaction-disjoint rather than product-disjoint.
This read-only stress test therefore keeps the trained models and saved test
scores fixed, then restricts evaluation to test products absent from the
corresponding train split (and, separately, absent from train plus validation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.condition_modeling import load_condition_rows
from prosys_shared.features import canonicalize_smiles
from prosys_shared.mainline import FAMILY_ORDER, evaluate_scored_frame_with_manifest, split_file_for_family


METHODS = (
    ('Product-Bernoulli Naive Bayes', 'product_naive_bayes'),
    ('Product-GNN', 'product_gnn'),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _manifest_products(cache_file: Path) -> pd.DataFrame:
    cache = _read_json(cache_file)
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for reaction in cache.get('reactions', []):
        sample_index = int(reaction['sample_index'])
        if sample_index in seen:
            raise ValueError(f'Duplicate sample_index={sample_index} in {cache_file}.')
        seen.add(sample_index)
        product = canonicalize_smiles(str(reaction.get('product', '')))
        if not product:
            raise ValueError(f'Unparseable cached product at sample_index={sample_index}: {cache_file}')
        records.append({'sample_index': sample_index, 'product_canonical': product})
    if not records:
        raise ValueError(f'Empty route cache: {cache_file}')
    return pd.DataFrame(records).sort_values('sample_index').reset_index(drop=True)


def _product_set(split_file: Path) -> set[str]:
    products = {
        canonicalize_smiles(str(row.product))
        for row in load_condition_rows(split_file)
    }
    products.discard('')
    return products


def _evaluate_subset(candidate_file: Path, sample_indices: list[int]) -> dict[str, Any]:
    if not sample_indices:
        return {
            'num_slates': 0,
            'system_top1_all': None,
            'system_top10_all': None,
            'context_top1_all': None,
            'context_top10_all': None,
        }
    needed = {'sample_index', 'label', 'route_match', 'context_match', 'system_score'}
    frame = pd.read_csv(candidate_file, usecols=lambda name: name in needed)
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f'{candidate_file} misses required columns: {sorted(missing)}')
    subset = frame.loc[frame['sample_index'].isin(sample_indices)].copy()
    return evaluate_scored_frame_with_manifest(
        subset,
        expected_sample_indices=sample_indices,
        score_column='system_score',
    )


def _macro(rows: pd.DataFrame, subset: str, method: str) -> dict[str, Any]:
    group = rows.loc[rows['subset'] == subset]
    values = {
        metric: group[metric].dropna().astype(float)
        for metric in ('context1', 'context10', 'sys1', 'sys10')
    }
    return {
        'method': method,
        'subset': subset,
        'families': int(group['family'].nunique()),
        'n_test': int(group['n_test'].sum()),
        'n_unique_products': int(group['n_unique_products'].sum()),
        **{metric: float(series.mean()) if not series.empty else None for metric, series in values.items()},
    }


def _pct(value: float | None) -> str:
    return 'NA' if value is None or pd.isna(value) else f'{100.0 * value:.2f}'


def _write_report(output_root: Path, rows: pd.DataFrame, macro_rows: list[dict[str, Any]]) -> None:
    lines = [
        '# Product-Disjoint Stress Test for Direct Baselines',
        '',
        'This is a post-hoc, fixed-model subset evaluation. It does not retrain either '
        'direct product-to-condition baseline and therefore does not replace a fully '
        'product-disjoint train/validation/test split.',
        '',
        '`train_unseen` excludes test products present in the family training split. '
        '`train_val_unseen` additionally excludes products present in the validation split, '
        'which is the stricter sensitivity set because validation participates in checkpoint '
        'and score-fusion selection.',
        '',
        '## Macro averages',
        '',
        '| Method | Subset | Test records | Unique products | Condition@1 | Condition@10 | Sys@1 | Sys@10 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in macro_rows:
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(row['method']),
                    str(row['subset']),
                    str(row['n_test']),
                    str(row['n_unique_products']),
                    _pct(row['context1']),
                    _pct(row['context10']),
                    _pct(row['sys1']),
                    _pct(row['sys10']),
                ]
            )
            + ' |'
        )
    lines.extend(['', '## Per-family results', ''])
    lines.append('| Method | Subset | Family | Test records | Unique products | Condition@1 | Condition@10 | Sys@1 | Sys@10 |')
    lines.append('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for _, row in rows.iterrows():
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(row['method']),
                    str(row['subset']),
                    str(row['family']),
                    str(int(row['n_test'])),
                    str(int(row['n_unique_products'])),
                    _pct(row['context1']),
                    _pct(row['context10']),
                    _pct(row['sys1']),
                    _pct(row['sys10']),
                ]
            )
            + ' |'
        )
    lines.append('')
    (output_root / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=str, default='.')
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--naive-bayes-root', type=str, required=True)
    parser.add_argument('--product-gnn-root', type=str, required=True)
    parser.add_argument('--route-root', type=str, default='outputs/stage1_routes')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    resolve = lambda value: (Path(value) if Path(value).is_absolute() else repo_root / value).resolve()
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    roots = {
        'Product-Bernoulli Naive Bayes': resolve(args.naive_bayes_root),
        'Product-GNN': resolve(args.product_gnn_root),
    }
    route_root = resolve(args.route_root)

    records: list[dict[str, Any]] = []
    subset_counts: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        manifest = _manifest_products(route_root / family / 'route_cache.json')
        train_products = _product_set(split_file_for_family(repo_root, family, 'train'))
        val_products = _product_set(split_file_for_family(repo_root, family, 'val'))
        subset_indices = {
            'train_unseen': manifest.loc[
                ~manifest['product_canonical'].isin(train_products), 'sample_index'
            ].astype(int).tolist(),
            'train_val_unseen': manifest.loc[
                ~manifest['product_canonical'].isin(train_products | val_products), 'sample_index'
            ].astype(int).tolist(),
        }
        for subset, indices in subset_indices.items():
            subset_manifest = manifest.loc[manifest['sample_index'].isin(indices)]
            subset_counts.append(
                {
                    'family': family,
                    'subset': subset,
                    'n_test': int(len(indices)),
                    'n_unique_products': int(subset_manifest['product_canonical'].nunique()),
                    'train_unique_products': int(len(train_products)),
                    'val_unique_products': int(len(val_products)),
                }
            )
            for method_label, method_dir in METHODS:
                candidate_file = roots[method_label] / method_dir / family / 'test_top10_candidates.csv.gz'
                metrics = _evaluate_subset(candidate_file, indices)
                records.append(
                    {
                        'method': method_label,
                        'family': family,
                        'subset': subset,
                        'n_test': int(len(indices)),
                        'n_unique_products': int(subset_manifest['product_canonical'].nunique()),
                        'context1': metrics.get('context_top1_all'),
                        'context10': metrics.get('context_top10_all'),
                        'sys1': metrics.get('system_top1_all'),
                        'sys10': metrics.get('system_top10_all'),
                    }
                )

    rows = pd.DataFrame(records)
    rows.to_csv(output_root / 'per_family_results.csv', index=False)
    pd.DataFrame(subset_counts).to_csv(output_root / 'subset_counts.csv', index=False)
    macro_rows = [
        _macro(rows.loc[rows['method'] == method], subset, method)
        for method, _ in METHODS
        for subset in ('train_unseen', 'train_val_unseen')
    ]
    pd.DataFrame(macro_rows).to_csv(output_root / 'macro_results.csv', index=False)
    _write_report(output_root, rows, macro_rows)
    _write_json(
        {
            'scope': 'fixed-model post-hoc direct-baseline product-disjoint stress test',
            'methods': [label for label, _ in METHODS],
            'subset_definitions': {
                'train_unseen': 'test product canonical SMILES absent from the family train split',
                'train_val_unseen': 'test product canonical SMILES absent from family train and validation splits',
            },
            'score_definition': 'saved final system_score evaluated against the fixed Stage-1 test route cache',
        },
        output_root / 'provenance.json',
    )


if __name__ == '__main__':
    main()
