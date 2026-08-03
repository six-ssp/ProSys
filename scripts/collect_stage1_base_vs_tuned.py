"""Collect Stage 1 base-vs-family-tuned route-recall comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.mainline import FAMILY_ORDER, display_family_name, parse_families_arg, stage1_route_recall


def _mean(values: list[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def _weighted_mean(rows: list[dict], value_key: str, weight_key: str) -> float | None:
    pairs = [
        (float(row[value_key]), float(row[weight_key]))
        for row in rows
        if row.get(value_key) is not None and row.get(weight_key) not in (None, 0)
    ]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def append_macro_and_weighted(rows: list[dict], *, sample_key: str) -> list[dict]:
    if not rows:
        return rows
    metric_keys = [key for key in rows[0].keys() if key not in {'family', 'display_family'}]
    macro = {'family': 'MACRO-AVG', 'display_family': 'MACRO-AVG'}
    for key in metric_keys:
        macro[key] = _mean([row.get(key) for row in rows])
    weighted = {'family': 'WEIGHTED-AVG', 'display_family': 'WEIGHTED-AVG'}
    for key in metric_keys:
        if key == sample_key:
            weighted[key] = sum(int(row.get(sample_key, 0) or 0) for row in rows)
        else:
            weighted[key] = _weighted_mean(rows, key, sample_key)
    return list(rows) + [macro, weighted]


def collect_rows(route_root: Path, base_route_root: Path, families: list[str]) -> list[dict]:
    rows: list[dict] = []
    for family in families:
        tuned_cache = route_root / family / 'route_cache.json'
        base_cache = base_route_root / family / 'route_cache.json'
        if not tuned_cache.exists() or not base_cache.exists():
            continue
        tuned = stage1_route_recall(tuned_cache)
        base = stage1_route_recall(base_cache)
        rows.append(
            {
                'family': family,
                'display_family': display_family_name(family),
                'test_products': tuned.get('n'),
                'base_route_at_1': base.get('route_recall_top1'),
                'base_route_at_3': base.get('route_recall_top3'),
                'base_route_at_5': base.get('route_recall_top5'),
                'base_route_at_10': base.get('route_recall_top10'),
                'family_tuned_route_at_1': tuned.get('route_recall_top1'),
                'family_tuned_route_at_3': tuned.get('route_recall_top3'),
                'family_tuned_route_at_5': tuned.get('route_recall_top5'),
                'family_tuned_route_at_10': tuned.get('route_recall_top10'),
                'delta_route_at_1': tuned.get('route_recall_top1', 0.0) - base.get('route_recall_top1', 0.0),
                'delta_route_at_3': tuned.get('route_recall_top3', 0.0) - base.get('route_recall_top3', 0.0),
                'delta_route_at_5': tuned.get('route_recall_top5', 0.0) - base.get('route_recall_top5', 0.0),
                'delta_route_at_10': tuned.get('route_recall_top10', 0.0) - base.get('route_recall_top10', 0.0),
            }
        )
    return append_macro_and_weighted(rows, sample_key='test_products')


def render_markdown(rows: list[dict]) -> str:
    frame = pd.DataFrame(rows)
    paper = frame.copy()
    if 'display_family' in paper.columns:
        paper['family'] = paper['display_family']
        paper = paper.drop(columns=['display_family'])
    ordered_columns = [
        'family',
        'test_products',
        'base_route_at_1',
        'family_tuned_route_at_1',
        'delta_route_at_1',
        'base_route_at_3',
        'family_tuned_route_at_3',
        'delta_route_at_3',
        'base_route_at_5',
        'family_tuned_route_at_5',
        'delta_route_at_5',
        'base_route_at_10',
        'family_tuned_route_at_10',
        'delta_route_at_10',
    ]
    paper = paper[ordered_columns]
    for column in paper.columns:
        if column == 'family':
            continue
        if column == 'test_products':
            paper[column] = paper[column].map(lambda value: int(round(float(value))) if pd.notna(value) else value)
        else:
            paper[column] = paper[column].map(lambda value: round(float(value) * 100.0, 1) if pd.notna(value) else value)

    lines = [
        '# Stage 1 Base vs Family-Tuned Route Recall',
        '',
        'All route-recall values are reported as percentages.',
        '',
        paper.to_markdown(index=False),
        '',
    ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect Stage 1 base-vs-family-tuned route recall results.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--base_route_root', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--output_md', type=str, required=True)
    args = parser.parse_args()

    families = parse_families_arg(args.families)
    route_root = Path(args.route_root).resolve()
    base_route_root = Path(args.base_route_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_md = Path(args.output_md).resolve()

    rows = collect_rows(route_root, base_route_root, families)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_md.write_text(render_markdown(rows), encoding='utf-8')
    print(f'[stage1-base-vs-tuned] wrote {output_csv}')
    print(f'[stage1-base-vs-tuned] wrote {output_md}')


if __name__ == '__main__':
    main()
