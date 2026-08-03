"""Collect family-level Non-Oracle Stage-2/Stage-3 results and render final tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILY_ORDER = [
    'Beckmann',
    'Buchwald-HartwigCross-Coupling',
    'Chan_LamCoupling',
    'DielsAlder',
    'Friedel-CraftsAcylation',
    'Friedel-CraftsAlkylation',
]

FAMILY_DISPLAY_NAMES = {
    'Beckmann': 'Beckmann',
    'Buchwald-HartwigCross-Coupling': 'Buchwald-Hartwig',
    'Chan_LamCoupling': 'Chan-Lam',
    'DielsAlder': 'Diels-Alder',
    'Friedel-CraftsAcylation': 'Friedel-Crafts Acyl.',
    'Friedel-CraftsAlkylation': 'Friedel-Crafts Alkyl.',
}

HISTORICAL_BASELINES = ['prototype_fnn', 'knn_xgb']
STAGE3_BASELINES = ['knn_xgb', 'knn_rf', 'knn_svm', 'knn_bayes']
STAGE2_BASELINES = ['knn_xgb', 'cluster_xgb']

BASELINE_LABELS = {
    'prototype_fnn': 'Original FNN',
    'knn_xgb': 'KNN+XGB',
    'knn_rf': 'KNN+RF',
    'knn_svm': 'KNN+SVM',
    'knn_bayes': 'KNN+Bayes',
    'cluster_xgb': 'Cluster+XGB',
}

REPORT_FILTER_BASELINE = 'knn_xgb'
REPORT_FILTER_SYS10_THRESHOLD = 0.20
TOPKS = (1, 3, 5, 10)
TEMPERATURE_METRICS = [
    ('temp_mae', 'Temp MAE', False),
    ('temp_within_5c', 'Temp±5C', True),
    ('temp_within_10c', 'Temp±10C', True),
    ('temp_within_20c', 'Temp±20C', True),
]


def display_family_name(family: str) -> str:
    return FAMILY_DISPLAY_NAMES.get(family, family)


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(FAMILY_ORDER), family)


def _format_metric(value: float | None, *, percent: bool) -> str:
    if value is None or pd.isna(value):
        return 'NA'
    if percent:
        return f'{value * 100.0:.1f}'
    return f'{value:.2f}'


def _mean_metric(values: list[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not np.isnan(float(value))]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def _evaluate_scored_frame(frame: pd.DataFrame, *, score_column: str, temperature_column: str | None = None) -> dict:
    if frame.empty:
        return {'num_slates': 0, 'pool_coverage': 0.0}

    work = frame.sort_values(['sample_index', score_column], ascending=[True, False]).copy()
    num_slates = 0
    covered_slates = 0
    hit_counters = {f'system_top{k}': 0 for k in TOPKS}
    hit_counters.update({f'context_top{k}': 0 for k in TOPKS})
    hit_counters.update({f'route_top{k}': 0 for k in TOPKS})
    covered_hit = {f'system_top{k}': 0 for k in TOPKS}
    temp_abs_errors: list[float] = []

    for _, group in work.groupby('sample_index', sort=True):
        num_slates += 1
        label = group['label'].to_numpy(dtype=np.float64)
        route_match = group['route_match'].to_numpy(dtype=np.float64)
        context_match = group['context_match'].to_numpy(dtype=np.float64)
        has_positive = bool(np.any(label > 0.5))
        if has_positive:
            covered_slates += 1

        for k in TOPKS:
            sys_hit = bool(np.any(label[:k] > 0.5))
            hit_counters[f'system_top{k}'] += int(sys_hit)
            hit_counters[f'context_top{k}'] += int(np.any(context_match[:k] > 0.5))
            hit_counters[f'route_top{k}'] += int(np.any(route_match[:k] > 0.5))
            if has_positive:
                covered_hit[f'system_top{k}'] += int(sys_hit)

        if temperature_column and temperature_column in group.columns:
            temperature_gold = group['temperature_gold'].to_numpy(dtype=np.float64)
            temperature_pred = group[temperature_column].to_numpy(dtype=np.float64)
            valid_positive = (label > 0.5) & np.isfinite(temperature_gold) & np.isfinite(temperature_pred)
            positive_rows = np.flatnonzero(valid_positive)
            if positive_rows.size > 0:
                first_idx = int(positive_rows[0])
                temp_abs_errors.append(abs(float(temperature_pred[first_idx]) - float(temperature_gold[first_idx])))

    metrics: dict[str, object] = {
        'num_slates': int(num_slates),
        'covered_slates': int(covered_slates),
        'pool_coverage': (covered_slates / num_slates if num_slates else 0.0),
    }
    for key, value in hit_counters.items():
        metrics[f'{key}_all'] = (value / num_slates if num_slates else 0.0)
    for k in TOPKS:
        key = f'system_top{k}'
        metrics[f'{key}_covered'] = (covered_hit[key] / covered_slates if covered_slates else 0.0)

    if temp_abs_errors:
        errors = np.asarray(temp_abs_errors, dtype=np.float64)
        metrics['temperature'] = {
            'definition': 'standalone_temperature_error_on_highest_ranked_full_match',
            'n': int(errors.size),
            'mae': float(np.mean(errors)),
            'mse': float(np.mean(errors ** 2)),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'support': 'highest_ranked_full_match_with_valid_temperature',
            'within_5c': float(np.mean(errors <= 5.0)),
            'within_10c': float(np.mean(errors <= 10.0)),
            'within_20c': float(np.mean(errors <= 20.0)),
        }
    else:
        metrics['temperature'] = {
            'definition': 'standalone_temperature_error_on_highest_ranked_full_match',
            'n': 0,
            'mae': None,
            'mse': None,
            'rmse': None,
            'support': 'highest_ranked_full_match_with_valid_temperature',
            'within_5c': None,
            'within_10c': None,
            'within_20c': None,
        }
    return metrics


def _refresh_metrics(row: dict) -> dict:
    frame_path = row.get('scored_test_file') or row.get('candidate_table')
    if not frame_path:
        return row

    path = Path(str(frame_path))
    if not path.exists():
        return row

    try:
        frame = pd.read_csv(path)
    except Exception:
        return row

    score_column = None
    temperature_column = None
    if 'xgb_score' in frame.columns:
        score_column = 'xgb_score'
        temperature_column = 'xgb_temperature_pred' if 'xgb_temperature_pred' in frame.columns else None
    elif 'model_score' in frame.columns:
        score_column = 'model_score'
        temperature_column = 'model_temperature_pred' if 'model_temperature_pred' in frame.columns else None
    elif 'legacy_score' in frame.columns:
        score_column = 'legacy_score'
        temperature_column = 'legacy_temperature_pred' if 'legacy_temperature_pred' in frame.columns else None

    if not score_column:
        return row

    updated = dict(row)
    updated['metrics'] = _evaluate_scored_frame(
        frame,
        score_column=score_column,
        temperature_column=temperature_column,
    )
    return updated


def _load_rows(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for result_file in sorted(output_root.glob('*/**/result.json')):
        try:
            row = json.loads(result_file.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        baseline = row.get('baseline')
        family = row.get('family')
        if not baseline or not family:
            continue
        if baseline not in BASELINE_LABELS:
            continue
        rows.append(_refresh_metrics(row))
    rows.sort(key=lambda row: (_family_sort_key(row['family']), row['baseline']))
    return rows


def _flatten_rows(rows: list[dict]) -> pd.DataFrame:
    flat_rows: list[dict] = []
    for row in rows:
        metrics = row.get('metrics', {})
        temp = metrics.get('temperature', {})
        recall = row.get('stage1_route_recall', {})
        flat_rows.append(
            {
                'family': row.get('family'),
                'baseline': row.get('baseline'),
                'rr10': recall.get('route_recall_top10'),
                'pool_coverage': metrics.get('pool_coverage'),
                'sys1': metrics.get('system_top1_all'),
                'sys5': metrics.get('system_top5_all'),
                'sys10': metrics.get('system_top10_all'),
                'temp_mae': temp.get('mae'),
                'temp_mse': temp.get('mse'),
                'temp_rmse': temp.get('rmse'),
                'temp_within_5c': temp.get('within_5c'),
                'temp_within_10c': temp.get('within_10c'),
                'temp_within_20c': temp.get('within_20c'),
                'temp_n': temp.get('n'),
            }
        )
    return pd.DataFrame(flat_rows)


def _report_filter_families(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    families = sorted(frame['family'].dropna().astype(str).unique().tolist(), key=_family_sort_key)
    if frame.empty or 'baseline' not in frame.columns or 'sys10' not in frame.columns:
        return families, []

    mainline = frame[frame['baseline'] == REPORT_FILTER_BASELINE].copy()
    if mainline.empty:
        return families, []

    sys10 = pd.to_numeric(mainline['sys10'], errors='coerce')
    keep = set(mainline.loc[sys10 > REPORT_FILTER_SYS10_THRESHOLD, 'family'].dropna().astype(str).tolist())
    kept = [family for family in families if family in keep]
    removed = [family for family in families if family not in keep]
    return kept, removed


def _filter_report_frame(frame: pd.DataFrame) -> pd.DataFrame:
    kept, _ = _report_filter_families(frame)
    if not kept:
        return frame.iloc[0:0].copy()
    return frame[frame['family'].isin(kept)].copy()


def _report_filter_note(frame: pd.DataFrame) -> str | None:
    _, removed = _report_filter_families(frame)
    if not removed:
        return None
    removed_text = ', '.join(display_family_name(family) for family in removed)
    return (
        f'Filtered to families with {BASELINE_LABELS[REPORT_FILTER_BASELINE]} sys@10 > '
        f'{REPORT_FILTER_SYS10_THRESHOLD * 100.0:.0f}%. Removed: {removed_text}.'
    )


def _macro_rows(frame: pd.DataFrame, baselines: list[str]) -> list[dict]:
    rows: list[dict] = []
    for baseline in baselines:
        block = frame[frame['baseline'] == baseline]
        if block.empty:
            continue
        rows.append(
            {
                'baseline': baseline,
                'label': BASELINE_LABELS.get(baseline, baseline),
                'families': int(block['family'].nunique()),
                'rr10': _mean_metric(block['rr10'].dropna().astype(float).tolist()),
                'pool_coverage': _mean_metric(block['pool_coverage'].dropna().astype(float).tolist()),
                'sys1': _mean_metric(block['sys1'].dropna().astype(float).tolist()),
                'sys5': _mean_metric(block['sys5'].dropna().astype(float).tolist()),
                'sys10': _mean_metric(block['sys10'].dropna().astype(float).tolist()),
                'temp_mae': _mean_metric(block['temp_mae'].dropna().astype(float).tolist()),
                'temp_within_5c': _mean_metric(block['temp_within_5c'].dropna().astype(float).tolist()),
                'temp_within_10c': _mean_metric(block['temp_within_10c'].dropna().astype(float).tolist()),
                'temp_within_20c': _mean_metric(block['temp_within_20c'].dropna().astype(float).tolist()),
            }
        )
    return rows


def _family_metric_matrix(frame: pd.DataFrame, baselines: list[str], *, metric: str) -> list[dict]:
    rows: list[dict] = []
    families = sorted(frame['family'].dropna().unique().tolist(), key=_family_sort_key)
    for family in families:
        row = {'family': display_family_name(family)}
        block = frame[frame['family'] == family]
        for baseline in baselines:
            match = block[block['baseline'] == baseline]
            row[baseline] = None if match.empty else match.iloc[0][metric]
        rows.append(row)

    macro = {'family': 'MACRO-AVG'}
    for baseline in baselines:
        values = [row[baseline] for row in rows if row.get(baseline) is not None]
        macro[baseline] = _mean_metric(values)
    rows.append(macro)
    return rows


def _render_macro_markdown(rows: list[dict]) -> list[str]:
    headers = ['Baseline', 'n_fam', 'rr@10', 'cover', 'sys@1', 'sys@5', 'sys@10']
    headers.extend(label for _, label, _ in TEMPERATURE_METRICS)
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        temp_cells = [
            _format_metric(row.get(metric), percent=percent)
            for metric, _, percent in TEMPERATURE_METRICS
        ]
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['label'],
                    str(row['families']),
                    _format_metric(row['rr10'], percent=True),
                    _format_metric(row['pool_coverage'], percent=True),
                    _format_metric(row['sys1'], percent=True),
                    _format_metric(row['sys5'], percent=True),
                    _format_metric(row['sys10'], percent=True),
                    *temp_cells,
                ]
            )
            + ' |'
        )
    return lines


def _render_family_metric_markdown(rows: list[dict], baselines: list[str], *, title: str, percent: bool) -> list[str]:
    labels = [BASELINE_LABELS.get(baseline, baseline) for baseline in baselines]
    lines = [f'## {title}']
    lines.append('| Family | ' + ' | '.join(labels) + ' |')
    lines.append('| --- | ' + ' | '.join(['---:' for _ in baselines]) + ' |')
    for row in rows:
        values = [_format_metric(row.get(baseline), percent=percent) for baseline in baselines]
        lines.append('| ' + ' | '.join([row['family'], *values]) + ' |')
    return lines


def _write_report(
    frame: pd.DataFrame,
    output_file: Path,
    *,
    title: str,
    baselines: list[str],
    include_coverage: bool,
    note: str | None = None,
) -> Path:
    lines = [f'# {title}', '']
    if note:
        lines.extend([note, ''])
    lines.extend(['## Macro Summary', ''])
    lines.extend(_render_macro_markdown(_macro_rows(frame, baselines)))
    lines.append('')
    if include_coverage:
        lines.extend(
            _render_family_metric_markdown(
                _family_metric_matrix(frame, baselines, metric='pool_coverage'),
                baselines,
                title='Coverage by Family',
                percent=True,
            )
        )
        lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='sys1'), baselines, title='sys@1 by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='sys5'), baselines, title='sys@5 by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='sys10'), baselines, title='sys@10 by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='temp_mae'), baselines, title='Temp MAE by Family', percent=False))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='temp_within_5c'), baselines, title='Temp±5C by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='temp_within_10c'), baselines, title='Temp±10C by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='temp_within_20c'), baselines, title='Temp±20C by Family', percent=True))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_file


def _write_overview(frame: pd.DataFrame, output_file: Path, *, note: str | None = None) -> Path:
    lines = [
        '# Stage23 Non-Oracle Results Overview',
        '',
    ]
    if note:
        lines.extend([note, ''])
    lines.extend(
        [
            '| Family | Baseline | rr@10 | cover | sys@1 | sys@5 | sys@10 | Temp MAE | Temp±5C | Temp±10C | Temp±20C |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
    )
    for _, row in frame.sort_values(['family', 'baseline']).iterrows():
        lines.append(
            '| '
            + ' | '.join(
                [
                    display_family_name(str(row['family'])),
                    BASELINE_LABELS.get(str(row['baseline']), str(row['baseline'])),
                    'NA' if pd.isna(row['rr10']) else f'{float(row["rr10"]) * 100.0:.1f}',
                    'NA' if pd.isna(row['pool_coverage']) else f'{float(row["pool_coverage"]) * 100.0:.1f}',
                    'NA' if pd.isna(row['sys1']) else f'{float(row["sys1"]) * 100.0:.1f}',
                    'NA' if pd.isna(row['sys5']) else f'{float(row["sys5"]) * 100.0:.1f}',
                    'NA' if pd.isna(row['sys10']) else f'{float(row["sys10"]) * 100.0:.1f}',
                    'NA' if pd.isna(row['temp_mae']) else f'{float(row["temp_mae"]):.1f}',
                    'NA' if pd.isna(row['temp_within_5c']) else f'{float(row["temp_within_5c"]) * 100.0:.1f}',
                    'NA' if pd.isna(row['temp_within_10c']) else f'{float(row["temp_within_10c"]) * 100.0:.1f}',
                    'NA' if pd.isna(row['temp_within_20c']) else f'{float(row["temp_within_20c"]) * 100.0:.1f}',
                ]
            )
            + ' |'
        )
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_file


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return 'NA'
    return f'{float(value) * 100.0:.1f}'


def _macro_table(frame: pd.DataFrame, baselines: list[str]) -> list[str]:
    metrics = ['rr10', 'pool_coverage', 'sys1', 'sys5', 'sys10', 'temp_mae', 'temp_within_5c', 'temp_within_10c', 'temp_within_20c']
    lines = [
        '| Baseline | n_fam | rr@10 | cover | sys@1 | sys@5 | sys@10 | Temp MAE | Temp±5C | Temp±10C | Temp±20C |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for baseline in baselines:
        block = frame[frame['baseline'] == baseline]
        if block.empty:
            continue
        means = {metric: block[metric].mean() for metric in metrics}
        lines.append(
            '| '
            + ' | '.join(
                [
                    BASELINE_LABELS.get(baseline, baseline),
                    str(int(block['family'].nunique())),
                    _fmt_pct(means['rr10']),
                    _fmt_pct(means['pool_coverage']),
                    _fmt_pct(means['sys1']),
                    _fmt_pct(means['sys5']),
                    _fmt_pct(means['sys10']),
                    'NA' if pd.isna(means['temp_mae']) else f'{float(means["temp_mae"]):.1f}',
                    _fmt_pct(means['temp_within_5c']),
                    _fmt_pct(means['temp_within_10c']),
                    _fmt_pct(means['temp_within_20c']),
                ]
            )
            + ' |'
        )
    return lines


def _delta_table(frame: pd.DataFrame, lhs: str, rhs: str) -> list[str]:
    metrics = [
        ('pool_coverage', 'cover', True),
        ('sys1', 'sys@1', True),
        ('sys5', 'sys@5', True),
        ('sys10', 'sys@10', True),
        ('temp_mae', 'Temp MAE', False),
        ('temp_within_5c', 'Temp±5C', True),
        ('temp_within_10c', 'Temp±10C', True),
        ('temp_within_20c', 'Temp±20C', True),
    ]
    left = frame[frame['baseline'] == lhs]
    right = frame[frame['baseline'] == rhs]
    if left.empty or right.empty:
        return []

    left_mean = left.set_index('family')
    right_mean = right.set_index('family')
    common = sorted(set(left_mean.index).intersection(set(right_mean.index)), key=_family_sort_key)
    if not common:
        return []

    lines = ['| Metric | Delta |', '| --- | ---: |']
    for metric, label, percent in metrics:
        delta = left_mean.loc[common, metric].mean() - right_mean.loc[common, metric].mean()
        if pd.isna(delta):
            lines.append(f'| {label} | NA |')
        elif percent:
            lines.append(f'| {label} | {delta * 100.0:+.1f} pp |')
        else:
            lines.append(f'| {label} | {delta:+.1f} |')
    return lines


def _write_average_effect(frame: pd.DataFrame, output_file: Path) -> Path:
    families = sorted(frame['family'].dropna().astype(str).unique().tolist(), key=_family_sort_key)
    family_text = ', '.join(display_family_name(family).rstrip('.') for family in families)
    lines = [
        '# Average Effect Summary',
        '',
        'Scope: families with `KNN+XGB sys@10 > 20%`.',
        '',
        f'Included families: {family_text}.',
        '',
        '## Historical Baseline Mean',
        '',
    ]
    lines.extend(_macro_table(frame, HISTORICAL_BASELINES))
    lines.extend(['', '## Mainline Delta vs Original FNN', ''])
    lines.extend(_delta_table(frame, 'knn_xgb', 'prototype_fnn'))
    lines.extend(['', '## Stage 3 Mean', ''])
    lines.extend(_macro_table(frame, STAGE3_BASELINES))
    lines.extend(['', '## Stage 2 Mean', ''])
    lines.extend(_macro_table(frame, STAGE2_BASELINES))
    lines.extend(['', 'Note: temperature metrics are evaluated separately on the highest-ranked full-match candidate with a valid gold/predicted temperature for each sample.'])
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description='Render final Stage23 Non-Oracle reports from result.json files.')
    parser.add_argument('--output_root', type=str, required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    rows = _load_rows(output_root)
    if not rows:
        raise SystemExit(f'No result.json files found under {output_root}')

    flat = _flatten_rows(rows)
    flat = flat.sort_values(['family', 'baseline']).reset_index(drop=True)
    flat.to_csv(output_root / 'results_flat.csv', index=False)
    (output_root / 'all_results.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    report_flat = _filter_report_frame(flat)
    report_note = _report_filter_note(flat)

    _write_overview(report_flat, output_root / 'overview.md', note=report_note)
    _write_report(
        report_flat[report_flat['baseline'].isin(HISTORICAL_BASELINES)].copy(),
        output_root / 'baseline_historical.md',
        title='Historical Baseline: Original FNN vs Mainline',
        baselines=HISTORICAL_BASELINES,
        include_coverage=True,
        note=report_note,
    )
    _write_report(
        report_flat[report_flat['baseline'].isin(STAGE3_BASELINES)].copy(),
        output_root / 'ablation_stage3.md',
        title='Stage 3 Ablation: Fixed KNN Pool, Different Rerankers',
        baselines=STAGE3_BASELINES,
        include_coverage=False,
        note=report_note,
    )
    _write_report(
        report_flat[report_flat['baseline'].isin(STAGE2_BASELINES)].copy(),
        output_root / 'ablation_stage2.md',
        title='Stage 2 Ablation: Fixed XGBoost, Different Candidate Pools',
        baselines=STAGE2_BASELINES,
        include_coverage=True,
        note=report_note,
    )
    _write_average_effect(report_flat, output_root / 'average_effect.md')

    print(f'rendered {len(rows)} rows to {output_root}')


if __name__ == '__main__':
    main()
