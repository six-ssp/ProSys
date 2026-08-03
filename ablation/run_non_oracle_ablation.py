"""Run the maintained ProSys Non-Oracle ablation suite.

This script executes the three ablations defined in `ablation/ablation.md`:

1. Stage 1: base route model vs family-tuned route model (`route@k`)
2. Stage 2: KNN candidate pool vs Top-K frequency candidate pool (`sys@k`)
3. Stage 3: keep XGBoost reranker vs remove Stage 3 and use heuristic ranking (`sys@k`)
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

from prosys_shared.mainline import (
    FAMILY_ORDER,
    base_candidate_row,
    display_family_name,
    evaluate_scored_frame,
    label_candidate_table,
    load_route_records,
    parse_families_arg,
    split_file_for_family,
    stable_sort_candidate_frame,
    stage1_route_recall,
)
from prosys_shared.product_memory import normalize_condition_labels
from prosys_shared.route_cache import load_route_records_from_cache
from stage3_XGBoost import score_table_with_xgb, train_xgb_ranker_and_temperature

TOPKS = (1, 3, 5, 10)
STAGE2_BASELINES = ('knn_xgb', 'frequency_topk_xgb')
STAGE3_BASELINES = ('no_stage3', 'knn_xgb')
STAGE2_LABELS = {
    'knn_xgb': 'KNN + XGBoost',
    'frequency_topk_xgb': 'Top-K Frequency + XGBoost',
}
STAGE3_LABELS = {
    'no_stage3': 'w/o Stage 3',
    'knn_xgb': 'w/ Stage 3',
}


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(FAMILY_ORDER), family)


def _write_json(data: dict | list, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output_file


def _mean_metric(values: list[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return 'NA'
    return f'{value * 100.0:.1f}'


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return 'NA'
    return f'{value:.{digits}f}'


def _shared_paths(shared_root: Path) -> dict[str, Path]:
    return {
        'candidate_train': shared_root / 'candidate_pool' / 'train.csv',
        'candidate_val': shared_root / 'candidate_pool' / 'val.csv',
        'candidate_test': shared_root / 'candidate_pool' / 'test.csv',
        'table_train': shared_root / 'training_tables' / 'train.csv',
        'table_val': shared_root / 'training_tables' / 'val.csv',
        'table_test': shared_root / 'training_tables' / 'test.csv',
    }


def _load_mainline_result(mainline_root: Path, family: str) -> dict:
    result_file = mainline_root / family / 'knn_xgb' / 'non_oracle' / 'result.json'
    if not result_file.exists():
        raise FileNotFoundError(f'Missing mainline result: {result_file}')
    return json.loads(result_file.read_text(encoding='utf-8'))


def _load_family_context_frequency(split_file: Path) -> list[dict]:
    counts: dict[tuple[str, str], dict[str, float]] = {}
    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 7:
                continue

            reagent_norm = normalize_condition_labels(parts[4])
            solvent_norm = normalize_condition_labels(parts[5])
            key = (reagent_norm, solvent_norm)
            bucket = counts.setdefault(
                key,
                {
                    'reagent_norm': reagent_norm,
                    'solvent_norm': solvent_norm,
                    'count': 0.0,
                    'temperature_sum': 0.0,
                    'temperature_count': 0.0,
                },
            )
            bucket['count'] += 1.0
            try:
                temperature = float(parts[6])
            except (TypeError, ValueError):
                temperature = float('nan')
            if pd.notna(temperature):
                bucket['temperature_sum'] += float(temperature)
                bucket['temperature_count'] += 1.0

    rows: list[dict] = []
    total = sum(float(bucket['count']) for bucket in counts.values()) or 1.0
    for bucket in counts.values():
        temperature_pred = (
            float(bucket['temperature_sum']) / float(bucket['temperature_count'])
            if float(bucket['temperature_count']) > 0.0
            else float('nan')
        )
        rows.append(
            {
                'reagent_norm': str(bucket['reagent_norm']),
                'solvent_norm': str(bucket['solvent_norm']),
                'context_count': float(bucket['count']),
                'context_relative_freq': float(bucket['count']) / float(total),
                'mode_temperature_pred': temperature_pred,
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row['context_count']),
            row['reagent_norm'],
            row['solvent_norm'],
        )
    )
    return rows


def _build_frequency_candidate_table(
    records: list,
    context_rows: list[dict],
    output_file: Path,
) -> Path:
    rows: list[dict] = []
    for record in records:
        base = base_candidate_row(record)
        for rank, context in enumerate(context_rows, start=1):
            rows.append(
                {
                    **base,
                    'reagent_norm': context['reagent_norm'],
                    'solvent_norm': context['solvent_norm'],
                    'from_frequency_topk': 1,
                    'mode_context_count': float(context['context_count']),
                    'mode_context_relative_freq': float(context['context_relative_freq']),
                    'mode_temperature_pred': float(context['mode_temperature_pred']),
                    'mode_rank': rank,
                    # Keep the feature space shape close to the maintained mainline.
                    'from_baseline_knn': 0,
                    'knn_similarity_sum': 0.0,
                    'knn_similarity_max': 0.0,
                    'knn_neighbor_count': 0.0,
                    'knn_weighted_mean_yield': 0.0,
                    'cluster_id': -1,
                    'cluster_context_count': 0.0,
                    'cluster_context_support': 0.0,
                    'cluster_context_mean_yield': 0.0,
                }
            )

    frame = stable_sort_candidate_frame(pd.DataFrame(rows))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)
    return output_file


def _ensure_frequency_tables(
    repo_root: Path,
    family: str,
    route_cache: Path,
    shared_root: Path,
    *,
    max_contexts: int,
    max_train_routes: int | None,
    max_val_routes: int | None,
    force_rebuild: bool,
) -> dict[str, Path]:
    paths = _shared_paths(shared_root)
    if not force_rebuild and all(path.exists() for path in paths.values()):
        return paths

    train_split = split_file_for_family(repo_root, family, 'train')
    val_split = split_file_for_family(repo_root, family, 'val')
    test_split = split_file_for_family(repo_root, family, 'test')

    context_rows = _load_family_context_frequency(train_split)[:max_contexts]
    train_records = load_route_records(train_split, family=family)
    val_records = load_route_records(val_split, family=family)
    test_records = load_route_records_from_cache(route_cache, family=family)

    if max_train_routes is not None:
        train_records = train_records[:max_train_routes]
    if max_val_routes is not None:
        val_records = val_records[:max_val_routes]

    _build_frequency_candidate_table(train_records, context_rows, paths['candidate_train'])
    _build_frequency_candidate_table(val_records, context_rows, paths['candidate_val'])
    _build_frequency_candidate_table(test_records, context_rows, paths['candidate_test'])

    label_candidate_table(paths['candidate_train'], train_split, paths['table_train'])
    label_candidate_table(paths['candidate_val'], val_split, paths['table_val'])
    label_candidate_table(paths['candidate_test'], test_split, paths['table_test'])
    return paths


def _run_frequency_xgb(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_root: Path,
    *,
    max_contexts: int,
    max_train_routes: int | None,
    max_val_routes: int | None,
    force_rebuild: bool,
) -> dict:
    result_dir = output_root / family / 'frequency_topk_xgb' / 'non_oracle'
    result_file = result_dir / 'result.json'
    if result_file.exists() and not force_rebuild:
        return json.loads(result_file.read_text(encoding='utf-8'))

    table_paths = _ensure_frequency_tables(
        repo_root=repo_root,
        family=family,
        route_cache=route_cache,
        shared_root=output_root / family / '_shared_frequency_topk',
        max_contexts=max_contexts,
        max_train_routes=max_train_routes,
        max_val_routes=max_val_routes,
        force_rebuild=force_rebuild,
    )

    artifacts = train_xgb_ranker_and_temperature(
        train_table_file=table_paths['table_train'],
        val_table_file=table_paths['table_val'],
        output_dir=result_dir / 'model',
    )
    scored = score_table_with_xgb(
        table_file=table_paths['table_test'],
        model_file=artifacts['model_file'],
        metadata_file=artifacts['metadata_file'],
        temperature_model_file=artifacts.get('temperature_model_file'),
        temperature_metadata_file=artifacts.get('temperature_metadata_file'),
    )
    scored_file = result_dir / 'test_scored.csv'
    scored_file.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_file, index=False)

    result = {
        'family': family,
        'baseline': 'frequency_topk_xgb',
        'candidate_table': str(table_paths['table_test']),
        'scored_test_file': str(scored_file),
        'model': artifacts,
        'metrics': evaluate_scored_frame(scored, score_column='xgb_score', temperature_column='xgb_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, result_file)
    return result


def _run_no_stage3(
    mainline_root: Path,
    family: str,
    route_cache: Path,
    output_root: Path,
    *,
    force_rebuild: bool,
) -> dict:
    result_dir = output_root / family / 'no_stage3'
    result_file = result_dir / 'result.json'
    if result_file.exists() and not force_rebuild:
        return json.loads(result_file.read_text(encoding='utf-8'))

    table_file = mainline_root / family / '_shared_knn' / 'training_tables' / 'test.csv'
    if not table_file.exists():
        raise FileNotFoundError(f'Missing KNN labeled test table for {family}: {table_file}')

    frame = pd.read_csv(table_file)
    sort_columns = [
        'sample_index',
        'retro_rank',
        'retro_probability',
        'knn_similarity_sum',
        'knn_similarity_max',
        'knn_neighbor_count',
        'knn_weighted_mean_yield',
        'reagent_norm',
        'solvent_norm',
    ]
    ascending = [True, True, False, False, False, False, False, True, True]
    ranked = frame.sort_values(sort_columns, ascending=ascending, kind='mergesort').reset_index(drop=True)
    ranked['heuristic_rank'] = ranked.groupby('sample_index', sort=False).cumcount() + 1
    ranked['heuristic_score'] = -ranked['heuristic_rank'].astype(np.float32)

    scored_file = result_dir / 'test_scored.csv'
    scored_file.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(scored_file, index=False)

    result = {
        'family': family,
        'baseline': 'no_stage3',
        'candidate_table': str(table_file),
        'scored_test_file': str(scored_file),
        'metrics': evaluate_scored_frame(ranked, score_column='heuristic_score'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, result_file)
    return result


def _load_stage1_rows(base_vs_tuned_csv: Path, families: list[str]) -> list[dict]:
    if not base_vs_tuned_csv.exists():
        raise FileNotFoundError(f'Missing Stage 1 comparison file: {base_vs_tuned_csv}')
    frame = pd.read_csv(base_vs_tuned_csv)
    frame = frame[frame['family'].isin(families)].copy()
    rows: list[dict] = []
    for family in sorted(frame['family'].dropna().astype(str).tolist(), key=_family_sort_key):
        row = frame.loc[frame['family'] == family].iloc[0]
        rows.append(
            {
                'family': family,
                'display_family': row.get('display_family', display_family_name(family)),
                'test_products': float(row.get('test_products', float('nan'))),
                'base_route_at_1': float(row['base_route_at_1']),
                'base_route_at_3': float(row['base_route_at_3']),
                'base_route_at_5': float(row['base_route_at_5']),
                'base_route_at_10': float(row['base_route_at_10']),
                'family_tuned_route_at_1': float(row['family_tuned_route_at_1']),
                'family_tuned_route_at_3': float(row['family_tuned_route_at_3']),
                'family_tuned_route_at_5': float(row['family_tuned_route_at_5']),
                'family_tuned_route_at_10': float(row['family_tuned_route_at_10']),
                'delta_route_at_1': float(row['delta_route_at_1']),
                'delta_route_at_3': float(row['delta_route_at_3']),
                'delta_route_at_5': float(row['delta_route_at_5']),
                'delta_route_at_10': float(row['delta_route_at_10']),
            }
        )
    return rows


def _append_stage1_macro(rows: list[dict]) -> list[dict]:
    macro = {'family': 'MACRO-AVG', 'display_family': 'MACRO-AVG', 'test_products': None}
    for column in [
        'base_route_at_1',
        'base_route_at_3',
        'base_route_at_5',
        'base_route_at_10',
        'family_tuned_route_at_1',
        'family_tuned_route_at_3',
        'family_tuned_route_at_5',
        'family_tuned_route_at_10',
        'delta_route_at_1',
        'delta_route_at_3',
        'delta_route_at_5',
        'delta_route_at_10',
    ]:
        macro[column] = _mean_metric([row[column] for row in rows])
    return [*rows, macro]


def _flatten_result_rows(rows: list[dict]) -> pd.DataFrame:
    flat_rows: list[dict] = []
    for row in rows:
        metrics = row.get('metrics', {})
        temp = metrics.get('temperature', {})
        recall = row.get('stage1_route_recall', {})
        flat_rows.append(
            {
                'family': row.get('family'),
                'display_family': display_family_name(row.get('family', '')),
                'baseline': row.get('baseline'),
                'rr1': recall.get('route_recall_top1'),
                'rr3': recall.get('route_recall_top3'),
                'rr5': recall.get('route_recall_top5'),
                'rr10': recall.get('route_recall_top10'),
                'pool_coverage': metrics.get('pool_coverage'),
                'sys1': metrics.get('system_top1_all'),
                'sys3': metrics.get('system_top3_all'),
                'sys5': metrics.get('system_top5_all'),
                'sys10': metrics.get('system_top10_all'),
                'temp_mae': temp.get('mae'),
                'temp_within_5c': temp.get('within_5c'),
                'temp_within_10c': temp.get('within_10c'),
                'temp_within_20c': temp.get('within_20c'),
                'temp_n': temp.get('n'),
            }
        )
    return pd.DataFrame(flat_rows)


def _render_stage1_markdown(rows: list[dict]) -> list[str]:
    lines = [
        '# Ablation A1: Stage 1 Base vs Finetuned',
        '',
        '| Family | Base@1 | Tuned@1 | Delta@1 | Base@3 | Tuned@3 | Delta@3 | Base@5 | Tuned@5 | Delta@5 | Base@10 | Tuned@10 | Delta@10 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['display_family'],
                    _format_percent(row.get('base_route_at_1')),
                    _format_percent(row.get('family_tuned_route_at_1')),
                    _format_percent(row.get('delta_route_at_1')),
                    _format_percent(row.get('base_route_at_3')),
                    _format_percent(row.get('family_tuned_route_at_3')),
                    _format_percent(row.get('delta_route_at_3')),
                    _format_percent(row.get('base_route_at_5')),
                    _format_percent(row.get('family_tuned_route_at_5')),
                    _format_percent(row.get('delta_route_at_5')),
                    _format_percent(row.get('base_route_at_10')),
                    _format_percent(row.get('family_tuned_route_at_10')),
                    _format_percent(row.get('delta_route_at_10')),
                ]
            )
            + ' |'
        )
    return lines


def _render_stage2_markdown(frame: pd.DataFrame) -> list[str]:
    lines = [
        '# Ablation A2: Stage 2 KNN vs Top-K Frequency Pool',
        '',
        '| Family | Cover (KNN) | Cover (Freq) | sys@1 (KNN) | sys@1 (Freq) | sys@3 (KNN) | sys@3 (Freq) | sys@5 (KNN) | sys@5 (Freq) | sys@10 (KNN) | sys@10 (Freq) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]

    families = sorted(frame['family'].dropna().astype(str).unique().tolist(), key=_family_sort_key)
    row_store: list[dict] = []
    for family in families:
        block = frame[frame['family'] == family]
        row = {
            'family': display_family_name(family),
            'knn_xgb_cover': block.loc[block['baseline'] == 'knn_xgb', 'pool_coverage'].iloc[0],
            'frequency_cover': block.loc[block['baseline'] == 'frequency_topk_xgb', 'pool_coverage'].iloc[0],
            'knn_xgb_sys1': block.loc[block['baseline'] == 'knn_xgb', 'sys1'].iloc[0],
            'frequency_sys1': block.loc[block['baseline'] == 'frequency_topk_xgb', 'sys1'].iloc[0],
            'knn_xgb_sys3': block.loc[block['baseline'] == 'knn_xgb', 'sys3'].iloc[0],
            'frequency_sys3': block.loc[block['baseline'] == 'frequency_topk_xgb', 'sys3'].iloc[0],
            'knn_xgb_sys5': block.loc[block['baseline'] == 'knn_xgb', 'sys5'].iloc[0],
            'frequency_sys5': block.loc[block['baseline'] == 'frequency_topk_xgb', 'sys5'].iloc[0],
            'knn_xgb_sys10': block.loc[block['baseline'] == 'knn_xgb', 'sys10'].iloc[0],
            'frequency_sys10': block.loc[block['baseline'] == 'frequency_topk_xgb', 'sys10'].iloc[0],
        }
        row_store.append(row)
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['family'],
                    _format_percent(row['knn_xgb_cover']),
                    _format_percent(row['frequency_cover']),
                    _format_percent(row['knn_xgb_sys1']),
                    _format_percent(row['frequency_sys1']),
                    _format_percent(row['knn_xgb_sys3']),
                    _format_percent(row['frequency_sys3']),
                    _format_percent(row['knn_xgb_sys5']),
                    _format_percent(row['frequency_sys5']),
                    _format_percent(row['knn_xgb_sys10']),
                    _format_percent(row['frequency_sys10']),
                ]
            )
            + ' |'
        )

    macro = {
        'family': 'MACRO-AVG',
        'knn_xgb_cover': _mean_metric([row['knn_xgb_cover'] for row in row_store]),
        'frequency_cover': _mean_metric([row['frequency_cover'] for row in row_store]),
        'knn_xgb_sys1': _mean_metric([row['knn_xgb_sys1'] for row in row_store]),
        'frequency_sys1': _mean_metric([row['frequency_sys1'] for row in row_store]),
        'knn_xgb_sys3': _mean_metric([row['knn_xgb_sys3'] for row in row_store]),
        'frequency_sys3': _mean_metric([row['frequency_sys3'] for row in row_store]),
        'knn_xgb_sys5': _mean_metric([row['knn_xgb_sys5'] for row in row_store]),
        'frequency_sys5': _mean_metric([row['frequency_sys5'] for row in row_store]),
        'knn_xgb_sys10': _mean_metric([row['knn_xgb_sys10'] for row in row_store]),
        'frequency_sys10': _mean_metric([row['frequency_sys10'] for row in row_store]),
    }
    lines.append(
        '| '
        + ' | '.join(
            [
                macro['family'],
                _format_percent(macro['knn_xgb_cover']),
                _format_percent(macro['frequency_cover']),
                _format_percent(macro['knn_xgb_sys1']),
                _format_percent(macro['frequency_sys1']),
                _format_percent(macro['knn_xgb_sys3']),
                _format_percent(macro['frequency_sys3']),
                _format_percent(macro['knn_xgb_sys5']),
                _format_percent(macro['frequency_sys5']),
                _format_percent(macro['knn_xgb_sys10']),
                _format_percent(macro['frequency_sys10']),
            ]
        )
        + ' |'
    )
    return lines


def _render_stage3_markdown(frame: pd.DataFrame) -> list[str]:
    lines = [
        '# Ablation A3: Keep Stage 3 vs Remove Stage 3',
        '',
        '| Family | sys@1 (w/o) | sys@1 (w/) | sys@3 (w/o) | sys@3 (w/) | sys@5 (w/o) | sys@5 (w/) | sys@10 (w/o) | sys@10 (w/) | Delta@10 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]

    families = sorted(frame['family'].dropna().astype(str).unique().tolist(), key=_family_sort_key)
    row_store: list[dict] = []
    for family in families:
        block = frame[frame['family'] == family]
        row = {
            'family': display_family_name(family),
            'no_stage3_sys1': block.loc[block['baseline'] == 'no_stage3', 'sys1'].iloc[0],
            'knn_xgb_sys1': block.loc[block['baseline'] == 'knn_xgb', 'sys1'].iloc[0],
            'no_stage3_sys3': block.loc[block['baseline'] == 'no_stage3', 'sys3'].iloc[0],
            'knn_xgb_sys3': block.loc[block['baseline'] == 'knn_xgb', 'sys3'].iloc[0],
            'no_stage3_sys5': block.loc[block['baseline'] == 'no_stage3', 'sys5'].iloc[0],
            'knn_xgb_sys5': block.loc[block['baseline'] == 'knn_xgb', 'sys5'].iloc[0],
            'no_stage3_sys10': block.loc[block['baseline'] == 'no_stage3', 'sys10'].iloc[0],
            'knn_xgb_sys10': block.loc[block['baseline'] == 'knn_xgb', 'sys10'].iloc[0],
        }
        row['delta_sys10'] = row['knn_xgb_sys10'] - row['no_stage3_sys10']
        row_store.append(row)
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['family'],
                    _format_percent(row['no_stage3_sys1']),
                    _format_percent(row['knn_xgb_sys1']),
                    _format_percent(row['no_stage3_sys3']),
                    _format_percent(row['knn_xgb_sys3']),
                    _format_percent(row['no_stage3_sys5']),
                    _format_percent(row['knn_xgb_sys5']),
                    _format_percent(row['no_stage3_sys10']),
                    _format_percent(row['knn_xgb_sys10']),
                    _format_percent(row['delta_sys10']),
                ]
            )
            + ' |'
        )

    macro = {
        'family': 'MACRO-AVG',
        'no_stage3_sys1': _mean_metric([row['no_stage3_sys1'] for row in row_store]),
        'knn_xgb_sys1': _mean_metric([row['knn_xgb_sys1'] for row in row_store]),
        'no_stage3_sys3': _mean_metric([row['no_stage3_sys3'] for row in row_store]),
        'knn_xgb_sys3': _mean_metric([row['knn_xgb_sys3'] for row in row_store]),
        'no_stage3_sys5': _mean_metric([row['no_stage3_sys5'] for row in row_store]),
        'knn_xgb_sys5': _mean_metric([row['knn_xgb_sys5'] for row in row_store]),
        'no_stage3_sys10': _mean_metric([row['no_stage3_sys10'] for row in row_store]),
        'knn_xgb_sys10': _mean_metric([row['knn_xgb_sys10'] for row in row_store]),
        'delta_sys10': _mean_metric([row['delta_sys10'] for row in row_store]),
    }
    lines.append(
        '| '
        + ' | '.join(
            [
                macro['family'],
                _format_percent(macro['no_stage3_sys1']),
                _format_percent(macro['knn_xgb_sys1']),
                _format_percent(macro['no_stage3_sys3']),
                _format_percent(macro['knn_xgb_sys3']),
                _format_percent(macro['no_stage3_sys5']),
                _format_percent(macro['knn_xgb_sys5']),
                _format_percent(macro['no_stage3_sys10']),
                _format_percent(macro['knn_xgb_sys10']),
                _format_percent(macro['delta_sys10']),
            ]
        )
        + ' |'
    )
    return lines


def _write_overview(
    stage1_rows: list[dict],
    stage2_frame: pd.DataFrame,
    stage3_frame: pd.DataFrame,
    output_root: Path,
) -> None:
    lines: list[str] = ['# Non-Oracle Ablation Results', '']
    lines.extend(_render_stage1_markdown(stage1_rows))
    lines.extend(['', ''])
    lines.extend(_render_stage2_markdown(stage2_frame))
    lines.extend(['', ''])
    lines.extend(_render_stage3_markdown(stage3_frame))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'overview.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output_root / 'stage1_ablation.md').write_text('\n'.join(_render_stage1_markdown(stage1_rows)) + '\n', encoding='utf-8')
    (output_root / 'stage2_ablation.md').write_text('\n'.join(_render_stage2_markdown(stage2_frame)) + '\n', encoding='utf-8')
    (output_root / 'stage3_ablation.md').write_text('\n'.join(_render_stage3_markdown(stage3_frame)) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run ProSys Non-Oracle ablation experiments.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--mainline_root', type=str, default='outputs/stage23_mainline')
    parser.add_argument('--stage1_csv', type=str, default='outputs/checklist_stats/07_stage1_base_vs_tuned.csv')
    parser.add_argument('--output_root', type=str, default='outputs/ablation')
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--max_train_routes', type=int, default=2000)
    parser.add_argument('--max_val_routes', type=int, default=500)
    parser.add_argument('--force_rebuild', action='store_true')
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    mainline_root = (repo_root / args.mainline_root).resolve()
    stage1_csv = (repo_root / args.stage1_csv).resolve()
    output_root = (repo_root / args.output_root).resolve()
    families = parse_families_arg(args.families)

    stage1_rows = _append_stage1_macro(_load_stage1_rows(stage1_csv, families))

    stage2_rows: list[dict] = []
    stage3_rows: list[dict] = []
    all_results: dict[str, object] = {
        'families': families,
        'stage1': stage1_rows,
        'stage2': [],
        'stage3': [],
    }

    for family in families:
        route_cache = route_root / family / 'route_cache.json'
        if not route_cache.exists():
            raise FileNotFoundError(f'Missing Stage 1 route cache for {family}: {route_cache}')

        print(f'[ablation] family={family} stage2 mainline reference', flush=True)
        mainline_result = _load_mainline_result(mainline_root, family)
        stage2_rows.append(mainline_result)
        stage3_rows.append(mainline_result)

        print(f'[ablation] family={family} stage2 frequency_topk_xgb', flush=True)
        freq_result = _run_frequency_xgb(
            repo_root=repo_root,
            family=family,
            route_cache=route_cache,
            output_root=output_root,
            max_contexts=args.max_contexts,
            max_train_routes=args.max_train_routes,
            max_val_routes=args.max_val_routes,
            force_rebuild=args.force_rebuild,
        )
        stage2_rows.append(freq_result)

        print(f'[ablation] family={family} stage3 no_stage3', flush=True)
        no_stage3_result = _run_no_stage3(
            mainline_root=mainline_root,
            family=family,
            route_cache=route_cache,
            output_root=output_root,
            force_rebuild=args.force_rebuild,
        )
        stage3_rows.append(no_stage3_result)

    all_results['stage2'] = stage2_rows
    all_results['stage3'] = stage3_rows
    _write_json(all_results, output_root / 'all_results.json')

    stage1_frame = pd.DataFrame(stage1_rows)
    stage1_frame.to_csv(output_root / 'stage1_route_ablation.csv', index=False)

    stage2_frame = _flatten_result_rows(stage2_rows)
    stage2_frame = stage2_frame[stage2_frame['baseline'].isin(STAGE2_BASELINES)].copy()
    stage2_frame = stage2_frame.sort_values(['family', 'baseline']).reset_index(drop=True)
    stage2_frame.to_csv(output_root / 'stage2_pool_ablation_long.csv', index=False)

    stage3_frame = _flatten_result_rows(stage3_rows)
    stage3_frame = stage3_frame[stage3_frame['baseline'].isin(STAGE3_BASELINES)].copy()
    stage3_frame = stage3_frame.sort_values(['family', 'baseline']).reset_index(drop=True)
    stage3_frame.to_csv(output_root / 'stage3_rerank_ablation_long.csv', index=False)

    _write_overview(stage1_rows, stage2_frame, stage3_frame, output_root)
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
