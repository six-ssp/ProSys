"""Run Non-Oracle Stage-2/Stage-3 baseline and ablation experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_omp_threads = os.environ.get('OMP_NUM_THREADS', '').strip()
try:
    if int(_omp_threads) <= 0:
        raise ValueError
except (TypeError, ValueError):
    os.environ['OMP_NUM_THREADS'] = '8'

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common import (
    FAMILY_ORDER,
    base_candidate_row,
    display_family_name,
    evaluate_scored_frame,
    label_candidate_table,
    load_route_records,
    parse_families_arg,
    score_table_with_xgb,
    split_file_for_family,
    stable_sort_candidate_frame,
    train_xgb_ranker,
)
from baseline.legacy_models import LegacyRankingEvaluator
from baseline.run_non_oracle_baselines import stage1_route_recall
from baseline.run_oracle_baselines import (
    ClusterContextPoolBuilder,
    _legacy_reaction_fp,
    _load_legacy_evaluators,
)
from baseline.tabular_models import (
    score_table_with_tabular_model,
    train_tabular_ranker_and_temperature,
)
from stage2_KNN import KNNContextPoolBuilder
from prosys_shared.product_memory import normalize_condition_labels
from prosys_shared.route_cache import load_route_records_from_cache

TOPKS = (1, 3, 5, 10)

HISTORICAL_BASELINES = ['prototype_fnn', 'knn_xgb']
STAGE3_BASELINES = ['knn_xgb', 'knn_rf', 'knn_svm', 'knn_bayes']
STAGE2_BASELINES = ['knn_xgb', 'cluster_xgb', 'fnnpool_xgb']
ALL_BASELINES = [
    'prototype_fnn',
    'knn_xgb',
    'knn_rf',
    'knn_svm',
    'knn_bayes',
    'cluster_xgb',
    'fnnpool_xgb',
]
ABLATION_BASELINES = [baseline for baseline in ALL_BASELINES if baseline != 'prototype_fnn']
RUN_SETS = {
    'all': ALL_BASELINES,
    'historical': HISTORICAL_BASELINES,
    'ablation': ABLATION_BASELINES,
    'stage2_ablation': STAGE2_BASELINES,
    'stage3_ablation': STAGE3_BASELINES,
}

BASELINE_LABELS = {
    'prototype_fnn': 'Original FNN',
    'knn_xgb': 'KNN+XGB',
    'knn_rf': 'KNN+RF',
    'knn_svm': 'KNN+SVM',
    'knn_bayes': 'KNN+Bayes',
    'cluster_xgb': 'Cluster+XGB',
    'fnnpool_xgb': 'FNN-pool+XGB',
}

REPORT_FILTER_BASELINE = 'knn_xgb'
REPORT_FILTER_SYS10_THRESHOLD = 0.20


def _write_json(data: dict | list, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output_file


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(FAMILY_ORDER), family)


def _format_metric(value: float | None, *, percent: bool) -> str:
    if value is None:
        return 'NA'
    if percent:
        return f'{value * 100.0:.1f}'
    return f'{value:.2f}'


def _mean_metric(values: list[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not np.isnan(float(value))]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


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


def _shared_paths(shared_root: Path) -> dict[str, Path]:
    return {
        'candidate_train': shared_root / 'candidate_pool' / 'train.csv',
        'candidate_val': shared_root / 'candidate_pool' / 'val.csv',
        'candidate_test': shared_root / 'candidate_pool' / 'test.csv',
        'table_train': shared_root / 'training_tables' / 'train.csv',
        'table_val': shared_root / 'training_tables' / 'val.csv',
        'table_test': shared_root / 'training_tables' / 'test.csv',
    }


def _ensure_knn_tables(
    repo_root: Path,
    family: str,
    route_cache: Path,
    shared_root: Path,
    *,
    top_k: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    max_train_routes: int,
    max_val_routes: int,
) -> dict[str, Path]:
    paths = _shared_paths(shared_root)
    if all(path.exists() for path in paths.values()):
        return paths

    builder = KNNContextPoolBuilder(
        repo_root=repo_root,
        family=family,
        top_k=top_k,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
    )
    builder.build_table('train', paths['candidate_train'], max_routes=max_train_routes)
    builder.build_table('val', paths['candidate_val'], max_routes=max_val_routes)
    builder.build_non_oracle_table(route_cache, paths['candidate_test'])

    label_candidate_table(paths['candidate_train'], split_file_for_family(repo_root, family, 'train'), paths['table_train'])
    label_candidate_table(paths['candidate_val'], split_file_for_family(repo_root, family, 'val'), paths['table_val'])
    label_candidate_table(paths['candidate_test'], split_file_for_family(repo_root, family, 'test'), paths['table_test'])
    return paths


def _ensure_cluster_tables(
    repo_root: Path,
    family: str,
    route_cache: Path,
    shared_root: Path,
    *,
    cluster_num: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    svd_dim: int,
    max_train_routes: int,
    max_val_routes: int,
) -> dict[str, Path]:
    paths = _shared_paths(shared_root)
    if all(path.exists() for path in paths.values()):
        return paths

    builder = ClusterContextPoolBuilder(
        repo_root=repo_root,
        family=family,
        cluster_num=cluster_num,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        svd_dim=svd_dim,
    )
    builder.build_table('train', paths['candidate_train'], max_routes=max_train_routes)
    builder.build_table('val', paths['candidate_val'], max_routes=max_val_routes)

    routes = load_route_records_from_cache(route_cache, family=family)
    rows: list[dict] = []
    for record in routes:
        base = base_candidate_row(record)
        for candidate in builder._candidate_rows(record):
            rows.append(
                {
                    **base,
                    'reagent_norm': candidate['reagent_norm'],
                    'solvent_norm': candidate['solvent_norm'],
                    'from_baseline_cluster': 1,
                    'knn_similarity_sum': 0.0,
                    'knn_similarity_max': 0.0,
                    'knn_neighbor_count': 0.0,
                    'knn_weighted_mean_yield': 0.0,
                    'cluster_id': int(candidate.get('cluster_id', -1)),
                    'cluster_context_count': float(candidate.get('cluster_context_count', 0.0)),
                    'cluster_context_support': float(candidate.get('cluster_context_support', 0.0)),
                    'cluster_context_mean_yield': float(candidate.get('cluster_context_mean_yield', 0.0)),
                }
            )
    stable_sort_candidate_frame(pd.DataFrame(rows)).to_csv(paths['candidate_test'], index=False)

    label_candidate_table(paths['candidate_train'], split_file_for_family(repo_root, family, 'train'), paths['table_train'])
    label_candidate_table(paths['candidate_val'], split_file_for_family(repo_root, family, 'val'), paths['table_val'])
    label_candidate_table(paths['candidate_test'], split_file_for_family(repo_root, family, 'test'), paths['table_test'])
    return paths


def _build_fnn_pool_non_oracle_table(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_file: Path,
    *,
    legacy_max_contexts: int,
) -> Path:
    routes = load_route_records_from_cache(route_cache, family=family)
    mt, _ = _load_legacy_evaluators(repo_root, with_ranker=False)
    context_builder = LegacyRankingEvaluator(mt.solvent_classes, mt.reagent_classes)
    total_routes = len(routes)

    rows: list[dict] = []
    with torch.inference_mode():
        for route_idx, record in enumerate(routes, start=1):
            base = base_candidate_row(record)
            base['from_fnn'] = 1
            rxn_fp = torch.as_tensor(
                _legacy_reaction_fp(
                    record.reactants,
                    record.product,
                    fpsize=mt.args_MT.fpsize,
                    radius=mt.args_MT.radius,
                ),
                dtype=torch.float32,
            )
            input_solvents, input_reagents = mt.make_input_rxn_condition(rxn_fp)
            contexts = context_builder.make_contexts(input_solvents, input_reagents)

            seen = set()
            for rank, (solvent_norm, reagent_norm) in enumerate(contexts, start=1):
                key = (normalize_condition_labels(reagent_norm), normalize_condition_labels(solvent_norm))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        **base,
                        'reagent_norm': key[0],
                        'solvent_norm': key[1],
                        'legacy_rank': rank,
                    }
                )
                if len(seen) >= legacy_max_contexts:
                    break
            if route_idx % 100 == 0 or route_idx == total_routes:
                print(f'[stage23-nonoracle] {family} fnnpool-test {route_idx}/{total_routes}', flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stable_sort_candidate_frame(pd.DataFrame(rows)).to_csv(output_file, index=False)
    return output_file


def _build_fnn_pool_split_table(
    repo_root: Path,
    family: str,
    split: str,
    output_file: Path,
    *,
    legacy_max_contexts: int,
    max_routes: int | None,
) -> Path:
    records = load_route_records(split_file_for_family(repo_root, family, split), family=family)
    if max_routes is not None:
        records = records[:max_routes]

    mt, _ = _load_legacy_evaluators(repo_root, with_ranker=False)
    context_builder = LegacyRankingEvaluator(mt.solvent_classes, mt.reagent_classes)
    total_records = len(records)
    rows: list[dict] = []
    with torch.inference_mode():
        for route_idx, record in enumerate(records, start=1):
            base = base_candidate_row(record)
            base['from_fnn'] = 1
            rxn_fp = torch.as_tensor(
                _legacy_reaction_fp(
                    record.reactants,
                    record.product,
                    fpsize=mt.args_MT.fpsize,
                    radius=mt.args_MT.radius,
                ),
                dtype=torch.float32,
            )
            input_solvents, input_reagents = mt.make_input_rxn_condition(rxn_fp)
            contexts = context_builder.make_contexts(input_solvents, input_reagents)

            seen = set()
            for rank, (solvent_norm, reagent_norm) in enumerate(contexts, start=1):
                key = (normalize_condition_labels(reagent_norm), normalize_condition_labels(solvent_norm))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        **base,
                        'reagent_norm': key[0],
                        'solvent_norm': key[1],
                        'legacy_rank': rank,
                    }
                )
                if len(seen) >= legacy_max_contexts:
                    break
            if route_idx % 100 == 0 or route_idx == total_records:
                print(f'[stage23-nonoracle] {family} fnnpool-{split} {route_idx}/{total_records}', flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stable_sort_candidate_frame(pd.DataFrame(rows)).to_csv(output_file, index=False)
    return output_file


def _ensure_fnnpool_tables(
    repo_root: Path,
    family: str,
    route_cache: Path,
    shared_root: Path,
    *,
    legacy_max_contexts: int,
    max_train_routes: int,
    max_val_routes: int,
) -> dict[str, Path]:
    paths = _shared_paths(shared_root)
    if all(path.exists() for path in paths.values()):
        return paths

    _build_fnn_pool_split_table(
        repo_root=repo_root,
        family=family,
        split='train',
        output_file=paths['candidate_train'],
        legacy_max_contexts=legacy_max_contexts,
        max_routes=max_train_routes,
    )
    _build_fnn_pool_split_table(
        repo_root=repo_root,
        family=family,
        split='val',
        output_file=paths['candidate_val'],
        legacy_max_contexts=legacy_max_contexts,
        max_routes=max_val_routes,
    )

    _build_fnn_pool_non_oracle_table(
        repo_root=repo_root,
        family=family,
        route_cache=route_cache,
        output_file=paths['candidate_test'],
        legacy_max_contexts=legacy_max_contexts,
    )

    label_candidate_table(paths['candidate_train'], split_file_for_family(repo_root, family, 'train'), paths['table_train'])
    label_candidate_table(paths['candidate_val'], split_file_for_family(repo_root, family, 'val'), paths['table_val'])
    label_candidate_table(paths['candidate_test'], split_file_for_family(repo_root, family, 'test'), paths['table_test'])
    return paths


def _run_prototype_fnn(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_dir: Path,
    *,
    legacy_max_contexts: int,
) -> dict:
    result_file = output_dir / 'result.json'
    if result_file.exists():
        return json.loads(result_file.read_text(encoding='utf-8'))

    candidate_file = output_dir / 'candidate_pool_test_scored.csv'
    table_file = output_dir / 'test_scored_labeled.csv'

    routes = load_route_records_from_cache(route_cache, family=family)
    mt, rk = _load_legacy_evaluators(repo_root, with_ranker=True)
    total_routes = len(routes)
    rows: list[dict] = []
    with torch.inference_mode():
        for route_idx, record in enumerate(routes, start=1):
            base = base_candidate_row(record)
            base['from_fnn'] = 1
            rxn_fp = torch.as_tensor(
                _legacy_reaction_fp(
                    record.reactants,
                    record.product,
                    fpsize=mt.args_MT.fpsize,
                    radius=mt.args_MT.radius,
                ),
                dtype=torch.float32,
            )
            input_solvents, input_reagents = mt.make_input_rxn_condition(rxn_fp)
            ranked = rk.rank_top_contexts(rxn_fp, input_solvents, input_reagents, top_k=legacy_max_contexts)
            for rank, (solvent_norm, reagent_norm, temp_pred, score) in enumerate(ranked, start=1):
                rows.append(
                    {
                        **base,
                        'reagent_norm': normalize_condition_labels(reagent_norm),
                        'solvent_norm': normalize_condition_labels(solvent_norm),
                        'legacy_score': float(score),
                        'legacy_temperature_pred': float(temp_pred),
                        'legacy_rank': rank,
                    }
                )
            if route_idx % 100 == 0 or route_idx == total_routes:
                print(f'[stage23-nonoracle] {family} prototype_fnn {route_idx}/{total_routes}', flush=True)
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    stable_sort_candidate_frame(pd.DataFrame(rows)).to_csv(candidate_file, index=False)
    label_candidate_table(candidate_file, split_file_for_family(repo_root, family, 'test'), table_file)
    frame = pd.read_csv(table_file)
    result = {
        'baseline': 'prototype_fnn',
        'family': family,
        'candidate_table': str(table_file),
        'metrics': evaluate_scored_frame(frame, score_column='legacy_score', temperature_column='legacy_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, result_file)
    return result


def _run_xgb_from_tables(
    family: str,
    route_cache: Path,
    output_dir: Path,
    table_paths: dict[str, Path],
    *,
    baseline_name: str,
) -> dict:
    result_file = output_dir / 'result.json'
    if result_file.exists():
        return json.loads(result_file.read_text(encoding='utf-8'))

    artifacts = train_xgb_ranker(table_paths['table_train'], table_paths['table_val'], output_dir / 'model')
    scored = score_table_with_xgb(table_paths['table_test'], artifacts.model_file, artifacts.metadata_file)
    scored_file = output_dir / 'test_scored.csv'
    scored_file.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_file, index=False)
    result = {
        'baseline': baseline_name,
        'family': family,
        'candidate_table': str(table_paths['table_test']),
        'scored_test_file': str(scored_file),
        'model': artifacts.to_dict(),
        'metrics': evaluate_scored_frame(scored, score_column='xgb_score', temperature_column='xgb_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, result_file)
    return result


def _run_tabular_from_tables(
    family: str,
    route_cache: Path,
    output_dir: Path,
    table_paths: dict[str, Path],
    *,
    baseline_name: str,
    kind: str,
) -> dict:
    result_file = output_dir / 'result.json'
    if result_file.exists():
        return json.loads(result_file.read_text(encoding='utf-8'))

    artifacts = train_tabular_ranker_and_temperature(
        table_paths['table_train'],
        table_paths['table_val'],
        output_dir / 'model',
        kind=kind,
    )
    scored = score_table_with_tabular_model(table_paths['table_test'], artifacts['model_file'], artifacts['metadata_file'])
    scored_file = output_dir / 'test_scored.csv'
    scored_file.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_file, index=False)
    result = {
        'baseline': baseline_name,
        'family': family,
        'candidate_table': str(table_paths['table_test']),
        'scored_test_file': str(scored_file),
        'model': artifacts,
        'metrics': evaluate_scored_frame(scored, score_column='model_score', temperature_column='model_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, result_file)
    return result


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
                'temp_within_10c': temp.get('within_10c'),
                'temp_within_20c': temp.get('within_20c'),
                'temp_n': temp.get('n'),
            }
        )
    return pd.DataFrame(flat_rows)


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
                'temp_within_10c': _mean_metric(block['temp_within_10c'].dropna().astype(float).tolist()),
                'temp_within_20c': _mean_metric(block['temp_within_20c'].dropna().astype(float).tolist()),
            }
        )
    return rows


def _family_metric_matrix(
    frame: pd.DataFrame,
    baselines: list[str],
    *,
    metric: str,
) -> list[dict]:
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
    lines = [
        '| Baseline | n_fam | rr@10 | cover | sys@1 | sys@5 | sys@10 | Temp@10C | Temp@20C |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
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
                    _format_metric(row['temp_within_10c'], percent=True),
                    _format_metric(row['temp_within_20c'], percent=True),
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
        lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='pool_coverage'), baselines, title='Coverage by Family', percent=True))
        lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='sys1'), baselines, title='sys@1 by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='sys5'), baselines, title='sys@5 by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='sys10'), baselines, title='sys@10 by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='temp_within_10c'), baselines, title='Temp@10C by Family', percent=True))
    lines.append('')
    lines.extend(_render_family_metric_markdown(_family_metric_matrix(frame, baselines, metric='temp_within_20c'), baselines, title='Temp@20C by Family', percent=True))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run Non-Oracle baseline/ablation experiments for ProSys.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--output_root', type=str, default='outputs/stage23_non_oracle')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--fpsize', type=int, default=4096)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--knn_top_k', type=int, default=20)
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--cluster_num', type=int, default=32)
    parser.add_argument('--svd_dim', type=int, default=64)
    parser.add_argument('--legacy_max_contexts', type=int, default=200)
    parser.add_argument('--max_train_routes', type=int, default=2000)
    parser.add_argument('--max_val_routes', type=int, default=500)
    parser.add_argument(
        '--run_set',
        type=str,
        default='all',
        choices=sorted(RUN_SETS),
        help='which experiment subset to execute',
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    families = parse_families_arg(args.families)
    selected_baselines = set(RUN_SETS[args.run_set])

    summary_rows: list[dict] = []
    for family in families:
        route_cache = route_root / family / 'route_cache.json'
        if not route_cache.exists():
            print(f'[stage23-nonoracle] skip {family}: missing {route_cache}')
            continue

        family_root = output_root / family
        print(f'[stage23-nonoracle] running {family}')

        knn_tables = _ensure_knn_tables(
            repo_root=repo_root,
            family=family,
            route_cache=route_cache,
            shared_root=family_root / '_shared_knn',
            top_k=args.knn_top_k,
            max_contexts=args.max_contexts,
            fpsize=args.fpsize,
            radius=args.radius,
            max_train_routes=args.max_train_routes,
            max_val_routes=args.max_val_routes,
        )
        cluster_tables = _ensure_cluster_tables(
            repo_root=repo_root,
            family=family,
            route_cache=route_cache,
            shared_root=family_root / '_shared_cluster',
            cluster_num=args.cluster_num,
            max_contexts=args.max_contexts,
            fpsize=args.fpsize,
            radius=args.radius,
            svd_dim=args.svd_dim,
            max_train_routes=args.max_train_routes,
            max_val_routes=args.max_val_routes,
        )
        fnnpool_tables = _ensure_fnnpool_tables(
            repo_root=repo_root,
            family=family,
            route_cache=route_cache,
            shared_root=family_root / '_shared_fnnpool',
            legacy_max_contexts=args.legacy_max_contexts,
            max_train_routes=args.max_train_routes,
            max_val_routes=args.max_val_routes,
        )

        if 'prototype_fnn' in selected_baselines:
            summary_rows.append(
                _run_prototype_fnn(
                    repo_root=repo_root,
                    family=family,
                    route_cache=route_cache,
                    output_dir=family_root / 'prototype_fnn' / 'non_oracle',
                    legacy_max_contexts=args.legacy_max_contexts,
                )
            )
        if 'knn_xgb' in selected_baselines:
            summary_rows.append(
                _run_xgb_from_tables(
                    family=family,
                    route_cache=route_cache,
                    output_dir=family_root / 'knn_xgb' / 'non_oracle',
                    table_paths=knn_tables,
                    baseline_name='knn_xgb',
                )
            )
        if 'knn_rf' in selected_baselines:
            summary_rows.append(
                _run_tabular_from_tables(
                    family=family,
                    route_cache=route_cache,
                    output_dir=family_root / 'knn_rf' / 'non_oracle',
                    table_paths=knn_tables,
                    baseline_name='knn_rf',
                    kind='rf',
                )
            )
        if 'knn_svm' in selected_baselines:
            summary_rows.append(
                _run_tabular_from_tables(
                    family=family,
                    route_cache=route_cache,
                    output_dir=family_root / 'knn_svm' / 'non_oracle',
                    table_paths=knn_tables,
                    baseline_name='knn_svm',
                    kind='svm',
                )
            )
        if 'knn_bayes' in selected_baselines:
            summary_rows.append(
                _run_tabular_from_tables(
                    family=family,
                    route_cache=route_cache,
                    output_dir=family_root / 'knn_bayes' / 'non_oracle',
                    table_paths=knn_tables,
                    baseline_name='knn_bayes',
                    kind='bayes',
                )
            )
        if 'cluster_xgb' in selected_baselines:
            summary_rows.append(
                _run_xgb_from_tables(
                    family=family,
                    route_cache=route_cache,
                    output_dir=family_root / 'cluster_xgb' / 'non_oracle',
                    table_paths=cluster_tables,
                    baseline_name='cluster_xgb',
                )
            )
        if 'fnnpool_xgb' in selected_baselines:
            summary_rows.append(
                _run_xgb_from_tables(
                    family=family,
                    route_cache=route_cache,
                    output_dir=family_root / 'fnnpool_xgb' / 'non_oracle',
                    table_paths=fnnpool_tables,
                    baseline_name='fnnpool_xgb',
                )
            )

    flat = _flatten_rows(summary_rows)
    flat = flat.sort_values(['family', 'baseline']).reset_index(drop=True)
    flat.to_csv(output_root / 'results_flat.csv', index=False)
    _write_json(summary_rows, output_root / 'all_results.json')

    report_flat = _filter_report_frame(flat)
    report_note = _report_filter_note(flat)

    if selected_baselines & set(HISTORICAL_BASELINES):
        _write_report(
            report_flat[report_flat['baseline'].isin(HISTORICAL_BASELINES)].copy(),
            output_root / 'baseline_historical.md',
            title='Historical Baseline: Original FNN vs Mainline',
            baselines=HISTORICAL_BASELINES,
            include_coverage=True,
            note=report_note,
        )
    if selected_baselines & set(STAGE3_BASELINES):
        _write_report(
            report_flat[report_flat['baseline'].isin(STAGE3_BASELINES)].copy(),
            output_root / 'ablation_stage3.md',
            title='Stage 3 Ablation: Fixed KNN Pool, Different Rerankers',
            baselines=STAGE3_BASELINES,
            include_coverage=False,
            note=report_note,
        )
    if selected_baselines & set(STAGE2_BASELINES):
        _write_report(
            report_flat[report_flat['baseline'].isin(STAGE2_BASELINES)].copy(),
            output_root / 'ablation_stage2.md',
            title='Stage 2 Ablation: Fixed XGBoost, Different Candidate Pools',
            baselines=STAGE2_BASELINES,
            include_coverage=True,
            note=report_note,
        )

    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
