"""Run the maintained ProSys ablation suite for the ReaFNN-GNN mainline.

The suite uses the existing family-tuned Stage 1 route caches and evaluates
every method against the complete cache identity manifest.  It intentionally
keeps the test path Non-Oracle: only Stage 1 predicted routes enter Stage 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.mainline import (
    FAMILY_ORDER,
    base_candidate_row,
    display_family_name,
    evaluate_scored_frame_with_manifest,
    label_candidate_table,
    load_route_records,
    load_split_rows,
    parse_families_arg,
    split_file_for_family,
    stable_sort_candidate_frame,
    stage1_route_recall,
)
from prosys_shared.product_memory import normalize_condition_labels, safe_float
from prosys_shared.route_cache import load_route_cache_sample_indices, load_route_records_from_cache
from stage2_KNN import KNNContextPoolBuilder
from stage3_XGBoost import score_table_with_xgb, train_xgb_ranker_and_temperature
from stage3_XGBoost.reaction_gnn_features import augment_table_with_reaction_gnn_features


TOPKS = (1, 3, 5, 10)
TARGET_COLUMNS = {
    'label',
    'route_match',
    'context_match',
    'rank_relevance',
    'sample_weight',
    'temperature_gold',
    'yield_gold',
}

METHOD_LABELS = {
    'full_mainline': 'KNN + ReaFNN + Reaction-GNN + XGBoost',
    'knn_only_xgb': 'KNN only + Reaction-GNN + XGBoost',
    'frequency_top20_xgb': 'Top-20 frequency + Reaction-GNN + XGBoost',
    'no_gnn_xgb': 'KNN + ReaFNN + XGBoost (w/o Reaction-GNN)',
    'knn_only_no_gnn_xgb': 'KNN only + XGBoost (w/o ReaFNN and Reaction-GNN)',
    'no_stage3': 'KNN + ReaFNN (w/o XGBoost)',
}


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(FAMILY_ORDER), family)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _write_json(data: dict | list, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + '\n',
        encoding='utf-8',
    )
    return output_file


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _paths_for_pool(pool_root: Path) -> dict[str, Path]:
    return {
        'candidate_train': pool_root / 'candidate_pool' / 'train.csv',
        'candidate_val': pool_root / 'candidate_pool' / 'val.csv',
        'candidate_test': pool_root / 'candidate_pool' / 'test.csv',
        'table_train': pool_root / 'training_tables' / 'train.csv',
        'table_val': pool_root / 'training_tables' / 'val.csv',
        'table_test': pool_root / 'training_tables' / 'test.csv',
    }


def _paths_for_gnn(gnn_root: Path) -> dict[str, Path]:
    return {
        'table_train': gnn_root / 'training_tables' / 'train.csv',
        'table_val': gnn_root / 'training_tables' / 'val.csv',
        'table_test': gnn_root / 'training_tables' / 'test.csv',
    }


def _all_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def _write_table(frame: pd.DataFrame, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)
    return output_file


def _candidate_budget(frame: pd.DataFrame, expected_sample_indices: Iterable[int]) -> dict:
    expected = list(dict.fromkeys(int(value) for value in expected_sample_indices))
    if frame.empty:
        return {
            'test_manifest_samples': len(expected),
            'candidate_slates': 0,
            'missing_candidate_slates': len(expected),
            'candidate_rows': 0,
            'mean_rows_per_candidate_slate': 0.0,
            'max_rows_per_candidate_slate': 0,
            'mean_contexts_per_route': 0.0,
            'max_contexts_per_route': 0,
        }

    slate_sizes = frame.groupby('sample_index', sort=True).size()
    route_keys = [column for column in ['sample_index', 'retro_rank', 'reactants', 'product'] if column in frame.columns]
    route_sizes = frame.groupby(route_keys, sort=True).size() if route_keys else pd.Series(dtype=np.int64)
    observed = set(int(value) for value in slate_sizes.index.tolist())
    return {
        'test_manifest_samples': len(expected),
        'candidate_slates': int(len(observed)),
        'missing_candidate_slates': int(len(expected) - len(observed)),
        'candidate_rows': int(len(frame)),
        'mean_rows_per_candidate_slate': float(slate_sizes.mean()),
        'max_rows_per_candidate_slate': int(slate_sizes.max()),
        'mean_contexts_per_route': float(route_sizes.mean()) if not route_sizes.empty else 0.0,
        'max_contexts_per_route': int(route_sizes.max()) if not route_sizes.empty else 0,
    }


def _assert_feature_contract(feature_columns: list[str]) -> None:
    leaked = sorted(set(feature_columns) & TARGET_COLUMNS)
    if leaked:
        raise ValueError(f'Leaky target columns entered the XGBoost feature matrix: {leaked}')


def _frequency_context_rows(train_split_file: Path, max_contexts: int) -> list[dict]:
    """Build a deterministic Top-K context pool using only the train split."""

    aggregates: dict[tuple[str, str], dict[str, float | str]] = {}
    for row in load_split_rows(train_split_file):
        reagent = normalize_condition_labels(row['reagent_norm'])
        solvent = normalize_condition_labels(row['solvent_norm'])
        key = (reagent, solvent)
        stats = aggregates.setdefault(
            key,
            {
                'reagent_norm': reagent,
                'solvent_norm': solvent,
                'context_count': 0.0,
                'yield_sum': 0.0,
                'yield_count': 0.0,
            },
        )
        stats['context_count'] = float(stats['context_count']) + 1.0
        yield_value = safe_float(row['yield'])
        if np.isfinite(yield_value):
            stats['yield_sum'] = float(stats['yield_sum']) + float(yield_value)
            stats['yield_count'] = float(stats['yield_count']) + 1.0

    total = sum(float(row['context_count']) for row in aggregates.values()) or 1.0
    contexts: list[dict] = []
    for row in aggregates.values():
        count = float(row['context_count'])
        yield_count = float(row['yield_count'])
        contexts.append(
            {
                'reagent_norm': str(row['reagent_norm']),
                'solvent_norm': str(row['solvent_norm']),
                'frequency_context_count': count,
                'frequency_context_support': count / total,
                'frequency_mean_yield': (
                    float(row['yield_sum']) / yield_count if yield_count > 0.0 else 0.0
                ),
            }
        )
    contexts.sort(
        key=lambda row: (
            -float(row['frequency_context_count']),
            str(row['reagent_norm']),
            str(row['solvent_norm']),
        )
    )
    return contexts[:max_contexts]


def _frequency_candidate_table(records: list, contexts: list[dict], output_file: Path) -> Path:
    rows: list[dict] = []
    for record in records:
        base = base_candidate_row(record)
        for rank, context in enumerate(contexts, start=1):
            rows.append(
                {
                    **base,
                    'reagent_norm': context['reagent_norm'],
                    'solvent_norm': context['solvent_norm'],
                    'from_baseline_knn': 0,
                    'knn_similarity_sum': 0.0,
                    'knn_similarity_max': 0.0,
                    'knn_neighbor_count': 0.0,
                    'knn_weighted_mean_yield': 0.0,
                    'reafnn_reagent_score': 0.0,
                    'reafnn_solvent_score': 0.0,
                    'reafnn_token_score': 0.0,
                    'reafnn_prior_score': 0.0,
                    'reafnn_historical_bonus': 0.0,
                    'reafnn_novelty_penalty': 0.0,
                    'reafnn_context_score': 0.0,
                    'reafnn_context_count': 0.0,
                    'reafnn_context_support': 0.0,
                    'reafnn_mean_yield': 0.0,
                    'from_reafnn_generated': 0,
                    'from_reafnn_novel': 0,
                    'reafnn_is_historical': 0,
                    'cluster_id': -1,
                    'cluster_context_count': 0.0,
                    'cluster_context_support': 0.0,
                    'cluster_context_mean_yield': 0.0,
                    **context,
                    'frequency_rank': int(rank),
                }
            )
    return _write_table(stable_sort_candidate_frame(pd.DataFrame(rows)), output_file)


def _build_knn_only_tables(
    *,
    repo_root: Path,
    family: str,
    route_cache: Path,
    pool_root: Path,
    top_k: int,
    max_contexts: int,
    prefilter_contexts: int,
    fpsize: int,
    radius: int,
    parallel_workers: int,
    force: bool,
) -> dict[str, Path]:
    paths = _paths_for_pool(pool_root)
    if not force and _all_exist(paths.values()):
        return paths

    builder = KNNContextPoolBuilder(
        repo_root=repo_root,
        family=family,
        top_k=top_k,
        max_contexts=max_contexts,
        prefilter_contexts=prefilter_contexts,
        fpsize=fpsize,
        radius=radius,
        reaffn_artifact_dir=None,
        parallel_workers=parallel_workers,
    )
    builder.build_table('train', paths['candidate_train'])
    builder.build_table('val', paths['candidate_val'])
    builder.build_non_oracle_table(route_cache, paths['candidate_test'])
    label_candidate_table(paths['candidate_train'], split_file_for_family(repo_root, family, 'train'), paths['table_train'])
    label_candidate_table(paths['candidate_val'], split_file_for_family(repo_root, family, 'val'), paths['table_val'])
    label_candidate_table(paths['candidate_test'], split_file_for_family(repo_root, family, 'test'), paths['table_test'])
    return paths


def _build_frequency_tables(
    *,
    repo_root: Path,
    family: str,
    route_cache: Path,
    pool_root: Path,
    max_contexts: int,
    force: bool,
) -> dict[str, Path]:
    paths = _paths_for_pool(pool_root)
    if not force and _all_exist(paths.values()):
        return paths

    train_split = split_file_for_family(repo_root, family, 'train')
    contexts = _frequency_context_rows(train_split, max_contexts=max_contexts)
    if not contexts:
        raise ValueError(f'No training contexts available for frequency ablation: {family}')

    _frequency_candidate_table(load_route_records(train_split, family), contexts, paths['candidate_train'])
    _frequency_candidate_table(
        load_route_records(split_file_for_family(repo_root, family, 'val'), family),
        contexts,
        paths['candidate_val'],
    )
    _frequency_candidate_table(
        load_route_records_from_cache(route_cache, family),
        contexts,
        paths['candidate_test'],
    )
    label_candidate_table(paths['candidate_train'], train_split, paths['table_train'])
    label_candidate_table(paths['candidate_val'], split_file_for_family(repo_root, family, 'val'), paths['table_val'])
    label_candidate_table(paths['candidate_test'], split_file_for_family(repo_root, family, 'test'), paths['table_test'])
    return paths


def _ensure_gnn_features(
    *,
    table_paths: dict[str, Path],
    source_gnn_model: Path,
    gnn_root: Path,
    device: str,
    force: bool,
) -> dict[str, Path]:
    paths = _paths_for_gnn(gnn_root)
    if not force and _all_exist(paths.values()):
        return paths
    if not source_gnn_model.exists():
        raise FileNotFoundError(f'Missing frozen mainline Reaction-GNN model: {source_gnn_model}')

    for split in ('train', 'val', 'test'):
        augment_table_with_reaction_gnn_features(
            table_file=table_paths[f'table_{split}'],
            artifact_dir=source_gnn_model,
            output_file=paths[f'table_{split}'],
            device=device,
        )
    return paths


def _stage1_result(family: str, route_cache: Path, base_route_cache: Path) -> dict:
    current = _read_json(route_cache)
    base = _read_json(base_route_cache)
    current_ids = [
        (int(row['sample_index']), str(row['reaction_id']), str(row['product']))
        for row in current.get('reactions', [])
    ]
    base_ids = [
        (int(row['sample_index']), str(row['reaction_id']), str(row['product']))
        for row in base.get('reactions', [])
    ]
    if current_ids != base_ids:
        raise ValueError(f'Stage 1 base/tuned test manifests differ for {family}')

    tuned = stage1_route_recall(route_cache)
    base_metrics = stage1_route_recall(base_route_cache)
    return {
        'family': family,
        'test_products': int(tuned['n']),
        'base_cache': str(base_route_cache),
        'tuned_cache': str(route_cache),
        **{f'base_route_at_{k}': float(base_metrics[f'route_recall_top{k}']) for k in TOPKS},
        **{f'tuned_route_at_{k}': float(tuned[f'route_recall_top{k}']) for k in TOPKS},
        **{
            f'delta_route_at_{k}': float(tuned[f'route_recall_top{k}'] - base_metrics[f'route_recall_top{k}'])
            for k in TOPKS
        },
    }


def _result_from_scored(
    *,
    family: str,
    method: str,
    route_cache: Path,
    scored: pd.DataFrame,
    score_column: str,
    temperature_column: str | None,
    candidate_table: Path | None,
    artifacts: dict | None = None,
    source: str | None = None,
) -> dict:
    expected = load_route_cache_sample_indices(route_cache)
    metrics = evaluate_scored_frame_with_manifest(
        scored,
        expected_sample_indices=expected,
        score_column=score_column,
        temperature_column=temperature_column,
    )
    return {
        'family': family,
        'method': method,
        'method_label': METHOD_LABELS[method],
        'route_cache': str(route_cache),
        'candidate_table': str(candidate_table) if candidate_table is not None else None,
        'source': source,
        'artifacts': artifacts or {},
        'candidate_budget': _candidate_budget(scored, expected),
        'metrics': metrics,
        'stage1_route_recall': stage1_route_recall(route_cache),
    }


def _run_xgb_for_pool(
    *,
    family: str,
    method: str,
    route_cache: Path,
    table_paths: dict[str, Path],
    result_root: Path,
    force: bool,
    source: str,
) -> dict:
    result_dir = result_root / family / method / 'non_oracle'
    result_file = result_dir / 'result.json'
    if result_file.exists() and not force:
        return _read_json(result_file)

    artifacts = train_xgb_ranker_and_temperature(
        train_table_file=table_paths['table_train'],
        val_table_file=table_paths['table_val'],
        output_dir=result_dir / 'model',
    )
    _assert_feature_contract(list(artifacts['feature_columns']))
    scored = score_table_with_xgb(
        table_file=table_paths['table_test'],
        model_file=artifacts['model_file'],
        metadata_file=artifacts['metadata_file'],
        temperature_model_file=artifacts.get('temperature_model_file'),
        temperature_metadata_file=artifacts.get('temperature_metadata_file'),
    )
    scored_file = result_dir / 'test_scored.csv'
    _write_table(scored, scored_file)
    result = _result_from_scored(
        family=family,
        method=method,
        route_cache=route_cache,
        scored=scored,
        score_column='xgb_score',
        temperature_column='xgb_temperature_pred',
        candidate_table=table_paths['table_test'],
        artifacts=artifacts,
        source=source,
    )
    result['scored_test_file'] = str(scored_file)
    _write_json(result, result_file)
    return result


def _mainline_reference(
    *,
    family: str,
    route_cache: Path,
    mainline_root: Path,
    output_root: Path,
    force: bool,
) -> dict:
    source_root = mainline_root / family / '_shared_reaction_gnn' / 'training_tables'
    table_paths = {
        'table_train': source_root / 'train.csv',
        'table_val': source_root / 'val.csv',
        'table_test': source_root / 'test.csv',
    }
    if not _all_exist(table_paths.values()):
        raise FileNotFoundError(f'Missing saved full-mainline tables for {family}: {source_root}')
    return _run_xgb_for_pool(
        family=family,
        method='full_mainline',
        route_cache=route_cache,
        table_paths=table_paths,
        result_root=output_root,
        force=force,
        source='saved current ReaFNN + Reaction-GNN tables; deterministic XGBoost retraining for matched ablation protocol',
    )


def _stage2_heuristic_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank only with Stage 1 outputs and Stage 2 evidence, never labels."""

    specs = [
        ('sample_index', True),
        ('retro_rank', True),
        ('retro_probability', False),
        ('reafnn_context_score', False),
        ('reafnn_is_historical', False),
        ('from_reafnn_novel', True),
        ('knn_similarity_sum', False),
        ('knn_similarity_max', False),
        ('knn_neighbor_count', False),
        ('reafnn_context_support', False),
        ('reagent_norm', True),
        ('solvent_norm', True),
    ]
    sort_columns = [column for column, _ in specs if column in frame.columns]
    ascending = [ascending for column, ascending in specs if column in frame.columns]
    ranked = frame.sort_values(sort_columns, ascending=ascending, kind='mergesort').reset_index(drop=True)
    ranked['stage2_prior_rank'] = ranked.groupby('sample_index', sort=False).cumcount() + 1
    ranked['stage2_prior_score'] = -ranked['stage2_prior_rank'].astype(np.float32)
    return ranked


def _run_no_stage3(
    *,
    family: str,
    route_cache: Path,
    mainline_root: Path,
    output_root: Path,
    force: bool,
) -> dict:
    result_dir = output_root / family / 'no_stage3' / 'non_oracle'
    result_file = result_dir / 'result.json'
    if result_file.exists() and not force:
        return _read_json(result_file)

    source_table = mainline_root / family / '_shared_knn' / 'training_tables' / 'test.csv'
    if not source_table.exists():
        raise FileNotFoundError(f'Missing Stage 2 table for {family}: {source_table}')
    ranked = _stage2_heuristic_score(pd.read_csv(source_table))
    scored_file = result_dir / 'test_scored.csv'
    _write_table(ranked, scored_file)
    result = _result_from_scored(
        family=family,
        method='no_stage3',
        route_cache=route_cache,
        scored=ranked,
        score_column='stage2_prior_score',
        temperature_column=None,
        candidate_table=source_table,
        source='deterministic Stage 1 + Stage 2 prior; no learned Stage 3 model',
    )
    result['scored_test_file'] = str(scored_file)
    _write_json(result, result_file)
    return result


def _run_no_gnn_xgb(
    *,
    family: str,
    route_cache: Path,
    mainline_root: Path,
    output_root: Path,
    force: bool,
) -> dict:
    source_root = mainline_root / family / '_shared_knn' / 'training_tables'
    table_paths = {
        'table_train': source_root / 'train.csv',
        'table_val': source_root / 'val.csv',
        'table_test': source_root / 'test.csv',
    }
    if not _all_exist(table_paths.values()):
        raise FileNotFoundError(f'Missing no-GNN source tables for {family}: {source_root}')
    return _run_xgb_for_pool(
        family=family,
        method='no_gnn_xgb',
        route_cache=route_cache,
        table_paths=table_paths,
        result_root=output_root,
        force=force,
        source='current full Stage 2 tables, with all route_gnn_feat_* columns absent',
    )


def _run_knn_only_no_gnn_xgb(
    *,
    family: str,
    route_cache: Path,
    output_root: Path,
    force: bool,
) -> dict:
    """Isolate the combined removal of ReaFNN and Reaction-GNN features."""

    source_root = output_root / family / '_shared_knn_only' / 'training_tables'
    table_paths = {
        'table_train': source_root / 'train.csv',
        'table_val': source_root / 'val.csv',
        'table_test': source_root / 'test.csv',
    }
    if not _all_exist(table_paths.values()):
        raise FileNotFoundError(
            f'Missing KNN-only source tables for {family}: {source_root}. '
            'Run the Stage 2 ablation before this interaction control.'
        )
    return _run_xgb_for_pool(
        family=family,
        method='knn_only_no_gnn_xgb',
        route_cache=route_cache,
        table_paths=table_paths,
        result_root=output_root,
        force=force,
        source='KNN-only candidate pool with all ReaFNN and Reaction-GNN features absent',
    )


def _metric_row(result: dict) -> dict:
    metrics = result['metrics']
    temperature = metrics.get('temperature', {})
    budget = result.get('candidate_budget', {})
    row = {
        'family': result['family'],
        'display_family': display_family_name(result['family']),
        'method': result['method'],
        'method_label': result['method_label'],
        'test_n': metrics.get('num_slates'),
        'candidate_slates': metrics.get('candidate_slates'),
        'missing_candidate_slates': metrics.get('missing_candidate_slates'),
        'candidate_rows': budget.get('candidate_rows'),
        'mean_rows_per_candidate_slate': budget.get('mean_rows_per_candidate_slate'),
        'mean_contexts_per_route': budget.get('mean_contexts_per_route'),
        'max_contexts_per_route': budget.get('max_contexts_per_route'),
        'pool_coverage': metrics.get('pool_coverage'),
        'pool_route_coverage': metrics.get('pool_route_coverage'),
        'pool_context_coverage': metrics.get('pool_context_coverage'),
        'sys1': metrics.get('system_top1_all'),
        'sys3': metrics.get('system_top3_all'),
        'sys5': metrics.get('system_top5_all'),
        'sys10': metrics.get('system_top10_all'),
        'mrr': metrics.get('system_mrr'),
        'ndcg10': metrics.get('system_ndcg10'),
        'temp_n': temperature.get('n'),
        'temp_mae': temperature.get('mae'),
        'temp_mse': temperature.get('mse'),
        'temp_rmse': temperature.get('rmse'),
        'temp_within_5c': temperature.get('within_5c'),
        'temp_within_10c': temperature.get('within_10c'),
        'temp_within_20c': temperature.get('within_20c'),
        'denominator': metrics.get('denominator'),
    }
    return row


def _macro_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    metric_columns = [
        'pool_coverage', 'pool_route_coverage', 'pool_context_coverage',
        'sys1', 'sys3', 'sys5', 'sys10', 'mrr', 'ndcg10',
        'mean_rows_per_candidate_slate', 'mean_contexts_per_route',
        'temp_mae', 'temp_mse', 'temp_rmse',
        'temp_within_5c', 'temp_within_10c', 'temp_within_20c',
    ]
    rows = []
    for method, bucket in frame.groupby('method', sort=False):
        macro = {
            'family': 'MACRO-AVG',
            'display_family': 'MACRO-AVG',
            'method': method,
            'method_label': METHOD_LABELS[method],
            'test_n': int(bucket['test_n'].sum()),
            'candidate_slates': int(bucket['candidate_slates'].sum()),
            'missing_candidate_slates': int(bucket['missing_candidate_slates'].sum()),
            'candidate_rows': int(bucket['candidate_rows'].sum()),
            'max_contexts_per_route': int(bucket['max_contexts_per_route'].max()),
            'temp_n': int(bucket['temp_n'].fillna(0).sum()),
            'denominator': 'all_test_manifest_samples',
        }
        for column in metric_columns:
            numeric = pd.to_numeric(bucket[column], errors='coerce').dropna()
            macro[column] = float(numeric.mean()) if not numeric.empty else np.nan
        rows.append(macro)
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _pct(value) -> str:
    return 'N/A' if pd.isna(value) else f'{float(value) * 100.0:.2f}%'


def _number(value, digits: int = 2) -> str:
    return 'N/A' if pd.isna(value) else f'{float(value):.{digits}f}'


def _render_stage1(rows: list[dict]) -> str:
    lines = [
        '# Stage 1 Ablation: Base vs Family-Tuned Route Recall',
        '',
        'Both cache sets were checked to contain the identical ordered test identity manifest for every family.',
        '',
        '| Family | Test n | Base@1 | Tuned@1 | Base@3 | Tuned@3 | Base@5 | Tuned@5 | Base@10 | Tuned@10 | Delta@10 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    ordered = sorted(rows, key=lambda row: _family_sort_key(row['family']))
    for row in ordered:
        lines.append(
            '| ' + ' | '.join([
                display_family_name(row['family']), str(row['test_products']),
                _pct(row['base_route_at_1']), _pct(row['tuned_route_at_1']),
                _pct(row['base_route_at_3']), _pct(row['tuned_route_at_3']),
                _pct(row['base_route_at_5']), _pct(row['tuned_route_at_5']),
                _pct(row['base_route_at_10']), _pct(row['tuned_route_at_10']),
                _pct(row['delta_route_at_10']),
            ]) + ' |'
        )
    if ordered:
        lines.append(
            '| MACRO-AVG | ' + ' | '.join([
                _number(np.mean([row['test_products'] for row in ordered]), 1),
                _pct(np.mean([row['base_route_at_1'] for row in ordered])),
                _pct(np.mean([row['tuned_route_at_1'] for row in ordered])),
                _pct(np.mean([row['base_route_at_3'] for row in ordered])),
                _pct(np.mean([row['tuned_route_at_3'] for row in ordered])),
                _pct(np.mean([row['base_route_at_5'] for row in ordered])),
                _pct(np.mean([row['tuned_route_at_5'] for row in ordered])),
                _pct(np.mean([row['base_route_at_10'] for row in ordered])),
                _pct(np.mean([row['tuned_route_at_10'] for row in ordered])),
                _pct(np.mean([row['delta_route_at_10'] for row in ordered])),
            ]) + ' |'
        )
    return '\n'.join(lines) + '\n'


def _render_method_table(frame: pd.DataFrame, title: str, introduction: str) -> str:
    lines = [f'# {title}', '', introduction, '']
    lines.extend([
        '| Family | Method | Cover | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 | Temp MAE | Temp+/-5C | Temp+/-10C | Temp+/-20C | Temp n | Candidate slates |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    family_order = {family: idx for idx, family in enumerate(FAMILY_ORDER)}
    work = frame.copy()
    work['_order'] = work['family'].map(lambda value: family_order.get(value, len(FAMILY_ORDER)))
    work = work.sort_values(['_order', 'method'], kind='mergesort')
    for row in work.itertuples(index=False):
        lines.append(
            '| ' + ' | '.join([
                row.display_family,
                row.method_label,
                _pct(row.pool_coverage), _pct(row.sys1), _pct(row.sys3), _pct(row.sys5), _pct(row.sys10),
                _pct(row.mrr), _pct(row.ndcg10), _number(row.temp_mae),
                _pct(row.temp_within_5c), _pct(row.temp_within_10c), _pct(row.temp_within_20c),
                _number(row.temp_n, 0),
                f'{int(row.candidate_slates)}/{int(row.test_n)}',
            ]) + ' |'
        )
    return '\n'.join(lines) + '\n'


def _write_reports(
    *,
    output_root: Path,
    stage1_rows: list[dict],
    stage2_results: list[dict],
    stage3_results: list[dict],
    interaction_results: list[dict],
) -> None:
    if stage1_rows:
        stage1_frame = pd.DataFrame(stage1_rows)
        _write_table(stage1_frame, output_root / 'stage1_route_ablation.csv')
        (output_root / 'stage1_route_ablation.md').write_text(_render_stage1(stage1_rows), encoding='utf-8')

    stage2_frame = _macro_rows(pd.DataFrame([_metric_row(result) for result in stage2_results])) if stage2_results else pd.DataFrame()
    stage3_frame = _macro_rows(pd.DataFrame([_metric_row(result) for result in stage3_results])) if stage3_results else pd.DataFrame()
    interaction_frame = _macro_rows(pd.DataFrame([_metric_row(result) for result in interaction_results])) if interaction_results else pd.DataFrame()
    if not stage2_frame.empty:
        _write_table(stage2_frame, output_root / 'stage2_pool_ablation.csv')
        (output_root / 'stage2_pool_ablation.md').write_text(
            _render_method_table(
                stage2_frame,
                'Stage 2 Ablation: Candidate Screening',
                'Only the candidate-pool construction changes. Each pool receives its own XGBoost model trained on the corresponding train/validation tables; the frozen family Reaction-GNN encoder is identical across rows.',
            ),
            encoding='utf-8',
        )
    if not stage3_frame.empty:
        _write_table(stage3_frame, output_root / 'stage3_reranking_ablation.csv')
        (output_root / 'stage3_reranking_ablation.md').write_text(
            _render_method_table(
                stage3_frame,
                'Stage 3 Ablation: Reaction-GNN and XGBoost',
                'All rows use the saved full KNN+ReaFNN Stage 2 candidate tables. The no-Stage-3 row has no temperature regressor, so its temperature cells are intentionally N/A.',
            ),
            encoding='utf-8',
        )
    if not interaction_frame.empty:
        _write_table(interaction_frame, output_root / 'stage23_interaction_ablation.csv')
        (output_root / 'stage23_interaction_ablation.md').write_text(
            _render_method_table(
                interaction_frame,
                'Stage 2/3 Interaction Control',
                'This factorial control separates the ReaFNN candidate-pool contribution from the Reaction-GNN feature contribution. All rows share the same Stage 1 routes, full-manifest denominator, candidate cap, and train/validation-only model fitting.',
            ),
            encoding='utf-8',
        )

    overview = [
        '# Current-Mainline Ablation Results',
        '',
        'This result root supersedes the legacy KNN-XGBoost ablation outputs. Every Stage 2/3 metric uses all reaction identities in the Stage 1 test cache as the denominator; samples without a Stage 1 route are retained as zero-hit slates.',
        '',
        'Generated tables:',
        '',
        '- `stage1_route_ablation.md`: base model versus family-tuned route recall.',
        '- `stage2_pool_ablation.md`: full KNN+ReaFNN pool, KNN-only pool, and Top-20 training-frequency pool.',
        '- `stage3_reranking_ablation.md`: full model, no Reaction-GNN feature, and no XGBoost reranker.',
        '- `stage23_interaction_ablation.md`: a 2x2 control across ReaFNN pooling and Reaction-GNN features.',
        '',
        'Training KNN candidate pools use leave-one-reaction-out retrieval and query-adjusted global support statistics. Validation and test routes are outside the train memory.',
        '',
        'Temperature is evaluated only for the highest-ranked exact route-and-condition match with a valid temperature label. It is therefore conditional and is not divided by missing Stage 1 slates.',
        '',
    ]
    (output_root / 'overview.md').write_text('\n'.join(overview), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run current-mainline ProSys ablations.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--run_set', choices=['all', 'stage1', 'stage2', 'stage3', 'interaction'], default='all')
    parser.add_argument('--output_root', type=str, default='outputs/ablation_reafnn_gnn_20260726')
    parser.add_argument('--mainline_root', type=str, default='outputs/stage23_mainline_reafnn_gnn_fused_20260723')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--base_route_root', type=str, default='outputs/stage1_base_vs_tuned/base_route_caches')
    parser.add_argument('--knn_top_k', type=int, default=64)
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--prefilter_contexts', type=int, default=64)
    parser.add_argument('--fpsize', type=int, default=4096)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--knn_workers', type=int, default=8)
    parser.add_argument('--gnn_device', type=str, default='cuda')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    mainline_root = (repo_root / args.mainline_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    base_route_root = (repo_root / args.base_route_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    families = parse_families_arg(args.families)

    _write_json(
        {
            'families': families,
            'run_set': args.run_set,
            'route_root': str(route_root),
            'base_route_root': str(base_route_root),
            'mainline_root': str(mainline_root),
            'knn_top_k': args.knn_top_k,
            'max_contexts': args.max_contexts,
            'prefilter_contexts': args.prefilter_contexts,
            'fpsize': args.fpsize,
            'radius': args.radius,
            'knn_workers': args.knn_workers,
            'gnn_device': args.gnn_device,
            'test_protocol': 'full Stage 1 route-cache identity manifest; no Stage 1 route means zero-hit slate',
            'selection_protocol': 'all learned model fitting and score-fusion selection use train/validation only',
            'knn_train_protocol': 'leave-one-reaction-out retrieval with query-adjusted context count, support, and mean-yield statistics',
        },
        output_root / 'run_config.json',
    )

    stage1_rows: list[dict] = []
    stage2_results: list[dict] = []
    stage3_results: list[dict] = []
    interaction_results: list[dict] = []

    for family in families:
        route_cache = route_root / family / 'route_cache.json'
        base_route_cache = base_route_root / family / 'route_cache.json'
        if not route_cache.exists():
            raise FileNotFoundError(f'Missing current Stage 1 route cache: {route_cache}')

        print(f'[ablation] {family}', flush=True)
        if args.run_set in {'all', 'stage1'}:
            if not base_route_cache.exists():
                raise FileNotFoundError(f'Missing base Stage 1 route cache: {base_route_cache}')
            stage1_rows.append(_stage1_result(family, route_cache, base_route_cache))

        need_full = args.run_set in {'all', 'stage2', 'stage3', 'interaction'}
        full = None
        if need_full:
            full = _mainline_reference(
                family=family,
                route_cache=route_cache,
                mainline_root=mainline_root,
                output_root=output_root,
                force=args.force,
            )

        if args.run_set in {'all', 'stage2'}:
            source_gnn_model = mainline_root / family / '_shared_reaction_gnn' / 'model'
            knn_pool = _build_knn_only_tables(
                repo_root=repo_root,
                family=family,
                route_cache=route_cache,
                pool_root=output_root / family / '_shared_knn_only',
                top_k=args.knn_top_k,
                max_contexts=args.max_contexts,
                prefilter_contexts=args.prefilter_contexts,
                fpsize=args.fpsize,
                radius=args.radius,
                parallel_workers=args.knn_workers,
                force=args.force,
            )
            knn_gnn = _ensure_gnn_features(
                table_paths=knn_pool,
                source_gnn_model=source_gnn_model,
                gnn_root=output_root / family / '_shared_knn_only_gnn',
                device=args.gnn_device,
                force=args.force,
            )
            knn_only = _run_xgb_for_pool(
                family=family,
                method='knn_only_xgb',
                route_cache=route_cache,
                table_paths=knn_gnn,
                result_root=output_root,
                force=args.force,
                source='KNN candidate pool without ReaFNN; frozen current-mainline Reaction-GNN encoder',
            )

            frequency_pool = _build_frequency_tables(
                repo_root=repo_root,
                family=family,
                route_cache=route_cache,
                pool_root=output_root / family / '_shared_frequency_top20',
                max_contexts=args.max_contexts,
                force=args.force,
            )
            frequency_gnn = _ensure_gnn_features(
                table_paths=frequency_pool,
                source_gnn_model=source_gnn_model,
                gnn_root=output_root / family / '_shared_frequency_top20_gnn',
                device=args.gnn_device,
                force=args.force,
            )
            frequency = _run_xgb_for_pool(
                family=family,
                method='frequency_top20_xgb',
                route_cache=route_cache,
                table_paths=frequency_gnn,
                result_root=output_root,
                force=args.force,
                source='Top-20 global context frequency from this family train split; frozen current-mainline Reaction-GNN encoder',
            )
            stage2_results.extend([full, knn_only, frequency])

        if args.run_set in {'all', 'stage3'}:
            no_gnn = _run_no_gnn_xgb(
                family=family,
                route_cache=route_cache,
                mainline_root=mainline_root,
                output_root=output_root,
                force=args.force,
            )
            no_stage3 = _run_no_stage3(
                family=family,
                route_cache=route_cache,
                mainline_root=mainline_root,
                output_root=output_root,
                force=args.force,
            )
            stage3_results.extend([full, no_gnn, no_stage3])

        if args.run_set == 'all':
            interaction = _run_knn_only_no_gnn_xgb(
                family=family,
                route_cache=route_cache,
                output_root=output_root,
                force=args.force,
            )
            interaction_results.extend([full, knn_only, no_gnn, interaction])
        elif args.run_set == 'interaction':
            knn_only_result = output_root / family / 'knn_only_xgb' / 'non_oracle' / 'result.json'
            if not knn_only_result.exists():
                raise FileNotFoundError(
                    f'Missing KNN-only comparison for {family}: {knn_only_result}. '
                    'Run --run_set stage2 or --run_set all first.'
                )
            no_gnn = _run_no_gnn_xgb(
                family=family,
                route_cache=route_cache,
                mainline_root=mainline_root,
                output_root=output_root,
                force=args.force,
            )
            interaction = _run_knn_only_no_gnn_xgb(
                family=family,
                route_cache=route_cache,
                output_root=output_root,
                force=args.force,
            )
            interaction_results.extend([full, _read_json(knn_only_result), no_gnn, interaction])

    _write_json(stage1_rows, output_root / 'stage1_results.json')
    _write_json(stage2_results, output_root / 'stage2_results.json')
    _write_json(stage3_results, output_root / 'stage3_results.json')
    _write_json(interaction_results, output_root / 'interaction_results.json')
    _write_reports(
        output_root=output_root,
        stage1_rows=stage1_rows,
        stage2_results=stage2_results,
        stage3_results=stage3_results,
        interaction_results=interaction_results,
    )
    print(f'[ablation] results written to {output_root}', flush=True)


if __name__ == '__main__':
    main()
