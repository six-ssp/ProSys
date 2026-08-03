"""Shared utilities for the maintained ProSys mainline pipeline.

This module intentionally contains only the Stage 1/2/3 helpers that are still
part of the current KNN + XGBoost workflow. It replaces the old dependency on
historical baseline wrappers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import RDLogger

from prosys_shared.constants import (
    CONTEXT_DENSE_COLUMNS_V2,
    PRODUCT_DESCRIPTOR_COLUMNS_V2,
    ROUTE_DENSE_COLUMNS_V2,
    ROUTE_GNN_FEATURE_COLUMNS_V2,
    SUPPORT_FEATURE_COLUMNS_V2,
)
from prosys_shared.features import (
    canonicalize_reaction_side,
    canonicalize_smiles,
    count_condition_tokens,
    molecule_graph_descriptors,
)
from prosys_shared.product_memory import normalize_condition_labels, safe_float
from prosys_shared.route_cache import RouteRecord

RDLogger.DisableLog('rdApp.error')

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

TEXT_COLUMNS = {
    'family',
    'sample_index',
    'reaction_id',
    'product',
    'reactants',
    'reagent_norm',
    'solvent_norm',
    'product_canonical',
    'route_canonical',
    'label_type',
}

TARGET_COLUMNS = {
    'label',
    'route_match',
    'context_match',
    'rank_relevance',
    'sample_weight',
    'temperature_gold',
    'yield_gold',
}

STANDARD_FEATURE_COLUMNS = (
    ROUTE_DENSE_COLUMNS_V2
    + CONTEXT_DENSE_COLUMNS_V2
    + PRODUCT_DESCRIPTOR_COLUMNS_V2
    + ROUTE_GNN_FEATURE_COLUMNS_V2
    + SUPPORT_FEATURE_COLUMNS_V2
)

CANDIDATE_SORT_SPECS = [
    ('sample_index', True),
    ('reaction_id', True),
    ('retro_rank', True),
    ('retro_score', False),
    ('retro_probability', False),
    ('product', True),
    ('reactants', True),
    ('reagent_norm', True),
    ('solvent_norm', True),
]


@dataclass
class GoldConditionIndex:
    route_keys: set[str] = field(default_factory=set)
    context_keys: set[tuple[str, str]] = field(default_factory=set)
    exact_keys: set[tuple[str, str, str]] = field(default_factory=set)
    temperatures: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)
    yields: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)

    def add(
        self,
        route_key: str,
        reagent_norm: str,
        solvent_norm: str,
        temperature: float,
        yield_value: float,
    ) -> None:
        exact_key = (route_key, reagent_norm, solvent_norm)
        self.route_keys.add(route_key)
        self.context_keys.add((reagent_norm, solvent_norm))
        self.exact_keys.add(exact_key)
        self.temperatures.setdefault(exact_key, []).append(temperature)
        self.yields.setdefault(exact_key, []).append(yield_value)


def parse_families_arg(families: str) -> list[str]:
    text = families.strip()
    if text.lower() == 'all':
        return list(FAMILY_ORDER)
    requested = [item.strip() for item in text.replace(',', ' ').split() if item.strip()]
    unknown = [item for item in requested if item not in FAMILY_ORDER]
    if unknown:
        raise ValueError(f'Unknown families: {unknown}')
    return requested


def display_family_name(family: str) -> str:
    return FAMILY_DISPLAY_NAMES.get(family, family)


def family_dir(repo_root: str | Path, family: str) -> Path:
    return Path(repo_root) / 'data' / f'reaction_processed_{family}_catmerge'


def split_file_for_family(repo_root: str | Path, family: str, split: str) -> Path:
    mapping = {
        'train': 'Splitted_second_train_labels_processed.txt',
        'val': 'Splitted_second_validate_labels_processed.txt',
        'test': 'Splitted_second_test_labels_processed.txt',
    }
    if split not in mapping:
        raise ValueError(f'Unsupported split: {split}')
    return family_dir(repo_root, family) / 'For_second_part_model' / mapping[split]


def load_split_rows(split_file: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 7:
                continue
            reaction_id, reactants, product, yield_value, reagent, solvent, temperature = parts[:7]
            rows.append(
                {
                    'reaction_id': str(reaction_id),
                    'reactants': str(reactants),
                    'product': str(product),
                    'yield': str(yield_value),
                    'reagent_norm': str(reagent),
                    'solvent_norm': str(solvent),
                    'temperature': str(temperature),
                }
            )
    return rows


def load_route_records(split_file: str | Path, family: str) -> list[RouteRecord]:
    rows = load_split_rows(split_file)
    records: list[RouteRecord] = []
    seen = set()
    for row in rows:
        dedup_key = (row['reaction_id'], row['reactants'], row['product'])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        records.append(
            RouteRecord(
                family=family,
                sample_index=len(records),
                reaction_id=row['reaction_id'],
                reactants=row['reactants'],
                product=row['product'],
            )
        )
    return records


def zero_support_fields() -> dict[str, float]:
    return {
        'from_fnn': 0,
        'from_product_exact': 0,
        'from_product_scaffold': 0,
        'from_product_knn': 0,
        'product_exact_pair_support': 0.0,
        'product_exact_reagent_support': 0.0,
        'product_exact_solvent_support': 0.0,
        'product_scaffold_pair_support': 0.0,
        'product_scaffold_reagent_support': 0.0,
        'product_scaffold_solvent_support': 0.0,
        'product_knn_pair_support': 0.0,
        'product_knn_reagent_support': 0.0,
        'product_knn_solvent_support': 0.0,
        'product_pair_freq': 0.0,
        'product_pair_mean_yield': 0.0,
    }


def base_candidate_row(record: RouteRecord) -> dict:
    return {
        'family': record.family,
        'sample_index': record.sample_index,
        'reaction_id': record.reaction_id,
        'product': record.product,
        'reactants': record.reactants,
        'retro_rank': record.retro_rank,
        'retro_score': record.retro_score,
        'retro_probability': record.retro_probability,
        **zero_support_fields(),
    }


def stable_sort_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, is_ascending in CANDIDATE_SORT_SPECS:
        if column in frame.columns:
            sort_columns.append(column)
            ascending.append(is_ascending)

    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, ascending=ascending, kind='mergesort').reset_index(drop=True)


def _ensure_text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([''] * len(frame), index=frame.index, dtype='object')
    return frame[column].fillna('').astype(str)


def _map_unique_strings(values: pd.Series, func) -> pd.Series:
    unique_values = pd.unique(values.to_numpy(dtype=object))
    cache = {value: func(value) for value in unique_values}
    return values.map(cache)


def _canonical_from_existing_or_source(
    frame: pd.DataFrame,
    *,
    source_column: str,
    existing_column: str,
    canonicalizer,
) -> pd.Series:
    source_values = _ensure_text_series(frame, source_column)
    if existing_column not in frame.columns:
        return _map_unique_strings(source_values, canonicalizer)

    existing = _ensure_text_series(frame, existing_column)
    missing = existing.eq('')
    if not missing.any():
        return existing

    filled = existing.copy()
    filled.loc[missing] = _map_unique_strings(source_values.loc[missing], canonicalizer)
    return filled


def _label_type(route_match: int, context_match: int, label: int) -> str:
    if label:
        return 'positive'
    if route_match:
        return 'route_only'
    if context_match:
        return 'context_only'
    return 'negative'


def _sample_weight(label_type: str) -> float:
    if label_type == 'positive':
        return 1.0
    if label_type == 'route_only':
        return 1.0
    if label_type == 'context_only':
        return 0.75
    return 0.25


def _rank_relevance(label_type: str) -> float:
    if label_type == 'positive':
        return 3.0
    if label_type == 'route_only':
        return 2.0
    if label_type == 'context_only':
        return 1.0
    return 0.0


def _mean_or_nan(values: list[float]) -> float:
    numeric = [value for value in values if not np.isnan(value)]
    if not numeric:
        return float('nan')
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def load_gold_condition_index(split_file: str | Path) -> dict[tuple[str, str], GoldConditionIndex]:
    index: dict[tuple[str, str], GoldConditionIndex] = {}
    product_cache: dict[str, str] = {}
    route_cache: dict[str, str] = {}
    reagent_cache: dict[str, str] = {}
    solvent_cache: dict[str, str] = {}

    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 7:
                continue

            reaction_id, reactants, product, yield_value, reagent, solvent, temperature = parts[:7]
            product_key = product_cache.setdefault(product, canonicalize_smiles(product))
            if not product_key:
                continue

            route_key = route_cache.setdefault(reactants, canonicalize_reaction_side(reactants))
            reagent_norm = reagent_cache.setdefault(reagent, normalize_condition_labels(reagent))
            solvent_norm = solvent_cache.setdefault(solvent, normalize_condition_labels(solvent))

            bucket = index.setdefault((str(reaction_id), product_key), GoldConditionIndex())
            bucket.add(
                route_key=route_key,
                reagent_norm=reagent_norm,
                solvent_norm=solvent_norm,
                temperature=safe_float(temperature),
                yield_value=safe_float(yield_value),
            )

    return index


def build_candidate_training_table(
    candidate_pool_file: str | Path,
    gold_split_file: str | Path,
) -> pd.DataFrame:
    frame = pd.read_csv(candidate_pool_file)
    if frame.empty:
        return frame

    for column in SUPPORT_FEATURE_COLUMNS_V2:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0).astype(np.float32)

    frame['reaction_id'] = _ensure_text_series(frame, 'reaction_id')
    frame['reagent_norm'] = _map_unique_strings(_ensure_text_series(frame, 'reagent_norm'), normalize_condition_labels)
    frame['solvent_norm'] = _map_unique_strings(_ensure_text_series(frame, 'solvent_norm'), normalize_condition_labels)
    frame['product_canonical'] = _canonical_from_existing_or_source(
        frame,
        source_column='product',
        existing_column='product_canonical',
        canonicalizer=canonicalize_smiles,
    )
    frame['route_canonical'] = _canonical_from_existing_or_source(
        frame,
        source_column='reactants',
        existing_column='route_canonical',
        canonicalizer=canonicalize_reaction_side,
    )

    gold_index = load_gold_condition_index(gold_split_file)
    product_descriptor_cache = {
        product_key: molecule_graph_descriptors(product_key)
        for product_key in pd.unique(frame['product_canonical'].to_numpy(dtype=object))
    }

    route_match_values = []
    context_match_values = []
    label_values = []
    label_types = []
    sample_weights = []
    rank_relevance = []
    temperature_gold = []
    yield_gold = []

    reaction_ids = frame['reaction_id'].to_numpy(dtype=object)
    product_keys = frame['product_canonical'].to_numpy(dtype=object)
    route_keys = frame['route_canonical'].to_numpy(dtype=object)
    reagent_norms = frame['reagent_norm'].to_numpy(dtype=object)
    solvent_norms = frame['solvent_norm'].to_numpy(dtype=object)

    for reaction_id, product_key, route_key, reagent_norm, solvent_norm in zip(
        reaction_ids,
        product_keys,
        route_keys,
        reagent_norms,
        solvent_norms,
    ):
        bucket = gold_index.get((str(reaction_id), str(product_key)))
        route_key = str(route_key)
        reagent_norm = str(reagent_norm)
        solvent_norm = str(solvent_norm)

        route_match = int(bucket is not None and route_key in bucket.route_keys)
        context_key = (reagent_norm, solvent_norm)
        context_match = int(bucket is not None and context_key in bucket.context_keys)
        exact_key = (route_key, reagent_norm, solvent_norm)
        label = int(bucket is not None and exact_key in bucket.exact_keys)
        label_type = _label_type(route_match, context_match, label)

        route_match_values.append(route_match)
        context_match_values.append(context_match)
        label_values.append(label)
        label_types.append(label_type)
        sample_weights.append(_sample_weight(label_type))
        rank_relevance.append(_rank_relevance(label_type))
        temperature_gold.append(_mean_or_nan(bucket.temperatures.get(exact_key, [])) if label else float('nan'))
        yield_gold.append(_mean_or_nan(bucket.yields.get(exact_key, [])) if label else 0.0)

    route_keys_series = frame['route_canonical']
    frame['num_reagents'] = _map_unique_strings(frame['reagent_norm'], count_condition_tokens).to_numpy(dtype=np.int32)
    frame['num_solvents'] = _map_unique_strings(frame['solvent_norm'], count_condition_tokens).to_numpy(dtype=np.int32)
    frame['route_component_count'] = route_keys_series.map(lambda value: len(value.split('.')) if value else 0).to_numpy(dtype=np.int32)
    frame['reactants_length'] = _ensure_text_series(frame, 'reactants').str.len().to_numpy(dtype=np.int32)
    frame['route_match'] = np.asarray(route_match_values, dtype=np.int32)
    frame['context_match'] = np.asarray(context_match_values, dtype=np.int32)
    frame['label'] = np.asarray(label_values, dtype=np.float32)
    frame['label_type'] = label_types
    frame['sample_weight'] = np.asarray(sample_weights, dtype=np.float32)
    frame['rank_relevance'] = np.asarray(rank_relevance, dtype=np.float32)
    frame['temperature_gold'] = np.asarray(temperature_gold, dtype=np.float32)
    frame['yield_gold'] = np.asarray(yield_gold, dtype=np.float32)

    product_feature_matrix = np.vstack(
        [product_descriptor_cache[str(product_key)] for product_key in frame['product_canonical'].to_numpy(dtype=object)]
    ).astype(np.float32)
    for idx, column in enumerate(PRODUCT_DESCRIPTOR_COLUMNS_V2):
        frame[column] = product_feature_matrix[:, idx]

    return stable_sort_candidate_frame(frame)


def label_candidate_table(candidate_pool_file: str | Path, gold_split_file: str | Path, output_file: str | Path) -> Path:
    frame = build_candidate_training_table(
        candidate_pool_file=candidate_pool_file,
        gold_split_file=gold_split_file,
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def stage1_route_recall(route_cache_file: str | Path, topks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    cache = json.loads(Path(route_cache_file).read_text(encoding='utf-8'))
    hits = {k: 0 for k in topks}
    n = 0
    for reaction in cache.get('reactions', []):
        gold_key = canonicalize_reaction_side(reaction.get('gold_reactants', ''))
        if not gold_key:
            continue
        n += 1
        routes = sorted(reaction.get('routes', []), key=lambda row: row.get('retro_rank', 1))
        pred_keys = [canonicalize_reaction_side(row['reactants']) for row in routes]
        for k in topks:
            if gold_key in pred_keys[:k]:
                hits[k] += 1
    return {
        'n': int(n),
        **{f'route_recall_top{k}': (hits[k] / n if n else 0.0) for k in topks},
    }


def _dcg_binary(labels: np.ndarray, k: int) -> float:
    if labels.size == 0 or k <= 0:
        return 0.0
    gains = labels[:k]
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains / discounts))


def _ndcg_binary(labels: np.ndarray, k: int) -> float:
    dcg = _dcg_binary(labels, k)
    ideal = _dcg_binary(np.sort(labels)[::-1], k)
    if ideal <= 0.0:
        return 0.0
    return dcg / ideal


def _safe_mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def evaluate_scored_frame(
    frame: pd.DataFrame,
    *,
    score_column: str,
    temperature_column: str | None = None,
    topks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    if frame.empty:
        metrics: dict[str, object] = {
            'num_slates': 0,
            'covered_slates': 0,
            'route_covered_slates': 0,
            'context_covered_slates': 0,
            'pool_route_coverage': 0.0,
            'pool_context_coverage': 0.0,
            'pool_coverage': 0.0,
            'system_mrr': 0.0,
            'system_ndcg10': 0.0,
            'temperature': {
                'definition': 'standalone_temperature_error_on_highest_ranked_full_match',
                'n': 0,
                'mae': None,
                'mse': None,
                'rmse': None,
                'support': 'highest_ranked_full_match_with_valid_temperature',
                'within_5c': None,
                'within_10c': None,
                'within_20c': None,
            },
        }
        for k in topks:
            metrics[f'system_top{k}_all'] = 0.0
            metrics[f'context_top{k}_all'] = 0.0
            metrics[f'route_top{k}_all'] = 0.0
            metrics[f'system_top{k}_covered'] = 0.0
        return metrics

    work = frame.sort_values(['sample_index', score_column], ascending=[True, False]).copy()
    num_slates = 0
    covered_slates = 0
    route_covered_slates = 0
    context_covered_slates = 0
    hit_counters = {f'system_top{k}': 0 for k in topks}
    hit_counters.update({f'context_top{k}': 0 for k in topks})
    hit_counters.update({f'route_top{k}': 0 for k in topks})
    covered_hit = {f'system_top{k}': 0 for k in topks}
    temp_abs_errors: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg10_values: list[float] = []

    for _, group in work.groupby('sample_index', sort=True):
        num_slates += 1
        label = group['label'].to_numpy(dtype=np.float64)
        route_match = group['route_match'].to_numpy(dtype=np.float64)
        context_match = group['context_match'].to_numpy(dtype=np.float64)

        has_positive = bool(np.any(label > 0.5))
        has_route = bool(np.any(route_match > 0.5))
        has_context = bool(np.any(context_match > 0.5))

        if has_positive:
            covered_slates += 1
        if has_route:
            route_covered_slates += 1
        if has_context:
            context_covered_slates += 1

        for k in topks:
            sys_hit = bool(np.any(label[:k] > 0.5))
            route_hit = bool(np.any(route_match[:k] > 0.5))
            context_hit = bool(np.any(context_match[:k] > 0.5))
            hit_counters[f'system_top{k}'] += int(sys_hit)
            hit_counters[f'route_top{k}'] += int(route_hit)
            hit_counters[f'context_top{k}'] += int(context_hit)
            if has_positive:
                covered_hit[f'system_top{k}'] += int(sys_hit)

        positive_rows = np.flatnonzero(label > 0.5)
        if positive_rows.size > 0:
            reciprocal_ranks.append(1.0 / float(positive_rows[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
        ndcg10_values.append(_ndcg_binary((label > 0.5).astype(np.float64), 10))

        if temperature_column and temperature_column in group.columns:
            temperature_gold = group['temperature_gold'].to_numpy(dtype=np.float64)
            temperature_pred = group[temperature_column].to_numpy(dtype=np.float64)
            valid_positive = (label > 0.5) & np.isfinite(temperature_gold) & np.isfinite(temperature_pred)
            positive_temp_rows = np.flatnonzero(valid_positive)
            if positive_temp_rows.size > 0:
                first_idx = int(positive_temp_rows[0])
                temp_abs_errors.append(abs(float(temperature_pred[first_idx]) - float(temperature_gold[first_idx])))

    metrics: dict[str, object] = {
        'num_slates': int(num_slates),
        'covered_slates': int(covered_slates),
        'route_covered_slates': int(route_covered_slates),
        'context_covered_slates': int(context_covered_slates),
        'pool_coverage': (covered_slates / num_slates if num_slates else 0.0),
        'pool_route_coverage': (route_covered_slates / num_slates if num_slates else 0.0),
        'pool_context_coverage': (context_covered_slates / num_slates if num_slates else 0.0),
        'system_mrr': float(np.mean(np.asarray(reciprocal_ranks, dtype=np.float64))) if reciprocal_ranks else 0.0,
        'system_ndcg10': float(np.mean(np.asarray(ndcg10_values, dtype=np.float64))) if ndcg10_values else 0.0,
    }
    for key, value in hit_counters.items():
        metrics[f'{key}_all'] = (value / num_slates if num_slates else 0.0)
    for k in topks:
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


def evaluate_scored_frame_with_manifest(
    frame: pd.DataFrame,
    *,
    expected_sample_indices: Iterable[int],
    score_column: str,
    temperature_column: str | None = None,
    topks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Evaluate against every test sample, including zero-candidate slates.

    Samples absent from ``frame`` occur when Stage 1 produces no route. They
    are legitimate end-to-end failures and must remain in the denominator for
    coverage, ranking, and top-k system metrics. Temperature remains
    conditional on valid, full-match candidates and is therefore not scaled.
    """

    expected = list(dict.fromkeys(int(value) for value in expected_sample_indices))
    if not expected:
        raise ValueError('The expected sample-index collection is empty.')
    if 'sample_index' not in frame.columns:
        raise ValueError('Candidate table is missing sample_index.')

    expected_set = set(expected)
    observed = (
        {int(value) for value in frame['sample_index'].dropna().tolist()}
        if not frame.empty
        else set()
    )
    unknown = sorted(observed - expected_set)
    if unknown:
        raise ValueError(
            'Candidate table contains sample indices absent from the test '
            f'manifest: {unknown[:5]}'
        )

    base = evaluate_scored_frame(
        frame,
        score_column=score_column,
        temperature_column=temperature_column,
        topks=topks,
    )
    candidate_slates = int(base.get('num_slates', 0))
    if candidate_slates != len(observed):
        raise ValueError(
            'Candidate slate count disagrees with unique sample_index count: '
            f'{candidate_slates} versus {len(observed)}.'
        )

    expected_count = len(expected)
    scale = candidate_slates / expected_count
    metrics = dict(base)
    metrics['num_slates'] = expected_count
    metrics['candidate_slates'] = candidate_slates
    metrics['missing_candidate_slates'] = expected_count - candidate_slates
    metrics['denominator'] = 'all_test_manifest_samples'
    for coverage_key, count_key in (
        ('pool_coverage', 'covered_slates'),
        ('pool_route_coverage', 'route_covered_slates'),
        ('pool_context_coverage', 'context_covered_slates'),
    ):
        metrics[coverage_key] = float(metrics.get(count_key, 0)) / expected_count
    metrics['system_mrr'] = float(base.get('system_mrr', 0.0)) * scale
    metrics['system_ndcg10'] = float(base.get('system_ndcg10', 0.0)) * scale
    for k in topks:
        for prefix in ('system', 'context', 'route'):
            key = f'{prefix}_top{k}_all'
            metrics[key] = float(base.get(key, 0.0)) * scale
    return metrics
