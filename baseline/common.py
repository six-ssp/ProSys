"""Shared helpers for ProSys baseline experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import RDLogger

from stage3_XGBoost.xgb_reranker import (
    infer_xgb_feature_columns as stage3_infer_xgb_feature_columns,
    load_xgb_ranker as stage3_load_xgb_ranker,
    score_table_with_xgb as stage3_score_table_with_xgb,
    train_xgb_ranker_and_temperature as stage3_train_xgb_ranker_and_temperature,
)
from prosys_shared.constants import (
    CONTEXT_DENSE_COLUMNS_V2,
    PRODUCT_DESCRIPTOR_COLUMNS_V2,
    ROUTE_DENSE_COLUMNS_V2,
    SUPPORT_FEATURE_COLUMNS_V2,
)
from prosys_shared.features import (
    canonicalize_reaction_side,
    canonicalize_smiles,
    count_condition_tokens,
    molecule_graph_descriptors,
)
from prosys_shared.product_memory import normalize_condition_labels, safe_float

RDLogger.DisableLog('rdApp.error')

FAMILY_ORDER = [
    'Beckmann',
    'Buchwald-HartwigCross-Coupling',
    'Chan_LamCoupling',
    'DielsAlder',
    'FischerIndoleSynthesis',
    'Friedel-CraftsAcylation',
    'Friedel-CraftsAlkylation',
    'GrignardReaction',
    'KumadaCoupling',
    'NegishiCoupling',
]

FAMILY_DISPLAY_NAMES = {
    'Beckmann': 'Beckmann',
    'Buchwald-HartwigCross-Coupling': 'Buchwald-Hartwig',
    'Chan_LamCoupling': 'Chan-Lam',
    'DielsAlder': 'Diels-Alder',
    'FischerIndoleSynthesis': 'Fischer-Indole',
    'Friedel-CraftsAcylation': 'Friedel-Crafts Acyl.',
    'Friedel-CraftsAlkylation': 'Friedel-Crafts Alkyl.',
    'GrignardReaction': 'Grignard',
    'KumadaCoupling': 'Kumada',
    'NegishiCoupling': 'Negishi',
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


@dataclass(frozen=True)
class RouteRecordLite:
    family: str
    sample_index: int
    reaction_id: str
    reactants: str
    product: str
    retro_rank: int = 1
    retro_score: float = 1.0
    retro_probability: float = 1.0


@dataclass
class XGBRunArtifacts:
    output_dir: str
    model_file: str
    metadata_file: str
    feature_columns: list[str]
    best_iteration: int | None
    temperature_model_file: str | None = None
    temperature_metadata_file: str | None = None
    temperature_num_train: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


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


def load_route_records(split_file: str | Path, family: str) -> list[RouteRecordLite]:
    rows = load_split_rows(split_file)
    records: list[RouteRecordLite] = []
    seen = set()
    for row in rows:
        dedup_key = (row['reaction_id'], row['reactants'], row['product'])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        records.append(
            RouteRecordLite(
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


def base_candidate_row(record: RouteRecordLite) -> dict:
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


def display_family_name(family: str) -> str:
    return FAMILY_DISPLAY_NAMES.get(family, family)


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
        return {'num_slates': 0, 'pool_coverage': 0.0}

    work = frame.sort_values(['sample_index', score_column], ascending=[True, False]).copy()
    num_slates = 0
    covered_slates = 0
    hit_counters = {f'system_top{k}': 0 for k in topks}
    hit_counters.update({f'context_top{k}': 0 for k in topks})
    hit_counters.update({f'route_top{k}': 0 for k in topks})
    covered_hit = {f'system_top{k}': 0 for k in topks}
    temp_hit_10c = {f'top{k}': 0 for k in topks}
    temp_hit_20c = {f'top{k}': 0 for k in topks}
    temp_abs_errors: list[float] = []

    for _, group in work.groupby('sample_index', sort=True):
        num_slates += 1
        label = group['label'].to_numpy(dtype=np.float64)
        route_match = group['route_match'].to_numpy(dtype=np.float64)
        context_match = group['context_match'].to_numpy(dtype=np.float64)
        has_positive = bool(np.any(label > 0.5))
        if has_positive:
            covered_slates += 1

        for k in topks:
            sys_hit = bool(np.any(label[:k] > 0.5))
            hit_counters[f'system_top{k}'] += int(sys_hit)
            hit_counters[f'context_top{k}'] += int(np.any(context_match[:k] > 0.5))
            hit_counters[f'route_top{k}'] += int(np.any(route_match[:k] > 0.5))
            if has_positive:
                covered_hit[f'system_top{k}'] += int(sys_hit)

        if temperature_column and temperature_column in group.columns:
            temperature_gold = group['temperature_gold'].to_numpy(dtype=np.float64)
            temperature_pred = group[temperature_column].to_numpy(dtype=np.float64)
            valid_temperature = np.isfinite(temperature_gold) & np.isfinite(temperature_pred)
            for k in topks:
                topk_mask = (label[:k] > 0.5) & valid_temperature[:k]
                if not np.any(topk_mask):
                    continue
                errors = np.abs(temperature_pred[:k][topk_mask] - temperature_gold[:k][topk_mask])
                temp_hit_10c[f'top{k}'] += int(np.any(errors <= 10.0))
                temp_hit_20c[f'top{k}'] += int(np.any(errors <= 20.0))

            top10_mask = (label[:10] > 0.5) & valid_temperature[:10]
            if np.any(top10_mask):
                first_idx = int(np.flatnonzero(top10_mask)[0])
                temp_abs_errors.append(abs(float(temperature_pred[first_idx]) - float(temperature_gold[first_idx])))

    metrics: dict[str, object] = {
        'num_slates': int(num_slates),
        'covered_slates': int(covered_slates),
        'pool_coverage': (covered_slates / num_slates if num_slates else 0.0),
    }
    for key, value in hit_counters.items():
        metrics[f'{key}_all'] = (value / num_slates if num_slates else 0.0)
    for k in topks:
        key = f'system_top{k}'
        metrics[f'{key}_covered'] = (covered_hit[key] / covered_slates if covered_slates else 0.0)
        metrics[f'temperature_top{k}_within_10c_all'] = (temp_hit_10c[f'top{k}'] / num_slates if num_slates else 0.0)
        metrics[f'temperature_top{k}_within_20c_all'] = (temp_hit_20c[f'top{k}'] / num_slates if num_slates else 0.0)

    if temp_abs_errors:
        errors = np.asarray(temp_abs_errors, dtype=np.float64)
        metrics['temperature'] = {
            'definition': 'topk_end_to_end_temperature_hit',
            'n': int(errors.size),
            'mae': float(np.mean(errors)),
            'mse': float(np.mean(errors ** 2)),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'mae_support': 'highest_ranked_full_match_with_valid_temperature_within_top10',
            'within_10c': (temp_hit_10c['top10'] / num_slates if num_slates else 0.0),
            'within_20c': (temp_hit_20c['top10'] / num_slates if num_slates else 0.0),
        }
    else:
        metrics['temperature'] = {
            'definition': 'topk_end_to_end_temperature_hit',
            'n': 0,
            'mae': None,
            'mse': None,
            'rmse': None,
            'mae_support': 'highest_ranked_full_match_with_valid_temperature_within_top10',
            'within_10c': 0.0,
            'within_20c': 0.0,
        }
    return metrics


def infer_xgb_feature_columns(frame: pd.DataFrame) -> list[str]:
    return stage3_infer_xgb_feature_columns(frame)


def train_xgb_ranker(
    train_table_file: str | Path,
    val_table_file: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 0,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
) -> XGBRunArtifacts:
    artifacts = stage3_train_xgb_ranker_and_temperature(
        train_table_file=train_table_file,
        val_table_file=val_table_file,
        output_dir=output_dir,
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
    )
    return XGBRunArtifacts(
        output_dir=str(artifacts['output_dir']),
        model_file=str(artifacts['model_file']),
        metadata_file=str(artifacts['metadata_file']),
        feature_columns=list(artifacts['feature_columns']),
        best_iteration=artifacts.get('best_iteration'),
        temperature_model_file=artifacts.get('temperature_model_file'),
        temperature_metadata_file=artifacts.get('temperature_metadata_file'),
        temperature_num_train=int(artifacts.get('temperature_num_train', 0)),
    )


def load_xgb_model(model_file: str | Path) -> xgb.XGBRanker:
    return stage3_load_xgb_ranker(model_file)


def score_table_with_xgb(table_file: str | Path, model_file: str | Path, metadata_file: str | Path) -> pd.DataFrame:
    return stage3_score_table_with_xgb(table_file, model_file, metadata_file)


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(FAMILY_ORDER), family)


def _macro_average_row(rows: list[dict]) -> dict:
    def mean_value(values: list[float | None]) -> float | None:
        numeric = [float(value) for value in values if value is not None]
        if not numeric:
            return None
        return float(np.mean(np.asarray(numeric, dtype=np.float64)))

    return {
        'family': 'MACRO-AVG',
        'rr10': mean_value([row['rr10'] for row in rows]),
        'cover': mean_value([row['cover'] for row in rows]),
        'sys1': mean_value([row['sys1'] for row in rows]),
        'sys5': mean_value([row['sys5'] for row in rows]),
        'sys10': mean_value([row['sys10'] for row in rows]),
    }


def _compact_metric_rows(rows: list[dict], *, baseline: str) -> list[dict]:
    selected = [row for row in rows if row.get('baseline') == baseline]
    selected = sorted(selected, key=lambda row: _family_sort_key(row['family']))

    compact_rows = []
    for row in selected:
        metrics = row.get('metrics', {})
        recall = row.get('stage1_route_recall', {})
        compact_rows.append(
            {
                'family': display_family_name(row['family']),
                'rr10': recall.get('route_recall_top10'),
                'cover': metrics.get('pool_coverage'),
                'sys1': metrics.get('system_top1_all'),
                'sys5': metrics.get('system_top5_all'),
                'sys10': metrics.get('system_top10_all'),
            }
        )

    if compact_rows:
        compact_rows.append(_macro_average_row(compact_rows))
    return compact_rows


def _format_percent(value: float | None) -> str:
    if value is None:
        return '  NA'
    return f'{value:>5.1f}'


def write_non_oracle_compact_tables(rows: list[dict], output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    baselines = sorted({row['baseline'] for row in rows})
    for baseline in baselines:
        compact_rows = _compact_metric_rows(rows, baseline=baseline)
        if not compact_rows:
            continue

        family_count = max(len(compact_rows) - 1, 0)
        family_width = max(len(row['family']) for row in compact_rows)
        lines = []
        lines.append(f'Non-Oracle results — {baseline} ({family_count} families)')
        lines.append(f'{"family":<{family_width}}  {"rr@10":>5}  {"cover":>5}  {"sys@1":>5}  {"sys@5":>5}  {"sys@10":>6}')
        for row in compact_rows:
            lines.append(
                f'{row["family"]:<{family_width}}  '
                f'{_format_percent(row["rr10"] * 100.0 if row["rr10"] is not None else None)}  '
                f'{_format_percent(row["cover"] * 100.0 if row["cover"] is not None else None)}  '
                f'{_format_percent(row["sys1"] * 100.0 if row["sys1"] is not None else None)}  '
                f'{_format_percent(row["sys5"] * 100.0 if row["sys5"] is not None else None)}  '
                f'{_format_percent(row["sys10"] * 100.0 if row["sys10"] is not None else None)}'
            )

        txt_file = output_path / f'{baseline}.txt'
        txt_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        outputs.append(txt_file)

    return outputs


def write_summary_table(rows: list[dict], output_file: str | Path) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text('No baseline rows.\n', encoding='utf-8')
        return output_path

    lines = []
    lines.append('=' * 140)
    lines.append('Oracle baseline summary')
    lines.append('=' * 140)
    header = f"{'baseline':<16} {'family':<34} {'cover':>7} {'sys@1':>7} {'sys@3':>7} {'sys@5':>7} {'sys@10':>8} {'temp_MAE':>10}"
    lines.append(header)
    for row in rows:
        metrics = row['metrics']
        temp = metrics.get('temperature', {})
        temp_mae = temp.get('mae')
        lines.append(
            f"{row['baseline']:<16} "
            f"{row['family']:<34} "
            f"{metrics.get('pool_coverage', 0.0) * 100:>6.1f} "
            f"{metrics.get('system_top1_all', 0.0) * 100:>6.1f} "
            f"{metrics.get('system_top3_all', 0.0) * 100:>6.1f} "
            f"{metrics.get('system_top5_all', 0.0) * 100:>6.1f} "
            f"{metrics.get('system_top10_all', 0.0) * 100:>7.1f} "
            f"{(f'{temp_mae:.1f}' if temp_mae is not None else 'NA'):>10}"
        )
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_path
