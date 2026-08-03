"""Run a product-only precedent-retrieval baseline for ProSys.

This baseline skips Stage 1 entirely. Given only a target product, it retrieves
historical full systems (reactants + reagents + solvents) from the training
split using three levels of precedent:

1. exact same canonical product
2. same Murcko scaffold
3. product fingerprint nearest neighbors

The retrieved systems are merged into one candidate slate and ranked by a
deterministic heuristic score. This lets us measure route/context/system top-k
accuracy when the model is not allowed to predict routes separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.features import (  # noqa: E402
    canonicalize_reaction_side,
    canonicalize_smiles,
    product_morgan_fp,
    product_scaffold_smiles,
    tanimoto_similarity_from_bitvect,
)
from prosys_shared.mainline import (  # noqa: E402
    display_family_name,
    evaluate_scored_frame,
    family_dir,
    label_candidate_table,
    load_split_rows,
    parse_families_arg,
    split_file_for_family,
)
from prosys_shared.product_memory import normalize_condition_labels, safe_float  # noqa: E402

TOPKS = (1, 3, 5, 10)


@dataclass(frozen=True)
class ProductOnlyQuery:
    family: str
    sample_index: int
    reaction_id: str
    product: str


@dataclass(frozen=True)
class SystemRecord:
    reaction_id: str
    reactants: str
    route_canonical: str
    product: str
    product_canonical: str
    product_scaffold: str
    reagent_norm: str
    solvent_norm: str
    yield_value: float
    temperature: float


def load_product_only_queries(split_file: str | Path, family: str) -> list[ProductOnlyQuery]:
    rows = load_split_rows(split_file)
    queries: list[ProductOnlyQuery] = []
    seen = set()
    for row in rows:
        dedup_key = (row['reaction_id'], row['reactants'], row['product'])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        queries.append(
            ProductOnlyQuery(
                family=family,
                sample_index=len(queries),
                reaction_id=row['reaction_id'],
                product=row['product'],
            )
        )
    return queries


def load_system_records(split_file: str | Path) -> list[SystemRecord]:
    records: list[SystemRecord] = []
    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 7:
                continue

            reaction_id, reactants, product, yield_value, reagent, solvent, temperature = parts[:7]
            product_canonical = canonicalize_smiles(product)
            route_canonical = canonicalize_reaction_side(reactants)
            if not product_canonical or not route_canonical:
                continue

            records.append(
                SystemRecord(
                    reaction_id=str(reaction_id),
                    reactants=str(reactants),
                    route_canonical=route_canonical,
                    product=str(product),
                    product_canonical=product_canonical,
                    product_scaffold=product_scaffold_smiles(product_canonical),
                    reagent_norm=normalize_condition_labels(reagent),
                    solvent_norm=normalize_condition_labels(solvent),
                    yield_value=safe_float(yield_value),
                    temperature=safe_float(temperature),
                )
            )
    return records


def compute_split_overlap_stats(train_file: str | Path, test_file: str | Path) -> dict[str, int]:
    def _product_set(split_file: str | Path) -> set[str]:
        values: set[str] = set()
        with open(split_file, 'r', encoding='utf-8') as handle:
            for line in handle:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue
                product_key = canonicalize_smiles(parts[2])
                if product_key:
                    values.add(product_key)
        return values

    def _route_set(split_file: str | Path) -> set[tuple[str, str]]:
        values: set[tuple[str, str]] = set()
        with open(split_file, 'r', encoding='utf-8') as handle:
            for line in handle:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3:
                    continue
                route_key = canonicalize_reaction_side(parts[1])
                product_key = canonicalize_smiles(parts[2])
                if route_key and product_key:
                    values.add((route_key, product_key))
        return values

    train_products = _product_set(train_file)
    test_products = _product_set(test_file)
    train_routes = _route_set(train_file)
    test_routes = _route_set(test_file)
    return {
        'train_product_count': int(len(train_products)),
        'test_product_count': int(len(test_products)),
        'product_overlap_count': int(len(train_products & test_products)),
        'train_route_count': int(len(train_routes)),
        'test_route_count': int(len(test_routes)),
        'route_overlap_count': int(len(train_routes & test_routes)),
    }


def _mean_or_nan(values: list[float]) -> float:
    numeric = [float(value) for value in values if not np.isnan(value)]
    if not numeric:
        return float('nan')
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def _aggregate_system_records(
    records: list[SystemRecord],
    *,
    key_builder,
    group_field_name: str,
) -> pd.DataFrame:
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}

    for record in records:
        group_value = key_builder(record)
        if not group_value:
            continue
        key = (
            str(group_value),
            record.route_canonical,
            record.reagent_norm,
            record.solvent_norm,
        )
        bucket = grouped.setdefault(
            key,
            {
                'group_value': str(group_value),
                'route_canonical': record.route_canonical,
                'reactants_repr': record.route_canonical,
                'reagent_norm': record.reagent_norm,
                'solvent_norm': record.solvent_norm,
                'count': 0,
                'yields': [],
                'temperatures': [],
            },
        )
        bucket['count'] = int(bucket['count']) + 1
        bucket['yields'].append(record.yield_value)
        bucket['temperatures'].append(record.temperature)

    rows: list[dict] = []
    for bucket in grouped.values():
        rows.append(
            {
                group_field_name: bucket['group_value'],
                'route_canonical': bucket['route_canonical'],
                'reactants_repr': bucket['reactants_repr'],
                'reagent_norm': bucket['reagent_norm'],
                'solvent_norm': bucket['solvent_norm'],
                'count': int(bucket['count']),
                'mean_yield': _mean_or_nan(bucket['yields']),
                'temperature_mean': _mean_or_nan(bucket['temperatures']),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        [group_field_name, 'count', 'route_canonical', 'reagent_norm', 'solvent_norm'],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)


def build_product_only_memory_artifacts(
    train_file: str | Path,
    output_dir: str | Path,
    *,
    family: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_system_records(train_file)
    if not records:
        raise ValueError(f'No valid system records found in {train_file}')

    exact_memory = _aggregate_system_records(
        records,
        key_builder=lambda record: record.product_canonical,
        group_field_name='product_canonical',
    )
    scaffold_memory = _aggregate_system_records(
        records,
        key_builder=lambda record: record.product_scaffold,
        group_field_name='product_scaffold',
    )

    unique_products = sorted({record.product_canonical for record in records})
    fps = np.stack([product_morgan_fp(product, n_bits=n_bits, radius=radius) for product in unique_products])
    knn_payload = {
        'product_smiles': np.asarray(unique_products),
        'packed_fps': np.packbits(fps, axis=1),
        'n_bits': np.asarray([n_bits], dtype=np.int32),
        'radius': np.asarray([radius], dtype=np.int32),
    }

    exact_path = output_dir / 'exact_product_system_memory.csv'
    scaffold_path = output_dir / 'scaffold_product_system_memory.csv'
    knn_path = output_dir / 'product_knn_index.npz'
    metadata_path = output_dir / 'product_only_memory_metadata.json'

    exact_memory.to_csv(exact_path, index=False)
    scaffold_memory.to_csv(scaffold_path, index=False)
    np.savez_compressed(knn_path, **knn_payload)

    metadata = {
        'family': family,
        'source_train_file': str(train_file),
        'num_records': len(records),
        'num_exact_rows': int(len(exact_memory)),
        'num_scaffold_rows': int(len(scaffold_memory)),
        'num_knn_products': int(len(unique_products)),
        'n_bits': int(n_bits),
        'radius': int(radius),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {
        'exact_memory': str(exact_path),
        'scaffold_memory': str(scaffold_path),
        'knn_index': str(knn_path),
        'metadata': str(metadata_path),
    }


class ProductOnlyPrecedentLookup:
    def __init__(self, memory_dir: str | Path):
        memory_dir = Path(memory_dir)
        self.exact_df = pd.read_csv(memory_dir / 'exact_product_system_memory.csv', keep_default_na=False)
        self.scaffold_df = pd.read_csv(memory_dir / 'scaffold_product_system_memory.csv', keep_default_na=False)

        knn = np.load(memory_dir / 'product_knn_index.npz', allow_pickle=True)
        self.knn_products = [str(value) for value in knn['product_smiles'].tolist()]
        self.knn_packed = knn['packed_fps']
        self.knn_n_bits = int(knn['n_bits'][0])
        self.knn_radius = int(knn['radius'][0])
        self.knn_matrix = np.unpackbits(self.knn_packed, axis=1)[:, :self.knn_n_bits].astype(np.uint8)

        self.exact_by_product: dict[str, list[dict]] = {}
        self.scaffold_by_scaffold: dict[str, list[dict]] = {}
        for row in self.exact_df.to_dict('records'):
            row = self._normalize_memory_row(row)
            self.exact_by_product.setdefault(str(row['product_canonical']), []).append(row)
        for row in self.scaffold_df.to_dict('records'):
            row = self._normalize_memory_row(row)
            self.scaffold_by_scaffold.setdefault(str(row['product_scaffold']), []).append(row)

    @staticmethod
    def _normalize_memory_row(row: dict) -> dict:
        row['route_canonical'] = str(row.get('route_canonical', ''))
        row['reactants_repr'] = str(row.get('reactants_repr', row['route_canonical']))
        row['reagent_norm'] = str(row.get('reagent_norm', ''))
        row['solvent_norm'] = str(row.get('solvent_norm', ''))
        row['count'] = int(row.get('count', 0))
        row['mean_yield'] = safe_float(row.get('mean_yield'))
        row['temperature_mean'] = safe_float(row.get('temperature_mean'))
        return row

    def exact_rows(self, product: str) -> list[dict]:
        product_canonical = canonicalize_smiles(product)
        rows = self.exact_by_product.get(product_canonical, [])
        return sorted(rows, key=lambda row: (-row['count'], row['route_canonical'], row['reagent_norm'], row['solvent_norm']))

    def scaffold_rows(self, product: str) -> list[dict]:
        scaffold = product_scaffold_smiles(product)
        rows = self.scaffold_by_scaffold.get(scaffold, [])
        return sorted(rows, key=lambda row: (-row['count'], row['route_canonical'], row['reagent_norm'], row['solvent_norm']))

    def knn_rows(self, product: str, *, top_products: int, max_systems: int) -> list[dict]:
        if self.knn_matrix.shape[0] == 0:
            return []

        query_fp = product_morgan_fp(product, n_bits=self.knn_n_bits, radius=self.knn_radius)
        similarities = tanimoto_similarity_from_bitvect(query_fp, self.knn_matrix)
        top_indices = np.argsort(similarities)[::-1][:top_products]

        aggregated: dict[tuple[str, str, str], dict] = {}
        for index in top_indices:
            sim = float(similarities[index])
            if sim <= 0:
                continue
            neighbor_product = self.knn_products[index]
            for row in self.exact_by_product.get(neighbor_product, []):
                key = (
                    str(row['route_canonical']),
                    str(row['reagent_norm']),
                    str(row['solvent_norm']),
                )
                stats = aggregated.setdefault(
                    key,
                    {
                        'route_canonical': str(row['route_canonical']),
                        'reactants_repr': str(row['reactants_repr']),
                        'reagent_norm': str(row['reagent_norm']),
                        'solvent_norm': str(row['solvent_norm']),
                        'knn_similarity_sum': 0.0,
                        'knn_weighted_support': 0.0,
                        'knn_similarity_max': 0.0,
                        'knn_neighbor_count': 0,
                        'weighted_yield_sum': 0.0,
                        'weighted_yield_weight': 0.0,
                        'weighted_temp_sum': 0.0,
                        'weighted_temp_weight': 0.0,
                    },
                )
                stats['knn_similarity_sum'] += sim
                stats['knn_weighted_support'] += sim * float(row['count'])
                stats['knn_similarity_max'] = max(float(stats['knn_similarity_max']), sim)
                stats['knn_neighbor_count'] = int(stats['knn_neighbor_count']) + 1
                weight = sim * float(row['count'])
                if not np.isnan(row['mean_yield']):
                    stats['weighted_yield_sum'] += weight * float(row['mean_yield'])
                    stats['weighted_yield_weight'] += weight
                if not np.isnan(row['temperature_mean']):
                    stats['weighted_temp_sum'] += weight * float(row['temperature_mean'])
                    stats['weighted_temp_weight'] += weight

        rows: list[dict] = []
        for stats in aggregated.values():
            yield_mean = (
                float(stats['weighted_yield_sum']) / float(stats['weighted_yield_weight'])
                if float(stats['weighted_yield_weight']) > 0
                else float('nan')
            )
            temp_mean = (
                float(stats['weighted_temp_sum']) / float(stats['weighted_temp_weight'])
                if float(stats['weighted_temp_weight']) > 0
                else float('nan')
            )
            rows.append(
                {
                    'route_canonical': stats['route_canonical'],
                    'reactants_repr': stats['reactants_repr'],
                    'reagent_norm': stats['reagent_norm'],
                    'solvent_norm': stats['solvent_norm'],
                    'knn_similarity_sum': float(stats['knn_similarity_sum']),
                    'knn_weighted_support': float(stats['knn_weighted_support']),
                    'knn_similarity_max': float(stats['knn_similarity_max']),
                    'knn_neighbor_count': int(stats['knn_neighbor_count']),
                    'mean_yield': yield_mean,
                    'temperature_mean': temp_mean,
                }
            )

        rows.sort(
            key=lambda row: (
                -float(row['knn_weighted_support']),
                -float(row['knn_similarity_max']),
                row['route_canonical'],
                row['reagent_norm'],
                row['solvent_norm'],
            )
        )
        return rows[:max_systems]


def _blank_candidate(route_canonical: str, reagent_norm: str, solvent_norm: str) -> dict:
    return {
        'reactants': route_canonical,
        'route_canonical': route_canonical,
        'reagent_norm': reagent_norm,
        'solvent_norm': solvent_norm,
        'from_product_exact': 0,
        'from_product_scaffold': 0,
        'from_product_knn': 0,
        'exact_system_count': 0.0,
        'scaffold_system_count': 0.0,
        'knn_similarity_sum': 0.0,
        'knn_weighted_support': 0.0,
        'knn_similarity_max': 0.0,
        'knn_neighbor_count': 0.0,
        'prior_mean_yield': float('nan'),
        'prior_temperature': float('nan'),
        '_prior_source_priority': 0,
        '_prior_source_support': 0.0,
    }


def _maybe_update_prior(candidate: dict, *, source_priority: int, source_support: float, mean_yield: float, temperature_mean: float) -> None:
    current_priority = int(candidate['_prior_source_priority'])
    current_support = float(candidate['_prior_source_support'])
    if source_priority < current_priority:
        return
    if source_priority == current_priority and source_support < current_support:
        return
    candidate['_prior_source_priority'] = int(source_priority)
    candidate['_prior_source_support'] = float(source_support)
    candidate['prior_mean_yield'] = float(mean_yield) if not np.isnan(mean_yield) else float('nan')
    candidate['prior_temperature'] = float(temperature_mean) if not np.isnan(temperature_mean) else float('nan')


def build_product_only_candidates(
    lookup: ProductOnlyPrecedentLookup,
    product: str,
    *,
    exact_limit: int,
    scaffold_limit: int,
    knn_top_products: int,
    knn_max_systems: int,
    max_total_candidates: int,
) -> list[dict]:
    candidates: dict[tuple[str, str, str], dict] = {}

    for row in lookup.exact_rows(product)[:exact_limit]:
        key = (str(row['route_canonical']), str(row['reagent_norm']), str(row['solvent_norm']))
        candidate = candidates.setdefault(key, _blank_candidate(*key))
        candidate['from_product_exact'] = 1
        candidate['exact_system_count'] = max(float(candidate['exact_system_count']), float(row['count']))
        _maybe_update_prior(
            candidate,
            source_priority=3,
            source_support=float(row['count']),
            mean_yield=float(row['mean_yield']),
            temperature_mean=float(row['temperature_mean']),
        )

    for row in lookup.scaffold_rows(product)[:scaffold_limit]:
        key = (str(row['route_canonical']), str(row['reagent_norm']), str(row['solvent_norm']))
        candidate = candidates.setdefault(key, _blank_candidate(*key))
        candidate['from_product_scaffold'] = 1
        candidate['scaffold_system_count'] = max(float(candidate['scaffold_system_count']), float(row['count']))
        _maybe_update_prior(
            candidate,
            source_priority=2,
            source_support=float(row['count']),
            mean_yield=float(row['mean_yield']),
            temperature_mean=float(row['temperature_mean']),
        )

    for row in lookup.knn_rows(product, top_products=knn_top_products, max_systems=knn_max_systems):
        key = (str(row['route_canonical']), str(row['reagent_norm']), str(row['solvent_norm']))
        candidate = candidates.setdefault(key, _blank_candidate(*key))
        candidate['from_product_knn'] = 1
        candidate['knn_similarity_sum'] = max(float(candidate['knn_similarity_sum']), float(row['knn_similarity_sum']))
        candidate['knn_weighted_support'] = max(float(candidate['knn_weighted_support']), float(row['knn_weighted_support']))
        candidate['knn_similarity_max'] = max(float(candidate['knn_similarity_max']), float(row['knn_similarity_max']))
        candidate['knn_neighbor_count'] = max(float(candidate['knn_neighbor_count']), float(row['knn_neighbor_count']))
        _maybe_update_prior(
            candidate,
            source_priority=1,
            source_support=float(row['knn_weighted_support']),
            mean_yield=float(row['mean_yield']),
            temperature_mean=float(row['temperature_mean']),
        )

    rows = list(candidates.values())
    for row in rows:
        prior_mean_yield = 0.0 if np.isnan(row['prior_mean_yield']) else float(row['prior_mean_yield'])
        row['precedent_score'] = (
            float(row['from_product_exact']) * 1_000_000_000.0
            + float(row['exact_system_count']) * 1_000_000.0
            + float(row['from_product_scaffold']) * 100_000.0
            + float(row['scaffold_system_count']) * 1_000.0
            + float(row['knn_weighted_support']) * 10.0
            + float(row['knn_similarity_max'])
            + prior_mean_yield / 1000.0
        )

    rows.sort(
        key=lambda row: (
            -float(row['precedent_score']),
            row['route_canonical'],
            row['reagent_norm'],
            row['solvent_norm'],
        )
    )
    for row in rows:
        row.pop('_prior_source_priority', None)
        row.pop('_prior_source_support', None)
    return rows[:max_total_candidates]


def build_candidate_pool_frame(
    queries: list[ProductOnlyQuery],
    lookup: ProductOnlyPrecedentLookup,
    *,
    exact_limit: int,
    scaffold_limit: int,
    knn_top_products: int,
    knn_max_systems: int,
    max_total_candidates: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    product_cache: dict[str, tuple[str, list[dict]]] = {}

    for query in queries:
        cached = product_cache.get(query.product)
        if cached is None:
            product_canonical = canonicalize_smiles(query.product)
            product_rows = build_product_only_candidates(
                lookup,
                query.product,
                exact_limit=exact_limit,
                scaffold_limit=scaffold_limit,
                knn_top_products=knn_top_products,
                knn_max_systems=knn_max_systems,
                max_total_candidates=max_total_candidates,
            )
            cached = (product_canonical, product_rows)
            product_cache[query.product] = cached

        product_canonical, product_rows = cached
        for row in product_rows:
            rows.append(
                {
                    'family': query.family,
                    'sample_index': query.sample_index,
                    'reaction_id': query.reaction_id,
                    'product': query.product,
                    'product_canonical': product_canonical,
                    'reactants': row['reactants'],
                    'route_canonical': row['route_canonical'],
                    'retro_rank': 1,
                    'retro_score': 1.0,
                    'retro_probability': 1.0,
                    'from_fnn': 0,
                    'from_product_exact': int(row['from_product_exact']),
                    'from_product_scaffold': int(row['from_product_scaffold']),
                    'from_product_knn': int(row['from_product_knn']),
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
                    'reagent_norm': row['reagent_norm'],
                    'solvent_norm': row['solvent_norm'],
                    'exact_system_count': float(row['exact_system_count']),
                    'scaffold_system_count': float(row['scaffold_system_count']),
                    'knn_similarity_sum': float(row['knn_similarity_sum']),
                    'knn_weighted_support': float(row['knn_weighted_support']),
                    'knn_similarity_max': float(row['knn_similarity_max']),
                    'knn_neighbor_count': float(row['knn_neighbor_count']),
                    'prior_mean_yield': float(row['prior_mean_yield']) if not np.isnan(row['prior_mean_yield']) else float('nan'),
                    'prior_temperature': float(row['prior_temperature']) if not np.isnan(row['prior_temperature']) else float('nan'),
                    'precedent_score': float(row['precedent_score']),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ['sample_index', 'precedent_score', 'route_canonical', 'reagent_norm', 'solvent_norm'],
        ascending=[True, False, True, True, True],
        kind='mergesort',
    ).reset_index(drop=True)


def _write_json(data: dict, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output_file


def _write_overview(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    flat_rows: list[dict] = []
    for row in rows:
        metrics = row['metrics']
        temp = metrics.get('temperature', {})
        overlap = row.get('overlap_stats', {})
        flat_rows.append(
            {
                'family': row['family'],
                'display_family': display_family_name(row['family']),
                'num_slates': metrics.get('num_slates'),
                'product_overlap_count': overlap.get('product_overlap_count'),
                'route_overlap_count': overlap.get('route_overlap_count'),
                'pool_route_coverage': metrics.get('pool_route_coverage'),
                'pool_context_coverage': metrics.get('pool_context_coverage'),
                'pool_coverage': metrics.get('pool_coverage'),
                'route_top1': metrics.get('route_top1_all'),
                'route_top3': metrics.get('route_top3_all'),
                'route_top5': metrics.get('route_top5_all'),
                'route_top10': metrics.get('route_top10_all'),
                'context_top1': metrics.get('context_top1_all'),
                'context_top3': metrics.get('context_top3_all'),
                'context_top5': metrics.get('context_top5_all'),
                'context_top10': metrics.get('context_top10_all'),
                'system_top1': metrics.get('system_top1_all'),
                'system_top3': metrics.get('system_top3_all'),
                'system_top5': metrics.get('system_top5_all'),
                'system_top10': metrics.get('system_top10_all'),
                'temperature_mae': temp.get('mae'),
                'temperature_within_5c': temp.get('within_5c'),
                'temperature_within_10c': temp.get('within_10c'),
                'temperature_within_20c': temp.get('within_20c'),
                'avg_candidates_per_slate': row.get('avg_candidates_per_slate'),
            }
        )

    frame = pd.DataFrame(flat_rows)
    csv_path = output_dir / 'results_flat.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    lines = []
    lines.append('# Product-only precedent baseline')
    lines.append('')
    lines.append('## Route + Context + System')
    lines.append('')
    lines.append('| Family | route@10 | context@10 | sys@1 | sys@3 | sys@5 | sys@10 | cover | avg cand. |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in flat_rows:
        lines.append(
            '| {family} | {route10:.1f} | {context10:.1f} | {sys1:.1f} | {sys3:.1f} | {sys5:.1f} | {sys10:.1f} | {cover:.1f} | {cand:.1f} |'.format(
                family=row['display_family'],
                route10=100.0 * float(row['route_top10'] or 0.0),
                context10=100.0 * float(row['context_top10'] or 0.0),
                sys1=100.0 * float(row['system_top1'] or 0.0),
                sys3=100.0 * float(row['system_top3'] or 0.0),
                sys5=100.0 * float(row['system_top5'] or 0.0),
                sys10=100.0 * float(row['system_top10'] or 0.0),
                cover=100.0 * float(row['pool_coverage'] or 0.0),
                cand=float(row['avg_candidates_per_slate'] or 0.0),
            )
        )

    if flat_rows:
        macro = pd.DataFrame(flat_rows).mean(numeric_only=True)
        lines.append(
            '| MACRO-AVG | {route10:.1f} | {context10:.1f} | {sys1:.1f} | {sys3:.1f} | {sys5:.1f} | {sys10:.1f} | {cover:.1f} | {cand:.1f} |'.format(
                route10=100.0 * float(macro.get('route_top10', 0.0)),
                context10=100.0 * float(macro.get('context_top10', 0.0)),
                sys1=100.0 * float(macro.get('system_top1', 0.0)),
                sys3=100.0 * float(macro.get('system_top3', 0.0)),
                sys5=100.0 * float(macro.get('system_top5', 0.0)),
                sys10=100.0 * float(macro.get('system_top10', 0.0)),
                cover=100.0 * float(macro.get('pool_coverage', 0.0)),
                cand=float(macro.get('avg_candidates_per_slate', 0.0)),
            )
        )

    lines.append('')
    lines.append('## Temperature')
    lines.append('')
    lines.append('| Family | Temp MAE | Temp±5C | Temp±10C | Temp±20C |')
    lines.append('| --- | ---: | ---: | ---: | ---: |')
    for row in flat_rows:
        mae = row['temperature_mae']
        lines.append(
            '| {family} | {mae} | {t5} | {t10} | {t20} |'.format(
                family=row['display_family'],
                mae='NA' if pd.isna(mae) else f'{float(mae):.2f}',
                t5='NA' if pd.isna(row['temperature_within_5c']) else f'{100.0 * float(row["temperature_within_5c"]):.1f}',
                t10='NA' if pd.isna(row['temperature_within_10c']) else f'{100.0 * float(row["temperature_within_10c"]):.1f}',
                t20='NA' if pd.isna(row['temperature_within_20c']) else f'{100.0 * float(row["temperature_within_20c"]):.1f}',
            )
        )

    md_path = output_dir / 'overview.md'
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return csv_path, md_path


def run_family(
    repo_root: Path,
    family: str,
    output_root: Path,
    *,
    force: bool,
    n_bits: int,
    radius: int,
    exact_limit: int,
    scaffold_limit: int,
    knn_top_products: int,
    knn_max_systems: int,
    max_total_candidates: int,
) -> dict:
    fam_dir = family_dir(repo_root, family)
    train_split = split_file_for_family(repo_root, family, 'train')
    test_split = split_file_for_family(repo_root, family, 'test')

    family_output = output_root / family / 'product_only_precedent'
    memory_dir = family_output / 'memory'
    candidate_file = family_output / 'candidate_pool_test.csv'
    labeled_file = family_output / 'test_labeled.csv'
    result_file = family_output / 'result.json'

    if force or not all(
        path.exists()
        for path in (
            memory_dir / 'exact_product_system_memory.csv',
            memory_dir / 'scaffold_product_system_memory.csv',
            memory_dir / 'product_knn_index.npz',
            memory_dir / 'product_only_memory_metadata.json',
        )
    ):
        build_product_only_memory_artifacts(
            train_file=train_split,
            output_dir=memory_dir,
            family=family,
            n_bits=n_bits,
            radius=radius,
        )

    lookup = ProductOnlyPrecedentLookup(memory_dir)
    queries = load_product_only_queries(test_split, family)
    candidate_frame = build_candidate_pool_frame(
        queries,
        lookup,
        exact_limit=exact_limit,
        scaffold_limit=scaffold_limit,
        knn_top_products=knn_top_products,
        knn_max_systems=knn_max_systems,
        max_total_candidates=max_total_candidates,
    )
    family_output.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(candidate_file, index=False)

    labeled_path = label_candidate_table(candidate_file, test_split, labeled_file)
    labeled_frame = pd.read_csv(labeled_path)
    metrics = evaluate_scored_frame(
        labeled_frame,
        score_column='precedent_score',
        temperature_column='prior_temperature',
        topks=TOPKS,
    )
    overlap_stats = compute_split_overlap_stats(train_split, test_split)
    result = {
        'family': family,
        'baseline': 'product_only_precedent',
        'candidate_table': str(labeled_path),
        'raw_candidate_file': str(candidate_file),
        'metrics': metrics,
        'overlap_stats': overlap_stats,
        'num_queries': int(len(queries)),
        'avg_candidates_per_slate': float(len(labeled_frame) / len(queries)) if queries else 0.0,
        'config': {
            'exact_limit': int(exact_limit),
            'scaffold_limit': int(scaffold_limit),
            'knn_top_products': int(knn_top_products),
            'knn_max_systems': int(knn_max_systems),
            'max_total_candidates': int(max_total_candidates),
            'n_bits': int(n_bits),
            'radius': int(radius),
        },
    }
    _write_json(result, result_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the product-only precedent baseline for ProSys.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--output_root', type=str, default='outputs/product_only_baseline')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--n_bits', type=int, default=2048)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--exact_limit', type=int, default=100)
    parser.add_argument('--scaffold_limit', type=int, default=100)
    parser.add_argument('--knn_top_products', type=int, default=20)
    parser.add_argument('--knn_max_systems', type=int, default=100)
    parser.add_argument('--max_total_candidates', type=int, default=200)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    families = parse_families_arg(args.families)

    results: list[dict] = []
    for family in families:
        print(f'[product_only] running {family}')
        result = run_family(
            repo_root,
            family,
            output_root,
            force=args.force,
            n_bits=args.n_bits,
            radius=args.radius,
            exact_limit=args.exact_limit,
            scaffold_limit=args.scaffold_limit,
            knn_top_products=args.knn_top_products,
            knn_max_systems=args.knn_max_systems,
            max_total_candidates=args.max_total_candidates,
        )
        metrics = result['metrics']
        print(
            f"[product_only] {family}: "
            f"route@10={metrics.get('route_top10_all', 0.0) * 100:.1f} "
            f"context@10={metrics.get('context_top10_all', 0.0) * 100:.1f} "
            f"sys@10={metrics.get('system_top10_all', 0.0) * 100:.1f} "
            f"cover={metrics.get('pool_coverage', 0.0) * 100:.1f}"
        )
        results.append(result)

    all_results = {row['family']: row for row in results}
    _write_json(all_results, output_root / 'all_results.json')
    csv_path, md_path = _write_overview(results, output_root)
    print(f'[product_only] wrote {csv_path}')
    print(f'[product_only] wrote {md_path}')


if __name__ == '__main__':
    main()
