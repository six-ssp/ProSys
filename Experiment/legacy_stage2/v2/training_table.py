"""Training-table builders for ProSys Stage 2 V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import PRODUCT_DESCRIPTOR_COLUMNS_V2, SUPPORT_FEATURE_COLUMNS_V2
from .features import (
    canonicalize_reaction_side,
    canonicalize_smiles,
    count_condition_tokens,
    count_reactant_components,
    molecule_graph_descriptors,
)
from .product_memory import normalize_condition_labels, safe_float


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


def _mean_or_nan(values: list[float]) -> float:
    numeric = [value for value in values if not np.isnan(value)]
    if not numeric:
        return float('nan')
    return float(np.mean(numeric))


def load_gold_condition_index(split_file: str | Path) -> dict[tuple[str, str], GoldConditionIndex]:
    index: dict[tuple[str, str], GoldConditionIndex] = {}
    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 7:
                continue

            reaction_id, reactants, product, yield_value, reagent, solvent, temperature = parts[:7]
            product_key = canonicalize_smiles(product)
            if not product_key:
                continue

            route_key = canonicalize_reaction_side(reactants)
            reagent_norm = normalize_condition_labels(reagent)
            solvent_norm = normalize_condition_labels(solvent)
            bucket = index.setdefault((reaction_id, product_key), GoldConditionIndex())
            bucket.add(
                route_key=route_key,
                reagent_norm=reagent_norm,
                solvent_norm=solvent_norm,
                temperature=safe_float(temperature),
                yield_value=safe_float(yield_value),
            )

    return index


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


def build_candidate_training_table(
    candidate_pool_file: str | Path,
    gold_split_file: str | Path,
) -> pd.DataFrame:
    frame = pd.read_csv(candidate_pool_file)
    if frame.empty:
        return frame

    gold_index = load_gold_condition_index(gold_split_file)
    product_cache: dict[str, np.ndarray] = {}

    for column in SUPPORT_FEATURE_COLUMNS_V2:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0).astype(np.float32)

    route_match_values = []
    context_match_values = []
    label_values = []
    label_types = []
    sample_weights = []
    rank_relevance = []
    temperature_gold = []
    yield_gold = []
    route_component_counts = []
    reactants_lengths = []
    num_reagents = []
    num_solvents = []
    product_keys = []
    route_keys = []

    product_feature_rows = []

    for row in frame.itertuples(index=False):
        product_key = canonicalize_smiles(row.product)
        route_key = canonicalize_reaction_side(row.reactants)
        bucket = gold_index.get((str(row.reaction_id), product_key))

        route_match = int(bucket is not None and route_key in bucket.route_keys)
        context_key = (normalize_condition_labels(row.reagent_norm), normalize_condition_labels(row.solvent_norm))
        context_match = int(bucket is not None and context_key in bucket.context_keys)
        exact_key = (route_key, context_key[0], context_key[1])
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

        route_component_counts.append(count_reactant_components(row.reactants))
        reactants_lengths.append(len(str(row.reactants)))
        num_reagents.append(count_condition_tokens(row.reagent_norm))
        num_solvents.append(count_condition_tokens(row.solvent_norm))
        product_keys.append(product_key)
        route_keys.append(route_key)

        if product_key not in product_cache:
            product_cache[product_key] = molecule_graph_descriptors(product_key)
        product_feature_rows.append(product_cache[product_key])

    frame['product_canonical'] = product_keys
    frame['route_canonical'] = route_keys
    frame['num_reagents'] = np.asarray(num_reagents, dtype=np.int32)
    frame['num_solvents'] = np.asarray(num_solvents, dtype=np.int32)
    frame['route_component_count'] = np.asarray(route_component_counts, dtype=np.int32)
    frame['reactants_length'] = np.asarray(reactants_lengths, dtype=np.int32)
    frame['route_match'] = np.asarray(route_match_values, dtype=np.int32)
    frame['context_match'] = np.asarray(context_match_values, dtype=np.int32)
    frame['label'] = np.asarray(label_values, dtype=np.float32)
    frame['label_type'] = label_types
    frame['sample_weight'] = np.asarray(sample_weights, dtype=np.float32)
    frame['rank_relevance'] = np.asarray(rank_relevance, dtype=np.float32)
    frame['temperature_gold'] = np.asarray(temperature_gold, dtype=np.float32)
    frame['yield_gold'] = np.asarray(yield_gold, dtype=np.float32)

    product_feature_matrix = np.vstack(product_feature_rows).astype(np.float32)
    for idx, column in enumerate(PRODUCT_DESCRIPTOR_COLUMNS_V2):
        frame[column] = product_feature_matrix[:, idx]

    return frame


def write_candidate_training_table(
    candidate_pool_file: str | Path,
    gold_split_file: str | Path,
    output_file: str | Path,
) -> Path:
    frame = build_candidate_training_table(candidate_pool_file, gold_split_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
