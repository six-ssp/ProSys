"""Shared helpers for reaction-condition token modeling.

These utilities are reused by the Stage 2 ReaFNN selector and the Stage 3
reaction-GNN feature encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prosys_shared.features import count_reactant_components, reaction_graph_descriptors, reaction_morgan_fp
from prosys_shared.product_memory import normalize_condition_labels, safe_float


@dataclass(frozen=True)
class ConditionRow:
    reaction_id: str
    reactants: str
    product: str
    yield_value: float
    reagent_norm: str
    solvent_norm: str
    temperature: float


@dataclass(frozen=True)
class AggregatedReactionExample:
    reaction_id: str
    reactants: str
    product: str
    reagent_tokens: tuple[str, ...]
    solvent_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ContextRecord:
    reagent_norm: str
    solvent_norm: str
    reagent_tokens: tuple[str, ...]
    solvent_tokens: tuple[str, ...]
    context_count: float
    context_support: float
    mean_yield: float


def split_condition_tokens(labels: str) -> tuple[str, ...]:
    normalized = normalize_condition_labels(labels)
    tokens = [token.strip() for token in normalized.split(';') if token.strip()]
    return tuple(tokens)


def load_condition_rows(split_file: str | Path) -> list[ConditionRow]:
    rows: list[ConditionRow] = []
    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 7:
                continue
            rows.append(
                ConditionRow(
                    reaction_id=str(parts[0]),
                    reactants=str(parts[1]),
                    product=str(parts[2]),
                    yield_value=safe_float(parts[3]),
                    reagent_norm=normalize_condition_labels(parts[4]),
                    solvent_norm=normalize_condition_labels(parts[5]),
                    temperature=safe_float(parts[6]),
                )
            )
    return rows


def aggregate_reaction_examples(rows: list[ConditionRow]) -> list[AggregatedReactionExample]:
    buckets: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        key = (row.reaction_id, row.reactants, row.product)
        if key not in buckets:
            buckets[key] = {
                'reagents': set(),
                'solvents': set(),
            }
            order.append(key)
        buckets[key]['reagents'].update(split_condition_tokens(row.reagent_norm))
        buckets[key]['solvents'].update(split_condition_tokens(row.solvent_norm))

    examples: list[AggregatedReactionExample] = []
    for reaction_id, reactants, product in order:
        bucket = buckets[(reaction_id, reactants, product)]
        examples.append(
            AggregatedReactionExample(
                reaction_id=reaction_id,
                reactants=reactants,
                product=product,
                reagent_tokens=tuple(sorted(bucket['reagents'])),
                solvent_tokens=tuple(sorted(bucket['solvents'])),
            )
        )
    return examples


def build_token_vocab(examples: list[AggregatedReactionExample], *, field: str) -> tuple[list[str], dict[str, int]]:
    if field not in {'reagent_tokens', 'solvent_tokens'}:
        raise ValueError(f'Unsupported field: {field}')

    counts: dict[str, int] = {}
    for example in examples:
        for token in getattr(example, field):
            counts[token] = counts.get(token, 0) + 1

    vocab = sorted(counts, key=lambda token: (-counts[token], token))
    return vocab, {token: idx for idx, token in enumerate(vocab)}


def multi_hot_from_tokens(tokens: tuple[str, ...], token_to_index: dict[str, int]) -> np.ndarray:
    vector = np.zeros((len(token_to_index),), dtype=np.float32)
    for token in tokens:
        index = token_to_index.get(token)
        if index is not None:
            vector[index] = 1.0
    return vector


def multi_hot_from_labels(labels: str, token_to_index: dict[str, int]) -> np.ndarray:
    return multi_hot_from_tokens(split_condition_tokens(labels), token_to_index)


def build_context_library(rows: list[ConditionRow]) -> list[ContextRecord]:
    agg: dict[tuple[str, str], dict[str, float | tuple[str, ...]]] = {}
    for row in rows:
        key = (row.reagent_norm, row.solvent_norm)
        stats = agg.setdefault(
            key,
            {
                'reagent_tokens': split_condition_tokens(row.reagent_norm),
                'solvent_tokens': split_condition_tokens(row.solvent_norm),
                'count': 0.0,
                'yield_sum': 0.0,
                'yield_n': 0.0,
            },
        )
        stats['count'] = float(stats['count']) + 1.0
        if np.isfinite(row.yield_value):
            stats['yield_sum'] = float(stats['yield_sum']) + float(row.yield_value)
            stats['yield_n'] = float(stats['yield_n']) + 1.0

    total = sum(float(stats['count']) for stats in agg.values()) or 1.0
    rows_out: list[ContextRecord] = []
    for (reagent_norm, solvent_norm), stats in agg.items():
        yield_n = float(stats['yield_n'])
        rows_out.append(
            ContextRecord(
                reagent_norm=reagent_norm,
                solvent_norm=solvent_norm,
                reagent_tokens=tuple(stats['reagent_tokens']),
                solvent_tokens=tuple(stats['solvent_tokens']),
                context_count=float(stats['count']),
                context_support=float(stats['count']) / float(total),
                mean_yield=(float(stats['yield_sum']) / yield_n if yield_n > 0 else 0.0),
            )
        )

    rows_out.sort(
        key=lambda row: (
            -row.context_count,
            row.reagent_norm,
            row.solvent_norm,
        )
    )
    return rows_out


def route_feature_vector(
    reactants: str,
    product: str,
    *,
    fpsize: int,
    radius: int,
) -> np.ndarray:
    route_fp = reaction_morgan_fp(reactants, product, fpsize=fpsize, radius=radius).astype(np.float32, copy=False)
    route_graph = reaction_graph_descriptors(reactants, product).astype(np.float32, copy=False)
    route_dense = np.asarray(
        [
            float(count_reactant_components(reactants)),
            float(len(str(reactants))),
        ],
        dtype=np.float32,
    )
    return np.concatenate((route_fp, route_graph, route_dense), axis=0).astype(np.float32, copy=False)
