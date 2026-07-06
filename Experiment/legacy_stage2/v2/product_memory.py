"""Product-memory builders for ProSys Stage 2 V2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .features import canonicalize_smiles, product_morgan_fp, product_scaffold_smiles

SENTINEL_LABELS = {'nan', 'none', 'not given', 'unknown', '-'}


@dataclass(frozen=True)
class ConditionRecord:
    reaction_id: str
    reactants: str
    product: str
    product_canonical: str
    reagent_norm: str
    solvent_norm: str
    yield_value: float
    temperature: float


def normalize_condition_labels(labels: str) -> str:
    parts = []
    for part in str(labels).split(';'):
        value = part.strip()
        if not value or value.lower() in SENTINEL_LABELS:
            continue
        parts.append(value)
    return '; '.join(sorted(dict.fromkeys(parts)))


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def load_condition_records(train_file: str | Path) -> list[ConditionRecord]:
    records: list[ConditionRecord] = []
    with open(train_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 7:
                continue

            reaction_id, reactants, product, yield_value, reagent, solvent, temperature = parts[:7]
            product_canonical = canonicalize_smiles(product)
            if not product_canonical:
                continue

            records.append(
                ConditionRecord(
                    reaction_id=reaction_id,
                    reactants=reactants,
                    product=product,
                    product_canonical=product_canonical,
                    reagent_norm=normalize_condition_labels(reagent),
                    solvent_norm=normalize_condition_labels(solvent),
                    yield_value=safe_float(yield_value),
                    temperature=safe_float(temperature),
                )
            )
    return records


def aggregate_temperature(values: Iterable[float]) -> tuple[float, float]:
    numeric = np.array([value for value in values if not math.isnan(value)], dtype=np.float32)
    if numeric.size == 0:
        return float('nan'), float('nan')
    return float(numeric.mean()), float(numeric.std(ddof=0))


def mean_or_nan(values: Iterable[float]) -> float:
    numeric = np.array([value for value in values if not math.isnan(value)], dtype=np.float32)
    if numeric.size == 0:
        return float('nan')
    return float(numeric.mean())


def build_exact_product_memory(records: list[ConditionRecord]) -> pd.DataFrame:
    grouped: dict[tuple[str, str, str], dict[str, list[float] | int]] = {}

    for record in records:
        key = (record.product_canonical, record.reagent_norm, record.solvent_norm)
        bucket = grouped.setdefault(
            key,
            {'count': 0, 'yields': [], 'temperatures': []},
        )
        bucket['count'] += 1
        bucket['yields'].append(record.yield_value)
        bucket['temperatures'].append(record.temperature)

    rows = []
    for (product_canonical, reagent_norm, solvent_norm), bucket in grouped.items():
        temp_mean, temp_std = aggregate_temperature(bucket['temperatures'])
        rows.append(
            {
                'product_canonical': product_canonical,
                'reagent_norm': reagent_norm,
                'solvent_norm': solvent_norm,
                'count': bucket['count'],
                'mean_yield': mean_or_nan(bucket['yields']),
                'temperature_mean': temp_mean,
                'temperature_std': temp_std,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ['product_canonical', 'count', 'reagent_norm', 'solvent_norm'],
        ascending=[True, False, True, True],
    )


def build_scaffold_memory(records: list[ConditionRecord]) -> pd.DataFrame:
    grouped: dict[tuple[str, str, str], dict[str, list[float] | int]] = {}

    for record in records:
        scaffold = product_scaffold_smiles(record.product_canonical)
        if not scaffold:
            continue

        key = (scaffold, record.reagent_norm, record.solvent_norm)
        bucket = grouped.setdefault(key, {'count': 0, 'yields': []})
        bucket['count'] += 1
        bucket['yields'].append(record.yield_value)

    rows = []
    for (product_scaffold, reagent_norm, solvent_norm), bucket in grouped.items():
        rows.append(
            {
                'product_scaffold': product_scaffold,
                'reagent_norm': reagent_norm,
                'solvent_norm': solvent_norm,
                'count': bucket['count'],
                'mean_yield': mean_or_nan(bucket['yields']),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ['product_scaffold', 'count', 'reagent_norm', 'solvent_norm'],
        ascending=[True, False, True, True],
    )


def build_product_knn_index(records: list[ConditionRecord], n_bits: int = 2048, radius: int = 2) -> dict[str, np.ndarray]:
    product_list = sorted({record.product_canonical for record in records})
    fps = np.stack([product_morgan_fp(product, n_bits=n_bits, radius=radius) for product in product_list])
    packed_fps = np.packbits(fps, axis=1)
    return {
        'product_smiles': np.array(product_list),
        'packed_fps': packed_fps,
        'n_bits': np.array([n_bits], dtype=np.int32),
        'radius': np.array([radius], dtype=np.int32),
    }


def infer_family_name(train_file: str | Path) -> str:
    train_path = Path(train_file)
    family_root = train_path.parents[1].name
    return family_root.replace('reaction_processed_', '').replace('_catmerge', '')


def build_product_memory_artifacts(
    train_file: str | Path,
    output_dir: str | Path,
    *,
    n_bits: int = 2048,
    radius: int = 2,
    family: str | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = load_condition_records(train_file)
    if not records:
        raise ValueError(f'No valid condition records found in {train_file}')

    exact_memory = build_exact_product_memory(records)
    scaffold_memory = build_scaffold_memory(records)
    knn_index = build_product_knn_index(records, n_bits=n_bits, radius=radius)

    exact_path = output_path / 'exact_product_memory.csv'
    scaffold_path = output_path / 'scaffold_product_memory.csv'
    knn_path = output_path / 'product_knn_index.npz'
    metadata_path = output_path / 'product_memory_metadata.json'

    exact_memory.to_csv(exact_path, index=False)
    scaffold_memory.to_csv(scaffold_path, index=False)
    np.savez_compressed(knn_path, **knn_index)

    metadata = {
        'family': family or infer_family_name(train_file),
        'source_train_file': str(train_file),
        'num_records': len(records),
        'num_exact_rows': int(len(exact_memory)),
        'num_scaffold_rows': int(len(scaffold_memory)),
        'num_knn_products': int(len(knn_index['product_smiles'])),
        'n_bits': n_bits,
        'radius': radius,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    return {
        'exact_memory': str(exact_path),
        'scaffold_memory': str(scaffold_path),
        'knn_index': str(knn_path),
        'metadata': str(metadata_path),
    }
