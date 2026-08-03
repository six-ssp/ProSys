"""Stable data contracts shared by the external baseline adapters.

The adapters deliberately depend on ProSys canonicalization helpers so that
third-party model outputs are evaluated under the same matching rules.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from rdkit import Chem

from prosys_shared.features import canonicalize_reaction_side, canonicalize_smiles
from prosys_shared.mainline import load_split_rows
from prosys_shared.product_memory import normalize_condition_labels, safe_float
from prosys_shared.route_cache import load_route_records_from_cache


SMILES_TOKEN_PATTERN = re.compile(
    r'(\[[^\[\]]+\]|Br|Cl|Si|Na|Li|Mg|Al|Ca|Fe|Zn|Cu|Pd|Pt|Ag|Au|Sn|Pb|Hg|Mn|Cr|Ni|Co|As|Se|Te|[A-Z][a-z]?|[bcnops]|%\d{2}|\d|\(|\)|\.|=|#|-|\+|\\|/|:|@|\?|\*|\$)'
)


def project_root() -> Path:
    """Return the ProSys repository root for module-based entry points."""

    return Path(__file__).resolve().parents[2]


def split_condition_tokens(labels: str) -> list[str]:
    """Return canonical, duplicate-free condition labels in deterministic order."""

    normalized = normalize_condition_labels(labels)
    return [token.strip() for token in normalized.split(';') if token.strip()]


def _label_space(prefix: str, labels: Iterable[str]) -> dict[str, Any]:
    ordered = sorted(set(labels))
    label_to_id = {label: f'{prefix}{index:04d}' for index, label in enumerate(ordered)}
    return {
        'none_id': f'{prefix}_NONE',
        'unknown_id': f'{prefix}_UNK',
        'label_to_id': label_to_id,
        'id_to_label': {identifier: label for label, identifier in label_to_id.items()},
    }


def build_label_vocabulary(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Build a train-only vocabulary for reagent and solvent condition tokens."""

    reagent_labels: set[str] = set()
    solvent_labels: set[str] = set()
    for row in rows:
        reagent_labels.update(split_condition_tokens(str(row.get('reagent_norm', ''))))
        solvent_labels.update(split_condition_tokens(str(row.get('solvent_norm', ''))))
    return {
        'schema_version': 1,
        'reagent': _label_space('R', reagent_labels),
        'solvent': _label_space('S', solvent_labels),
    }


def encode_condition_labels(labels: str, vocabulary: dict[str, Any], kind: str) -> list[str]:
    """Map a normalized condition set to stable model IDs without using test labels."""

    if kind not in {'reagent', 'solvent'}:
        raise ValueError(f'Unsupported condition kind: {kind}')
    space = vocabulary[kind]
    label_to_id = space['label_to_id']
    values = [label_to_id.get(token, space['unknown_id']) for token in split_condition_tokens(labels)]
    return values or [space['none_id']]


def decode_condition_ids(values: Iterable[str], vocabulary: dict[str, Any], kind: str) -> str:
    """Map model IDs back to the normalized labels consumed by the evaluator."""

    if kind not in {'reagent', 'solvent'}:
        raise ValueError(f'Unsupported condition kind: {kind}')
    space = vocabulary[kind]
    ignored = {space['none_id'], space['unknown_id'], '', None}
    labels = [space['id_to_label'][value] for value in values if value not in ignored and value in space['id_to_label']]
    return normalize_condition_labels('; '.join(labels))


def tokenize_smiles(smiles: str) -> list[str]:
    """Tokenize a SMILES string with the convention used by MolecularTransformer."""

    text = str(smiles).strip()
    if not text:
        return []
    tokens = SMILES_TOKEN_PATTERN.findall(text)
    if ''.join(tokens) != text:
        raise ValueError(f'Cannot losslessly tokenize SMILES: {text}')
    return tokens


def detokenize_smiles(tokens: Iterable[str]) -> str:
    """Invert ``tokenize_smiles`` for a single dot-separated reaction side."""

    return ''.join(str(token) for token in tokens).strip()


def canonicalize_without_atom_maps(smiles: str, *, sort_fragments: bool = False) -> str:
    """Canonicalize a SMILES side after removing atom-map annotations."""

    fragments: list[str] = []
    for fragment in str(smiles).split('.'):
        fragment = fragment.strip()
        if not fragment:
            continue
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            return ''
        for atom in molecule.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        fragments.append(Chem.MolToSmiles(molecule, canonical=True))
    if sort_fragments:
        fragments.sort()
    return '.'.join(fragments)


def normalize_split_rows(split_file: str | Path) -> list[dict[str, str]]:
    """Load one official split and apply the same condition-label normalization as ProSys."""

    rows = []
    for row in load_split_rows(split_file):
        rows.append(
            {
                **row,
                'reagent_norm': normalize_condition_labels(row['reagent_norm']),
                'solvent_norm': normalize_condition_labels(row['solvent_norm']),
            }
        )
    return rows


def route_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row['reaction_id']), str(row['reactants']), str(row['product']))


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return output


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    return output


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL in {path}:{line_number}') from exc
            if not isinstance(value, dict):
                raise ValueError(f'Expected a JSON object in {path}:{line_number}')
            yield value


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return output


def load_stage1_route_rows(route_cache_file: str | Path, family: str) -> list[dict[str, Any]]:
    """Convert a Stage 1 cache into the explicit Non-Oracle route manifest."""

    rows: list[dict[str, Any]] = []
    for record in load_route_records_from_cache(route_cache_file, family):
        rows.append(
            {
                'family': record.family,
                'sample_index': int(record.sample_index),
                'reaction_id': str(record.reaction_id),
                'reactants': str(record.reactants),
                'product': str(record.product),
                'retro_rank': int(record.retro_rank),
                'retro_score': float(record.retro_score),
                'retro_probability': float(record.retro_probability),
            }
        )
    return rows


def finite_mean(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    return sum(numeric) / len(numeric) if numeric else None


def number_or_none(value: Any) -> float | None:
    numeric = safe_float(value)
    return float(numeric) if math.isfinite(numeric) else None


def canonical_route_or_empty(reactants: str) -> str:
    return canonicalize_reaction_side(reactants)


def canonical_product_or_empty(product: str) -> str:
    return canonicalize_smiles(product)
