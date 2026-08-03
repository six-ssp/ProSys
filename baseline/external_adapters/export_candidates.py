"""Convert external baseline predictions into the common ProSys candidate table.

For Sequential FNN and Reaction-GCNN, predictions must be JSONL records of the
form ``{sample_index, retro_rank, candidates: [...]}``. Each candidate contains
``reagent_ids``, ``solvent_ids``, ``condition_score`` and optional
``temperature_pred``. Direct Transformer records use ``beams`` with a tagged
sequence: ``<REACTANT> ... <REAGENT> ... <SOLVENT> ...``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from prosys_shared.product_memory import normalize_condition_labels

from .contracts import (
    canonical_route_or_empty,
    decode_condition_ids,
    detokenize_smiles,
    read_jsonl,
)


METHODS = ('sequential_fnn', 'reaction_gcnn', 'molecular_transformer')
CANDIDATE_COLUMNS = [
    'family',
    'sample_index',
    'reaction_id',
    'product',
    'reactants',
    'reagent_norm',
    'solvent_norm',
    'retro_rank',
    'retro_score',
    'retro_probability',
    'condition_score',
    'route_score_z',
    'condition_score_z',
    'system_score',
    'temperature_pred',
    'source_method',
    'candidate_rank',
]
ID_LIST_PATTERN = re.compile(r'^[RS](?:\d{4}|_(?:NONE|UNK))(?:\s+[RS](?:\d{4}|_(?:NONE|UNK)))*$')


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _as_optional_float(value: Any) -> float | None:
    numeric = _as_float(value, float('nan'))
    return numeric if math.isfinite(numeric) else None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith('['):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
    if ID_LIST_PATTERN.fullmatch(text):
        return text.split()
    if ';' in text:
        return [part.strip() for part in text.split(';') if part.strip()]
    return [text]


def _decode_condition(candidate: dict[str, Any], vocabulary: dict[str, Any], kind: str) -> str:
    normalized_key = f'{kind}_norm'
    if normalized_key in candidate:
        return normalize_condition_labels(str(candidate[normalized_key]))
    values = _as_list(candidate.get(f'{kind}_ids', candidate.get(f'{kind}s', [])))
    known_ids = set(vocabulary[kind]['id_to_label']) | {
        vocabulary[kind]['none_id'],
        vocabulary[kind]['unknown_id'],
    }
    if values and all(value in known_ids for value in values):
        return decode_condition_ids(values, vocabulary, kind)
    return normalize_condition_labels('; '.join(values))


def _candidate_payloads(record: dict[str, Any], *, direct: bool) -> list[dict[str, Any]]:
    values = record.get('beams') if direct else record.get('candidates')
    if values is None:
        values = record.get('candidates', [])
    if not isinstance(values, list):
        raise ValueError('Prediction records must contain a list in candidates or beams.')
    result = []
    for value in values:
        result.append({'text': value} if isinstance(value, str) else dict(value))
    return result


def _parse_direct_sequence(text: str, vocabulary: dict[str, Any]) -> dict[str, str] | None:
    tokens = str(text).strip().split()
    markers = ('<REACTANT>', '<REAGENT>', '<SOLVENT>')
    if any(marker not in tokens for marker in markers):
        return None
    reactant_start = tokens.index('<REACTANT>') + 1
    reagent_start = tokens.index('<REAGENT>')
    solvent_start = tokens.index('<SOLVENT>')
    if not (reactant_start <= reagent_start <= solvent_start):
        return None
    reactants = detokenize_smiles(tokens[reactant_start:reagent_start])
    if not canonical_route_or_empty(reactants):
        return None
    return {
        'reactants': reactants,
        'reagent_norm': decode_condition_ids(tokens[reagent_start + 1:solvent_start], vocabulary, 'reagent'),
        'solvent_norm': decode_condition_ids(tokens[solvent_start + 1:], vocabulary, 'solvent'),
    }


def _load_routes(route_manifest: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    by_sample_rank: dict[tuple[int, int], dict[str, Any]] = {}
    by_sample: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(route_manifest):
        sample_index = int(row['sample_index'])
        by_sample[sample_index] = row
        if 'retro_rank' in row:
            by_sample_rank[(sample_index, int(row['retro_rank']))] = row
    return by_sample_rank, by_sample


def _raw_rows_from_predictions(
    predictions: Iterable[dict[str, Any]],
    *,
    method: str,
    vocabulary: dict[str, Any],
    route_by_sample_rank: dict[tuple[int, int], dict[str, Any]],
    route_by_sample: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    direct = method == 'molecular_transformer'
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        sample_index = int(prediction['sample_index'])
        candidates = _candidate_payloads(prediction, direct=direct)
        for candidate_rank, candidate in enumerate(candidates, start=1):
            score = _as_float(
                candidate.get('condition_score', candidate.get('sequence_score', candidate.get('score', -candidate_rank))),
                -float(candidate_rank),
            )
            temperature = _as_optional_float(candidate.get('temperature_pred'))
            if direct:
                manifest = route_by_sample.get(sample_index)
                if manifest is None:
                    raise KeyError(f'No direct-transformer manifest row for sample_index={sample_index}')
                parsed = _parse_direct_sequence(str(candidate.get('text', candidate.get('prediction', ''))), vocabulary)
                if parsed is None:
                    continue
                rows.append(
                    {
                        'family': manifest['family'],
                        'sample_index': sample_index,
                        'reaction_id': manifest['reaction_id'],
                        'product': manifest['product'],
                        'reactants': parsed['reactants'],
                        'reagent_norm': parsed['reagent_norm'],
                        'solvent_norm': parsed['solvent_norm'],
                        'retro_rank': candidate_rank,
                        'retro_score': score,
                        'retro_probability': 1.0,
                        'condition_score': score,
                        'temperature_pred': temperature,
                        'source_method': method,
                        'candidate_rank': candidate_rank,
                    }
                )
                continue

            retro_rank = int(candidate.get('retro_rank', prediction.get('retro_rank', 1)))
            route = route_by_sample_rank.get((sample_index, retro_rank))
            if route is None:
                raise KeyError(f'No Stage 1 route for sample_index={sample_index}, retro_rank={retro_rank}')
            rows.append(
                {
                    'family': route['family'],
                    'sample_index': sample_index,
                    'reaction_id': route['reaction_id'],
                    'product': route['product'],
                    'reactants': route['reactants'],
                    'reagent_norm': _decode_condition(candidate, vocabulary, 'reagent'),
                    'solvent_norm': _decode_condition(candidate, vocabulary, 'solvent'),
                    'retro_rank': retro_rank,
                    'retro_score': _as_float(route.get('retro_score'), 0.0),
                    'retro_probability': _as_float(route.get('retro_probability'), 0.0),
                    'condition_score': score,
                    'temperature_pred': temperature,
                    'source_method': method,
                    'candidate_rank': candidate_rank,
                }
            )
    return rows


def _deduplicate_and_cap(rows: Iterable[dict[str, Any]], max_contexts_per_route: int) -> list[dict[str, Any]]:
    best: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        route_key = canonical_route_or_empty(row['reactants']) or str(row['reactants'])
        key = (int(row['sample_index']), route_key, row['reagent_norm'], row['solvent_norm'])
        current = best.get(key)
        if current is None or float(row['condition_score']) > float(current['condition_score']):
            best[key] = row

    per_route: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in best.values():
        per_route[(int(row['sample_index']), int(row['retro_rank']))].append(row)
    capped: list[dict[str, Any]] = []
    for group in per_route.values():
        group.sort(key=lambda row: (-float(row['condition_score']), int(row['candidate_rank'])))
        capped.extend(group[:max_contexts_per_route])
    return capped


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std <= 1e-12:
        return [0.0] * len(values)
    return [(value - mean) / std for value in values]


def _fuse_scores(rows: list[dict[str, Any]], route_weight: float, condition_weight: float, method: str) -> None:
    if method == 'molecular_transformer':
        for row in rows:
            row['route_score_z'] = 0.0
            row['condition_score_z'] = float(row['condition_score'])
            row['system_score'] = float(row['condition_score'])
        return

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row['sample_index'])].append(row)
    for group in groups.values():
        route_scores = _zscore([float(row['retro_score']) for row in group])
        condition_scores = _zscore([float(row['condition_score']) for row in group])
        for row, route_score, condition_score in zip(group, route_scores, condition_scores):
            row['route_score_z'] = route_score
            row['condition_score_z'] = condition_score
            row['system_score'] = route_weight * route_score + condition_weight * condition_score


def export_candidates(
    *,
    method: str,
    predictions_file: Path,
    vocabulary_file: Path,
    route_manifest: Path,
    output_file: Path,
    max_contexts_per_route: int,
    route_weight: float,
    condition_weight: float,
) -> int:
    if method not in METHODS:
        raise ValueError(f'Unsupported method: {method}')
    vocabulary = json.loads(vocabulary_file.read_text(encoding='utf-8'))
    route_by_sample_rank, route_by_sample = _load_routes(route_manifest)
    rows = _raw_rows_from_predictions(
        read_jsonl(predictions_file),
        method=method,
        vocabulary=vocabulary,
        route_by_sample_rank=route_by_sample_rank,
        route_by_sample=route_by_sample,
    )
    rows = _deduplicate_and_cap(rows, max_contexts_per_route)
    _fuse_scores(rows, route_weight, condition_weight, method)
    rows.sort(
        key=lambda row: (
            int(row['sample_index']),
            -float(row['system_score']),
            int(row['retro_rank']),
            int(row['candidate_rank']),
        )
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--route-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-contexts-per-route', type=int, default=20)
    parser.add_argument('--route-weight', type=float, default=0.0)
    parser.add_argument('--condition-weight', type=float, default=1.0)
    args = parser.parse_args()
    count = export_candidates(
        method=args.method,
        predictions_file=args.predictions,
        vocabulary_file=args.vocabulary,
        route_manifest=args.route_manifest,
        output_file=args.output,
        max_contexts_per_route=args.max_contexts_per_route,
        route_weight=args.route_weight,
        condition_weight=args.condition_weight,
    )
    print(f'Wrote {count} candidates to {args.output}')


if __name__ == '__main__':
    main()
