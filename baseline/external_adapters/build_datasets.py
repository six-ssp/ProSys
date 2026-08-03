"""Build reproducible ProSys input packages for the three external baselines.

Run from the repository root, for example:
    python -m baseline.external_adapters.build_datasets --families Beckmann
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from prosys_shared.mainline import (
    parse_families_arg,
    load_route_records,
    split_file_for_family,
)
from prosys_shared.product_memory import normalize_condition_labels

from .contracts import (
    build_label_vocabulary,
    canonicalize_without_atom_maps,
    encode_condition_labels,
    finite_mean,
    load_stage1_route_rows,
    normalize_split_rows,
    number_or_none,
    project_root,
    route_key,
    split_condition_tokens,
    tokenize_smiles,
    write_csv,
    write_json,
    write_jsonl,
)


MODEL_NAMES = ('molecular_transformer', 'sequential_fnn', 'reaction_gcnn')
ROUTE_FIELDS = [
    'family',
    'sample_index',
    'reaction_id',
    'reactants',
    'product',
    'retro_rank',
    'retro_score',
    'retro_probability',
]


def _parse_models(value: str) -> list[str]:
    if value.strip().lower() == 'all':
        return list(MODEL_NAMES)
    requested = [item.strip() for item in value.replace(',', ' ').split() if item.strip()]
    unknown = [item for item in requested if item not in MODEL_NAMES]
    if unknown:
        raise ValueError(f'Unknown baseline model(s): {unknown}')
    return requested


def _require_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f'{path} already contains generated data. Choose a new output root instead of overwriting it.'
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for line in lines:
            handle.write(f'{line}\n')


def _smiles_for_sequence(smiles: str, *, reaction_side: bool) -> str:
    canonical = canonicalize_without_atom_maps(smiles, sort_fragments=reaction_side)
    if not canonical:
        raise ValueError(f'Cannot build a sequence example from invalid SMILES: {smiles}')
    return ' '.join(tokenize_smiles(canonical))


def _direct_source(product: str) -> str:
    return '<PRODUCT> ' + _smiles_for_sequence(product, reaction_side=False)


def _direct_target(row: dict[str, str], vocabulary: dict[str, Any]) -> str:
    reactants = _smiles_for_sequence(row['reactants'], reaction_side=True)
    reagent_ids = ' '.join(encode_condition_labels(row['reagent_norm'], vocabulary, 'reagent'))
    solvent_ids = ' '.join(encode_condition_labels(row['solvent_norm'], vocabulary, 'solvent'))
    return f'<REACTANT> {reactants} <REAGENT> {reagent_ids} <SOLVENT> {solvent_ids}'


def _labeled_route_entries(
    repo_root: Path,
    family: str,
    split: str,
    vocabulary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    split_file = split_file_for_family(repo_root, family, split)
    rows = normalize_split_rows(split_file)
    rows_by_route: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_route[route_key(row)].append(row)

    entries: list[dict[str, Any]] = []
    for record in load_route_records(split_file, family):
        key = (record.reaction_id, record.reactants, record.product)
        route_rows = rows_by_route.get(key, [])
        contexts = []
        all_reagent_ids: set[str] = set()
        all_solvent_ids: set[str] = set()
        temperatures: list[float] = []
        for row in route_rows:
            reagent_ids = encode_condition_labels(row['reagent_norm'], vocabulary, 'reagent')
            solvent_ids = encode_condition_labels(row['solvent_norm'], vocabulary, 'solvent')
            all_reagent_ids.update(identifier for identifier in reagent_ids if not identifier.endswith('_NONE'))
            all_solvent_ids.update(identifier for identifier in solvent_ids if not identifier.endswith('_NONE'))
            temperature = number_or_none(row['temperature'])
            if temperature is not None:
                temperatures.append(temperature)
            contexts.append(
                {
                    'reagent_norm': row['reagent_norm'],
                    'solvent_norm': row['solvent_norm'],
                    'reagent_ids': reagent_ids,
                    'solvent_ids': solvent_ids,
                    'temperature': temperature,
                    'yield': number_or_none(row['yield']),
                }
            )
        entries.append(
            {
                'family': family,
                'split': split,
                'sample_index': int(record.sample_index),
                'reaction_id': str(record.reaction_id),
                'reactants': str(record.reactants),
                'product': str(record.product),
                'reagent_ids': sorted(all_reagent_ids) or [vocabulary['reagent']['none_id']],
                'solvent_ids': sorted(all_solvent_ids) or [vocabulary['solvent']['none_id']],
                'temperature_mean': finite_mean(temperatures),
                'contexts': contexts,
            }
        )
    return entries, rows


def _context_library(rows: Iterable[dict[str, str]], vocabulary: dict[str, Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        reagent_norm = normalize_condition_labels(row['reagent_norm'])
        solvent_norm = normalize_condition_labels(row['solvent_norm'])
        key = (reagent_norm, solvent_norm)
        bucket = buckets.setdefault(
            key,
            {
                'reagent_norm': reagent_norm,
                'solvent_norm': solvent_norm,
                'reagent_ids': encode_condition_labels(reagent_norm, vocabulary, 'reagent'),
                'solvent_ids': encode_condition_labels(solvent_norm, vocabulary, 'solvent'),
                'count': 0,
                'temperatures': [],
            },
        )
        bucket['count'] += 1
        temperature = number_or_none(row['temperature'])
        if temperature is not None:
            bucket['temperatures'].append(temperature)

    result = []
    for bucket in buckets.values():
        result.append(
            {
                'reagent_norm': bucket['reagent_norm'],
                'solvent_norm': bucket['solvent_norm'],
                'reagent_ids': bucket['reagent_ids'],
                'solvent_ids': bucket['solvent_ids'],
                'count': int(bucket['count']),
                'temperature_mean': finite_mean(bucket['temperatures']),
            }
        )
    return sorted(result, key=lambda item: (-item['count'], item['reagent_norm'], item['solvent_norm']))


def _sample_manifest(repo_root: Path, family: str, split: str) -> list[dict[str, Any]]:
    """Expose formal split identities, including samples without Stage 1 routes."""

    return [
        {
            'family': family,
            'sample_index': int(record.sample_index),
            'reaction_id': str(record.reaction_id),
            'product': str(record.product),
        }
        for record in load_route_records(split_file_for_family(repo_root, family, split), family)
    ]


def _build_molecular_transformer_family(
    repo_root: Path,
    output_root: Path,
    family: str,
) -> dict[str, Any]:
    output_dir = output_root / 'molecular_transformer' / family
    _require_empty_directory(output_dir)
    train_rows = normalize_split_rows(split_file_for_family(repo_root, family, 'train'))
    vocabulary = build_label_vocabulary(train_rows)
    write_json(output_dir / 'label_vocabulary.json', vocabulary)

    counts: dict[str, int] = {}
    for split in ('train', 'val'):
        rows = normalize_split_rows(split_file_for_family(repo_root, family, split))
        _write_lines(output_dir / f'{split}.src', (_direct_source(row['product']) for row in rows))
        _write_lines(output_dir / f'{split}.tgt', (_direct_target(row, vocabulary) for row in rows))
        counts[split] = len(rows)

    test_entries, _test_rows = _labeled_route_entries(repo_root, family, 'test', vocabulary)
    _write_lines(output_dir / 'test.src', (_direct_source(row['product']) for row in test_entries))
    write_jsonl(
        output_dir / 'test_manifest.jsonl',
        (
            {
                'family': row['family'],
                'sample_index': row['sample_index'],
                'reaction_id': row['reaction_id'],
                'product': row['product'],
            }
            for row in test_entries
        ),
    )
    write_jsonl(
        output_dir / 'test_gold_systems.jsonl',
        (
            {
                'family': row['family'],
                'sample_index': row['sample_index'],
                'reaction_id': row['reaction_id'],
                'product': row['product'],
                'systems': [
                    {
                        'reactants': row['reactants'],
                        'reagent_norm': context['reagent_norm'],
                        'solvent_norm': context['solvent_norm'],
                    }
                    for context in row['contexts']
                ],
            }
            for row in test_entries
        ),
    )
    counts['test_inputs'] = len(test_entries)
    metadata = {
        'family': family,
        'method': 'molecular_transformer',
        'source_format': '<PRODUCT> product_smiles',
        'target_format': '<REACTANT> reactants <REAGENT> reagent_ids <SOLVENT> solvent_ids',
        'test_inference_file': 'test.src',
        'test_labels_are_separate': True,
        'counts': counts,
    }
    write_json(output_dir / 'metadata.json', metadata)
    return metadata


def _gcnn_rows(entries: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for entry in entries:
        yield {
            'sample_index': entry['sample_index'],
            'reaction_id': entry['reaction_id'],
            'family': entry['family'],
            'Reactant': entry['reactants'],
            'Product': entry['product'],
            'reagent_ids': json.dumps(entry['reagent_ids']),
            'solvent_ids': json.dumps(entry['solvent_ids']),
            'temperature_mean': entry['temperature_mean'],
        }


def _build_indirect_fnn_and_gcnn(
    repo_root: Path,
    route_root: Path,
    validation_route_root: Path | None,
    output_root: Path,
    family: str,
    requested_models: list[str],
) -> dict[str, dict[str, Any]]:
    train_rows = normalize_split_rows(split_file_for_family(repo_root, family, 'train'))
    vocabulary = build_label_vocabulary(train_rows)
    train_entries, _ = _labeled_route_entries(repo_root, family, 'train', vocabulary)
    val_entries, _ = _labeled_route_entries(repo_root, family, 'val', vocabulary)
    route_cache = route_root / family / 'route_cache.json'
    if not route_cache.exists():
        raise FileNotFoundError(f'Missing Non-Oracle Stage 1 route cache: {route_cache}')
    test_stage1_routes = load_stage1_route_rows(route_cache, family)
    test_manifest = _sample_manifest(repo_root, family, 'test')
    val_manifest = _sample_manifest(repo_root, family, 'val')
    validation_route_cache: Path | None = None
    val_stage1_routes: list[dict[str, Any]] = []
    if validation_route_root is not None:
        validation_route_cache = validation_route_root / family / 'route_cache.json'
        if not validation_route_cache.exists():
            raise FileNotFoundError(
                f'Missing validation-only Stage 1 route cache: {validation_route_cache}'
            )
        val_stage1_routes = load_stage1_route_rows(validation_route_cache, family)
    context_library = _context_library(train_rows, vocabulary)
    results: dict[str, dict[str, Any]] = {}

    if 'sequential_fnn' in requested_models:
        fnn_dir = output_root / 'sequential_fnn' / family
        _require_empty_directory(fnn_dir)
        write_json(fnn_dir / 'label_vocabulary.json', vocabulary)
        write_jsonl(fnn_dir / 'train_routes.jsonl', train_entries)
        write_jsonl(fnn_dir / 'val_routes.jsonl', val_entries)
        write_jsonl(fnn_dir / 'test_stage1_routes.jsonl', test_stage1_routes)
        write_jsonl(fnn_dir / 'val_manifest.jsonl', val_manifest)
        write_jsonl(fnn_dir / 'val_stage1_routes.jsonl', val_stage1_routes)
        write_jsonl(fnn_dir / 'test_manifest.jsonl', test_manifest)
        write_jsonl(fnn_dir / 'train_context_library.jsonl', context_library)
        metadata = {
            'family': family,
            'method': 'sequential_fnn',
            'feature_spec': {
                'name': 'product_fp_plus_delta_fp',
                'fingerprint': 'Morgan',
                'radius': 2,
                'fp_size': 4096,
                'input_dim': 8192,
            },
            'max_reagents': 3,
            'max_solvents': 2,
            'route_cache': str(route_cache),
            'validation_route_cache': str(validation_route_cache) if validation_route_cache else None,
            'counts': {
                'train_routes': len(train_entries),
                'val_routes': len(val_entries),
                'val_samples': len(val_manifest),
                'val_stage1_routes': len(val_stage1_routes),
                'test_stage1_routes': len(test_stage1_routes),
                'test_samples': len(test_manifest),
                'train_contexts': len(context_library),
            },
        }
        write_json(fnn_dir / 'metadata.json', metadata)
        results['sequential_fnn'] = metadata

    if 'reaction_gcnn' in requested_models:
        gcnn_dir = output_root / 'reaction_gcnn' / family
        _require_empty_directory(gcnn_dir)
        write_json(gcnn_dir / 'label_vocabulary.json', vocabulary)
        gcnn_fields = [
            'sample_index',
            'reaction_id',
            'family',
            'Reactant',
            'Product',
            'reagent_ids',
            'solvent_ids',
            'temperature_mean',
        ]
        write_csv(gcnn_dir / 'train.csv', _gcnn_rows(train_entries), gcnn_fields)
        write_csv(gcnn_dir / 'val.csv', _gcnn_rows(val_entries), gcnn_fields)
        write_csv(gcnn_dir / 'test_stage1_routes.csv', test_stage1_routes, ROUTE_FIELDS)
        # Keep the upstream-style CSVs and emit the shared JSONL contract used
        # by the PyTorch adapter for training and Stage 1-only inference.
        write_jsonl(gcnn_dir / 'train_routes.jsonl', train_entries)
        write_jsonl(gcnn_dir / 'val_routes.jsonl', val_entries)
        write_jsonl(gcnn_dir / 'test_stage1_routes.jsonl', test_stage1_routes)
        write_jsonl(gcnn_dir / 'val_manifest.jsonl', val_manifest)
        write_jsonl(gcnn_dir / 'val_stage1_routes.jsonl', val_stage1_routes)
        write_jsonl(gcnn_dir / 'test_manifest.jsonl', test_manifest)
        write_jsonl(gcnn_dir / 'train_context_library.jsonl', context_library)
        metadata = {
            'family': family,
            'method': 'reaction_gcnn',
            'graph_inputs': ['Reactant', 'Product'],
            'label_heads': ['reagent_ids', 'solvent_ids'],
            'adapter_inputs': ['train_routes.jsonl', 'val_routes.jsonl', 'test_stage1_routes.jsonl'],
            'route_cache': str(route_cache),
            'validation_route_cache': str(validation_route_cache) if validation_route_cache else None,
            'counts': {
                'train_routes': len(train_entries),
                'val_routes': len(val_entries),
                'val_samples': len(val_manifest),
                'val_stage1_routes': len(val_stage1_routes),
                'test_stage1_routes': len(test_stage1_routes),
                'test_samples': len(test_manifest),
                'train_contexts': len(context_library),
            },
        }
        write_json(gcnn_dir / 'metadata.json', metadata)
        results['reaction_gcnn'] = metadata

    return results


def _build_retro_pretraining_data(repo_root: Path, output_root: Path) -> dict[str, Any]:
    """Prepare product-to-reactant pretraining data from the same base route corpus as Stage 1."""

    output_dir = output_root / 'molecular_transformer' / 'base_retro_pretrain'
    _require_empty_directory(output_dir)
    raw_dir = repo_root / 'data' / 'editretro' / 'datasets' / 'USPTO_STAGE2_FILTERED' / 'raw'
    counts: dict[str, int] = {}
    for split, raw_name in (('train', 'raw_train.csv'), ('val', 'raw_val.csv')):
        source_lines: list[str] = []
        target_lines: list[str] = []
        with (raw_dir / raw_name).open('r', encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                reaction = str(row.get('reactants>reagents>production', ''))
                parts = reaction.split('>')
                if len(parts) != 3:
                    continue
                reactants = canonicalize_without_atom_maps(parts[0], sort_fragments=True)
                product = canonicalize_without_atom_maps(parts[2], sort_fragments=False)
                if not reactants or not product:
                    continue
                source_lines.append('<PRODUCT> ' + ' '.join(tokenize_smiles(product)))
                target_lines.append(
                    '<REACTANT> ' + ' '.join(tokenize_smiles(reactants)) + ' <REAGENT> R_NONE <SOLVENT> S_NONE'
                )
        _write_lines(output_dir / f'{split}.src', source_lines)
        _write_lines(output_dir / f'{split}.tgt', target_lines)
        counts[split] = len(source_lines)
    metadata = {
        'method': 'molecular_transformer_base_retro_pretrain',
        'source_dataset': str(raw_dir),
        'target_format': '<REACTANT> reactants <REAGENT> R_NONE <SOLVENT> S_NONE',
        'counts': counts,
    }
    write_json(output_dir / 'metadata.json', metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=project_root())
    parser.add_argument('--output-root', type=Path, default=Path('outputs/baseline_inputs'))
    parser.add_argument('--route-root', type=Path, default=Path('outputs/stage1_routes'))
    parser.add_argument(
        '--validation-route-root',
        type=Path,
        default=None,
        help='optional validation-only Stage 1 route-cache root for fusion selection',
    )
    parser.add_argument('--families', default='all')
    parser.add_argument('--models', default='all')
    parser.add_argument('--include-retro-pretrain', action='store_true')
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    route_root = args.route_root if args.route_root.is_absolute() else repo_root / args.route_root
    validation_route_root = (
        None
        if args.validation_route_root is None
        else (
            args.validation_route_root
            if args.validation_route_root.is_absolute()
            else repo_root / args.validation_route_root
        )
    )
    families = parse_families_arg(args.families)
    models = _parse_models(args.models)

    summary: dict[str, Any] = {'families': {}, 'models': models}
    for family in families:
        family_summary: dict[str, Any] = {}
        if 'molecular_transformer' in models:
            family_summary['molecular_transformer'] = _build_molecular_transformer_family(repo_root, output_root, family)
        family_summary.update(_build_indirect_fnn_and_gcnn(repo_root, route_root, validation_route_root, output_root, family, models))
        summary['families'][family] = family_summary
    if args.include_retro_pretrain and 'molecular_transformer' in models:
        summary['base_retro_pretrain'] = _build_retro_pretraining_data(repo_root, output_root)
    write_json(output_root / 'build_summary.json', summary)


if __name__ == '__main__':
    main()
