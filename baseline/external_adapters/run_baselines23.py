"""Run the formal Non-Oracle Sequential FNN and Reaction-GCNN baselines.

The runner trains on family-specific train rows, uses gold-route validation rows
only for early stopping, chooses the route/condition fusion ratio on a separate
validation-only Stage 1 cache, and evaluates the untouched test cache once.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from prosys_shared.mainline import build_candidate_training_table, parse_families_arg, split_file_for_family

from .contracts import project_root, read_jsonl, write_json
from .evaluate_candidates import evaluate_with_manifest, label_and_evaluate
from .export_candidates import export_candidates
from . import run_reaction_gcnn, run_sequential_fnn


METHODS = ('sequential_fnn', 'reaction_gcnn')
DEFAULT_ROUTE_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
CONDITION_WEIGHT = 1.0
TOPKS = (1, 3, 5, 10)


def _parse_methods(value: str) -> list[str]:
    if value.strip().lower() == 'all':
        return list(METHODS)
    methods = [item.strip() for item in value.replace(',', ' ').split() if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f'Unsupported method(s): {unknown}')
    return methods


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _manifest_indices(path: Path) -> list[int]:
    values = [int(row['sample_index']) for row in read_jsonl(path)]
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate sample indices in {path}.')
    if not values:
        raise ValueError(f'Empty manifest: {path}.')
    return values


def _select_fusion(
    frame: pd.DataFrame,
    manifest_file: Path,
    *,
    route_weights: tuple[float, ...],
    temperature_column: str | None,
) -> dict[str, Any]:
    """Choose the route-score ratio using validation full-system Top-10 accuracy, then full-system Top-1 accuracy."""

    required = {'route_score_z', 'condition_score_z'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Validation candidate table misses score columns: {missing}')
    expected_indices = _manifest_indices(manifest_file)
    records: list[dict[str, Any]] = []
    for route_weight in route_weights:
        frame['system_score'] = (
            float(route_weight) * frame['route_score_z'].astype(float)
            + CONDITION_WEIGHT * frame['condition_score_z'].astype(float)
        )
        metrics = evaluate_with_manifest(
            frame,
            expected_sample_indices=expected_indices,
            score_column='system_score',
            temperature_column=temperature_column,
            topks=TOPKS,
        )
        records.append(
            {
                'route_weight': float(route_weight),
                'condition_weight': CONDITION_WEIGHT,
                'metrics': metrics,
            }
        )
    selected = sorted(
        records,
        key=lambda item: (
            -float(item['metrics']['system_top10_all']),
            -float(item['metrics']['system_top1_all']),
            float(item['route_weight']),
        ),
    )[0]
    return {
        'selection_rule': 'max validation full-system Top-10 accuracy, then full-system Top-1 accuracy, then smaller route weight',
        'condition_weight': CONDITION_WEIGHT,
        'candidates': records,
        'selected': selected,
    }


def _train_and_predict(
    *,
    method: str,
    input_dir: Path,
    family_root: Path,
    device: str,
    max_epochs: int,
    patience: int,
    seed: int,
    top_contexts: int,
) -> tuple[dict[str, Any], Path, Path]:
    artifact_dir = family_root / 'artifacts'
    validation_prediction = family_root / 'validation_predictions.jsonl'
    test_prediction = family_root / 'test_predictions.jsonl'
    validation_routes = input_dir / 'val_stage1_routes.jsonl'
    test_routes = input_dir / 'test_stage1_routes.jsonl'

    if not validation_routes.exists() or not test_routes.exists():
        raise FileNotFoundError(f'Missing Stage 1 route package in {input_dir}.')

    if method == 'sequential_fnn':
        config = run_sequential_fnn.SequentialFNNConfig(
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            random_state=seed,
        )
        train_metadata = run_sequential_fnn.train(input_dir, artifact_dir, config)
        run_sequential_fnn.predict(
            input_dir,
            artifact_dir,
            validation_prediction,
            top_contexts=top_contexts,
            device_name=device,
            routes_file=validation_routes,
        )
        run_sequential_fnn.predict(
            input_dir,
            artifact_dir,
            test_prediction,
            top_contexts=top_contexts,
            device_name=device,
            routes_file=test_routes,
        )
    elif method == 'reaction_gcnn':
        config = run_reaction_gcnn.ReactionGCNNConfig(
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            random_state=seed,
        )
        train_metadata = run_reaction_gcnn.train(input_dir, artifact_dir, config)
        run_reaction_gcnn.predict(
            input_dir,
            artifact_dir,
            validation_prediction,
            top_contexts=top_contexts,
            device_name=device,
            routes_file=validation_routes,
        )
        run_reaction_gcnn.predict(
            input_dir,
            artifact_dir,
            test_prediction,
            top_contexts=top_contexts,
            device_name=device,
            routes_file=test_routes,
        )
    else:
        raise ValueError(f'Unsupported method: {method}')
    return train_metadata, validation_prediction, test_prediction


def _export(
    *,
    method: str,
    predictions: Path,
    input_dir: Path,
    route_manifest: Path,
    output: Path,
    route_weight: float,
    top_contexts: int,
) -> int:
    return export_candidates(
        method=method,
        predictions_file=predictions,
        vocabulary_file=input_dir / 'label_vocabulary.json',
        route_manifest=route_manifest,
        output_file=output,
        max_contexts_per_route=top_contexts,
        route_weight=route_weight,
        condition_weight=CONDITION_WEIGHT,
    )


def _run_family(
    *,
    repo_root: Path,
    input_root: Path,
    output_root: Path,
    method: str,
    family: str,
    device: str,
    max_epochs: int,
    patience: int,
    seed: int,
    top_contexts: int,
    route_weights: tuple[float, ...],
    resume: bool,
) -> dict[str, Any]:
    input_dir = input_root / method / family
    family_root = output_root / method / family
    result_file = family_root / 'run_metadata.json'
    if result_file.exists() and resume:
        return json.loads(result_file.read_text(encoding='utf-8'))
    if family_root.exists() and any(family_root.iterdir()) and not resume:
        raise FileExistsError(f'Refusing to overwrite existing run directory: {family_root}')
    family_root.mkdir(parents=True, exist_ok=True)

    train_metadata, validation_prediction, test_prediction = _train_and_predict(
        method=method,
        input_dir=input_dir,
        family_root=family_root,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        top_contexts=top_contexts,
    )

    validation_candidate = family_root / 'validation_candidates.csv'
    _export(
        method=method,
        predictions=validation_prediction,
        input_dir=input_dir,
        route_manifest=input_dir / 'val_stage1_routes.jsonl',
        output=validation_candidate,
        route_weight=0.0,
        top_contexts=top_contexts,
    )
    validation_frame = build_candidate_training_table(
        validation_candidate,
        split_file_for_family(repo_root, family, 'val'),
    )
    temperature_column = 'temperature_pred' if method == 'sequential_fnn' else None
    fusion = _select_fusion(
        validation_frame,
        input_dir / 'val_manifest.jsonl',
        route_weights=route_weights,
        temperature_column=temperature_column,
    )
    write_json(family_root / 'fusion_selection.json', fusion)

    selected_route_weight = float(fusion['selected']['route_weight'])
    test_candidate = family_root / 'test_candidates.csv'
    test_candidate_count = _export(
        method=method,
        predictions=test_prediction,
        input_dir=input_dir,
        route_manifest=input_dir / 'test_stage1_routes.jsonl',
        output=test_candidate,
        route_weight=selected_route_weight,
        top_contexts=top_contexts,
    )
    test_metrics = label_and_evaluate(
        candidate_file=test_candidate,
        gold_split_file=split_file_for_family(repo_root, family, 'test'),
        test_manifest_file=input_dir / 'test_manifest.jsonl',
        labeled_output_file=family_root / 'test_labeled_candidates.csv',
        score_column='system_score',
        temperature_column=temperature_column,
        topks=TOPKS,
    )
    result = {
        'family': family,
        'method': method,
        'input_dir': str(input_dir),
        'train_metadata': train_metadata,
        'validation_prediction_routes': sum(1 for _ in read_jsonl(validation_prediction)),
        'test_prediction_routes': sum(1 for _ in read_jsonl(test_prediction)),
        'test_candidate_rows': int(test_candidate_count),
        'fusion': fusion,
        'test_metrics': test_metrics,
        'temperature_column': temperature_column,
    }
    write_json(result_file, result)
    return result


def _flat_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record['test_metrics']
        temperature = metrics.get('temperature', {})
        rows.append(
            {
                'method': record['method'],
                'family': record['family'],
                'candidate_slates': metrics.get('candidate_slates'),
                'missing_candidate_slates': metrics.get('missing_candidate_slates'),
                'cover': metrics.get('pool_coverage'),
                'sys1': metrics.get('system_top1_all'),
                'sys3': metrics.get('system_top3_all'),
                'sys5': metrics.get('system_top5_all'),
                'sys10': metrics.get('system_top10_all'),
                'mrr': metrics.get('system_mrr'),
                'ndcg10': metrics.get('system_ndcg10'),
                'temperature_n': temperature.get('n'),
                'temperature_mae': temperature.get('mae'),
                'temperature_within_5c': temperature.get('within_5c'),
                'temperature_within_10c': temperature.get('within_10c'),
                'temperature_within_20c': temperature.get('within_20c'),
                'route_weight': record['fusion']['selected']['route_weight'],
                'condition_weight': record['fusion']['selected']['condition_weight'],
                'last_epoch': record['train_metadata']['last_epoch'],
                'n_families': 1,
            }
        )
    return rows


def _macro_average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append one equal-family aggregate row for each baseline method."""

    mean_fields = (
        'cover',
        'sys1',
        'sys3',
        'sys5',
        'sys10',
        'mrr',
        'ndcg10',
        'temperature_mae',
        'temperature_within_5c',
        'temperature_within_10c',
        'temperature_within_20c',
    )
    count_fields = ('candidate_slates', 'missing_candidate_slates', 'temperature_n')
    aggregates: list[dict[str, Any]] = []
    for method in sorted({str(row['method']) for row in rows}):
        method_rows = [row for row in rows if row['method'] == method]
        aggregate = {field: None for field in method_rows[0]}
        aggregate.update({'method': method, 'family': 'MACRO-AVG', 'n_families': len(method_rows)})
        for field in mean_fields:
            values = [float(row[field]) for row in method_rows if row.get(field) is not None]
            aggregate[field] = sum(values) / len(values) if values else None
        for field in count_fields:
            values = [float(row[field]) for row in method_rows if row.get(field) is not None]
            aggregate[field] = int(sum(values)) if values else 0
        aggregates.append(aggregate)
    return rows + aggregates


def _write_summary(output_root: Path, records: list[dict[str, Any]]) -> None:
    rows = _macro_average_rows(_flat_rows(records))
    fieldnames = list(rows[0]) if rows else ['method', 'family']
    with (output_root / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_root / 'summary.json', {'records': records, 'rows': rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=project_root())
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--families', default='all')
    parser.add_argument('--methods', default='all')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--top-contexts', type=int, default=20)
    parser.add_argument('--route-weights', type=float, nargs='+', default=list(DEFAULT_ROUTE_WEIGHTS))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    input_root = _resolve(repo_root, args.input_root)
    output_root = _resolve(repo_root, args.output_root)
    families = parse_families_arg(args.families)
    methods = _parse_methods(args.methods)
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f'Output root is not empty: {output_root}')
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / 'run_config.json',
        {
            'families': families,
            'methods': methods,
            'device': args.device,
            'max_epochs': args.max_epochs,
            'patience': args.patience,
            'seed': args.seed,
            'top_contexts': args.top_contexts,
            'route_weights': args.route_weights,
            'condition_weight': CONDITION_WEIGHT,
            'protocol': 'validation-only fusion selection; untouched fixed-manifest test evaluation',
        },
    )

    records: list[dict[str, Any]] = []
    for method in methods:
        for family in families:
            result = _run_family(
                repo_root=repo_root,
                input_root=input_root,
                output_root=output_root,
                method=method,
                family=family,
                device=args.device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                seed=args.seed,
                top_contexts=args.top_contexts,
                route_weights=tuple(args.route_weights),
                resume=args.resume,
            )
            records.append(result)
            _write_summary(output_root, records)
            print(
                json.dumps(
                    {
                        'method': method,
                        'family': family,
                        'system_top10': result['test_metrics']['system_top10_all'],
                        'candidate_slates': result['test_metrics']['candidate_slates'],
                    }
                )
            )


if __name__ == '__main__':
    main()
