"""Evaluate external-baseline candidates with a fixed Non-Oracle denominator.

The shared ProSys evaluator scores only slates present in a candidate table.
This adapter additionally consumes the full test identity manifest, so samples
without a Stage 1 route are retained as zero-hit slates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from prosys_shared.mainline import build_candidate_training_table, evaluate_scored_frame

from .contracts import read_jsonl, write_json


DEFAULT_TOPKS = (1, 3, 5, 10)


def _manifest_sample_indices(path: Path) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for row in read_jsonl(path):
        try:
            sample_index = int(row['sample_index'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Manifest row in {path} has no integer sample_index.') from exc
        if sample_index in seen:
            raise ValueError(f'Duplicate sample_index={sample_index} in manifest {path}.')
        seen.add(sample_index)
        indices.append(sample_index)
    if not indices:
        raise ValueError(f'Test manifest {path} is empty.')
    return indices


def _empty_metrics(topks: tuple[int, ...]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        'num_slates': 0,
        'covered_slates': 0,
        'route_covered_slates': 0,
        'context_covered_slates': 0,
        'pool_coverage': 0.0,
        'pool_route_coverage': 0.0,
        'pool_context_coverage': 0.0,
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


def evaluate_with_manifest(
    frame: pd.DataFrame,
    *,
    expected_sample_indices: Iterable[int],
    score_column: str,
    temperature_column: str | None,
    topks: tuple[int, ...] = DEFAULT_TOPKS,
) -> dict[str, Any]:
    """Evaluate candidate slates while assigning zero score to absent samples."""

    expected = list(dict.fromkeys(int(value) for value in expected_sample_indices))
    if not expected:
        raise ValueError('The expected sample-index collection is empty.')
    expected_set = set(expected)
    observed = (
        {int(value) for value in frame['sample_index'].dropna().tolist()}
        if not frame.empty
        else set()
    )
    unknown = sorted(observed - expected_set)
    if unknown:
        raise ValueError(
            f'Candidate table contains {len(unknown)} sample indices absent from the test manifest: '
            f'{unknown[:5]}'
        )

    base = (
        evaluate_scored_frame(
            frame,
            score_column=score_column,
            temperature_column=temperature_column,
            topks=topks,
        )
        if not frame.empty
        else _empty_metrics(topks)
    )
    candidate_slates = int(base['num_slates'])
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
        metrics[coverage_key] = float(metrics[count_key]) / expected_count
    metrics['system_mrr'] = float(base['system_mrr']) * scale
    metrics['system_ndcg10'] = float(base['system_ndcg10']) * scale
    for k in topks:
        for prefix in ('system', 'context', 'route'):
            key = f'{prefix}_top{k}_all'
            metrics[key] = float(base[key]) * scale
    return metrics


def label_and_evaluate(
    *,
    candidate_file: Path,
    gold_split_file: Path,
    test_manifest_file: Path,
    labeled_output_file: Path,
    score_column: str,
    temperature_column: str | None,
    topks: tuple[int, ...],
) -> dict[str, Any]:
    """Label a candidate table with gold systems and evaluate it against a manifest."""

    frame = build_candidate_training_table(candidate_file, gold_split_file)
    labeled_output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(labeled_output_file, index=False)
    return evaluate_with_manifest(
        frame,
        expected_sample_indices=_manifest_sample_indices(test_manifest_file),
        score_column=score_column,
        temperature_column=temperature_column,
        topks=topks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--gold-split', type=Path, required=True)
    parser.add_argument('--test-manifest', type=Path, required=True)
    parser.add_argument('--labeled-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-column', default='system_score')
    parser.add_argument('--temperature-column', default='')
    parser.add_argument('--topks', type=int, nargs='+', default=list(DEFAULT_TOPKS))
    args = parser.parse_args()

    topks = tuple(args.topks)
    temperature_column = args.temperature_column.strip() or None
    metrics = label_and_evaluate(
        candidate_file=args.candidates,
        gold_split_file=args.gold_split,
        test_manifest_file=args.test_manifest,
        labeled_output_file=args.labeled_output,
        score_column=args.score_column,
        temperature_column=temperature_column,
        topks=topks,
    )
    payload = {
        'candidates': str(args.candidates),
        'gold_split': str(args.gold_split),
        'test_manifest': str(args.test_manifest),
        'labeled_output': str(args.labeled_output),
        'metrics': metrics,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
