"""Independently audit outputs from the maintained current-mainline ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.mainline import display_family_name, parse_families_arg
from prosys_shared.route_cache import load_route_cache_sample_indices


TARGET_COLUMNS = {
    'label',
    'route_match',
    'context_match',
    'rank_relevance',
    'sample_weight',
    'temperature_gold',
    'yield_gold',
}

METHODS = (
    'full_mainline',
    'knn_only_xgb',
    'frequency_top20_xgb',
    'no_gnn_xgb',
    'no_stage3',
    'knn_only_no_gnn_xgb',
)

SCORE_METRICS = (
    'pool_coverage',
    'system_top1_all',
    'system_top3_all',
    'system_top5_all',
    'system_top10_all',
    'system_mrr',
    'system_ndcg10',
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _result_path(output_root: Path, family: str, method: str) -> Path:
    return output_root / family / method / 'non_oracle' / 'result.json'


def _metadata_feature_check(result: dict) -> dict:
    artifacts = result.get('artifacts') or result.get('model') or {}
    metadata_file = artifacts.get('metadata_file')
    if not metadata_file:
        return {'checked': False, 'reason': 'no_xgboost_metadata'}
    path = Path(metadata_file)
    if not path.exists():
        return {'checked': False, 'reason': f'missing_metadata:{path}'}
    metadata = _read_json(path)
    features = [str(value) for value in metadata.get('feature_columns', [])]
    leaked = sorted(TARGET_COLUMNS.intersection(features))
    return {
        'checked': True,
        'feature_count': len(features),
        'leaked_target_columns': leaked,
        'reaction_gnn_feature_count': sum(column.startswith('route_gnn_feat_') for column in features),
    }


def _audit_result(result_path: Path, expected_indices: set[int], max_contexts: int) -> dict:
    result = _read_json(result_path)
    metrics = result.get('metrics') or {}
    budget = result.get('candidate_budget') or {}
    observed_n = int(metrics.get('num_slates', -1))
    candidate_slates = int(metrics.get('candidate_slates', -1))
    missing = int(metrics.get('missing_candidate_slates', -1))
    feature_check = _metadata_feature_check(result)
    checks = {
        'full_manifest_denominator': observed_n == len(expected_indices),
        'candidate_partition': candidate_slates + missing == len(expected_indices),
        'context_cap': int(budget.get('max_contexts_per_route', 0)) <= max_contexts,
        'no_target_feature_leakage': not feature_check.get('leaked_target_columns', []),
    }
    return {
        'family': str(result.get('family')),
        'method': str(result.get('method')),
        'result_file': str(result_path),
        'method_label': result.get('method_label'),
        'test_manifest_samples': len(expected_indices),
        'num_slates': observed_n,
        'candidate_slates': candidate_slates,
        'missing_candidate_slates': missing,
        'max_contexts_per_route': int(budget.get('max_contexts_per_route', 0)),
        'feature_check': feature_check,
        'checks': checks,
    }


def _compare_full_reference(
    *,
    output_root: Path,
    mainline_root: Path,
    family: str,
) -> dict:
    ablation_result = _read_json(_result_path(output_root, family, 'full_mainline'))
    mainline_path = mainline_root / family / 'knn_xgb' / 'non_oracle' / 'result.json'
    if not mainline_path.exists():
        return {'checked': False, 'reason': f'missing_mainline_result:{mainline_path}'}
    mainline_result = _read_json(mainline_path)
    differences = {}
    for name in SCORE_METRICS:
        ablation_value = float((ablation_result.get('metrics') or {}).get(name, float('nan')))
        mainline_value = float((mainline_result.get('metrics') or {}).get(name, float('nan')))
        differences[name] = abs(ablation_value - mainline_value)
    return {
        'checked': True,
        'mainline_result_file': str(mainline_path),
        'max_abs_metric_difference': max(differences.values(), default=0.0),
        'metric_differences': differences,
        'matches': max(differences.values(), default=0.0) <= 1e-12,
    }


def _render_markdown(audit: dict) -> str:
    lines = [
        '# Current-Mainline Ablation Audit',
        '',
        'The audit checks full-manifest denominators, candidate caps, XGBoost target-feature exclusion, and equality between the reported full ablation reference and the rebuilt mainline result.',
        '',
        f"Overall status: **{'PASS' if audit['passed'] else 'FAIL'}**",
        '',
        '| Family | Method | Manifest | Candidate partition | Context cap | Target leakage | XGB features | GNN features |',
        '| --- | --- | --- | --- | --- | --- | ---: | ---: |',
    ]
    for row in audit['rows']:
        checks = row['checks']
        features = row['feature_check']
        lines.append(
            '| ' + ' | '.join([
                display_family_name(row['family']),
                str(row['method']),
                'PASS' if checks['full_manifest_denominator'] else 'FAIL',
                'PASS' if checks['candidate_partition'] else 'FAIL',
                'PASS' if checks['context_cap'] else 'FAIL',
                'PASS' if checks['no_target_feature_leakage'] else 'FAIL',
                str(features.get('feature_count', 'N/A')),
                str(features.get('reaction_gnn_feature_count', 'N/A')),
            ]) + ' |'
        )

    lines.extend([
        '',
        '## Full Reference Match',
        '',
        '| Family | Match | Maximum absolute metric difference |',
        '| --- | --- | ---: |',
    ])
    for row in audit['full_reference_checks']:
        status = 'PASS' if row.get('matches') else 'FAIL'
        maximum = row.get('max_abs_metric_difference')
        rendered = f'{maximum:.3g}' if maximum is not None else 'N/A'
        lines.append(f"| {display_family_name(row['family'])} | {status} | {rendered} |")

    lines.extend([
        '',
        '## Protocol Notes',
        '',
        '- Training KNN pools use leave-one-reaction-out retrieval and query-adjusted global condition statistics.',
        '- Test metrics retain every Stage 1 cache identity, including products without a candidate slate.',
        '- Temperature metrics remain conditional on a highest-ranked exact route-and-condition match with a valid label.',
        '',
    ])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit current-mainline ProSys ablation outputs.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--output_root', type=str, default='outputs/ablation_reafnn_gnn_20260726')
    parser.add_argument('--mainline_root', type=str, default='outputs/stage23_mainline_reafnn_gnn_fused_20260723')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--max_contexts', type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    mainline_root = (repo_root / args.mainline_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    families = parse_families_arg(args.families)

    rows = []
    full_reference_checks = []
    for family in families:
        route_cache = route_root / family / 'route_cache.json'
        expected_indices = set(load_route_cache_sample_indices(route_cache))
        for method in METHODS:
            path = _result_path(output_root, family, method)
            if not path.exists():
                raise FileNotFoundError(f'Missing ablation result: {path}')
            rows.append(_audit_result(path, expected_indices, int(args.max_contexts)))
        reference = _compare_full_reference(
            output_root=output_root,
            mainline_root=mainline_root,
            family=family,
        )
        reference['family'] = family
        full_reference_checks.append(reference)

    passed = all(all(bool(value) for value in row['checks'].values()) for row in rows)
    passed = passed and all(bool(row.get('matches')) for row in full_reference_checks)
    audit = {
        'families': families,
        'max_contexts': int(args.max_contexts),
        'rows': rows,
        'full_reference_checks': full_reference_checks,
        'passed': bool(passed),
    }
    _write_json(audit, output_root / 'audit.json')
    (output_root / 'audit.md').write_text(_render_markdown(audit), encoding='utf-8')
    print(f"[ablation-audit] {'PASS' if passed else 'FAIL'}: {output_root / 'audit.md'}")


if __name__ == '__main__':
    main()
