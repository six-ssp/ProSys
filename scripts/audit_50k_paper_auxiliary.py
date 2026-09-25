#!/usr/bin/env python3
"""Replay direct Condition@k and validation fusion, without fitting models."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from prosys_shared.cache_integrity import file_sha256 as sha
from prosys_shared.mainline import (FAMILY_ORDER, canonicalize_smiles,
    canonicalize_reaction_side, load_gold_condition_index, split_file_for_family)


def condition_metrics(queries, predictions, gold):
    """Retain every query, including those without any Stage 1 route."""
    identity = lambda q: (int(q['sample_index']), str(q['reaction_id']), q['product'])
    expected = [identity(q) for q in queries]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Empty or duplicate query manifest')
    if [identity(q) for q in predictions] != expected:
        raise ValueError('Prediction identity/order differs from query manifest')
    first = []
    for query, prediction in zip(queries, predictions):
        key = (str(query['reaction_id']), canonicalize_smiles(query['product']))
        if key not in gold:
            raise ValueError('Query has no matching reference bucket')
        contexts = prediction['contexts']
        if [c['context_rank'] for c in contexts] != list(range(1, len(contexts) + 1)):
            raise ValueError('Noncontiguous condition ranks')
        pairs = [(c['reagent_norm'], c['solvent_norm']) for c in contexts]
        if len(pairs) != len(set(pairs)) or len(pairs) > 20:
            raise ValueError('Duplicate contexts or violated per-route budget')
        first.append(next((i for i, pair in enumerate(pairs, 1) if pair in gold[key]), np.inf))
    values = np.asarray(first)
    return {'num_product_queries': len(expected),
        'condition_pool_coverage': float(np.isfinite(values).mean()),
        **{f'condition_top{k}_all': float((values <= k).mean()) for k in (1, 3, 5, 10)}}


def selected_weight(grid, coverages):
    if (len(grid) != len(coverages) or not grid or len(set(grid)) != len(grid)
            or not {0., 1.}.issubset(grid) or not np.isfinite(grid + coverages).all()
            or any(x < 0 or x > 1 for x in grid + coverages)):
        raise ValueError('Invalid fusion grid/coverage')
    best = max(coverages)
    return max(w for w, c in zip(grid, coverages) if abs(c - best) <= 1e-12)


def fusion_replay(builder, records, queries, gold, calibration):
    ids = [int(q['sample_index']) for q in queries]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Invalid validation query manifest')
    grid = calibration['protocol']['weight_grid']
    hit_sets = [set() for _ in grid]
    matchable = 0
    for route in records:
        if route.sample_index not in ids:
            raise ValueError('Route outside validation manifest')
        key = (str(route.reaction_id), canonicalize_smiles(route.product))
        route_key = canonicalize_reaction_side(route.reactants)
        bucket = gold.get(key)
        if bucket is None or route_key not in bucket.route_keys:
            continue
        matchable += 1
        state = builder._independent_post_fusion_state(route, leave_one_reaction_out=False)
        for weight, hits in zip(grid, hit_sets):
            candidates = builder._select_independent_post_fusion_contexts(state, knn_weight=weight)
            if any((route_key, c['reagent_norm'], c['solvent_norm']) in bucket.exact_keys for c in candidates):
                hits.add(route.sample_index)
    coverage = [len(hits) / len(ids) for hits in hit_sets]
    chosen = selected_weight(grid, coverage)
    expected_rows = calibration['weight_candidates']
    if [r['knn_weight'] for r in expected_rows] != grid:
        raise ValueError('Calibration grid differs')
    for row, hits, value in zip(expected_rows, hit_sets, coverage):
        if (row['validation_sample_identities'] != len(ids)
                or row['validation_exact_candidate_hits'] != len(hits)
                or row['validation_route_records'] != len(records)
                or row['validation_matchable_route_records'] != matchable):
            raise ValueError('Validation fusion counts differ')
        np.testing.assert_allclose(row['validation_stage2_candidate_coverage'], value, rtol=0, atol=1e-12)
    if chosen != calibration['selected_knn_weight']:
        raise ValueError('Validation-selected weight differs')
    np.testing.assert_allclose(calibration['selected_reafnn_weight'], 1 - chosen, rtol=0, atol=1e-12)
    np.testing.assert_allclose(calibration['validation_stage2_candidate_coverage'], max(coverage), rtol=0, atol=1e-12)
    return {'n_validation_queries': len(ids), 'validation_routes': len(records),
        'selected_knn_weight': chosen, 'selected_reafnn_weight': 1 - chosen,
        'reafnn_only_coverage': coverage[grid.index(0.)],
        'knn_only_coverage': coverage[grid.index(1.)], 'fused_coverage': max(coverage)}


def main():
    from scripts.summarize_50k_downstream_evidence import verify_run
    from scripts.predict_product import readonly_builder
    from prosys_shared.route_cache import load_route_records_from_cache
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mainline', type=Path, required=True)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--routes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', nargs='+', choices=FAMILY_ORDER, default=FAMILY_ORDER)
    args = parser.parse_args()
    if len(args.families) != len(set(args.families)):
        raise ValueError('Duplicate requested families')
    output = args.output.resolve()
    if output.exists() or any(output == p.resolve() or p.resolve() in output.parents
            or output in p.resolve().parents for p in (args.mainline, args.baselines, args.routes, ROOT / 'data')):
        raise ValueError('Require a new output directory separate from inputs')
    for family in args.families:
        if not (args.baselines / family / 'independent_replay.json').exists():
            raise ValueError('Missing completed baseline audit: ' + family)
        for seed in (0, 1, 2):
            if not (args.mainline / 'compact' / f'seed_{seed}' / family / 'evidence_manifest.json').exists():
                raise ValueError('Missing completed mainline evidence: ' + family)
    output.mkdir(parents=True)
    bindings = {str(Path(__file__).resolve()): sha(Path(__file__))}
    def track(path):
        path = Path(path).resolve()
        digest = sha(path)
        if str(path) in bindings and bindings[str(path)] != digest:
            raise ValueError('Input changed during read: ' + str(path))
        bindings[str(path)] = digest
        return path
    def read(path):
        return json.loads(track(path).read_text())
    conditions, fusion, receipts = [], [], []
    for family in args.families:
        baseline = args.baselines / family
        with (output / (family + '_baseline_audit.log')).open('x') as log:
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/audit_50k_baseline_evidence.py'),
                '--study', str(baseline), '--route-study', str(args.routes)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        read(baseline / 'independent_replay.json')
        jobs = [('product_naive_bayes', 0, baseline / 'deterministic_b1/product_naive_bayes' / family)]
        jobs += [('product_gnn', s, baseline / f'seed_{s}/product_gnn/product_gnn' / family) for s in (0, 1, 2)]
        for method, seed, folder in jobs:
            metadata = read(folder / 'run_metadata.json')
            for split, label in (('val', 'validation'), ('test', 'test')):
                queries = read(args.routes / split / family / 'route_cache.json')['reactions']
                path = track(folder / (label + '_condition_predictions.jsonl'))
                predictions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                gold = load_gold_condition_index(track(split_file_for_family(ROOT, family, split)))
                metrics = condition_metrics(queries, predictions, {k: set(v.context_keys) for k, v in gold.items()})
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, metadata[label + '_condition_metrics'][key], rtol=0, atol=1e-12)
                conditions.append({'family': family, 'method': method, 'seed': seed, 'split': split, **metrics})
        for seed in (0, 1, 2):
            _, receipt = verify_run(args.mainline.resolve(), args.routes.resolve(), family, seed)
            receipts.append(receipt)
            folder = args.mainline / 'compact' / f'seed_{seed}' / family
            track(folder / 'evidence_manifest.json')
            calibration = read(folder / 'bundle/reafnn/post_fusion_calibration.json')
            builder = readonly_builder(ROOT, family, folder, 'cpu')
            cache = track(args.routes / 'val' / family / 'route_cache.json')
            queries = read(cache)['reactions']
            records = load_route_records_from_cache(cache, family)
            gold = load_gold_condition_index(track(split_file_for_family(ROOT, family, 'val')))
            values = fusion_replay(builder, records, queries, gold, calibration)
            fusion.append({'family': family, 'seed': seed, **values})
            print('Verified auxiliary fields', family, seed, flush=True)
    if any(sha(Path(p)) != h for p, h in bindings.items()):
        raise ValueError('Tracked input changed during auxiliary replay')
    pd.DataFrame(conditions).to_csv(output / 'direct_condition_metrics.csv', index=False)
    pd.DataFrame(fusion).to_csv(output / 'validation_fusion.csv', index=False)
    report = {'pass': True, 'families': args.families,
        'six_family_scope': set(args.families) == set(FAMILY_ORDER),
        'mainline_receipts': receipts, 'input_sha256': bindings,
        'output_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        'scope': 'Condition@k independently recounted before route pairing; fusion replay on validation only, without fitting',
        'limits': ['Fusion replay reuses retained branch scoring/selection primitives.',
            'Per-family/seed rows only; average families equally before seed statistics.',
            'Not six-family evidence unless six_family_scope is true.']}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
