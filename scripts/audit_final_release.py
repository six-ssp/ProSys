#!/usr/bin/env python3
"""Read-only replay of retained release evidence; write a separate audit receipt."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'Experiment/mainline_evidence_completion_20260913'
OUT = ROOT / 'Experiment/final_release_audit_20260915'


def read(path):
    return json.loads(path.read_text())


def replay(pair):
    import numpy as np
    import pandas as pd
    from prosys_shared.cache_integrity import file_sha256
    from prosys_shared.evidence import identity_hash, require_same_temperature_support
    from prosys_shared.mainline import evaluate_scored_frame_with_manifest, split_file_for_family
    from scripts.summarize_mainline_evidence import align_control_rows, flat

    family, seed = pair
    directory = STUDY / 'compact' / f'seed_{seed}' / family
    saved = read(directory / 'result.json')
    controls = read(directory / 'control_metrics.json')
    prior_audit = read(directory / 'exact_control_audit.json')
    provenance = read(directory / 'provenance.json')
    for split, expected in provenance['splits'].items():
        assert file_sha256(split_file_for_family(ROOT, family, split)) == expected, (family, seed, split)
    assert file_sha256(ROOT / 'outputs/stage1_routes_validation' / family / 'route_cache.json') == provenance['route_validation']
    for name, expected in provenance['models'].items():
        assert file_sha256(directory / 'bundle' / name) == expected, (family, seed, name)
    route_path = ROOT / 'outputs/stage1_routes' / family / 'route_cache.json'
    assert file_sha256(route_path) == provenance['route_test']
    manifest = [int(r['sample_index']) for r in read(route_path)['reactions']]
    frame = pd.read_csv(directory / 'full_candidates.csv.gz', float_precision='round_trip')
    assert identity_hash(frame) == prior_audit['candidate_identity_sha256']
    no_ltr = align_control_rows(frame, pd.read_csv(directory / 'no_ltr_scores.csv.gz', float_precision='round_trip'))
    no_graph = align_control_rows(frame, pd.read_csv(directory / 'no_graph_predictions.csv.gz', float_precision='round_trip'))
    np.testing.assert_array_equal(frame.xgb_score, no_graph.xgb_score)
    results = {}
    for arm, gold in [('full', saved['metrics']), ('no_ltr', controls['no_ltr']),
                      ('no_rgnn_temperature', controls['no_rgnn_temperature'])]:
        current = frame.copy()
        score = 'xgb_score'
        if arm == 'no_ltr':
            current['stage2_prior_score'] = no_ltr.stage2_prior_score.to_numpy()
            score = 'stage2_prior_score'
        elif arm == 'no_rgnn_temperature':
            current['xgb_temperature_pred'] = no_graph.xgb_temperature_pred.to_numpy()
            support = require_same_temperature_support(frame, current,
                left_column='xgb_temperature_pred', right_column='xgb_temperature_pred')
            assert support == prior_audit['temperature_support']
        metrics = flat(evaluate_scored_frame_with_manifest(current,
            expected_sample_indices=manifest, score_column=score,
            temperature_column='xgb_temperature_pred' if arm != 'no_ltr' else None))
        expected = flat(gold)
        keys = ['cover', 'sys1', 'sys3', 'sys5', 'sys10', 'mrr', 'ndcg10']
        if arm != 'no_ltr':
            keys += ['temp_n', 'temp_mae', 'temp_within_5c', 'temp_within_10c', 'temp_within_20c']
        for key in keys:
            np.testing.assert_allclose(metrics[key], expected[key], rtol=1e-6, atol=1e-6,
                err_msg=f'{family}/{seed}/{arm}/{key}')
        results[arm] = {key: metrics[key] for key in keys}
    return {'family': family, 'seed': seed, 'candidate_rows': len(frame),
        'query_count': len(manifest), 'model_hashes_pass': True,
        'training_validation_test_hashes_pass': True,
        'identity_and_replay_pass': True, 'temperature_support': support,
        'metrics': results}


def check_splits():
    from prosys_shared.mainline import FAMILY_ORDER, load_split_rows, split_file_for_family
    rows = []
    for family in FAMILY_ORDER:
        keys = {s: {(str(r['reaction_id']), r['product']) for r in
            load_split_rows(split_file_for_family(ROOT, family, s))} for s in ('train', 'val', 'test')}
        for split in ('val', 'test'):
            folder = 'stage1_routes_validation' if split == 'val' else 'stage1_routes'
            cache = read(ROOT / 'outputs' / folder / family / 'route_cache.json')
            identities = {(str(r['reaction_id']), r['product']) for r in cache['reactions']}
            assert identities == keys[split]
            assert len(identities) == len(cache['reactions'])
            other = {s: len(identities & k) for s, k in keys.items() if s != split}
            assert not any(other.values())
            rows.append({'family': family, 'split': split, 'queries': len(identities),
                'exact_split_membership_pass': True, 'other_split_identity_overlaps': other})
    return rows


def main():
    import pandas as pd
    from prosys_shared.mainline import FAMILY_ORDER
    from prosys_shared.cache_integrity import file_sha256

    OUT.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(ROOT).free > 5 * 2**30
    splits = check_splits()
    pairs = [(f, s) for s in (0, 1, 2) for f in FAMILY_ORDER]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        records = []
        for row in executor.map(replay, pairs):
            records.append(row)
            print(json.dumps({'family': row['family'], 'seed': row['seed'], 'pass': True}), flush=True)
    metrics = [{'family': r['family'], 'seed': r['seed'], 'arm': arm, **values}
        for r in records for arm, values in r['metrics'].items()]
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / 'replayed_per_family_seed.csv', index=False)
    columns = [c for c in frame if c not in ('family', 'seed', 'arm', 'temp_n')]
    macro = frame.groupby(['arm', 'seed'])[columns].mean()
    macro.to_csv(OUT / 'replayed_macro_by_seed.csv')
    macro.groupby('arm').agg(['mean', 'std']).to_csv(OUT / 'replayed_macro_mean_std.csv')
    record = {'pass': True, 'scope': 'retained mainline and exact paired controls; no retraining',
        'split_membership': splits, 'runs': records,
        'source_sha256': file_sha256(Path(__file__)),
        'limits': ['Existing test reuse and USPTO validation overlap are not removed by this replay.',
                   'Historical promoted models are not retroactively refitted with repaired canonicalization.']}
    (OUT / 'replay_audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print('PASS: 18 retained mainline runs and 36 paired controls', flush=True)


if __name__ == '__main__':
    main()
