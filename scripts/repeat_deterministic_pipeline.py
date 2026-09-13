#!/usr/bin/env python3
"""Repeat a full corrected Stage-2/3 fit and compare identity-aligned predictions."""

import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'Experiment/project_completion_20260913/deterministic_pipeline_repeat'
REFERENCE = ROOT / 'Experiment/project_completion_20260913/canonical_sensitivity/corrected/seed_0'


def compare_predictions(left, right):
    import pandas as pd
    from prosys_shared.evidence import IDENTITY_COLUMNS, require_same_candidates, require_same_temperature_support
    ids = list(IDENTITY_COLUMNS)
    for frame in (left, right):
        if frame.duplicated(ids).any():
            raise ValueError('Ambiguous duplicated candidate identities')
    digest = require_same_candidates(left, right)
    a = left.sort_values(ids, kind='mergesort').reset_index(drop=True)
    b = right.sort_values(ids, kind='mergesort').reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    support = require_same_temperature_support(left, right,
        left_column='xgb_temperature_pred', right_column='xgb_temperature_pred')
    return {'rows': len(a), 'candidate_identity_sha256': digest,
            'all_retained_candidate_columns_exactly_equal': True, 'temperature_support': support}


def main():
    from scripts.run_stage1_multiseed import write, sha
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = OUTPUT / 'compact'
    scratch = OUTPUT / 'scratch'
    if not (result / 'completion.json').exists():
        if result.exists() or scratch.exists():
            raise FileExistsError('Partial repeat exists; inspect rather than overwrite')
        if shutil.disk_usage(ROOT).free < 5 * 1024**3:
            raise RuntimeError('Less than 5 GiB free')
        write(OUTPUT / 'status.json', {'phase': 'full_repeat', 'pid': os.getpid()})
        from scripts.run_mainline_evidence import child
        child('DielsAlder', 0, scratch, result)
        shutil.rmtree(scratch)
    if not (REFERENCE / 'completion.json').exists():
        write(OUTPUT / 'status.json', {'phase': 'awaiting_reference_completion', 'pid': os.getpid()})
        print('Repeat complete; reference not complete yet. Rerun comparison after it finishes.', flush=True)
        return
    import pandas as pd
    import torch
    left = pd.read_csv(REFERENCE / 'full_candidates.csv.gz')
    right = pd.read_csv(result / 'full_candidates.csv.gz')
    proof = compare_predictions(left, right)
    a = torch.load(REFERENCE / 'bundle/rgnn/reaction_gnn.pt', map_location='cpu', weights_only=False)
    b = torch.load(result / 'bundle/rgnn/reaction_gnn.pt', map_location='cpu', weights_only=False)
    assert a['config'] == b['config'] and a['config']['deterministic']
    assert a['model_state'].keys() == b['model_state'].keys()
    for name in a['model_state']:
        assert torch.equal(a['model_state'][name], b['model_state'][name]), name
    prov_a = json.loads((REFERENCE / 'provenance.json').read_text())
    prov_b = json.loads((result / 'provenance.json').read_text())
    for key in ('splits', 'route_test', 'route_validation'):
        assert prov_a[key] == prov_b[key], key
    proof.update({'family': 'DielsAlder', 'seed': 0, 'rgnn_tensors_exactly_equal': True,
                  'same_data_and_route_hashes': True,
                  'reference_result_sha256': sha(REFERENCE / 'result.json'),
                  'repeat_result_sha256': sha(result / 'result.json'),
                  'scope': 'fresh full corrected Stage-2/3 fit, same seed and fixed old validation; current environment only'})
    write(OUTPUT / 'comparison.json', proof)
    write(OUTPUT / 'status.json', {'phase': 'complete', 'pass': True})
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
