#!/usr/bin/env python3
"""Compare read-only inference with a retained scored query by chemical identity."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    from prosys_shared.features import canonicalize_smiles, canonicalize_reaction_side
    from scripts.run_stage1_multiseed import sha, write
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--route-cache', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.prediction / 'prediction.json').read_text())
    cache = json.loads(args.route_cache.read_text())
    matched = [r for r in cache['reactions'] if canonicalize_smiles(r['product']) == report['product']]
    if len(matched) != 1:
        raise ValueError('Parity smoke requires an unambiguous query identity')
    query = matched[0]
    full = pd.read_csv(args.artifact / 'full_candidates.csv.gz')
    left = full[full.sample_index == query['sample_index']].copy()
    right = pd.read_csv(args.prediction / 'ranked_candidates.csv.gz')
    if {'label', 'yield_gold', 'temperature_gold', 'rank_relevance', 'route_match', 'context_match'} & set(right.columns):
        raise ValueError('Inference output includes gold evaluation fields')
    ids = ['product', 'reactants', 'reagent_norm', 'solvent_norm']
    for frame in (left, right):
        frame['product'] = frame['product'].map(canonicalize_smiles)
        frame['reactants'] = frame['reactants'].map(canonicalize_reaction_side)
        if frame.duplicated(ids).any():
            raise ValueError('Ambiguous canonical candidate identities')
    left = left.sort_values(ids).reset_index(drop=True)
    right = right.sort_values(ids).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[ids], right[ids])
    metadata = json.loads((args.artifact / 'bundle/ranker/xgb_ranker_meta.json').read_text())
    features = metadata['feature_columns']
    assert len(features) == 52
    np.testing.assert_array_equal(left[features].to_numpy(float), right[features].to_numpy(float))
    differences = {}
    for column in ('xgb_score', 'xgb_temperature_pred'):
        a, b = left[column].to_numpy(float), right[column].to_numpy(float)
        differences[column] = float(np.max(np.abs(a - b)))
        np.testing.assert_array_equal(a, b)
    proof = {'sample_index': query['sample_index'], 'rows': len(left),
             'canonical_candidate_identities_equal': True, 'all_52_ltr_features_exactly_equal': True,
             'prediction_max_absolute_error': differences, 'no_gold_inputs_or_output_labels': True,
             'route_cache_sha256': sha(args.route_cache),
             'artifact_result_sha256': sha(args.artifact / 'result.json'),
             'scope': 'one unique product, cached Stage-1 routes, corrected model bundle',
             'identity_note': 'CLI canonicalizes its product string; raw string equality is not chemical identity equality.'}
    write(args.prediction / 'parity.json', proof)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
