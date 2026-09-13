#!/usr/bin/env python3
"""Replay and summarize the versioned Diels-Alder canonicalization intervention."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'Experiment/project_completion_20260913/canonical_sensitivity'


def main():
    import numpy as np
    import pandas as pd
    from prosys_shared.evidence import identity_hash, temperature_support
    from scripts.run_stage1_multiseed import sha, write
    p = argparse.ArgumentParser()
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    rows, paired, missing = [], [], []
    expected = json.loads((ROOT / 'outputs/stage1_routes/DielsAlder/route_cache.json').read_text())['reactions']
    expected_ids = sorted(r['sample_index'] for r in expected)
    for seed in (0, 1, 2):
        runs = {}
        for mode in ('legacy', 'corrected'):
            folder = STUDY / mode / f'seed_{seed}'
            if not (folder / 'completion.json').exists():
                missing.append([mode, seed])
                continue
            metrics = json.loads((folder / 'result.json').read_text())['metrics']
            queries = pd.read_csv(folder / 'queries.csv')
            if queries.sample_index.duplicated().any() or sorted(queries.sample_index.tolist()) != expected_ids:
                raise ValueError('Wrong evaluation query identities')
            row = {'mode': mode, 'seed': seed, 'n_queries': len(queries)}
            for k in (1, 3, 5, 10):
                measured = float(queries[f'sys{k}'].mean())
                assert abs(measured - metrics[f'system_top{k}_all']) < 1e-12
                row[f'sys{k}_percent'] = 100 * measured
            scored = pd.read_csv(folder / 'full_candidates.csv.gz')
            support = temperature_support(scored, 'xgb_temperature_pred')
            # The scorer evaluates float32 predictions before CSV serialization.
            # Restore that dtype, rather than interpreting short float32 decimal
            # strings as new float64 predictions and weakening the comparison.
            predictions = support.xgb_temperature_pred.to_numpy(np.float32).astype(np.float64)
            errors = np.abs(predictions - support.temperature_gold.to_numpy(float))
            assert len(errors) == metrics['temperature']['n']
            np.testing.assert_allclose(errors.mean(), metrics['temperature']['mae'], rtol=1e-12, atol=1e-12)
            row['temperature_n'] = len(errors)
            row['temperature_mae_c'] = float(errors.mean())
            row['temperature_within10_percent'] = float((errors <= 10).mean() * 100)
            row['result_sha256'] = sha(folder / 'result.json')
            rows.append(row)
            runs[mode] = {'row': row, 'scored': scored, 'support': support,
                          'provenance': json.loads((folder / 'provenance.json').read_text()),
                          'runtime': json.loads((folder / 'runtime_manifest.json').read_text())}
        if len(runs) == 2:
            old, new = runs['legacy'], runs['corrected']
            for field in ('splits', 'route_test', 'route_validation'):
                assert old['provenance'][field] == new['provenance'][field], field
            assert old['runtime']['source_sha256'] == new['runtime']['source_sha256']
            same_support = identity_hash(old['support'], ordered=True) == identity_hash(new['support'], ordered=True)
            if same_support:
                np.testing.assert_array_equal(old['support'].temperature_gold, new['support'].temperature_gold)
            paired.append({'seed': seed, 'same_raw_splits_and_routes': True,
                           'same_recorded_model_source_hashes': True,
                           'same_candidate_identities': identity_hash(old['scored']) == identity_hash(new['scored']),
                           'same_selected_temperature_identities': same_support,
                           **{f'sys{k}_delta_pp': new['row'][f'sys{k}_percent'] - old['row'][f'sys{k}_percent']
                              for k in (1, 3, 5, 10)}})
    complete = len(rows) == 6 and not missing
    if args.require_complete and not complete:
        raise RuntimeError('Incomplete sensitivity study: ' + str(missing))
    frame = pd.DataFrame(rows)
    frame.to_csv(STUDY / 'per_seed_metrics.csv', index=False)
    stats = []
    if complete:
        for mode in ('legacy', 'corrected'):
            group = frame[frame['mode'] == mode]
            for metric in [c for c in frame if c.startswith('sys') or c in ('temperature_mae_c', 'temperature_within10_percent')]:
                stats.append({'mode': mode, 'metric': metric, 'mean': float(group[metric].mean()),
                              'sample_sd': float(group[metric].std(ddof=1)), 'training_seeds': 3})
        pd.DataFrame(stats).to_csv(STUDY / 'mean_std.csv', index=False)
    write(STUDY / 'summary_audit.json', {'complete': complete, 'missing': missing, 'rows': rows,
           'paired_checks': paired, 'mean_std': stats,
           'scope': 'single-family whole-pipeline representation repair; original validation fixed in both arms',
           'not_a_new_six_family_mainline': True})
    lines = ['# Diels-Alder canonicalization sensitivity', '',
             'Status: ' + ('complete (six fits, three seeds per arm).' if complete else 'partial; no three-seed aggregate is claimed.'), '',
             'Raw data, splits and Stage-1 predictions are fixed. Both arms use deterministic R-GNN.',
             'The legacy arm explicitly restores the old helper in an isolated process; model source hashes otherwise agree.',
             'This is a complete-pipeline repair intervention, not an isolated temperature-head ablation.', '',
             frame.drop(columns=['result_sha256']).to_markdown(index=False, floatfmt='.4f'), '',
             'Candidate and conditional-temperature identity checks:', '',
             pd.DataFrame(paired).to_markdown(index=False, floatfmt='.4f'), '',
             'Different selected temperature identities, where present, preclude treating the two aggregate MAEs as an exact same-support temperature-only comparison.',
             'No result here replaces the promoted six-family mainline; clean-validation selection and expert repeats remain separate work.', '']
    (STUDY / 'SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps({'complete': complete, 'completed_fits': len(rows), 'missing': missing}))


if __name__ == '__main__':
    main()
