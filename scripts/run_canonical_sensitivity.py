#!/usr/bin/env python3
"""Versioned old/new canonicalizer intervention; paired deterministic R-GNN."""

import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'Experiment/project_completion_20260913/canonical_sensitivity'


def legacy_side(smiles):
    from prosys_shared.features import canonicalize_smiles
    fragments = []
    for fragment in str(smiles).split('.'):
        fragment = fragment.strip()
        if not fragment:
            continue
        canonical = canonicalize_smiles(fragment)
        if canonical:
            fragments.append(canonical)
    return '.'.join(sorted(fragments))


def main():
    from scripts.run_stage1_multiseed import write, sha
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('legacy', 'corrected'))
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.mode:
        from prosys_shared import features
        if args.mode == 'legacy':
            features.canonicalize_reaction_side = legacy_side
        destination = STUDY / args.mode / f'seed_{args.seed}'
        scratch = STUDY / 'scratch' / args.mode / f'seed_{args.seed}'
        if destination.exists():
            raise FileExistsError(destination)
        write(destination / 'intervention.json', {'mode': args.mode, 'seed': args.seed,
              'active_canonicalizer_source': inspect.getsource(features.canonicalize_reaction_side),
              'runner_sha256': sha(Path(__file__)),
              'rgnn_protocol': 'deterministic training and embedding inference in both arms',
              'not_a_historical_reproduction': True})
        from scripts.run_mainline_evidence import child
        child('DielsAlder', args.seed, scratch, destination)
        if (destination / 'completion.json').exists():
            shutil.rmtree(scratch)
        return
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
    for seed in (0, 1, 2):
        for mode in ('legacy', 'corrected'):
            if (STUDY / mode / f'seed_{seed}/completion.json').exists():
                continue
            if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                raise RuntimeError('Less than 5 GiB free')
            print('Starting', mode, seed, flush=True)
            with (STUDY / f'{mode}_{seed}.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__)), '--mode', mode, '--seed', str(seed)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = []
    for seed in (0, 1, 2):
        for mode in ('legacy', 'corrected'):
            path = STUDY / mode / f'seed_{seed}'
            result = json.loads((path / 'result.json').read_text())
            rows.append({'mode': mode, 'seed': seed, 'metrics': result['metrics'],
                         'result_sha256': sha(path / 'result.json')})
    write(STUDY / 'comparison.json', {'family': 'DielsAlder', 'rows': rows,
           'scope': 'whole-side canonicalization intervention; raw splits fixed; both arms freshly refit with deterministic R-GNN'})
    print('Six sensitivity fits complete', flush=True)


if __name__ == '__main__':
    main()
