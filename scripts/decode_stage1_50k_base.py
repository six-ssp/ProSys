#!/usr/bin/env python3
"""Evaluate the completed scratch-50K base on all original family test queries."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.admit_stage1_50k_experts import completed_base
from scripts.run_stage1_multiseed import sha, write
from scripts.run_verified_mainline import verify_route_manifest


def main():
    from prosys_shared.mainline import FAMILY_ORDER, split_file_for_family, stage1_route_recall
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guarded-inference', action='store_true')
    parser.add_argument('--study-root', type=Path)
    args = parser.parse_args()
    study = (args.study_root or ROOT / ('Experiment/stage1_50k_guarded_base_test_20260924'
              if args.guarded_inference else 'Experiment/stage1_50k_base_test_20260924')).resolve()
    base_study = ROOT / 'Experiment/stage1_50k_from_scratch_20260924'
    checkpoint = completed_base(base_study)
    databin = ROOT / 'data/editretro/datasets/USPTO_50K_FILTERED/aug10/data-bin'
    study.mkdir(parents=True, exist_ok=True)
    lock = (study / '.queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (study / 'protocol.json').exists():
        raise FileExistsError('Explicit recovery required; refusing to overwrite a base decode study')
    sources = [Path(__file__), checkpoint, base_study / 'completion.json']
    route_builder = ROOT / 'stage1_retrosynthesis/build_route_cache.py'
    if args.guarded_inference:
        route_builder = ROOT / 'scripts/build_stage1_guarded_routes.py'
        sources += [route_builder, ROOT / 'scripts/stage1_interactive_guarded.py',
                    ROOT / 'scripts/stage1_identity_guard.py', ROOT / 'scripts/audit_stage1_base_augmented.py']
    sources += list((ROOT / 'stage1_retrosynthesis').rglob('*.py'))
    sources += [databin / ('dict.' + side + '.txt') for side in ('src', 'tgt')]
    sources += [split_file_for_family(ROOT, f, 'test') for f in FAMILY_ORDER]
    sources += [ROOT / 'scripts/run_verified_mainline.py', ROOT / 'prosys_shared/mainline.py',
                ROOT / 'prosys_shared/features.py']
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))}
    write(study / 'protocol.json', {'families': FAMILY_ORDER, 'checkpoint': str(checkpoint),
          'source_sha256': bindings, 'aug': 10, 'beam': [5, 2, 1], 'n_best': 10,
          'batch_size': 64, 'max_tokens': 4000,
          'inference_augmentation': 'product_identity_fallback_v1' if args.guarded_inference else 'stock_diagnostic',
          'scope': 'fixed completed base; full original family condition test queries; no expert metrics yet'})
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    rows = []
    try:
        for family in FAMILY_ORDER:
            if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                raise RuntimeError('Less than 5 GiB free; refusing next base decode')
            output = study / family
            if output.exists():
                raise FileExistsError(output)
            output.mkdir()
            split = split_file_for_family(ROOT, family, 'test')
            command = [sys.executable, '-B', route_builder,
                '--repo_root', ROOT, '--family', family, '--checkpoint', checkpoint,
                '--databin', databin, '--gold_split', split, '--output', output,
                '--aug', '10', '--topk', '10', '--repos_beam', '5', '--token_beam', '2',
                '--mask_beam', '1', '--n_best', '10', '--score_alpha', '0.1',
                '--batch_size', '64', '--buffer_size', '2000', '--max_tokens', '4000',
                '--device', '0', '--processes', '2']
            write(output / 'command.json', {'command': list(map(str, command)),
                  'environment': {k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}})
            write(study / 'status.json', {'phase': 'decoding', 'family': family,
                  'pid': os.getpid(), 'completed_families': [r['family'] for r in rows]})
            start = time.time()
            with (output / 'decode.log').open('x') as handle:
                subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            cache_path = output / 'route_cache.json'
            verify_route_manifest(cache_path, split, family)
            cache = json.loads(cache_path.read_text())
            if Path(cache['checkpoint']).resolve() != checkpoint:
                raise ValueError('Decoded with an unexpected checkpoint')
            if args.guarded_inference:
                guard_path = output / 'augmentation_guard.json'
                guard = json.loads(guard_path.read_text())
                if (guard.get('pass') is not True or guard.get('query_count') != len(cache['reactions']) or
                        cache.get('augmentation_guard', {}).get('receipt_sha256') != sha(guard_path)):
                    raise ValueError('Missing or mismatched product identity guard')
            if any(sha(ROOT / name) != digest for name, digest in bindings.items()):
                raise ValueError('Base decode inputs or sources changed during execution')
            metrics = stage1_route_recall(cache_path)
            write(output / 'metrics.json', metrics)
            write(output / 'completion.json', {'family': family, 'wall_seconds': time.time() - start,
                'checkpoint_sha256': sha(checkpoint), 'original_query_count': len(cache['reactions']),
                'output_sha256': {p.name: sha(p) for p in output.iterdir()
                                  if p.is_file() and p.name != 'completion.json'}})
            rows.append({'family': family, **metrics})
        write(study / 'base_metrics.json', {'families': rows, 'complete': True,
              'protocol_sha256': sha(study / 'protocol.json'),
              'scope': 'base only; paired fine-tuned comparison not yet complete'})
        write(study / 'status.json', {'phase': 'complete', 'families': len(rows)})
    except Exception as exc:
        write(study / 'status.json', {'phase': 'failed', 'error': str(exc), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
