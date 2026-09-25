#!/usr/bin/env python3
"""Publish fixed-seed expert test/validation routes after completed-job checks.

This controller may wait for an identified existing expert queue. It never
starts or resumes training, chooses seeds by accuracy, or replaces old routes.
"""

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
from scripts.audit_stage1_base_augmented import sha, write
from scripts.continue_stage1_50k_experts import process_identity
from scripts.run_verified_mainline import verify_route_manifest
from scripts.stage1_route_admission import PROTOCOL, require_paired_checkpoints, verify_guard
from scripts.summarize_stage1_multiseed import check_invalidation, verify_job


def require_queue_state(completed, failed, observed, expected):
    if failed:
        raise RuntimeError('Expert queue failed; no automatic training restart')
    if completed:
        return True
    if expected is None or observed != expected:
        raise RuntimeError('Expert completion missing and identified queue is no longer live')
    return False


def main():
    from prosys_shared.mainline import FAMILY_ORDER, split_file_for_family
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-pid', type=int)
    args = parser.parse_args()
    study, output = args.expert_study.resolve(), args.output.resolve()
    check_invalidation(study)
    protocol_path = study / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if (set(protocol['families']) != set(FAMILY_ORDER) or len(protocol['families']) != len(FAMILY_ORDER) or
            protocol['seeds'] != [0, 1, 2] or protocol.get('inference_augmentation') != PROTOCOL):
        raise ValueError('Not the complete guarded six-family three-seed expert study')
    observed = process_identity(args.wait_for_pid) if args.wait_for_pid else None
    if args.wait_for_pid:
        if observed is None:
            raise RuntimeError('Specified expert queue is not live')
        command = (Path('/proc') / str(args.wait_for_pid) / 'cmdline').read_bytes().split(b'\0')
        if (not any(Path(os.fsdecode(token)).name == 'run_stage1_multiseed.py' for token in command if token) or
                str(study).encode() not in command):
            raise ValueError('PID is not the specified expert study runner')
    if output.exists():
        raise FileExistsError('Use a new publication root; no silent partial reuse')
    output.mkdir(parents=True)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path(__file__), protocol_path, ROOT / 'scripts/stage1_route_admission.py',
        ROOT / 'scripts/summarize_stage1_multiseed.py', ROOT / 'scripts/continue_stage1_50k_experts.py',
        ROOT / 'scripts/build_stage1_guarded_routes.py', ROOT / 'scripts/stage1_interactive_guarded.py',
        ROOT / 'scripts/stage1_identity_guard.py', ROOT / 'scripts/run_verified_mainline.py']
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    write(output / 'protocol.json', {'seed': 1, 'families': protocol['families'],
        'expert_study': str(study), 'queue_identity': observed, 'source_sha256': bindings,
        'seed_selection': 'fixed historical seed 1, declared before new test evaluation',
        'scope': 'new expert test/validation routes only, not downstream fitted models'})
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    published = []
    try:
        for family in protocol['families']:
            job = study / 'seed_1' / family
            while True:
                check_invalidation(study)
                queue_path = study / 'queue_status.json'
                queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
                live = process_identity(args.wait_for_pid) if args.wait_for_pid else None
                ready = require_queue_state((job / 'completion.json').exists(), queue.get('phase') == 'failed', live, observed)
                if ready:
                    break
                write(output / 'status.json', {'phase': 'waiting_for_completed_seed1', 'family': family,
                    'checked_at': time.time(), 'observed_queue_identity': live, 'controller_pid': os.getpid()})
                time.sleep(30)
            verified = verify_job(study, family, 1)
            if any(sha(ROOT / name) != digest for name, digest in bindings.items()):
                raise ValueError('Downstream route publication sources changed')
            if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                raise RuntimeError('Less than 5 GiB free before validation decoding')
            source = job / 'routes/route_cache.json'
            test_dir, val_dir = output / 'test' / family, output / 'val' / family
            test_dir.mkdir(parents=True)
            for name in ('route_cache.json', 'augmentation_guard.json'):
                os.link(source.parent / name, test_dir / name)
            training = json.loads((job / 'training_complete.json').read_text())
            checkpoint = Path(json.loads(source.read_text())['checkpoint'])
            val_split = split_file_for_family(ROOT, family, 'val')
            command = [sys.executable, '-B', ROOT / 'scripts/build_stage1_guarded_routes.py',
                '--repo_root', ROOT, '--family', family, '--checkpoint', checkpoint,
                '--databin', training['actual_config']['data'], '--gold_split', val_split,
                '--output', val_dir, '--device', '0', '--processes', '2']
            write(output / 'status.json', {'phase': 'decoding_validation', 'family': family,
                                          'controller_pid': os.getpid()})
            log = output / 'logs' / (family + '.log')
            log.parent.mkdir(exist_ok=True)
            with log.open('x') as handle:
                subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            test, val = test_dir / 'route_cache.json', val_dir / 'route_cache.json'
            verify_route_manifest(test, split_file_for_family(ROOT, family, 'test'), family)
            verify_route_manifest(val, val_split, family)
            guards = {'test': verify_guard(test), 'val': verify_guard(val)}
            if guards['test']['source_sha256'] != guards['val']['source_sha256']:
                raise ValueError('Validation/test guard sources differ')
            checkpoint_sha = require_paired_checkpoints(test, val)
            if checkpoint_sha != verified['best_sha256']:
                raise ValueError('Expert checkpoint changed during validation decoding')
            if any(sha(ROOT / name) != digest for name, digest in bindings.items()):
                raise ValueError('Publication sources changed during decoding')
            item = {'family': family, 'seed': 1, 'checkpoint_sha256': checkpoint_sha,
                    'expert_verification': verified, 'guarded_routes': guards,
                    'expert_completion_sha256': sha(job / 'completion.json'),
                    'split_sha256': {s: sha(split_file_for_family(ROOT, family, s)) for s in ('train', 'val', 'test')}}
            write(output / 'receipts' / (family + '.json'), item)
            published.append(item)
            print('Published verified seed-1 route pair:', family, flush=True)
        write(output / 'summary.json', {'pass': True, 'complete': True, 'seed': 1,
            'families': published, 'protocol_sha256': sha(output / 'protocol.json'),
            'scope': 'same-checkpoint validation/test route pairs, fixed original query support'})
        write(output / 'status.json', {'phase': 'complete', 'families': len(published)})
    except Exception as exc:
        write(output / 'status.json', {'phase': 'failed', 'error': str(exc), 'controller_pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
