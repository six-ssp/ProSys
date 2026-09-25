#!/usr/bin/env python3
"""Wait for one identified live base job, then admit/train/verify expert seeds.

Never restart base training or restore an intermediate checkpoint. PID start
ticks prevent a reused process number from being mistaken for the base job.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stage1_multiseed import sha, write


def process_identity(pid, proc_root=Path('/proc')):
    try:
        fields = (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] in ('Z', 'X'):
        return None
    return (pid, fields[19])


def waiting_state(completed, failed, observed, expected):
    if failed:
        raise RuntimeError('Base coordinator failed; no automatic restart or expert launch')
    if completed:
        return 'ready_for_verification'
    if observed is None or observed != expected:
        raise RuntimeError('Identified base process is gone without a completion receipt')
    return 'waiting_for_base'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-pid', type=int, required=True)
    parser.add_argument('--base-study', type=Path, default=ROOT / 'Experiment/stage1_50k_from_scratch_20260924')
    parser.add_argument('--study-root', type=Path, default=ROOT / 'Experiment/stage1_50k_expert_multiseed_20260924')
    args = parser.parse_args()
    base_study, study = args.base_study.resolve(), args.study_root.resolve()
    expected = process_identity(args.base_pid)
    if expected is None:
        raise RuntimeError('Handoff requires a currently live identified base coordinator')
    command = (Path('/proc') / str(args.base_pid) / 'cmdline').read_bytes().split(b'\0')
    if not any(Path(os.fsdecode(part)).name == 'run_stage1_50k_from_scratch.py' for part in command if part):
        raise ValueError('Specified PID is not the scratch-50K base coordinator')
    study.mkdir(parents=True, exist_ok=True)
    lock = (study / '.handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = study / 'handoff.json'
    if record.exists():
        raise FileExistsError('Existing handoff requires explicit recovery; refusing duplicate queue')
    write(record, {'base_process_identity': expected, 'base_study': str(base_study),
                   'script_sha256': sha(Path(__file__)), 'seeds': [0, 1, 2],
                   'initialization': 'shared completed scratch-50K base',
                   'scope': 'six-family expert seeds and test routes; downstream remains separate'})
    try:
        while True:
            status = json.loads((base_study / 'status.json').read_text())
            phase = waiting_state((base_study / 'completion.json').exists(),
                                  status.get('phase') == 'failed',
                                  process_identity(args.base_pid), expected)
            write(study / 'handoff_status.json', {'phase': phase, 'pid': os.getpid(),
                  'observed_base_identity': process_identity(args.base_pid), 'checked_at': time.time()})
            if phase != 'waiting_for_base':
                break
            time.sleep(60)
        admitted = base_study / 'admitted_expert_inputs'
        base = base_study / 'training/USPTO_50K_FILTERED/run/checkpoints/checkpoint_best.pt'
        commands = [
            ('admitting', [sys.executable, '-B', ROOT / 'scripts/admit_stage1_50k_experts.py',
                           '--base-study', base_study, '--output', admitted]),
            ('training_experts', [sys.executable, '-B', '-u', ROOT / 'scripts/run_stage1_multiseed.py',
                '--prepared-input-root', admitted, '--base-checkpoint', base,
                '--validation-protocol', 'strict_post_augmentation', '--study-root', study, '--seeds', '0,1,2']),
            ('verifying_results', [sys.executable, '-B', ROOT / 'scripts/summarize_stage1_multiseed.py',
                                   '--study-root', study])]
        env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        for phase, command in commands:
            write(study / 'handoff_status.json', {'phase': phase, 'pid': os.getpid(), 'started': time.time()})
            with (study / (phase + '.log')).open('x') as handle:
                subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
        write(study / 'handoff_status.json', {'phase': 'expert_results_verified', 'finished': time.time(),
              'summary_receipt_sha256': sha(study / 'summary_verification.json')})
    except Exception as exc:
        write(study / 'handoff_status.json', {'phase': 'failed', 'error': str(exc), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
