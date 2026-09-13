#!/usr/bin/env python3
"""Fixed-base expert-training replicates with isolated outputs and provenance."""

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'Experiment/stage1_finetune_multiseed_20260913'
FAMILIES = ['Beckmann', 'Chan_LamCoupling', 'Friedel-CraftsAcylation',
            'Friedel-CraftsAlkylation', 'Buchwald-HartwigCross-Coupling', 'DielsAlder']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def logged_arguments(path):
    with path.open() as handle:
        line = next(x[x.index('Namespace('):] for x in handle if 'Namespace(' in x)
    expression = ast.parse(line.strip(), mode='eval').body
    return {k.arg: ast.literal_eval(k.value) for k in expression.keywords}


def run_job(family, seed):
    from prosys_shared.mainline import split_file_for_family, stage1_route_recall
    ds = 'REAXYS_' + family + '_SINGLE_CATMERGE'
    destination = STUDY / ('seed_' + str(seed)) / family
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = ROOT / 'stage1_retrosynthesis/checkpoints/checkpoint_USPTO_STAGE2_FILTERED_best.pt'
    databin = ROOT / 'data/editretro/datasets' / ds / 'aug10/data-bin'
    historical = list((ROOT / 'stage1_retrosynthesis/results/family_finetune' / ds).glob('*/train.log'))
    if len(historical) != 1:
        raise ValueError('Ambiguous historical config: ' + str(historical))
    old = logged_arguments(historical[0])
    inputs = [base, historical[0], Path(old['data']) / 'dict.src.txt',
              ROOT / 'stage1_retrosynthesis/scripts/run_family_finetune_one.sh',
              ROOT / 'stage1_retrosynthesis/build_route_cache.py']
    inputs += sorted(p for p in databin.iterdir() if p.is_file())
    inputs += [split_file_for_family(ROOT, family, s) for s in ('train', 'val', 'test')]
    for module in ('editretro', 'fairseq/fairseq', 'fairseq/fairseq_cli', 'preprocess'):
        inputs += sorted((ROOT / 'stage1_retrosynthesis' / module).rglob('*.py'))
    provenance = {'family': family, 'seed': seed, 'historical_config': old,
                  'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(inputs))},
                  'scope': 'fresh family finetuning; fixed base, augmented inputs, splits and decode seed'}
    manifest = destination / 'inputs.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise RuntimeError('Inputs changed; use a new study root, not a stale run')
    write(manifest, provenance)
    finished = destination / 'completion.json'
    if finished.exists():
        for name, digest in json.loads(finished.read_text())['output_sha256'].items():
            if sha(destination / name) != digest:
                raise RuntimeError('Changed completed output: ' + name)
        return
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError('Less than 5 GiB free; refusing job admission')
    env = dict(os.environ, PYTHON_BIN=sys.executable, GPU_ID='0', SEED=str(seed),
               RESULTS_ROOT=str(destination / 'training'), RUN_NAME='run',
               BASE_CKPT=str(base), SKIP_PREPARE='1', MAX_TOKENS=str(old['max_tokens']),
               MAX_EPOCH=str(old['max_epoch']), MAX_UPDATE=str(old['max_update']),
               LR=str(old['lr'][0]), WARMUP=str(old['warmup_updates']),
               UPDATE_FREQ=str(old['update_freq'][0]), PATIENCE=str(old['patience']),
               NUM_WORKERS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', USE_FP16='1', NO_EPOCH_CHECKPOINTS='1',
               SAVE_INTERVAL_UPDATES='0', KEEP_LAST_EPOCHS='1')
    trained = destination / 'training' / ds / 'run'
    training_done = destination / 'training_complete.json'
    started = time.time()
    if not training_done.exists():
        # A failed partial run is deliberately not silently resumed from a new base.
        if (trained / 'train.log').exists():
            raise RuntimeError('Partial training needs explicit checkpoint-aware recovery: ' + str(trained))
        write(destination / 'status.json', {'phase': 'training', 'pid': os.getpid(), 'started': started})
        command = ['bash', str(ROOT / 'stage1_retrosynthesis/scripts/run_family_finetune_one.sh'), str(ROOT), ds]
        write(destination / 'command.json', {'command': command, 'environment': {k: env[k] for k in
            ('SEED', 'MAX_TOKENS', 'NUM_WORKERS', 'OMP_NUM_THREADS', 'RESULTS_ROOT', 'SKIP_PREPARE')}})
        with (destination / 'launcher.log').open('w') as handle:
            subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        text = (trained / 'train.log').read_text()
        if 'done training in' not in text:
            raise RuntimeError('Training did not finish normally')
        actual = logged_arguments(trained / 'train.log')
        for key in ('lr', 'max_tokens', 'max_epoch', 'max_update', 'patience', 'warmup_updates',
                    'update_freq', 'fp16', 'dropout', 'attention_dropout', 'weight_decay', 'arch'):
            if actual[key] != old[key]:
                raise RuntimeError('Unexpected training change: ' + key)
        assert actual['seed'] == seed and actual['fixed_validation_seed'] == 7
        write(training_done, {'wall_seconds': time.time() - started,
              'checkpoints': {p.name: sha(p) for p in (trained / 'checkpoints').glob('*.pt')},
              'actual_config': actual})
    for name, digest in json.loads(training_done.read_text())['checkpoints'].items():
        if sha(trained / 'checkpoints' / name) != digest:
            raise RuntimeError('Training checkpoint changed: ' + name)
    checkpoint = trained / 'checkpoints/checkpoint_best.pt'
    write(destination / 'status.json', {'phase': 'decoding', 'pid': os.getpid()})
    routes = destination / 'routes'
    command = [sys.executable, str(ROOT / 'stage1_retrosynthesis/build_route_cache.py'),
               '--repo_root', str(ROOT), '--family', family, '--checkpoint', str(checkpoint),
               '--output', str(routes), '--device', '0', '--processes', '2']
    with (destination / 'decode.log').open('w') as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    cache = json.loads((routes / 'route_cache.json').read_text())
    original = json.loads((ROOT / 'outputs/stage1_routes' / family / 'route_cache.json').read_text())
    keys = lambda c: [(r['sample_index'], r['reaction_id'], r['product'], r['gold_reactants']) for r in c['reactions']]
    if keys(cache) != keys(original):
        raise RuntimeError('Evaluation query identities changed')
    metrics = stage1_route_recall(routes / 'route_cache.json')
    write(destination / 'metrics.json', metrics)
    kept = [destination / 'metrics.json', routes / 'route_cache.json', training_done]
    write(finished, {'family': family, 'seed': seed, 'status': 'complete',
                     'output_sha256': {str(p.relative_to(destination)): sha(p) for p in kept}})
    write(destination / 'status.json', {'phase': 'complete'})
    lock.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--families', default=','.join(FAMILIES))
    p.add_argument('--seeds', default='0,1,2')
    args = p.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    for family in args.families.split(','):
        if family not in FAMILIES:
            raise ValueError(family)
        for seed in map(int, args.seeds.split(',')):
            print('Starting', family, seed, flush=True)
            run_job(family, seed)
            print('Complete', family, seed, flush=True)


if __name__ == '__main__':
    main()
