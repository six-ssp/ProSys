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


def prepared_databins(prepared_root, families):
    """Validate the frozen builder receipts before admitting any training job."""
    prepared_root = Path(prepared_root).resolve()
    summary_path = prepared_root / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if not summary.get('preparatory_copy_complete'):
        raise ValueError('Prepared input copy is incomplete')
    entries = {r['dataset']: r for r in summary['families']}
    result = {}
    for family in families:
        ds = 'REAXYS_' + family + '_SINGLE_CATMERGE'
        entry = entries[ds]
        folder = prepared_root / 'artifacts' / ds
        manifest = folder / 'manifest.json'
        if sha(manifest) != entry['manifest_sha256']:
            raise ValueError('Prepared manifest changed: ' + ds)
        record = json.loads(manifest.read_text())
        if record['dataset'] != ds or not record.get('output_sha256'):
            raise ValueError('Invalid prepared manifest: ' + ds)
        for name, digest in record['output_sha256'].items():
            path = (folder / name).resolve()
            path.relative_to(folder.resolve())
            if sha(path) != digest:
                raise ValueError('Prepared output changed: ' + str(path))
        for split in record['splits']:
            if not (split['all_input_text_bin_pairs_equal'] and
                    split['all_retained_token_tensors_unchanged']):
                raise ValueError('Prepared tensor integrity failed: ' + ds)
        result[family] = folder / 'data-bin'
    return result


def require_training_admission(databin, base_checkpoint=None, validation_protocol=None):
    if databin is None:
        raise ValueError('Legacy raw inputs are not admitted; use independently audited prepared inputs')
    manifest_path = Path(databin).parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('ready_for_formal_three_seed_training') is not True:
        raise ValueError('Prepared tensors are not scientific training admission; split/base/validation checks remain pending')
    evidence = manifest.get('training_admission_evidence', {})
    expected = {'expert_post_augmentation', 'base_heldout_boundary', 'condition_validation_propagation'}
    if set(evidence) != expected:
        raise ValueError('Missing scientific admission evidence')
    for name, entry in evidence.items():
        path = ROOT / entry['path']
        receipt = json.loads(path.read_text())
        if (sha(path) != entry['sha256'] or receipt.get('pass') is not True or
                receipt.get('kind') != name or receipt.get('training_admission_eligible') is not True):
            raise ValueError('Unverified scientific admission evidence: ' + name)
        if name == 'base_heldout_boundary' and base_checkpoint is not None:
            if receipt.get('base_checkpoint_sha256') != sha(base_checkpoint):
                raise ValueError('Admission is for a different base checkpoint')
        if name == 'expert_post_augmentation':
            bound_outputs = receipt.get('dataset_output_sha256', {}).get(manifest.get('dataset'))
            if not manifest.get('output_sha256') or bound_outputs != manifest['output_sha256']:
                raise ValueError('Admission does not bind these expert outputs')
        if name == 'condition_validation_propagation':
            from prosys_shared.mainline import split_file_for_family
            family = next((f for f in FAMILIES if manifest.get('dataset') == 'REAXYS_' + f + '_SINGLE_CATMERGE'), None)
            if family is None:
                raise ValueError('Unknown family in admission manifest')
            if validation_protocol is not None and receipt.get('validation_protocol') != validation_protocol:
                raise ValueError('Admission validation protocol differs')
            bindings = receipt.get('condition_split_sha256', {})
            for split in ('train', 'val', 'test'):
                path = split_file_for_family(ROOT, family, split)
                if bindings.get(str(path.relative_to(ROOT))) != sha(path):
                    raise ValueError('Admission condition split changed: ' + split)


def run_job(family, seed, study=STUDY, databin_override=None, base_checkpoint=None,
            validation_protocol='fixed_original', guarded_inference=False):
    for marker in ('SCIENTIFIC_INVALIDATION.json', 'PAUSED_FOR_SPLIT_AUDIT.json'):
        if (study / marker).exists():
            raise ValueError('Refusing invalidated study: ' + str(study / marker))
    base = Path(base_checkpoint or ROOT / 'stage1_retrosynthesis/checkpoints/checkpoint_USPTO_50K_FILTERED_best.pt').resolve()
    require_training_admission(databin_override, base, validation_protocol)
    from prosys_shared.mainline import split_file_for_family, stage1_route_recall
    ds = 'REAXYS_' + family + '_SINGLE_CATMERGE'
    destination = study / ('seed_' + str(seed)) / family
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    databin = databin_override or ROOT / 'data/editretro/datasets' / ds / 'aug10/data-bin'
    historical = list((ROOT / 'stage1_retrosynthesis/results/family_finetune' / ds).glob('*/train.log'))
    if len(historical) != 1:
        raise ValueError('Ambiguous historical config: ' + str(historical))
    old = logged_arguments(historical[0])
    inputs = [base, historical[0], Path(old['data']) / 'dict.src.txt',
              ROOT / 'stage1_retrosynthesis/scripts/run_family_finetune_one.sh',
              ROOT / 'stage1_retrosynthesis/build_route_cache.py']
    route_builder = ROOT / 'stage1_retrosynthesis/build_route_cache.py'
    if guarded_inference:
        route_builder = ROOT / 'scripts/build_stage1_guarded_routes.py'
        inputs += [route_builder, ROOT / 'scripts/stage1_interactive_guarded.py',
                   ROOT / 'scripts/stage1_identity_guard.py', ROOT / 'scripts/audit_stage1_base_augmented.py']
    inputs += sorted(p for p in databin.iterdir() if p.is_file())
    if databin_override is not None:
        inputs += [databin.parent / 'manifest.json', Path(__file__).resolve()]
    inputs += [split_file_for_family(ROOT, family, s) for s in ('train', 'val', 'test')]
    for module in ('editretro', 'fairseq/fairseq', 'fairseq/fairseq_cli', 'preprocess'):
        inputs += sorted((ROOT / 'stage1_retrosynthesis' / module).rglob('*.py'))
    provenance = {'family': family, 'seed': seed, 'historical_config': old,
                  'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(inputs))},
                  'scope': 'fresh family finetuning; fixed base and query splits; fixed decode configuration',
                  'prepared_nonempty_copy': databin_override is not None,
                  'validation_protocol': validation_protocol,
                  'base_checkpoint': str(base),
                  'limitations': ('Fixed-base expert seeds, not independent base-pretraining seeds. '
                                 'Input and checkpoint scientific admission is separate from numerical replay.')}
    if guarded_inference:
        provenance['inference_augmentation'] = 'product_identity_fallback_v1'
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
               BASE_CKPT=str(base), SKIP_PREPARE='1', DATA_BIN=str(databin), MAX_TOKENS=str(old['max_tokens']),
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
            ('SEED', 'MAX_TOKENS', 'NUM_WORKERS', 'OMP_NUM_THREADS', 'RESULTS_ROOT', 'SKIP_PREPARE', 'DATA_BIN', 'BASE_CKPT')}})
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
    command = [sys.executable, str(route_builder),
               '--repo_root', str(ROOT), '--family', family, '--checkpoint', str(checkpoint),
               '--databin', str(databin),
               '--output', str(routes), '--device', '0', '--processes', '2']
    with (destination / 'decode.log').open('w') as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    cache = json.loads((routes / 'route_cache.json').read_text())
    original = json.loads((ROOT / 'outputs/stage1_routes' / family / 'route_cache.json').read_text())
    keys = lambda c: [(r['sample_index'], r['reaction_id'], r['product'], r['gold_reactants']) for r in c['reactions']]
    if keys(cache) != keys(original):
        raise RuntimeError('Evaluation query identities changed')
    metrics = stage1_route_recall(routes / 'route_cache.json')
    if guarded_inference:
        guard = json.loads((routes / 'augmentation_guard.json').read_text())
        if (guard.get('pass') is not True or guard.get('query_count') != len(cache['reactions']) or
                cache.get('augmentation_guard', {}).get('receipt_sha256') != sha(routes / 'augmentation_guard.json')):
            raise RuntimeError('Missing or mismatched inference guard receipt')
    for name, digest in provenance['input_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Training/decoding inputs changed: ' + name)
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
    p.add_argument('--study-root', type=Path, default=STUDY)
    p.add_argument('--prepared-input-root', type=Path)
    p.add_argument('--validation-protocol', choices=['fixed_original', 'strict_post_augmentation'])
    p.add_argument('--base-checkpoint', type=Path)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--guarded-inference', action='store_true',
                   help='Preserve product identity in every inference augmentation')
    args = p.parse_args()
    families = args.families.split(',')
    seeds = list(map(int, args.seeds.split(',')))
    if len(set(families)) != len(families) or len(set(seeds)) != len(seeds):
        p.error('Duplicate families or seeds')
    for family in families:
        if family not in FAMILIES:
            raise ValueError(family)
    if args.prepared_input_root and args.validation_protocol is None:
        p.error('Prepared copies require an explicit --validation-protocol')
    if args.validation_protocol == 'strict_post_augmentation' and args.base_checkpoint is None:
        p.error('Strict repaired inputs require an explicit freshly trained --base-checkpoint')
    study = args.study_root.resolve()
    if args.prepared_input_root and study == STUDY:
        p.error('Prepared copies require a new --study-root; historical jobs stay immutable')
    databins = prepared_databins(args.prepared_input_root, families) if args.prepared_input_root else {}
    base = (args.base_checkpoint or ROOT / 'stage1_retrosynthesis/checkpoints/checkpoint_USPTO_50K_FILTERED_best.pt').resolve()
    for family in families:
        require_training_admission(databins.get(family), base, args.validation_protocol)
    print('Input preflight passed:', len(families), 'families;', len(seeds), 'seeds', flush=True)
    if args.preflight_only:
        return
    study.mkdir(parents=True, exist_ok=True)
    queue_lock = (study / '.queue.lock').open('a')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {'families': families, 'seeds': seeds, 'validation_protocol': args.validation_protocol,
                'base_checkpoint': str(base), 'base_checkpoint_sha256': sha(base),
                'prepared_input_root': str(args.prepared_input_root.resolve()) if args.prepared_input_root else None,
                'scope': 'fixed-base expert finetuning, not independent base-pretraining repeats',
                'replace_mainline': False}
    if args.guarded_inference:
        protocol['inference_augmentation'] = 'product_identity_fallback_v1'
    protocol_path = study / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Study protocol changed; use a separate study root')
    write(protocol_path, protocol)
    for family in families:
        for seed in seeds:
            print('Starting', family, seed, flush=True)
            try:
                run_job(family, seed, study, databins.get(family), base, args.validation_protocol,
                        guarded_inference=args.guarded_inference)
            except Exception as error:
                write(study / 'queue_status.json', {'phase': 'failed', 'family': family,
                      'seed': seed, 'error': str(error), 'pid': os.getpid()})
                raise
            print('Complete', family, seed, flush=True)
    write(study / 'queue_status.json', {'phase': 'complete', 'jobs': len(families) * len(seeds)})


if __name__ == '__main__':
    main()
