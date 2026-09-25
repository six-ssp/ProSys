#!/usr/bin/env python3
"""Build an audited USPTO-50K base and optionally train without restored weights.

Uses the class-stratified, non-shuffled split of the upstream RetroSim loader.
Upstream test records are reserved, never used for training/model selection.
Reaxys query files and previously trained checkpoints are never overwritten.
"""

import argparse
from collections import Counter, defaultdict
import csv
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
from scripts.audit_stage1_base_augmented import ZERO, load_targets, reaction_key, sha, write
from scripts.trace_stage1_augmented_collisions import transformed_keys
from scripts.run_stage1_multiseed import logged_arguments

SOURCE_COMMIT = '0a272f0b5de833c448f41491e81e4dc00b4d85b0'
SOURCE_BLOB = '6baad99c6d412ce06a84c6144bf70a16bb1bad89'
SOURCE_URL = f'https://raw.githubusercontent.com/connorcoley/retrosim/{SOURCE_COMMIT}/retrosim/data/data_processed.csv'
DATASET = 'USPTO_50K_FILTERED'
EXPERT_INPUTS = ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs/artifacts'


def upstream_splits(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row['class'])].append(index)
    result = {'train': [], 'val': [], 'test': []}
    for label in sorted(groups):
        indices = groups[label]
        train_end, val_end = int(0.8 * len(indices)), int(0.9 * len(indices))
        result['train'].extend(indices[:train_end])
        result['val'].extend(indices[train_end:val_end])
        result['test'].extend(indices[val_end:])
    return {split: sorted(indices) for split, indices in result.items()}


def identities(reaction):
    parts = reaction.split('>')
    if len(parts) != 3:
        return set()
    key = reaction_key(parts[0], parts[2])
    transformed = transformed_keys(reaction)
    if key == ZERO or not transformed:
        return set()
    return {key, *(item[0] for item in transformed)}


def filter_indices(indices, keys, protected, reserved, validation):
    kept, exclusions, seen = [], [], set()
    for index in indices:
        ids = keys[index]
        reasons = []
        if not ids:
            reasons.append('invalid_or_no_mapped_precursors')
        if ids & protected:
            reasons.append('reaxys_heldout_identity')
        if ids & reserved:
            reasons.append('upstream_test_identity')
        if ids & validation:
            reasons.append('upstream_validation_identity')
        if ids & seen:
            reasons.append('duplicate_identity_within_split')
        if reasons:
            exclusions.append({'source_csv_data_index': index, 'reasons': reasons})
        else:
            kept.append(index)
            seen.update(ids)
    return kept, exclusions, seen


def run(command, log, env):
    print('Running:', ' '.join(map(str, command)), flush=True)
    with log.open('w') as handle:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def prepare_raw(source, dataset, study):
    data = source.read_bytes()
    blob = hashlib.sha1(('blob %d\0' % len(data)).encode() + data).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError('USPTO-50K source does not match pinned upstream Git blob')
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50016:
        raise ValueError('Unexpected upstream corpus length')
    splits = upstream_splits(rows)
    print('Loading Reaxys held-out identities', flush=True)
    targets, paths = load_targets(EXPERT_INPUTS)
    paths += [source, Path(__file__), ROOT / 'scripts/trace_stage1_augmented_collisions.py',
              ROOT / 'scripts/audit_stage1_base_augmented.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    protected = set().union(*targets.values())
    keys = []
    for i, row in enumerate(rows):
        keys.append(identities(row['rxn_smiles']))
        if i % 5000 == 0:
            print('Raw chemistry:', i, '/', len(rows), flush=True)
    reserved = set().union(*(keys[i] for i in splits['test']))
    validation = set().union(*(keys[i] for i in splits['val']))
    kept, removed, memberships = {}, {}, {}
    for split in ('val', 'train'):
        kept[split], removed[split], memberships[split] = filter_indices(
            splits[split], keys, protected, reserved, validation if split == 'train' else set())
    if not all(kept.values()) or memberships['train'] & memberships['val']:
        raise ValueError('Empty or overlapping filtered 50K splits')
    raw = dataset / 'raw'
    raw.mkdir(parents=True)
    for split in ('train', 'val'):
        with (raw / f'raw_{split}.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=['pair_id', 'reactants>reagents>production'])
            writer.writeheader()
            for index in kept[split]:
                writer.writerow({'pair_id': str(index), 'reactants>reagents>production': rows[index]['rxn_smiles']})
    if {p: sha(ROOT / p) for p in hashes} != hashes:
        raise ValueError('Source inputs changed during raw filtering')
    write(study / 'raw_lineage.json', {'kept_source_indices': kept, 'excluded_rows': removed,
          'reserved_upstream_test_indices': splits['test']})
    result = {'source_url': SOURCE_URL, 'source_commit': SOURCE_COMMIT, 'source_git_blob': blob,
        'source_sha256': hashes, 'source_rows': len(rows),
        'upstream_split_rule': 'RetroSim get_data.py, class-stratified 80/10/10, shuffle=False; original row order',
        'source_split_rows': {s: len(v) for s, v in splits.items()},
        'retained_raw_rows': {s: len(v) for s, v in kept.items()},
        'removed_raw_rows': {s: len(v) for s, v in removed.items()},
        'exclusion_reason_counts_nonexclusive': {s: dict(Counter(reason for r in v for reason in r['reasons']))
                                                for s, v in removed.items()},
        'raw_output_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(raw.glob('*.csv'))},
        'uses_full_data_or_weights': False, 'uses_upstream_test_for_fitting': False,
        'raw_lineage_sha256': sha(study / 'raw_lineage.json')}
    write(study / 'raw_preparation.json', result)
    return result


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/uspto_50k_source/data_processed.csv')
    parser.add_argument('--study-root', type=Path, default=ROOT / 'Experiment/stage1_50k_from_scratch_20260924')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--train', action='store_true', help='Start fresh training only after the full input audit passes')
    args = parser.parse_args()
    study, source = args.study_root.resolve(), args.source.resolve()
    dataset = ROOT / 'data/editretro/datasets' / DATASET
    if study.exists() or dataset.exists():
        raise FileExistsError('Refusing overwrite or implicit resume; study/dataset must be new')
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError('Less than 6 GiB free; refusing preparation/training')
    study.mkdir(parents=True)
    write(study / 'status.json', {'phase': 'preparing_raw', 'pid': os.getpid(), 'started': time.time()})
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES='0')
    try:
        raw = prepare_raw(source, dataset, study)
        write(study / 'status.json', {'phase': 'preprocessing_and_binarizing', 'pid': os.getpid()})
        run([sys.executable, 'stage1_retrosynthesis/scripts/prepare_family_binarized.py',
             '--dataset', DATASET, '--augmentation', '10', '--processes', args.workers,
             '--repo_root', ROOT], study / 'prepare.log', env)
        write(study / 'status.json', {'phase': 'full_augmented_audit', 'pid': os.getpid()})
        run([sys.executable, 'scripts/audit_stage1_base_augmented.py', '--dataset-folder', dataset / 'aug10',
             '--prepared', EXPERT_INPUTS, '--output', study / 'base_audit', '--workers', args.workers],
            study / 'audit.log', env)
        audit = json.loads((study / 'base_audit/audit.json').read_text())
        if audit.get('pass') is not True:
            raise ValueError('Full augmented audit has not passed')
        inputs = dict(raw['source_sha256'], **raw['raw_output_sha256'], **audit['source_sha256'])
        for module in ('scripts', 'editretro', 'fairseq/fairseq', 'fairseq/fairseq_cli', 'preprocess'):
            for p in (ROOT / 'stage1_retrosynthesis' / module).rglob('*'):
                if p.is_file() and p.suffix in ('.py', '.sh', '.txt'):
                    inputs[str(p.relative_to(ROOT))] = sha(p)
        if any(sha(ROOT / name) != value for name, value in inputs.items()):
            raise ValueError('Training inputs no longer match audited preparation')
        write(study / 'inputs.json', {'source_sha256': inputs, 'base_audit_sha256': sha(study / 'base_audit/audit.json'),
                                     'initialization': 'random; no pretrained checkpoint'})
        if not args.train:
            write(study / 'status.json', {'phase': 'inputs_verified_not_trained'})
            return
        if shutil.disk_usage(ROOT).free < 5 * 1024**3:
            raise RuntimeError('Less than 5 GiB free; refusing training')
        trained = study / 'training' / DATASET / 'run'
        if trained.exists():
            raise FileExistsError('Fresh training requires an empty destination')
        env.update(SKIP_PREPARE='1', DATA_BIN=str(dataset / 'aug10/data-bin'), RESULTS_ROOT=str(study / 'training'),
            RUN_NAME='run', RESTORE_CKPT='', UPDATE_ALIAS='0', GPU_ID='0', SEED='1',
            MAX_EPOCH='50', MAX_UPDATE='200000', LR='0.0003', WARMUP='10000', MAX_TOKENS='16384',
            UPDATE_FREQ='1', USE_FP16='1', NUM_WORKERS='4', PATIENCE='10',
            NO_EPOCH_CHECKPOINTS='1', SAVE_INTERVAL_UPDATES='0', KEEP_LAST_EPOCHS='1')
        write(study / 'training_config.json', {k: env[k] for k in (
            'DATA_BIN', 'RESULTS_ROOT', 'RUN_NAME', 'RESTORE_CKPT', 'UPDATE_ALIAS', 'SEED', 'MAX_EPOCH',
            'MAX_UPDATE', 'LR', 'WARMUP', 'MAX_TOKENS', 'UPDATE_FREQ', 'USE_FP16', 'NUM_WORKERS', 'PATIENCE',
            'NO_EPOCH_CHECKPOINTS', 'SAVE_INTERVAL_UPDATES', 'KEEP_LAST_EPOCHS')})
        write(study / 'status.json', {'phase': 'training_from_scratch', 'pid': os.getpid(), 'started': time.time()})
        run(['bash', ROOT / 'stage1_retrosynthesis/scripts/run_base_train.sh', ROOT, DATASET], study / 'launcher.log', env)
        log = trained / 'train.log'
        actual = logged_arguments(log)
        if 'done training in' not in log.read_text() or actual['restore_file'] != 'checkpoint_last.pt':
            raise ValueError('Missing normal fresh-training completion evidence')
        if actual['data'] != str(dataset / 'aug10/data-bin') or actual['seed'] != 1:
            raise ValueError('Training used unexpected data/seed')
        if any(sha(ROOT / name) != value for name, value in inputs.items()):
            raise ValueError('Training sources changed during fitting')
        checkpoints = {name: sha(trained / 'checkpoints' / name)
                       for name in ('checkpoint_best.pt', 'checkpoint_last.pt')}
        write(study / 'completion.json', {'dataset': DATASET, 'initialization': 'random',
              'uses_full_data_or_weights': False, 'checkpoints': checkpoints, 'actual_config': actual,
              'inputs_sha256': sha(study / 'inputs.json'), 'train_log_sha256': sha(log)})
        write(study / 'status.json', {'phase': 'base_training_complete_not_promoted'})
    except Exception as exc:
        write(study / 'status.json', {'phase': 'failed', 'error': str(exc), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
