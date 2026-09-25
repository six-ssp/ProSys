#!/usr/bin/env python3
"""Exclude source-inconsistent train/validation augmentations in new copies.

Original condition queries and stored test tensors are unchanged. Test
augmentation diagnostics are disclosed separately, not certified as repaired.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import sha, write
from scripts.build_stage1_nonempty_inputs import build_one


def exclusion_plan(report, rows):
    excluded = {split: [] for split in ('train', 'val', 'test')}
    observed = {split: set() for split in excluded}
    for row in rows:
        split, index = row['split'], row['augmented_row']
        if (split not in observed or type(index) is not int or index < 0 or
                index >= report['splits'][split]['augmented_rows'] or index in observed[split]):
            raise ValueError('Invalid or duplicate mismatch row')
        observed[split].add(index)
    for split, indices in observed.items():
        if len(indices) != report['splits'][split]['unmatched_augmented_pairs']:
            raise ValueError('Mismatch details/count discrepancy')
        if split != 'test':
            excluded[split] = sorted(indices)
    return excluded


def prepare(task):
    dataset, source_root, audit_root, output = task
    source = source_root / 'artifacts' / dataset
    receipt = audit_root / (dataset + '.json')
    details = audit_root / (dataset + '.unmatched.jsonl.gz')
    report = json.loads(receipt.read_text())
    if report['dataset'] != dataset or sha(details) != report['details_sha256']:
        raise ValueError('Mismatched audit receipt')
    if any(s['invalid_augmented_pairs'] for s in report['splits'].values()):
        raise ValueError('Unparseable input requires separate investigation')
    for name, digest in report['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Chemical audit source changed: ' + name)
    with gzip.open(details, 'rt') as handle:
        excluded = exclusion_plan(report, [json.loads(line) for line in handle])
    old = json.loads((source / 'manifest.json').read_text())
    for name, digest in old['output_sha256'].items():
        if sha(source / name) != digest:
            raise ValueError('Prepared source changed')
    for split in excluded:
        for side in ('src', 'tgt'):
            path = source / (split + '.' + side)
            if report['source_sha256'].get(str(path.relative_to(ROOT))) != sha(path):
                raise ValueError('Audit is for a different augmented source')
    destination = output / 'artifacts' / dataset
    if any(excluded.values()):
        result = build_one(source, destination, excluded_indices=excluded)
        manifest = json.loads((destination / 'manifest.json').read_text())
    else:
        destination.mkdir(parents=True)
        for name in old['output_sha256']:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source / name, target)
        manifest = dict(old)
        result = {'dataset': dataset, 'path': str(destination), 'splits': old['splits']}
    for side in ('src', 'tgt'):
        if sha(source / ('test.' + side)) != sha(destination / ('test.' + side)):
            raise ValueError('Stored test augmentations changed')
    manifest.pop('training_admission_evidence', None)
    manifest.update(ready_for_formal_three_seed_training=False, protocol_selection_pending=True,
        augmentation_fidelity_repair={
            'source_audit_sha256': sha(receipt), 'source_audit': str(receipt.relative_to(ROOT)),
            'source_details_sha256': sha(details), 'excluded_indices': excluded,
            'rule': 'remove all same-split raw-transformed identity mismatches from train/val, independently of model scores',
            'stored_test_mismatches_not_modified': report['splits']['test']['unmatched_augmented_pairs'],
            'original_condition_queries_modified': False,
            'retained_training_identities_pass_by_exhaustive_exclusion_and_unchanged_tensor_copy': True})
    write(destination / 'manifest.json', manifest)
    result.update(manifest_sha256=sha(destination / 'manifest.json'),
                  fidelity_exclusions={s: len(rows) for s, rows in excluded.items()})
    print(json.dumps({'dataset': dataset, 'excluded': result['fidelity_exclusions']}), flush=True)
    return result


def main():
    from prosys_shared.mainline import FAMILY_ORDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'Experiment/stage1_50k_from_scratch_20260924/admitted_expert_inputs')
    parser.add_argument('--audit', type=Path, default=ROOT / 'Experiment/stage1_augmentation_identity_20260924')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError('Less than 5 GiB free')
    # Each family must have finished its audit; the base receipt is a separate
    # checkpoint-admission condition, never inferred from family checks.
    tasks = [('REAXYS_' + f + '_SINGLE_CATMERGE', args.source.resolve(),
              args.audit.resolve(), args.output.resolve()) for f in FAMILY_ORDER]
    for dataset, _, audit, _ in tasks:
        if not (audit / (dataset + '.json')).is_file():
            raise FileNotFoundError('Incomplete family audit: ' + dataset)
    args.output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(prepare, tasks))
    write(args.output / 'summary.json', {'preparatory_copy_complete': True,
        'protocol_selection_pending': True, 'ready_for_formal_three_seed_training': False,
        'families': rows, 'training_launched': False, 'original_inputs_modified': False,
        'scope': 'Train/validation augmentation identity repair; stored test artifacts unchanged; requires independent copy/boundary/base admission.'})


if __name__ == '__main__':
    main()
