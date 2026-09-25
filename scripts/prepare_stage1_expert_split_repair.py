#!/usr/bin/env python3
"""Prepare versioned expert inputs with post-transformation held-out exclusions.

This preparatory command never launches training or certifies the base model.
The original condition test query set and original source files stay untouched.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import ZERO, paired_lines, reaction_key, sha, write
from scripts.build_stage1_nonempty_inputs import build_one


def exclusion_reasons(split, key, condition, augmented, base_validation_block):
    if split not in ('train', 'val', 'test'):
        raise ValueError('Unknown augmented split: ' + split)
    if key == ZERO:
        return ['invalid_reaction_identity']
    reasons = []
    heldouts = ('val', 'test') if split == 'train' else ('test',) if split == 'val' else ()
    for heldout in heldouts:
        if key in condition[heldout]:
            reasons.append('condition_' + heldout)
        if key in augmented[heldout]:
            reasons.append('augmented_' + heldout)
    if split == 'val' and key in base_validation_block:
        reasons.append('base_train_or_validation_membership')
    return reasons


def prepare_one(task):
    family, prepared, output, base_block, base_receipt, base_plan_receipt = task
    from rdkit import RDLogger
    from prosys_shared.mainline import load_split_rows, split_file_for_family
    RDLogger.DisableLog('rdApp.*')
    dataset = 'REAXYS_' + family + '_SINGLE_CATMERGE'
    source = prepared / 'artifacts' / dataset
    record = json.loads((source / 'manifest.json').read_text())
    for name, expected in record['output_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError('Prepared source changed: ' + str(source / name))
    paths = [source / 'manifest.json']
    paths += [p for p in source.rglob('*') if p.is_file() and p.name != 'progress.json']
    condition = {}
    for split in ('val', 'test'):
        path = split_file_for_family(ROOT, family, split)
        paths.append(path)
        condition[split] = {reaction_key(row['reactants'], row['product']) for row in load_split_rows(path)}
        if ZERO in condition[split]:
            raise ValueError('Unparseable condition held-out reaction')
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in set(paths)}
    keys = {split: [reaction_key(''.join(r.split()), ''.join(p.split()))
                    for _, p, r in paired_lines(source, split)] for split in ('train', 'val', 'test')}
    augmented = {split: set(values) for split, values in keys.items()}
    excluded, reasons, remaining = {}, {}, {}
    details = output / 'exclusion_lineage' / (family + '.jsonl.gz')
    details.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(details, 'wt') as handle:
        for split, values in keys.items():
            excluded[split], reasons[split], remaining[split] = [], {}, set()
            for index, key in enumerate(values):
                causes = exclusion_reasons(split, key, condition, augmented, base_block)
                if causes:
                    excluded[split].append(index)
                    for cause in causes:
                        reasons[split][cause] = reasons[split].get(cause, 0) + 1
                    handle.write(json.dumps({'source_split': split, 'source_augmented_row': index,
                        'reaction_sha256': key.hex(), 'reasons': causes}) + '\n')
                else:
                    remaining[split].add(key)
    if excluded['test']:
        raise ValueError('Refusing to change the prepared test split; repair invalid test inputs separately')
    post = {a + '_' + b: len(remaining[a] & remaining[b])
            for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    if any(post.values()) or any(remaining['train'] & condition[s] for s in ('val', 'test')) or \
            remaining['val'] & condition['test'] or remaining['val'] & base_block:
        raise ValueError('Exclusion policy did not remove every specified collision')
    for name, expected in hashes.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source changed while planning exclusions')
    destination = output / 'artifacts' / dataset
    result = build_one(source, destination, expected_hashes=hashes, excluded_indices=excluded)
    for side in ('src', 'tgt'):
        if sha(source / ('test.' + side)) != sha(destination / ('test.' + side)):
            raise ValueError('Prepared test sequence changed')
    manifest_path = destination / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['split_exclusion_plan'] = {
        'source_sha256': hashes, 'lineage_sha256': sha(details),
        'exclusions_by_split': {s: len(v) for s, v in excluded.items()},
        'reason_occurrences_by_split': reasons, 'post_filter_cross_split_overlaps': post,
        'base_audit_sha256': base_receipt,
        'strict_base_refilter_plan_sha256': base_plan_receipt,
        'base_eligibility_and_independent_output_audit_pending': True,
        'original_condition_test_manifest_modified': False}
    write(manifest_path, manifest)
    result['manifest_sha256'] = sha(manifest_path)
    result['exclusions_by_split'] = {s: len(v) for s, v in excluded.items()}
    print(json.dumps({'family': family, 'excluded': result['exclusions_by_split']}), flush=True)
    return result


def main():
    from prosys_shared.mainline import FAMILY_ORDER
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-audit', type=Path, required=True)
    parser.add_argument('--base-refilter-plan', type=Path,
                        help='Prepare for a future freshly retrained strict base, not the existing checkpoint')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    audit = json.loads((args.base_audit / 'audit.json').read_text())
    if not all(r['all_text_binary_tensors_equal'] for r in audit['splits'].values()):
        raise ValueError('Base binary identity is unverified')
    for name, expected in audit['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Base-audit source changed')
    base_block = set()
    planned_exclusions, base_plan_receipt = {}, None
    if args.base_refilter_plan:
        plan_path = args.base_refilter_plan / 'summary.json'
        plan = json.loads(plan_path.read_text())
        if plan.get('pass') is not True or plan['base_audit_sha256'] != sha(args.base_audit / 'audit.json'):
            raise ValueError('Strict base refilter plan does not match audited base inputs')
        indices_path = args.base_refilter_plan / 'exclusion_indices.json'
        if sha(indices_path) != plan['exclusion_indices_sha256']:
            raise ValueError('Strict base exclusion indices changed')
        planned_exclusions = json.loads(indices_path.read_text())['splits']
        if set(planned_exclusions) != {'train', 'val'}:
            raise ValueError('Strict base exclusions must cover both splits')
        base_plan_receipt = sha(plan_path)
    for split in ('train', 'val'):
        match_path = args.base_audit / (split + '.matches.jsonl')
        with match_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                # Only observed held-out matches are needed; no sampled membership.
                base_block.add(bytes.fromhex(row['reaction_sha256']))
        if sha(args.base_audit / (split + '.keys.bin')) != audit['splits'][split]['keys_sha256']:
            raise ValueError('Base full-membership keys changed')
    # Validate the small matched-key set independently against all base row keys.
    all_base_keys, retained_base_keys = set(), set()
    for split in ('train', 'val'):
        excluded = set(planned_exclusions.get(split, []))
        if any(type(i) is not int or i < 0 or i >= audit['splits'][split]['augmented_rows'] for i in excluded):
            raise ValueError('Strict base exclusion index outside source split')
        index = 0
        with (args.base_audit / (split + '.keys.bin')).open('rb') as handle:
            for block in iter(lambda: handle.read(32 * 65536), b''):
                if len(block) % 32:
                    raise ValueError('Truncated base membership digest')
                for offset in range(0, len(block), 32):
                    key = block[offset:offset + 32]
                    all_base_keys.add(key)
                    if index not in excluded:
                        retained_base_keys.add(key)
                    index += 1
        if index != audit['splits'][split]['augmented_rows']:
            raise ValueError('Base key count differs from audit')
    if not base_block <= all_base_keys:
        raise ValueError('Base match file contains unknown identities')
    # Use the full membership rather than trusting completeness of the match file.
    base_block = retained_base_keys - {ZERO}
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError('Less than 6 GiB available for safe versioned preparation')
    args.output.mkdir(parents=True)
    receipt = sha(args.base_audit / 'audit.json')
    tasks = [(family, args.prepared.resolve(), args.output.resolve(), base_block, receipt, base_plan_receipt) for family in FAMILY_ORDER]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(prepare_one, tasks))
    write(args.output / 'summary.json', {'preparatory_copy_complete': len(results) == len(FAMILY_ORDER),
        'protocol_selection_pending': True, 'ready_for_formal_three_seed_training': False,
        'families': results, 'base_audit_sha256': receipt, 'training_launched': False,
        'strict_base_refilter_plan_sha256': base_plan_receipt,
        'original_inputs_modified': False,
        'pending': ['independent output membership audit', 'base checkpoint eligibility',
                    'condition-validation propagation', 'training protocol admission']})


if __name__ == '__main__':
    main()
