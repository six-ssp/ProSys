#!/usr/bin/env python3
"""Audit actual nonempty augmented reaction membership, not just raw splits."""

from concurrent.futures import ProcessPoolExecutor
import argparse
from functools import lru_cache
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREPARED = ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs/artifacts'
OUT = ROOT / 'Experiment/final_release_audit_20260915/augmented_split_audit.json'


@lru_cache(maxsize=1)
def base_membership(folder):
    from scripts.run_stage1_multiseed import sha
    folder = Path(folder)
    audit = json.loads((folder / 'audit.json').read_text())
    keys = set()
    for split in ('train', 'val'):
        path = folder / (split + '.keys.bin')
        if sha(path) != audit['splits'][split]['keys_sha256']:
            raise ValueError('Base membership digest changed')
        if path.stat().st_size != 32 * audit['splits'][split]['augmented_rows']:
            raise ValueError('Base membership row count differs from full audit')
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(32 * 65536), b''):
                if len(block) % 32:
                    raise ValueError('Truncated base membership digest')
                keys.update(block[i:i + 32] for i in range(0, len(block), 32))
    if bytes(32) in keys:
        raise ValueError('Invalid reaction identity in base membership')
    return keys


def audit_family(task):
    family, prepared, base_folder = task
    from rdkit import RDLogger
    from prosys_shared.features import canonicalize_reaction_side
    from prosys_shared.mainline import load_split_rows, split_file_for_family
    from scripts.run_stage1_multiseed import sha

    RDLogger.DisableLog('rdApp.*')
    canonical = lru_cache(maxsize=150000)(canonicalize_reaction_side)
    folder = prepared / ('REAXYS_' + family + '_SINGLE_CATMERGE')
    condition = {s: {(canonical(r['reactants']), canonical(r['product']))
                     for r in load_split_rows(split_file_for_family(ROOT, family, s))}
                 for s in ('train', 'val', 'test')}
    if any(not keys or any(not all(key) for key in keys) for keys in condition.values()):
        raise ValueError('Empty or unparseable condition split')
    sets, details, files = {}, {}, []
    for split in ('train', 'val', 'test'):
        paths = [folder / (split + '.' + side) for side in ('src', 'tgt')]
        files += paths
        reactions, invalid, rows = set(), [], 0
        with paths[0].open() as src, paths[1].open() as tgt:
            for index, (product, reactants) in enumerate(zip_longest(src, tgt)):
                if product is None or reactants is None:
                    raise ValueError('Unpaired augmented text rows')
                key = (canonical(''.join(reactants.split())), canonical(''.join(product.split())))
                rows += 1
                if not all(key):
                    invalid.append(index)
                else:
                    reactions.add(key)
        sets[split] = reactions
        details[split] = {'augmented_pairs': rows, 'parseable_unique_reactions': len(reactions),
                          'unparseable_pair_count': len(invalid),
                          'condition_overlap': {s: len(reactions & keys) for s, keys in condition.items()}}
    cross = {a + '_' + b: len(sets[a] & sets[b])
             for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    base_overlap = None
    if base_folder is not None:
        base_keys = base_membership(str(base_folder))
        base_overlap = sum(hashlib.sha256((r + '>>' + p).encode()).digest() in base_keys
                           for r, p in sets['val'])
    return {'family': family, 'splits': details, 'augmented_cross_split_overlaps': cross,
            'expert_validation_vs_base_train_or_validation': base_overlap,
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
            'training_vs_condition_test_overlap': len(sets['train'] & condition['test']),
            'training_vs_condition_validation_overlap': len(sets['train'] & condition['val'])}


def main():
    from prosys_shared.mainline import FAMILY_ORDER, split_file_for_family
    from scripts.run_stage1_multiseed import sha, write
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--base-audit', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.prepared = args.prepared.resolve()
    args.output = args.output.resolve()
    if args.base_audit is not None:
        args.base_audit = args.base_audit.resolve()
    if args.output.exists():
        raise FileExistsError('Use a new audit receipt: ' + str(args.output))
    base_receipt = None
    base_pass = None
    paths = [Path(__file__), ROOT / 'prosys_shared/features.py', ROOT / 'prosys_shared/mainline.py',
             ROOT / 'scripts/run_stage1_multiseed.py']
    for family in FAMILY_ORDER:
        paths.extend(split_file_for_family(ROOT, family, split) for split in ('train', 'val', 'test'))
        folder = args.prepared / ('REAXYS_' + family + '_SINGLE_CATMERGE')
        paths.extend(folder / (split + '.' + side) for split in ('train', 'val', 'test') for side in ('src', 'tgt'))
    if args.base_audit:
        audit_path = args.base_audit / 'audit.json'
        base_receipt = sha(audit_path)
        base_audit = json.loads(audit_path.read_text())
        base_pass = base_audit.get('pass') is True
        for name, digest in base_audit['source_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Base audit source changed: ' + name)
            paths.append(ROOT / name)
        paths.extend([audit_path, args.base_audit / 'train.keys.bin', args.base_audit / 'val.keys.bin'])
    source_hashes = {str(p.resolve().relative_to(ROOT)): sha(p) for p in set(paths)}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(audit_family, [(f, args.prepared, args.base_audit) for f in FAMILY_ORDER]))
    if {name: sha(ROOT / name) for name in source_hashes} != source_hashes:
        raise ValueError('Chemical audit inputs changed during the scan')
    passed = base_pass is not False and all(not r['training_vs_condition_test_overlap'] and
                 not r['training_vs_condition_validation_overlap'] and
                 not r['splits']['val']['condition_overlap']['test'] and
                 r['expert_validation_vs_base_train_or_validation'] in (None, 0) and
                 not any(r['augmented_cross_split_overlaps'].values()) and
                 not any(s['unparseable_pair_count'] for s in r['splits'].values()) for r in results)
    write(args.output, {'pass': passed, 'families': results,
                'base_membership_audit_sha256': base_receipt,
                'base_audit_pass': base_pass,
                'source_sha256': source_hashes,
                'inputs_unchanged_during_audit': True,
                'training_admission_eligible': False,
                'scope': 'actual nonempty augmented text membership and exact canonical reaction overlap',
                'limitations': 'Checks current input membership only; does not repair existing weights or establish absent historical raw-to-bin lineage.'})
    print(json.dumps({'pass': passed, 'families': [
        {k: v for k, v in r.items() if k != 'source_sha256'} for r in results]}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
