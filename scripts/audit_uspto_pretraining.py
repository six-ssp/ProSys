#!/usr/bin/env python3
"""Independently audit retained filtered USPTO reactions, without changing data."""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def canonical_unmapped_side(value):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(value.strip())
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return '.'.join(sorted(Chem.MolToSmiles(m, canonical=True)
                         for m in Chem.GetMolFrags(mol, asMols=True)))


def key(reactants, product):
    r, p = canonical_unmapped_side(reactants), canonical_unmapped_side(product)
    if not r or not p:
        return None
    return hashlib.sha256((r + '>>' + p).encode()).digest(), hashlib.sha256(p.encode()).digest()


def worker(batch):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    result = []
    for identity, reaction in batch:
        parts = reaction.split('>')
        pair = key(parts[0], parts[2]) if len(parts) == 3 else None
        result.append((identity, pair))
    return result


def batches(path, size=256):
    with path.open() as handle:
        batch = []
        for row in csv.DictReader(handle):
            batch.append((row['id'], row['reactants>reagents>production']))
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


def main():
    from prosys_shared.mainline import FAMILY_ORDER, split_file_for_family
    from scripts.run_stage1_multiseed import sha, write, logged_arguments
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, default=ROOT / 'Experiment/project_completion_20260913/uspto_audit')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'audit.json').exists():
        raise FileExistsError(args.output / 'audit.json')
    targets, products, paths = {}, {}, []
    for family in FAMILY_ORDER:
        for split in ('val', 'test'):
            path = split_file_for_family(ROOT, family, split)
            paths.append(path)
            reactions, product_keys = set(), set()
            with path.open() as handle:
                for line in handle:
                    fields = line.rstrip('\n').split('\t')
                    pair = key(fields[1], fields[2])
                    if pair is None:
                        raise ValueError('Invalid held-out reaction: ' + str(path))
                    reactions.add(pair[0])
                    product_keys.add(pair[1])
            targets[(family, split)] = reactions
            products[(family, split)] = product_keys
    target_union = set().union(*targets.values())
    product_union = set().union(*products.values())
    raw_root = ROOT / 'data/editretro/datasets/USPTO_STAGE2_FILTERED/raw'
    records, sets, matches = {}, {}, []
    started = time.time()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for split in ('train', 'val'):
            path = raw_root / ('raw_' + split + '.csv')
            paths.append(path)
            unique, seen_products = set(), set()
            n, invalid, duplicates = 0, [], 0
            hit_keys, hit_products = set(), set()
            for result in pool.imap(worker, batches(path), chunksize=1):
                for identity, pair in result:
                    n += 1
                    if pair is None:
                        invalid.append(identity)
                        continue
                    r, product = pair
                    duplicates += int(r in unique)
                    unique.add(r)
                    seen_products.add(product)
                    if r in target_union:
                        hit_keys.add(r)
                        matches.append({'base_split': split, 'id': identity,
                                        'reaction_sha256': r.hex(), 'heldout_sets':
                                        [list(k) for k, v in targets.items() if r in v]})
                    if product in product_union:
                        hit_products.add(product)
                if n % 25600 == 0:
                    write(args.output / 'progress.json', {'split': split, 'rows': n,
                          'elapsed_seconds': time.time() - started})
            sets[split] = unique
            records[split] = {'rows': n, 'unique_reactions': len(unique), 'duplicates': duplicates,
                             'invalid_ids': invalid, 'unique_products': len(seen_products),
                             'heldout_overlap': [{'family': fam, 'condition_split': s,
                                'exact_reactions': len(hit_keys & targets[(fam, s)]),
                                'products': len(hit_products & products[(fam, s)]),
                                'target_unique_reactions': len(targets[(fam, s)]),
                                'target_unique_products': len(products[(fam, s)])}
                                for fam, s in targets]}
            write(args.output / ('base_' + split + '.json'), records[split])
    log = next((ROOT / 'stage1_retrosynthesis/results/base_train/USPTO_STAGE2_FILTERED').glob('*/train.log'))
    config = logged_arguments(log)
    data_bin = Path(config['data'])
    paths += [log] + sorted(p for p in data_bin.iterdir() if p.is_file())
    paths += sorted(p for p in data_bin.parent.iterdir() if p.is_file())
    checkpoint = ROOT / 'stage1_retrosynthesis/checkpoints/checkpoint_USPTO_STAGE2_FILTERED_best.pt'
    paths.append(checkpoint)
    write(args.output / 'audit.json', {'scope': 'retained filtered USPTO raw train/val versus six condition val/test splits',
          'canonicalization': 'RDKit whole-side parse; clear atom maps; sorted connected components; retain stereochemistry',
          'base_splits': records, 'base_train_val_reaction_overlap': len(sets['train'] & sets['val']),
          'exact_overlap_records': matches, 'base_training_config': config,
          'current_artifact_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
          'historical_linkage_limit': 'Training log identifies retained data-bin. Current hashes do not reconstruct missing historical raw-to-bin hashes.',
          'wall_seconds': time.time() - started, 'data_modified': False})
    print(json.dumps({'complete': True, 'exact_overlap_records': len(matches),
                      'train_val_overlap': len(sets['train'] & sets['val'])}), flush=True)


if __name__ == '__main__':
    main()
