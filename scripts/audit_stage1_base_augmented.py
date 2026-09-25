#!/usr/bin/env python3
"""Full augmented USPTO membership and text/binary audit, without altering inputs."""

import argparse
from functools import lru_cache
import hashlib
from itertools import zip_longest
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ZERO = bytes(32)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


@lru_cache(maxsize=30000)
def canonical(side):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(side)
    if mol is None or not mol.GetNumAtoms():
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return '.'.join(sorted(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
                           for fragment in Chem.GetMolFrags(mol, asMols=True)))


def reaction_key(reactants, product):
    r, p = canonical(reactants), canonical(product)
    return hashlib.sha256((r + '>>' + p).encode()).digest() if r and p else ZERO


def paired_lines(folder, split):
    with (folder / (split + '.src')).open() as src, (folder / (split + '.tgt')).open() as tgt:
        for index, (product, reactants) in enumerate(zip_longest(src, tgt)):
            if product is None or reactants is None:
                raise ValueError('Unpaired text rows: ' + str(folder))
            yield index, product, reactants


def batches(folder, split, size=512):
    batch = []
    for item in paired_lines(folder, split):
        batch.append(item)
        if len(batch) == size:
            yield split, batch
            batch = []
    if batch:
        yield split, batch


def initialize_worker(folder):
    global DICTIONARIES, BINS
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
    from fairseq.data import Dictionary, indexed_dataset
    folder = Path(folder) / 'data-bin'
    DICTIONARIES = {s: Dictionary.load(str(folder / ('dict.' + s + '.txt'))) for s in ('src', 'tgt')}
    BINS = {(split, side): indexed_dataset.make_dataset(
        str(folder / (('valid' if split == 'val' else split) + '.src-tgt.' + side)), 'mmap')
        for split in ('train', 'val') for side in ('src', 'tgt')}
    if any(value is None for value in BINS.values()):
        raise FileNotFoundError('Missing binary dataset')


def worker(task):
    import torch
    split, batch = task
    keys = []
    for index, product, reactants in batch:
        for side, text in (('src', product), ('tgt', reactants)):
            expected = DICTIONARIES[side].encode_line(text, add_if_not_exist=False).long()
            if index >= len(BINS[(split, side)]) or not torch.equal(expected, BINS[(split, side)][index].long()):
                raise ValueError('Actual binary/text mismatch: %s/%s/%d' % (split, side, index))
        keys.append(reaction_key(''.join(reactants.split()), ''.join(product.split())))
    return batch[0][0], keys, {side: len(BINS[(split, side)]) for side in ('src', 'tgt')}


def load_targets(prepared):
    from prosys_shared.mainline import FAMILY_ORDER, load_split_rows, split_file_for_family
    targets, paths = {}, []
    for family in FAMILY_ORDER:
        for split in ('val', 'test'):
            path = split_file_for_family(ROOT, family, split)
            paths.append(path)
            keys = {reaction_key(row['reactants'], row['product']) for row in load_split_rows(path)}
            if ZERO in keys:
                raise ValueError('Invalid held-out condition reaction')
            targets['condition/' + family + '/' + split] = keys
            folder = prepared / ('REAXYS_' + family + '_SINGLE_CATMERGE')
            paths.extend(folder / (split + '.' + side) for side in ('src', 'tgt'))
            keys = {reaction_key(''.join(r.split()), ''.join(p.split())) for _, p, r in paired_lines(folder, split)}
            if ZERO in keys:
                raise ValueError('Invalid held-out augmented expert reaction')
            targets['expert_augmented/' + family + '/' + split] = keys
    return targets, paths


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--dataset-folder', type=Path, default=ROOT / 'data/editretro/datasets/USPTO_STAGE2_FILTERED/aug10')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs/artifacts')
    parser.add_argument('--output', type=Path, default=ROOT / 'Experiment/stage1_split_repair_20260915/base_augmented')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.time()
    write(args.output / 'progress.json', {'phase': 'loading_heldout_targets'})
    targets, paths = load_targets(args.prepared)
    owners = {}
    for name, keys in targets.items():
        for key in keys:
            owners.setdefault(key, []).append(name)
    paths += [args.dataset_folder / (split + '.' + side) for split in ('train', 'val') for side in ('src', 'tgt')]
    paths += sorted((args.dataset_folder / 'data-bin').glob('*'))
    paths.append(Path(__file__))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}
    write(args.output / 'input_manifest.json', {'source_sha256': hashes, 'workers': args.workers})
    records, membership = {}, {}
    with mp.get_context('spawn').Pool(args.workers, initializer=initialize_worker, initargs=(str(args.dataset_folder),)) as pool:
        for split in ('train', 'val'):
            unique, hits = set(), {name: set() for name in targets}
            occurrences = dict.fromkeys(targets, 0)
            count, invalid, invalid_examples = 0, 0, []
            with (args.output / (split + '.keys.bin')).open('wb') as keys_file, \
                 (args.output / (split + '.matches.jsonl')).open('w') as matches:
                for offset, keys, lengths in pool.imap(worker, batches(args.dataset_folder, split), chunksize=1):
                    if offset != count:
                        raise ValueError('Unexpected augmented row order')
                    keys_file.write(b''.join(keys))
                    for local_index, key in enumerate(keys):
                        if key == ZERO:
                            invalid += 1
                            if len(invalid_examples) < 20:
                                invalid_examples.append(count + local_index)
                            continue
                        unique.add(key)
                        if key in owners:
                            matches.write(json.dumps({'augmented_row': count + local_index,
                                'reaction_sha256': key.hex(), 'heldout_sets': owners[key]}) + '\n')
                            for name in owners[key]:
                                hits[name].add(key)
                                occurrences[name] += 1
                    count += len(keys)
                    if count % 51200 == 0:
                        status = {'phase': 'full_text_binary_and_membership_scan', 'split': split,
                                  'rows': count, 'elapsed_seconds': time.time() - started,
                                  'heldout_unique_hits': {name: len(value) for name, value in hits.items()}}
                        write(args.output / 'progress.json', status)
                        print(split, count, 'rows;', round(time.time() - started, 1), 'seconds', flush=True)
                if not count or any(n != count for n in lengths.values()):
                    raise ValueError('Binary has different row count from paired text')
            membership[split] = unique
            records[split] = {'augmented_rows': count, 'unique_parseable_reactions': len(unique),
                'unparseable_pairs': invalid, 'unparseable_example_indices': invalid_examples,
                'all_text_binary_tensors_equal': True,
                'heldout_overlap': {name: {'unique_reactions': len(hits[name]), 'augmented_pairs': occurrences[name],
                                         'heldout_unique_reactions': len(targets[name])} for name in targets},
                'keys_sha256': sha(args.output / (split + '.keys.bin'))}
            write(args.output / (split + '.audit.json'), records[split])
    current = {path: sha(ROOT / path) for path in hashes}
    if current != hashes:
        raise ValueError('Inputs changed during audit')
    cross = len(membership['train'] & membership['val'])
    passed = not cross and all(not record['unparseable_pairs'] and
        not any(row['unique_reactions'] for row in record['heldout_overlap'].values()) for record in records.values())
    write(args.output / 'audit.json', {'pass': passed, 'splits': records,
        'base_augmented_train_validation_unique_overlap': cross,
        'source_sha256': hashes, 'wall_seconds': time.time() - started,
        'canonicalization': 'RDKit whole-side parse, clear maps, canonical connected components sorted, retain stereo',
        'scope': 'every augmented text pair and corresponding actual binary tensor; exact reaction identity, not product disjointness',
        'historical_limit': 'current artifacts only; no absent historical raw-to-binary manifest is reconstructed',
        'inputs_modified': False})
    print(json.dumps({'pass': passed, 'cross_split': cross, 'wall_seconds': time.time() - started}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
