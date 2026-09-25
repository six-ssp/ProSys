#!/usr/bin/env python3
"""Independently confirm one exact test overlap per family in binary inputs."""

from functools import lru_cache
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


@lru_cache(maxsize=40000)
def direct_canonical(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return '.'.join(sorted(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
                           for fragment in Chem.GetMolFrags(mol, asMols=True)))


def main():
    import torch
    from rdkit import RDLogger
    from fairseq.data import Dictionary, indexed_dataset
    from prosys_shared.mainline import FAMILY_ORDER, load_split_rows, split_file_for_family

    RDLogger.DisableLog('rdApp.*')
    rows = []
    for family in FAMILY_ORDER:
        test_path = split_file_for_family(ROOT, family, 'test')
        keys = {(direct_canonical(r['reactants']), direct_canonical(r['product']))
                for r in load_split_rows(test_path)}
        folder = ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs/artifacts' / ('REAXYS_' + family + '_SINGLE_CATMERGE')
        dictionaries = {s: Dictionary.load(str(folder / 'data-bin' / f'dict.{s}.txt')) for s in ('src', 'tgt')}
        bins = {s: indexed_dataset.make_dataset(str(folder / 'data-bin' / f'train.src-tgt.{s}'), 'mmap') for s in dictionaries}
        with (folder / 'train.src').open() as src, (folder / 'train.tgt').open() as tgt:
            found = False
            for index, (product, reactants) in enumerate(zip_longest(src, tgt)):
                assert product is not None and reactants is not None
                key = (direct_canonical(''.join(reactants.split())), direct_canonical(''.join(product.split())))
                if None not in key and key in keys:
                    for side, text in (('src', product), ('tgt', reactants)):
                        encoded = dictionaries[side].encode_line(text, add_if_not_exist=False)
                        assert torch.equal(encoded.long(), bins[side][index].long())
                    rows.append({'family': family, 'augmented_training_row': index,
                                 'reaction_sha256': hashlib.sha256(json.dumps(key).encode()).hexdigest(),
                                 'condition_test_sha256': sha(test_path),
                                 'prepared_manifest_sha256': sha(folder / 'manifest.json'),
                                 'direct_rdkit_exact_reaction_match': True, 'both_binary_tensors_match_text': True})
                    found = True
                    print('Confirmed binary overlap:', family, 'training row', index, flush=True)
                    break
            assert found, family
    write(ROOT / 'Experiment/final_release_audit_20260915/augmented_overlap_examples.json',
          {'positive_overlap_confirmed_in_all_six_families': True, 'examples': rows,
           'scope': 'one positive per family, independent whole-side RDKit canonicalization and actual binary tensor checks; not a recount of all overlaps'})


if __name__ == '__main__':
    main()
