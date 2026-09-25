#!/usr/bin/env python3
"""Diagnose unmatched augmentation identities without changing training data."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import canonical, sha, write


def structural_key(reactants, product):
    from rdkit import Chem
    sides = []
    for text in (reactants, product):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError('Unparseable reaction side')
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        sides.append('.'.join(sorted(Chem.MolToSmiles(fragment, isomericSmiles=False)
                                    for fragment in Chem.GetMolFrags(mol, asMols=True))))
    return '>>'.join(sides)


def mapped_pairs(mapped):
    from rdkit import Chem
    parts = mapped.split('>')
    if len(parts) != 3:
        return
    reactants = Chem.MolFromSmiles(parts[0])
    product = Chem.MolFromSmiles(parts[2].split(' ')[0])
    if reactants is None or product is None:
        return
    fragments = Chem.GetMolFrags(reactants, asMols=True)
    for p in Chem.GetMolFrags(product, asMols=True):
        maps = {atom.GetAtomMapNum() for atom in p.GetAtoms() if atom.GetAtomMapNum()}
        selected = [r for r in fragments if any(a.GetAtomMapNum() in maps for a in r.GetAtoms())]
        if selected:
            yield '.'.join(Chem.MolToSmiles(r) for r in selected), Chem.MolToSmiles(p)


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--audit-root', type=Path, default=ROOT / 'Experiment/stage1_augmentation_identity_20260924')
    args = parser.parse_args()
    audit = args.audit_root.resolve()
    source = audit / (args.dataset + '.unmatched.jsonl.gz')
    receipt = audit / (args.dataset + '.json')
    output = audit / (args.dataset + '.trace.json')
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(receipt.read_text())
    if sha(source) != report['details_sha256']:
        raise ValueError('Unmatched details do not match audit receipt')
    with gzip.open(source, 'rt') as handle:
        unmatched = [json.loads(line) for line in handle]
    sources = [source, receipt, Path(__file__), ROOT / 'scripts/audit_stage1_base_augmented.py']
    expected = {}
    for split in sorted({row['split'] for row in unmatched}):
        targets = {structural_key(row['reactants'], row['product'])
                   for row in unmatched if row['split'] == split}
        raw = ROOT / 'data/editretro/datasets' / args.dataset / 'raw' / ('raw_' + split + '.csv')
        sources.append(raw)
        if sha(raw) != report['source_sha256'][str(raw.relative_to(ROOT))]:
            raise ValueError('Raw source has changed since membership audit')
        with raw.open() as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                mapped = row.get('mapped_reaction_smiles', row['reactants>reagents>production'])
                for r, p in mapped_pairs(mapped):
                    key = structural_key(r, p)
                    if key in targets:
                        expected.setdefault((split, key), []).append({
                            'raw_csv_data_index': index, 'pair_id': row.get('pair_id'),
                            'reactants': canonical(r), 'product': canonical(p)})
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    findings = []
    for row in unmatched:
        candidates = expected.get((row['split'], structural_key(row['reactants'], row['product'])), [])
        r, p = canonical(row['reactants']), canonical(row['product'])
        findings.append({**row, 'connectivity_matched_raw_transforms': [
            {**item, 'reactant_stereo_identity_equal': r == item['reactants'],
             'product_stereo_identity_equal': p == item['product']}
            for item in candidates]})
    if {name: sha(ROOT / name) for name in before} != before:
        raise ValueError('Tracing sources changed')
    write(output, {'dataset': args.dataset, 'unmatched_augmented_pairs': len(findings),
        'pairs_with_same_nonstereo_raw_transform': sum(bool(r['connectivity_matched_raw_transforms']) for r in findings),
        'source_sha256': before, 'findings': findings,
        'limitations': 'Nonstereo serialization discards isotope/stereo information. A matching raw transform is a candidate source, not exact historical row lineage. No data or checkpoint was changed.'})
    print(json.dumps({'output': str(output), 'unmatched': len(findings),
                      'nonstereo_matches': sum(bool(r['connectivity_matched_raw_transforms']) for r in findings)}))


if __name__ == '__main__':
    main()
