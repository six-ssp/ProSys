#!/usr/bin/env python3
"""Trace observed expert train/held-out collisions to mapped raw transformations."""

from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import ZERO, canonical, paired_lines, reaction_key, sha, write


def transformed_keys(mapped_reaction):
    """Reproduce chemical membership of splitting/alignment, not random root order."""
    from rdkit import Chem
    parts = mapped_reaction.split('>')
    if len(parts) != 3:
        return []
    reactants = Chem.MolFromSmiles(parts[0])
    product = Chem.MolFromSmiles(parts[2].split(' ')[0])
    if reactants is None or product is None:
        return []
    rfrags = Chem.GetMolFrags(reactants, asMols=True)
    pfrags = Chem.GetMolFrags(product, asMols=True)
    result = []
    for pfrag in pfrags:
        maps = {a.GetAtomMapNum() for a in pfrag.GetAtoms() if a.GetAtomMapNum()}
        selected = [frag for frag in rfrags if any(a.GetAtomMapNum() in maps for a in frag.GetAtoms())]
        if not selected:
            continue
        key = reaction_key('.'.join(Chem.MolToSmiles(frag) for frag in selected), Chem.MolToSmiles(pfrag))
        if key != ZERO:
            result.append((key, len(pfrags), len(rfrags) - len(selected)))
    return result


def trace(family):
    from rdkit import RDLogger
    from prosys_shared.mainline import load_split_rows, split_file_for_family
    RDLogger.DisableLog('rdApp.*')
    prepared = ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs/artifacts' / ('REAXYS_' + family + '_SINGLE_CATMERGE')
    heldout, paths = {}, []
    for split in ('val', 'test'):
        path = split_file_for_family(ROOT, family, split)
        paths.append(path)
        for row in load_split_rows(path):
            key = reaction_key(row['reactants'], row['product'])
            if key == ZERO:
                raise ValueError('Unparseable held-out row')
            heldout.setdefault(key, set()).add(split)
    collisions = {}
    for index, p, r in paired_lines(prepared, 'train'):
        key = reaction_key(''.join(r.split()), ''.join(p.split()))
        if key in heldout:
            item = collisions.setdefault(key, {'first_augmented_index': index, 'augmented_pairs': 0, 'raw_sources': []})
            item['augmented_pairs'] += 1
    raw = ROOT / 'data/editretro/datasets' / ('REAXYS_' + family + '_SINGLE_CATMERGE') / 'raw/raw_train.csv'
    paths += [raw, prepared / 'train.src', prepared / 'train.tgt', Path(__file__)]
    with raw.open() as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            original = row['reactants>reagents>production'].split('>')
            full_key = reaction_key(original[0], original[2]) if len(original) == 3 else ZERO
            for key, products, removed in transformed_keys(row.get('mapped_reaction_smiles', '')):
                if key in collisions:
                    collisions[key]['raw_sources'].append({'raw_csv_data_index': index,
                        'pair_id': row['pair_id'], 'original_full_reaction_sha256': full_key.hex(),
                        'original_full_reaction_in_heldout': full_key in heldout,
                        'product_components_before_splitting': products,
                        'reactant_components_removed_by_mapping_selection': removed,
                        'transformation_changes_full_reaction_identity': key != full_key})
    rows = [{'reaction_sha256': key.hex(), 'heldout_splits': sorted(heldout[key]), **value}
            for key, value in sorted(collisions.items())]
    report = {'family': family, 'unique_train_condition_heldout_collisions': len(rows),
        'collisions_with_matching_mapped_raw_transform': sum(bool(r['raw_sources']) for r in rows),
        'collisions_without_matching_raw_transform': sum(not r['raw_sources'] for r in rows),
        'examples': rows, 'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'scope': 'actual augmented collisions matched to deterministic product splitting / mapped precursor selection on retained raw training records',
        'limitation': 'matching transformed chemistry explains candidate source records; absent historical row-order manifest is not reconstructed'}
    out = ROOT / 'Experiment/stage1_split_repair_20260915/raw_lineage' / (family + '.json')
    write(out, report)
    summary = {k: v for k, v in report.items() if k not in ('examples', 'source_sha256')}
    summary['detailed_receipt_sha256'] = sha(out)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    from prosys_shared.mainline import FAMILY_ORDER
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(trace, FAMILY_ORDER))
    write(ROOT / 'Experiment/stage1_split_repair_20260915/raw_transformation_trace.json', {'families': rows})


if __name__ == '__main__':
    main()
