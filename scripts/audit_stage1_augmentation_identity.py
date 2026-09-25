#!/usr/bin/env python3
"""Check actual augmented reaction identities against deterministic raw transforms.

This checks membership, not missing historical row-order lineage. It does not
edit training inputs, checkpoints, split membership or model selection.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import ZERO, paired_lines, reaction_key, sha, write
from scripts.trace_stage1_augmented_collisions import transformed_keys


def inspect_dataset(task):
    dataset, augmented, splits, output = task
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    raw = ROOT / 'data/editretro/datasets' / dataset / 'raw'
    sources = [Path(__file__), ROOT / 'scripts/trace_stage1_augmented_collisions.py',
               ROOT / 'scripts/audit_stage1_base_augmented.py', ROOT / 'prosys_shared/features.py']
    sources += [raw / ('raw_' + split + '.csv') for split in splits]
    sources += [augmented / (split + '.' + side) for split in splits for side in ('src', 'tgt')]
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    reports = {}
    details = output / (dataset + '.unmatched.jsonl.gz')
    with gzip.open(details, 'wt') as handle:
        for split in splits:
            expected, raw_rows, raw_without_keys = set(), 0, 0
            with (raw / ('raw_' + split + '.csv')).open() as source:
                for row in csv.DictReader(source):
                    raw_rows += 1
                    # Family files carry the mapper result in a separate column;
                    # the upstream 50K source already stores mapped reactions.
                    mapped = (row.get('mapped_reaction_smiles', '') if 'mapped_reaction_smiles' in row
                              else row['reactants>reagents>production'])
                    keys = transformed_keys(mapped)
                    raw_without_keys += not bool(keys)
                    expected.update(item[0] for item in keys)
            observed, missing, count, invalid = set(), set(), 0, 0
            mismatch_rows = 0
            for index, product, reactants in paired_lines(augmented, split):
                product, reactants = ''.join(product.split()), ''.join(reactants.split())
                key = reaction_key(reactants, product)
                observed.add(key)
                invalid += key == ZERO
                count += 1
                if key not in expected:
                    mismatch_rows += 1
                    missing.add(key)
                    handle.write(json.dumps({'split': split, 'augmented_row': index,
                        'reaction_sha256': key.hex(), 'product': product,
                        'reactants': reactants}) + '\n')
            reports[split] = {'raw_rows': raw_rows, 'raw_without_transformed_keys': raw_without_keys,
                'expected_unique_transformed_reactions': len(expected), 'augmented_rows': count,
                'actual_unique_reactions': len(observed), 'invalid_augmented_pairs': invalid,
                'unmatched_augmented_pairs': mismatch_rows, 'unmatched_unique_reactions': len(missing)}
    if {name: sha(ROOT / name) for name in before} != before:
        raise ValueError('Inputs changed during augmentation identity audit')
    result = {'dataset': dataset, 'splits': reports, 'source_sha256': before,
              'details_sha256': sha(details),
              'pass': all(r['unmatched_augmented_pairs'] == 0 and r['invalid_augmented_pairs'] == 0
                          for r in reports.values()),
              'scope': 'retained augmented identities are members of same-split raw mapped product/precursor transforms',
              'limitations': 'Membership only, not historical raw-row alignment; unmatched keys need cause tracing, not automatic deletion.'}
    write(output / (dataset + '.json'), result)
    print(json.dumps({'dataset': dataset, 'pass': result['pass'], 'splits': reports}), flush=True)
    return result


def main():
    from prosys_shared.mainline import FAMILY_ORDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'Experiment/stage1_augmentation_identity_20260924')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new audit root; no failed receipt is overwritten')
    output.mkdir(parents=True)
    base_ds = 'USPTO_50K_FILTERED'
    tasks = [(base_ds, ROOT / 'data/editretro/datasets' / base_ds / 'aug10', ('train', 'val'), output)]
    for family in FAMILY_ORDER:
        ds = 'REAXYS_' + family + '_SINGLE_CATMERGE'
        tasks.append((ds, ROOT / 'Experiment/stage1_50k_from_scratch_20260924/admitted_expert_inputs/artifacts' / ds,
                      ('train', 'val', 'test'), output))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(inspect_dataset, tasks))
    write(output / 'summary.json', {'pass': all(r['pass'] for r in results),
          'datasets': [{'dataset': r['dataset'], 'pass': r['pass'], 'splits': r['splits'],
                       'receipt_sha256': sha(output / (r['dataset'] + '.json'))} for r in results],
          'scope': 'source-to-augmented chemical membership, not a model-performance certificate'})


if __name__ == '__main__':
    main()
