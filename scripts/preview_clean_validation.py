#!/usr/bin/env python3
"""Read-only preview of base-disjoint validation; audit retained text/bin pairs."""

import csv
from functools import lru_cache
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'Experiment/project_completion_20260913/validation_preview'


@lru_cache(maxsize=65536)
def reaction_key(reactants, product):
    from scripts.audit_uspto_pretraining import key
    pair = key(reactants, product)
    if pair is None:
        return None
    return pair[0].hex()


def inspect_augmented(aug, excluded):
    import torch
    from fairseq.data import Dictionary, indexed_dataset
    dictionaries = {s: Dictionary.load(str(aug / 'data-bin' / f'dict.{s}.txt')) for s in ('src', 'tgt')}
    bins = {s: indexed_dataset.make_dataset(str(aug / 'data-bin' / f'valid.src-tgt.{s}'), 'mmap')
            for s in ('src', 'tgt')}
    if any(b is None for b in bins.values()):
        raise FileNotFoundError('Missing validation data-bin under ' + str(aug))
    indices, invalid, rows = [], [], 0
    unknown_rows = {'src': 0, 'tgt': 0}
    reaction_keys = set()
    with (aug / 'val.src').open() as source, (aug / 'val.tgt').open() as target:
        for index, (src, tgt) in enumerate(zip_longest(source, target)):
            if src is None or tgt is None:
                raise ValueError('Validation source/target text length mismatch')
            for side, line in (('src', src), ('tgt', tgt)):
                encoded = dictionaries[side].encode_line(line, add_if_not_exist=False)
                if index >= len(bins[side]) or not torch.equal(encoded.long(), bins[side][index].long()):
                    raise ValueError(f'Text/bin disagreement at {aug} {side} row {index}')
                unknown_rows[side] += int(bool((encoded == dictionaries[side].unk()).any()))
            k = reaction_key(''.join(tgt.split()), ''.join(src.split()))
            if k is None:
                invalid.append(index)
            else:
                reaction_keys.add(k)
            if k in excluded:
                indices.append(index)
            rows += 1
    if any(len(b) != rows for b in bins.values()):
        raise ValueError('Validation bin has extra rows')
    return {'rows': rows, 'remove_indices': indices, 'remove_rows': len(indices),
            'remaining_rows': rows - len(indices), 'unparseable_text_rows': invalid,
            'unknown_token_rows': unknown_rows, 'text_equals_bin_all_rows': True,
            'unique_reaction_keys': len(reaction_keys), 'present_excluded_keys': sorted(reaction_keys & excluded)}


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from prosys_shared.mainline import FAMILY_ORDER, split_file_for_family
    from scripts.run_stage1_multiseed import sha, write
    sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
    if (OUTPUT / 'preview.json').exists():
        raise FileExistsError(OUTPUT / 'preview.json')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit_file = ROOT / 'Experiment/project_completion_20260913/uspto_audit/audit.json'
    audit = json.loads(audit_file.read_text())
    excluded = {r['reaction_sha256'] for r in audit['exact_overlap_records']}
    rows, paths = [], [audit_file, Path(__file__)]
    for family in FAMILY_ORDER:
        val = split_file_for_family(ROOT, family, 'val')
        paths.append(val)
        total, remaining, dropped = set(), set(), []
        n = 0
        with val.open() as handle:
            for index, line in enumerate(handle):
                fields = line.rstrip('\n').split('\t')
                k = reaction_key(fields[1], fields[2])
                if k is None:
                    raise ValueError('Invalid condition validation reaction')
                total.add(k)
                if k in excluded:
                    dropped.append(index)
                else:
                    remaining.add(k)
                n += 1
        dataset = ROOT / 'data/editretro/datasets' / ('REAXYS_' + family + '_SINGLE_CATMERGE')
        raw = dataset / 'raw/raw_val.csv'
        paths.append(raw)
        raw_count, raw_dropped, raw_invalid = 0, [], []
        with raw.open() as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                parts = row['reactants>reagents>production'].split('>')
                k = reaction_key(parts[0], parts[2]) if len(parts) == 3 else None
                if k is None:
                    raw_invalid.append(index)
                if k in excluded:
                    raw_dropped.append(index)
                raw_count += 1
        aug = dataset / 'aug10'
        binary = inspect_augmented(aug, excluded)
        paths += [aug / 'val.src', aug / 'val.tgt']
        paths += sorted(p for p in (aug / 'data-bin').iterdir()
                        if p.name.startswith(('dict.', 'valid.')))
        row = {'family': family, 'condition_val_rows': n,
               'condition_val_unique_reactions': len(total), 'excluded_unique_reactions': len(total & excluded),
               'remaining_unique_reactions': len(remaining), 'condition_remove_indices': dropped,
               'condition_remove_rows': len(dropped), 'condition_remaining_rows': n - len(dropped),
               'stage1_raw_val_rows': raw_count, 'stage1_raw_remove_indices': raw_dropped,
               'stage1_raw_invalid_indices': raw_invalid, 'augmented_validation': binary}
        rows.append(row)
        write(OUTPUT / f'{family}.json', row)
        print(family, row['excluded_unique_reactions'], 'reactions;', len(dropped),
              'condition rows;', binary['remove_rows'], 'augmented rows', flush=True)
    write(OUTPUT / 'preview.json', {'scope': 'read-only candidate clean validation subset; no split/model modified',
          'family_rows': rows, 'excluded_unique_base_reactions': len(excluded),
          'current_input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
          'historical_membership_claim': False,
          'limitation': 'Text/bin equality binds currently retained family validation artifacts, not absent historical raw-to-bin manifests.'})
    fields = ['family', 'condition_val_unique_reactions', 'excluded_unique_reactions',
              'remaining_unique_reactions', 'condition_val_rows', 'condition_remove_rows',
              'condition_remaining_rows']
    with (OUTPUT / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)


if __name__ == '__main__':
    main()
