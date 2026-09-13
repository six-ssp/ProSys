#!/usr/bin/env python3
"""Replay the retained EditRetro input guard on audited invalid USPTO records."""

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from rdkit import Chem, RDLogger
    from scripts.audit_uspto_pretraining import key
    from scripts.run_stage1_multiseed import sha, write
    RDLogger.DisableLog('rdApp.*')
    output = ROOT / 'Experiment/project_completion_20260913/uspto_audit'
    target = output / 'invalid_rows_guard_replay.json'
    if target.exists():
        raise FileExistsError(target)
    audit = json.loads((output / 'audit.json').read_text())
    source = ROOT / 'stage1_retrosynthesis/preprocess/preprocess_data.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'is_parseable_reaction_sides')
    namespace = {'Chem': Chem}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    guard = namespace['is_parseable_reaction_sides']
    records = []
    for split in ('train', 'val'):
        invalid_ids = audit['base_splits'][split]['invalid_ids']
        wanted = set(invalid_ids)
        n = 0
        raw = ROOT / 'data/editretro/datasets/USPTO_STAGE2_FILTERED/raw' / f'raw_{split}.csv'
        with raw.open() as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                if row['id'] not in wanted:
                    continue
                rxn = row['reactants>reagents>production']
                parts = rxn.split('>')
                if len(parts) == 3 and key(parts[0], parts[2]) is not None:
                    continue
                accepted = guard(rxn)
                records.append({'base_split': split, 'csv_line': row_number, 'id': row['id'],
                    'raw_reaction_sha256': hashlib.sha256(rxn.encode()).hexdigest(),
                    'accepted_by_retained_preprocessing_guard': accepted})
                n += 1
        if n != len(invalid_ids):
            raise ValueError('Invalid row identity/count replay mismatch')
    write(target, {'rows': records, 'count': len(records),
          'rejected_count': sum(not r['accepted_by_retained_preprocessing_guard'] for r in records),
          'preprocessor_sha256': sha(source), 'guard_source': ast.get_source_segment(source.read_text(), function),
          'scope': 'Replay of the currently retained preprocessing input guard on exact original raw rows',
          'historical_raw_to_bin_membership_proven': False,
          'limitation': 'No per-row historical augmentation manifest was retained. This proves current guard rejection, not retrospective membership in the July base-training tensors.'})
    print(json.dumps({'count': len(records), 'rejected': sum(not r['accepted_by_retained_preprocessing_guard'] for r in records)}))


if __name__ == '__main__':
    main()
