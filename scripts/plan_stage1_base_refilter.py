#!/usr/bin/env python3
"""Create an exact exclusion plan for a fresh base; do not alter data or train."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import ZERO, load_targets, sha, write


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    folder = ROOT / 'Experiment/stage1_split_repair_20260915/base_augmented'
    output = ROOT / 'Experiment/stage1_split_repair_20260915/base_refilter_plan'
    if output.exists():
        raise FileExistsError(output)
    receipt = folder / 'audit.json'
    audit = json.loads(receipt.read_text())
    for name, expected in audit['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Original base-audit input changed: ' + name)
    targets, _ = load_targets(ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs/artifacts')
    forbidden = set().union(*targets.values())
    records, lineage, remaining = [], {}, {}
    for split in ('train', 'val'):
        path = folder / (split + '.keys.bin')
        if sha(path) != audit['splits'][split]['keys_sha256']:
            raise ValueError('Full augmented membership file changed')
        count, excluded, seen, retained = 0, [], set(), set()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(32 * 65536), b''):
                if len(block) % 32:
                    raise ValueError('Truncated membership keys')
                for offset in range(0, len(block), 32):
                    key = block[offset:offset + 32]
                    if key == ZERO or key in forbidden:
                        excluded.append(count)
                        seen.add(key)
                    else:
                        retained.add(key)
                    count += 1
        if count != audit['splits'][split]['augmented_rows']:
            raise ValueError('Membership row count disagrees with full audit')
        if retained & forbidden or ZERO in retained:
            raise ValueError('Excluded plan still has forbidden reactions')
        lineage[split] = excluded
        remaining[split] = retained
        records.append({'split': split, 'source_augmented_pairs': count,
            'excluded_augmented_pairs': len(excluded), 'excluded_unique_reactions': len(seen),
            'retained_augmented_pairs': count - len(excluded), 'retained_unique_reactions': len(retained),
            'planned_remaining_heldout_overlaps': 0})
    if remaining['train'] & remaining['val']:
        raise ValueError('Further base train/validation collision repair is required')
    write(output / 'exclusion_indices.json', {'source_augmented_index_space': True, 'splits': lineage,
                                            'base_audit_sha256': sha(receipt)})
    write(output / 'summary.json', {'pass': True, 'scope': 'exclusion-plan arithmetic, not rebuilt data or model eligibility',
        'training_admission_eligible': False, 'splits': records, 'base_audit_sha256': sha(receipt),
        'exclusion_indices_sha256': sha(output / 'exclusion_indices.json'),
        'source_sha256': audit['source_sha256'], 'planned_base_train_validation_overlap': 0,
        'inputs_modified': False, 'training_launched': False,
        'checkpoint_warning': 'An existing checkpoint is not repaired by changing its dataset. Fresh training is required to implement this strict option.'})
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
