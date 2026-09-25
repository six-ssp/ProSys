#!/usr/bin/env python3
"""Apply the audited strict base exclusion plan to a new, unpromoted copy."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import sha, write
from scripts.build_stage1_nonempty_inputs import build_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=ROOT / 'Experiment/stage1_split_repair_20260915/base_refilter_plan')
    parser.add_argument('--output', type=Path, default=ROOT / 'Experiment/stage1_split_repair_20260915/base_inputs')
    args = parser.parse_args()
    summary_path, indices_path = args.plan / 'summary.json', args.plan / 'exclusion_indices.json'
    plan = json.loads(summary_path.read_text())
    if plan.get('pass') is not True or plan.get('training_launched') is not False:
        raise ValueError('Expected a verified, unexecuted exclusion plan')
    if sha(indices_path) != plan['exclusion_indices_sha256']:
        raise ValueError('Exclusion indices changed')
    indices = json.loads(indices_path.read_text())
    if indices['base_audit_sha256'] != plan['base_audit_sha256']:
        raise ValueError('Exclusion plan/audit linkage differs')
    for name, expected in plan['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Exclusion plan source changed: ' + name)
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(ROOT).free < 7 * 1024**3:
        raise RuntimeError('Less than 7 GiB free; refusing a base-data copy')
    args.output.mkdir(parents=True)
    write(args.output / 'status.json', {'phase': 'building', 'training_launched': False,
                                      'exclusion_plan_sha256': sha(summary_path)})
    source = ROOT / 'data/editretro/datasets/USPTO_STAGE2_FILTERED/aug10'
    destination = args.output / 'USPTO_STAGE2_FILTERED'
    result = build_one(source, destination, expected_hashes=plan['source_sha256'],
                       expected_empty={'train': [], 'val': []}, excluded_indices=indices['splits'])
    expected = {r['split']: r for r in plan['splits']}
    for row in result['splits']:
        target = expected[row['split']]
        if row['retained_pairs'] != target['retained_augmented_pairs'] or \
                row['excluded_by_external_plan_pairs'] != target['excluded_augmented_pairs']:
            raise ValueError('Built dataset counts differ from exclusion plan')
    if sha(summary_path) != json.loads((args.output / 'status.json').read_text())['exclusion_plan_sha256']:
        raise ValueError('Exclusion plan changed during preparation')
    write(args.output / 'summary.json', {'preparatory_copy_complete': True, 'dataset': result,
        'exclusion_plan_sha256': sha(summary_path), 'base_audit_sha256': plan['base_audit_sha256'],
        'training_launched': False, 'training_admission_eligible': False, 'original_inputs_modified': False,
        'pending': ['independent rebuilt-input audit', 'training decision', 'fresh base training'],
        'warning': 'The existing base checkpoint is not repaired by this filtered input copy.'})
    write(args.output / 'status.json', {'phase': 'prepared_not_admitted', 'training_launched': False})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
