#!/usr/bin/env python3
"""Admit repaired experts only against a completed, audited scratch-50K base.

Input-only checking is safe during base training. Publication requires the
completed base receipt, preserves old manifests, and hard-links immutable data
into a new versioned root without duplicating the large tensors.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stage1_multiseed import prepared_databins, require_training_admission, sha, write
from scripts.verify_stage1_repaired_copy import verify_dataset


def checked_hashes(bindings):
    if not bindings:
        raise ValueError('Missing source bindings')
    for name, digest in bindings.items():
        path = (ROOT / name).resolve()
        path.relative_to(ROOT)
        if sha(path) != digest:
            raise ValueError('Evidence source changed: ' + name)


def completed_base(study):
    path = study / 'completion.json'
    if not path.is_file():
        raise ValueError('Base training has not completed; intermediate best is not admitted')
    record = json.loads(path.read_text())
    if (record.get('dataset') != 'USPTO_50K_FILTERED' or
            record.get('initialization') != 'random' or
            record.get('uses_full_data_or_weights') is not False):
        raise ValueError('Not a scratch-50K completion receipt')
    inputs_path = study / 'inputs.json'
    if record['inputs_sha256'] != sha(inputs_path):
        raise ValueError('Base training input receipt changed')
    inputs = json.loads(inputs_path.read_text())
    checked_hashes(inputs['source_sha256'])
    if inputs['base_audit_sha256'] != sha(study / 'base_audit/audit.json'):
        raise ValueError('Base training audit changed')
    trained = study / 'training/USPTO_50K_FILTERED/run'
    log = trained / 'train.log'
    if record['train_log_sha256'] != sha(log):
        raise ValueError('Base training log changed')
    text = log.read_text()
    if 'done training in' not in text or 'no existing checkpoint found' not in text:
        raise ValueError('Missing fresh-start or normal-completion log evidence')
    actual = record['actual_config']
    if (actual.get('restore_file') != 'checkpoint_last.pt' or
            Path(actual['data']).resolve() != ROOT / 'data/editretro/datasets/USPTO_50K_FILTERED/aug10/data-bin'):
        raise ValueError('Unexpected base data or checkpoint restoration')
    if set(record['checkpoints']) != {'checkpoint_best.pt', 'checkpoint_last.pt'}:
        raise ValueError('Missing best/last provenance')
    for name, digest in record['checkpoints'].items():
        if sha(trained / 'checkpoints' / name) != digest:
            raise ValueError('Base checkpoint changed: ' + name)
    return trained / 'checkpoints/checkpoint_best.pt'


def check_experts(study, prepared, workers, combined_path=None):
    from prosys_shared.mainline import FAMILY_ORDER, split_file_for_family
    base_path = study / 'base_audit/audit.json'
    combined_path = combined_path or study / 'expert_combined_audit.json'
    base = json.loads(base_path.read_text())
    combined = json.loads(combined_path.read_text())
    if (base.get('pass') is not True or combined.get('pass') is not True or
            combined.get('base_audit_pass') is not True or
            combined['base_membership_audit_sha256'] != sha(base_path)):
        raise ValueError('Missing passing combined expert/50K audit')
    checked_hashes(base['source_sha256'])
    checked_hashes(combined['source_sha256'])
    summary_path = prepared / 'summary.json'
    summary_hash = sha(summary_path)
    summary = json.loads(summary_path.read_text())
    expected = {'REAXYS_' + f + '_SINGLE_CATMERGE' for f in FAMILY_ORDER}
    if ({r['dataset'] for r in summary['families']} != expected or
            len(summary['families']) != len(expected) or
            {r['family'] for r in combined['families']} != set(FAMILY_ORDER) or
            len(combined['families']) != len(expected)):
        raise ValueError('Incomplete or duplicate family coverage')
    prepared_databins(prepared, FAMILY_ORDER)
    outputs, conditions = {}, {}
    for row in combined['families']:
        family = row['family']
        ds = 'REAXYS_' + family + '_SINGLE_CATMERGE'
        folder = prepared / 'artifacts' / ds
        manifest = json.loads((folder / 'manifest.json').read_text())
        outputs[ds] = manifest['output_sha256']
        if (row['expert_validation_vs_base_train_or_validation'] != 0 or
                row['training_vs_condition_test_overlap'] != 0 or
                row['training_vs_condition_validation_overlap'] != 0 or
                row['splits']['val']['condition_overlap']['test'] != 0 or
                any(row['augmented_cross_split_overlaps'].values()) or
                any(s['unparseable_pair_count'] for s in row['splits'].values())):
            raise ValueError('Expert boundary violation: ' + family)
        for split in manifest['splits']:
            name = split['split']
            if row['splits'][name]['augmented_pairs'] != split['retained_pairs']:
                raise ValueError('Audited expert count differs from prepared tensors')
            for side in ('src', 'tgt'):
                path = folder / (name + '.' + side)
                if combined['source_sha256'].get(str(path.relative_to(ROOT))) != sha(path):
                    raise ValueError('Combined audit belongs to different expert inputs')
        for split in ('train', 'val', 'test'):
            path = split_file_for_family(ROOT, family, split)
            name = str(path.relative_to(ROOT))
            conditions[name] = sha(path)
            if combined['source_sha256'].get(name) != conditions[name]:
                raise ValueError('Original condition split is not bound by combined audit')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        verified = list(pool.map(verify_dataset, summary['families']))
    checked_hashes(combined['source_sha256'])
    if sha(summary_path) != summary_hash:
        raise ValueError('Expert summary changed during verification')
    return {'pass': True, 'source_summary_sha256': summary_hash,
            'combined_audit_sha256': sha(combined_path), 'base_audit_sha256': sha(base_path),
            'dataset_output_sha256': outputs, 'condition_split_sha256': conditions,
            'independent_copy_checks': verified, 'training_admission_eligible': False,
            'scope': 'Input verification only; completed base weights still required'}


def publish(study, prepared, output, report, base):
    if report.get('pass') is not True:
        raise ValueError('Cannot publish failed expert checks')
    if sha(prepared / 'summary.json') != report['source_summary_sha256']:
        raise ValueError('Expert summary changed before publication')
    if output.exists():
        raise FileExistsError('Refusing to replace admitted inputs: ' + str(output))
    output.mkdir(parents=True)
    evidence = {}
    for kind, extra in (
            ('expert_post_augmentation', {'dataset_output_sha256': report['dataset_output_sha256']}),
            ('base_heldout_boundary', {'base_checkpoint_sha256': sha(base),
                                     'base_completion_sha256': sha(study / 'completion.json')}),
            ('condition_validation_propagation', {
                'validation_protocol': 'strict_post_augmentation',
                'condition_split_sha256': report['condition_split_sha256'],
                'condition_splits_changed': False,
                'reason': 'Fresh base and repaired experts exclude original condition held-outs; no validation subset change is needed.'})):
        path = output / (kind + '.json')
        write(path, {'kind': kind, 'pass': True, 'training_admission_eligible': True,
                     'base_audit_sha256': report['base_audit_sha256'],
                     'combined_audit_sha256': report['combined_audit_sha256'],
                     'source_summary_sha256': report['source_summary_sha256'], **extra})
        evidence[kind] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
    summary = json.loads((prepared / 'summary.json').read_text())
    for entry in summary['families']:
        source = prepared / 'artifacts' / entry['dataset']
        destination = output / 'artifacts' / entry['dataset']
        manifest = json.loads((source / 'manifest.json').read_text())
        for name, digest in manifest['output_sha256'].items():
            old, new = source / name, destination / name
            old.resolve().relative_to(source.resolve())
            new.parent.mkdir(parents=True, exist_ok=True)
            os.link(old, new)
            if sha(new) != digest:
                raise ValueError('Expert data changed during admission')
        manifest.update(ready_for_formal_three_seed_training=True,
                        training_admission_evidence=evidence,
                        admission_source_manifest_sha256=entry['manifest_sha256'])
        write(destination / 'manifest.json', manifest)
        entry.update(path=str(destination), manifest_sha256=sha(destination / 'manifest.json'))
    summary.update(protocol_selection_pending=False, ready_for_formal_three_seed_training=True,
                   base_checkpoint=str(base), base_checkpoint_sha256=sha(base))
    write(output / 'summary.json', summary)
    from prosys_shared.mainline import FAMILY_ORDER
    databins = prepared_databins(output, FAMILY_ORDER)
    for databin in databins.values():
        require_training_admission(databin, base, 'strict_post_augmentation')
    write(output / 'admission_complete.json', {'pass': True, 'source_checks': report,
          'base_checkpoint_sha256': sha(base), 'summary_sha256': sha(output / 'summary.json'),
          'scope': 'Fixed completed 50K base, fresh family training; no old model/result promotion'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-study', type=Path, default=ROOT / 'Experiment/stage1_50k_from_scratch_20260924')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'Experiment/stage1_split_repair_20260915/expert_inputs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--combined-audit', type=Path,
                        help='Versioned combined audit for a new prepared expert copy')
    parser.add_argument('--base-fidelity-audit', type=Path,
                        help='Full raw-to-augmentation identity receipt for the completed base')
    parser.add_argument('--check-inputs-only', action='store_true')
    args = parser.parse_args()
    study, prepared = args.base_study.resolve(), args.prepared.resolve()
    if args.combined_audit and not args.base_fidelity_audit:
        parser.error('New expert copies require --base-fidelity-audit')
    fidelity = None
    if args.base_fidelity_audit:
        fidelity = json.loads(args.base_fidelity_audit.read_text())
        if fidelity.get('pass') is not True or fidelity.get('dataset') != 'USPTO_50K_FILTERED':
            raise ValueError('Base augmentation identity audit failed')
        checked_hashes(fidelity['source_sha256'])
    base = None if args.check_inputs_only else completed_base(study)
    report = check_experts(study, prepared, args.workers,
                           args.combined_audit.resolve() if args.combined_audit else None)
    if fidelity is not None:
        report['base_fidelity_audit_sha256'] = sha(args.base_fidelity_audit)
    if args.check_inputs_only:
        write(study / 'expert_preflight.json', report)
        print('Expert inputs verified; no training admission or checkpoint promotion')
    else:
        if completed_base(study) != base:
            raise ValueError('Completed base changed during expert verification')
        output = args.output.resolve() if args.output else study / 'admitted_expert_inputs'
        publish(study, prepared, output, report, base)
        print('Expert training admission verified:', output)


if __name__ == '__main__':
    main()
