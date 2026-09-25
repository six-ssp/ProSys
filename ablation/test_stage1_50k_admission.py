"""Synthetic guards for scratch-50K completion; these do not certify real weights."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import admit_stage1_50k_experts as admission


class ScratchBaseAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patch = patch.object(admission, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.study = self.root / 'study'
        self.study.mkdir()
        self.trained = self.study / 'training/USPTO_50K_FILTERED/run'
        (self.trained / 'checkpoints').mkdir(parents=True)
        self.source = self.root / 'source.txt'
        self.source.write_text('fixed source\n')
        (self.study / 'base_audit').mkdir()
        self.audit = self.study / 'base_audit/audit.json'
        self.audit.write_text('{"pass": true}\n')
        self.inputs = self.study / 'inputs.json'
        self.inputs.write_text(json.dumps({
            'source_sha256': {'source.txt': admission.sha(self.source)},
            'base_audit_sha256': admission.sha(self.audit)}))
        self.log = self.trained / 'train.log'
        self.log.write_text('no existing checkpoint found\ndone training in 1 seconds\n')
        for name in ('checkpoint_best.pt', 'checkpoint_last.pt'):
            (self.trained / 'checkpoints' / name).write_bytes(b'synthetic fixture, not a model')
        self.record = {
            'dataset': 'USPTO_50K_FILTERED', 'initialization': 'random',
            'uses_full_data_or_weights': False, 'inputs_sha256': admission.sha(self.inputs),
            'train_log_sha256': admission.sha(self.log),
            'actual_config': {'restore_file': 'checkpoint_last.pt',
                'data': str(self.root / 'data/editretro/datasets/USPTO_50K_FILTERED/aug10/data-bin')},
            'checkpoints': {p.name: admission.sha(p) for p in (self.trained / 'checkpoints').iterdir()}}
        self.persist()

    def persist(self):
        (self.study / 'completion.json').write_text(json.dumps(self.record))

    def test_complete_bound_fixture_is_accepted(self):
        self.assertEqual(admission.completed_base(self.study), self.trained / 'checkpoints/checkpoint_best.pt')

    def test_intermediate_best_is_not_completion(self):
        (self.study / 'completion.json').unlink()
        with self.assertRaisesRegex(ValueError, 'has not completed'):
            admission.completed_base(self.study)

    def test_full_initialization_is_refused(self):
        self.record['uses_full_data_or_weights'] = True
        self.persist()
        with self.assertRaisesRegex(ValueError, 'scratch-50K'):
            admission.completed_base(self.study)

    def test_changed_training_source_is_refused(self):
        self.source.write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            admission.completed_base(self.study)

    def test_changed_checkpoint_is_refused(self):
        (self.trained / 'checkpoints/checkpoint_best.pt').write_bytes(b'new weights')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            admission.completed_base(self.study)

    def test_changed_audit_is_refused(self):
        self.audit.write_text('{"pass": false}\n')
        with self.assertRaisesRegex(ValueError, 'audit changed'):
            admission.completed_base(self.study)

    def test_restored_weights_are_refused(self):
        self.record['actual_config']['restore_file'] = 'pretrain.pt'
        self.persist()
        with self.assertRaisesRegex(ValueError, 'checkpoint restoration'):
            admission.completed_base(self.study)

    def test_missing_fresh_start_log_is_refused(self):
        self.log.write_text('done training in 1 seconds\n')
        self.record['train_log_sha256'] = admission.sha(self.log)
        self.persist()
        with self.assertRaisesRegex(ValueError, 'fresh-start'):
            admission.completed_base(self.study)

    def test_best_without_last_is_refused(self):
        del self.record['checkpoints']['checkpoint_last.pt']
        self.persist()
        with self.assertRaisesRegex(ValueError, 'best/last'):
            admission.completed_base(self.study)


class AdmissionPublicationTests(unittest.TestCase):
    def test_hardlinked_inputs_preserve_old_manifest_and_pass_runner_checks(self):
        from scripts import run_stage1_multiseed as runner
        from prosys_shared import mainline
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, output, study = root / 'prepared', root / 'new', root / 'study'
            study.mkdir()
            (study / 'completion.json').write_text('synthetic completion')
            base = root / 'base.pt'
            base.write_bytes(b'synthetic checkpoint')
            dataset = 'REAXYS_Beckmann_SINGLE_CATMERGE'
            folder = prepared / 'artifacts' / dataset
            (folder / 'data-bin').mkdir(parents=True)
            data = folder / 'data-bin/dict.src.txt'
            data.write_text('synthetic vocabulary')
            outputs = {'data-bin/dict.src.txt': admission.sha(data)}
            manifest = {'dataset': dataset, 'output_sha256': outputs, 'splits': [],
                        'ready_for_formal_three_seed_training': False}
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            old_hash = admission.sha(folder / 'manifest.json')
            summary = {'preparatory_copy_complete': True, 'families': [
                {'dataset': dataset, 'path': str(folder), 'manifest_sha256': old_hash}]}
            (prepared / 'summary.json').write_text(json.dumps(summary))
            conditions = {}
            for split in ('train', 'val', 'test'):
                path = mainline.split_file_for_family(root, 'Beckmann', split)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic split ' + split)
                conditions[str(path.relative_to(root))] = admission.sha(path)
            report = {'pass': True, 'source_summary_sha256': admission.sha(prepared / 'summary.json'),
                      'base_audit_sha256': 'synthetic-base-audit',
                      'combined_audit_sha256': 'synthetic-combined-audit',
                      'dataset_output_sha256': {dataset: outputs}, 'condition_split_sha256': conditions}
            with patch.object(admission, 'ROOT', root), patch.object(runner, 'ROOT', root), \
                    patch.object(mainline, 'FAMILY_ORDER', ['Beckmann']):
                admission.publish(study, prepared, output, report, base)
                linked = output / 'artifacts' / dataset / 'data-bin/dict.src.txt'
                self.assertEqual(data.stat().st_ino, linked.stat().st_ino)
                self.assertEqual(admission.sha(folder / 'manifest.json'), old_hash)
                self.assertTrue(json.loads((output / 'admission_complete.json').read_text())['pass'])
                with self.assertRaises(FileExistsError):
                    admission.publish(study, prepared, output, report, base)

    def test_failed_report_is_refused_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'new'
            with self.assertRaisesRegex(ValueError, 'failed expert checks'):
                admission.publish(root, root, output, {'pass': False}, root / 'base.pt')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
