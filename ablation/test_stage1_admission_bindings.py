"""Synthetic admission-binding tests; no real model or split is approved here."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_stage1_multiseed as runner


class AdmissionBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(runner, 'ROOT', self.root).start()
        self.folder = self.root / 'expert'
        self.folder.mkdir()
        self.base = self.root / 'base.pt'
        self.base.write_bytes(b'fixture, not a real checkpoint')
        from prosys_shared.mainline import split_file_for_family
        self.paths = [split_file_for_family(self.root, 'Beckmann', s) for s in ('train', 'val', 'test')]
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture\n')
        self.manifest = {'dataset': 'REAXYS_Beckmann_SINGLE_CATMERGE',
                         'output_sha256': {'data-bin/train.bin': 'toy-output-hash'},
                         'ready_for_formal_three_seed_training': True,
                         'training_admission_evidence': {}}
        self.receipts = {
            'base_heldout_boundary': {'base_checkpoint_sha256': runner.sha(self.base)},
            'expert_post_augmentation': {'dataset_output_sha256': {
                self.manifest['dataset']: self.manifest['output_sha256'].copy()}},
            'condition_validation_propagation': {
                'validation_protocol': 'strict_post_augmentation',
                'condition_split_sha256': {str(p.relative_to(self.root)): runner.sha(p) for p in self.paths}}}
        self.persist()

    def persist(self):
        for name, fields in self.receipts.items():
            path = self.root / (name + '.json')
            path.write_text(json.dumps({'kind': name, 'pass': True, 'training_admission_eligible': True, **fields}))
            self.manifest['training_admission_evidence'][name] = {'path': path.name, 'sha256': runner.sha(path)}
        (self.folder / 'manifest.json').write_text(json.dumps(self.manifest))

    def check(self, protocol='strict_post_augmentation'):
        runner.require_training_admission(self.folder / 'data-bin', self.base, protocol)

    def test_bound_fixture_is_accepted(self):
        self.check()

    def test_different_checkpoint_is_refused(self):
        self.base.write_bytes(b'different fixture')
        with self.assertRaisesRegex(ValueError, 'different base checkpoint'):
            self.check()

    def test_different_expert_outputs_are_refused(self):
        self.manifest['output_sha256']['data-bin/train.bin'] = 'changed-output-hash'
        self.persist()
        with self.assertRaisesRegex(ValueError, 'bind these expert outputs'):
            self.check()

    def test_changed_condition_split_is_refused(self):
        self.paths[1].write_text('changed validation fixture\n')
        with self.assertRaisesRegex(ValueError, 'condition split changed'):
            self.check()

    def test_wrong_validation_protocol_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'validation protocol differs'):
            self.check('fixed_original')

    def test_plan_or_replay_receipt_is_not_training_admission(self):
        self.receipts['base_heldout_boundary']['training_admission_eligible'] = False
        self.persist()
        with self.assertRaisesRegex(ValueError, 'Unverified scientific admission'):
            self.check()


if __name__ == '__main__':
    unittest.main()
