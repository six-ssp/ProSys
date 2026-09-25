"""Bounded Stage-1 wiring tests; do not claim to train EditRetro."""

import json
import os
from pathlib import Path
import hashlib
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Stage1SeedTests(unittest.TestCase):
    def test_tensor_preparation_alone_cannot_admit_training(self):
        from scripts.run_stage1_multiseed import require_training_admission
        with self.assertRaisesRegex(ValueError, 'Legacy raw inputs'):
            require_training_admission(None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps({'ready_for_formal_three_seed_training': False}))
            with self.assertRaisesRegex(ValueError, 'not scientific training admission'):
                require_training_admission(root / 'data-bin')
            (root / 'manifest.json').write_text(json.dumps({'ready_for_formal_three_seed_training': True}))
            with self.assertRaisesRegex(ValueError, 'Missing scientific admission evidence'):
                require_training_admission(root / 'data-bin')

    def test_explicit_seed_fixed_prepared_inputs_and_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ds = 'REAXYS_Beckmann_SINGLE_CATMERGE'
            databin = root / 'verified_copy' / ds / 'data-bin'
            databin.mkdir(parents=True)
            for name in ('dict.src.txt', 'dict.tgt.txt'):
                (databin / name).write_text('token 1\n')
            for split in ('train', 'valid'):
                for side in ('src', 'tgt'):
                    for extension in ('bin', 'idx'):
                        (databin / f'{split}.src-tgt.{side}.{extension}').write_bytes(b'x')
            base = root / 'base.pt'
            base.write_bytes(b'fake checkpoint; not used for training')
            recorder = root / 'python_stub'
            recorder.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                'open(os.environ["CALLS"],"w").write(json.dumps(sys.argv[1:]))\n')
            recorder.chmod(0o755)
            calls = root / 'calls.json'
            env = dict(os.environ, PYTHON_BIN=str(recorder), CALLS=str(calls), BASE_CKPT=str(base),
                       SKIP_PREPARE='1', DATA_BIN=str(databin), SEED='2', RUN_NAME='fixed', MAX_TOKENS='8192')
            script = ROOT / 'stage1_retrosynthesis/scripts/run_family_finetune_one.sh'
            subprocess.run(['bash', str(script), str(root), ds], env=env, check=True, capture_output=True)
            command = json.loads(calls.read_text())
            self.assertEqual(command[command.index('--seed') + 1], '2')
            self.assertEqual(command[command.index('--fixed-validation-seed') + 1], '7')
            self.assertEqual(command[command.index('--max-tokens') + 1], '8192')
            self.assertIn('--reset-optimizer', command)
            self.assertIn('--no-epoch-checkpoints', command)
            self.assertEqual(command[1], str(databin))

    def test_prepared_receipt_rejects_changed_output(self):
        from scripts.run_stage1_multiseed import prepared_databins
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ds = 'REAXYS_Beckmann_SINGLE_CATMERGE'
            dataset = root / 'artifacts' / ds
            databin = dataset / 'data-bin'
            databin.mkdir(parents=True)
            payload = databin / 'train.src-tgt.src.bin'
            payload.write_bytes(b'fixed input')
            manifest = dataset / 'manifest.json'
            manifest.write_text(json.dumps({'dataset': ds, 'splits': [
                {'all_input_text_bin_pairs_equal': True, 'all_retained_token_tensors_unchanged': True}],
                'output_sha256': {'data-bin/' + payload.name: hashlib.sha256(payload.read_bytes()).hexdigest()}}))
            (root / 'summary.json').write_text(json.dumps({'preparatory_copy_complete': True,
                'families': [{'dataset': ds, 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}]}))
            self.assertEqual(prepared_databins(root, ['Beckmann'])['Beckmann'], databin)
            payload.write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError, 'Prepared output changed'):
                prepared_databins(root, ['Beckmann'])


class UnmappedAuditTests(unittest.TestCase):
    def test_atom_maps_removed_before_reaction_comparison(self):
        from scripts.audit_uspto_pretraining import key
        self.assertEqual(key('[CH3:1][OH:2]', '[CH2:1]=[O:2]'), key('CO', 'C=O'))

    def test_shared_product_is_not_same_reaction(self):
        from scripts.audit_uspto_pretraining import key
        a, b = key('CO', 'C=O'), key('CCl', 'C=O')
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])

    def test_cross_dot_ring_and_invalid_side(self):
        from scripts.audit_uspto_pretraining import key
        self.assertEqual(key('C1.C1', 'CCO'), key('CC', 'CCO'))
        self.assertIsNone(key('CC.C1', 'CCO'))


if __name__ == '__main__':
    unittest.main()
