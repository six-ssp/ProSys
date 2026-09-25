"""Shell wiring smoke tests with a fake trainer, not scientific training."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BaseWiringTests(unittest.TestCase):
    def fixture(self, root):
        scripts = root / 'stage1_retrosynthesis/scripts'
        scripts.mkdir(parents=True)
        extension = scripts / 'ensure_fairseq_extensions.sh'
        extension.write_text('#!/bin/bash\nexit 0\n')
        extension.chmod(0o755)
        databin = root / 'versioned_inputs/data-bin'
        databin.mkdir(parents=True)
        for name in ('dict.src.txt', 'dict.tgt.txt'):
            (databin / name).write_text('token 1\n')
        for split in ('train', 'valid'):
            for side in ('src', 'tgt'):
                for suffix in ('bin', 'idx'):
                    (databin / f'{split}.src-tgt.{side}.{suffix}').write_bytes(b'fixture')
        alias = root / 'stage1_retrosynthesis/checkpoints/checkpoint_USPTO_STAGE2_FILTERED_best.pt'
        alias.parent.mkdir(parents=True)
        alias.write_bytes(b'original alias; not a real model')
        stub = root / 'fake_python'
        stub.write_text('#!/usr/bin/env python3\nimport sys,os,json,pathlib\n'
            'if sys.argv[1].endswith("fairseq_cli/train.py"):\n'
            ' pathlib.Path(os.environ["CALLS"]).write_text(json.dumps(sys.argv[1:]))\n'
            ' folder=pathlib.Path(sys.argv[sys.argv.index("--save-dir")+1])\n'
            ' (folder/"checkpoint_best.pt").write_bytes(b"fake best")\n'
            ' (folder/"checkpoint_last.pt").write_bytes(b"fake last")\n')
        stub.chmod(0o755)
        env = dict(os.environ, PYTHON_BIN=str(stub), CALLS=str(root / 'calls.json'),
                   SKIP_PREPARE='1', DATA_BIN=str(databin), UPDATE_ALIAS='0', SEED='2', RUN_NAME='fixed')
        return env, alias, databin

    def test_new_inputs_seed_and_no_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, alias, databin = self.fixture(root)
            command = ['bash', str(ROOT / 'stage1_retrosynthesis/scripts/run_base_train.sh'), str(root)]
            subprocess.run(command, env=env, check=True, capture_output=True)
            call = json.loads((root / 'calls.json').read_text())
            self.assertEqual(call[1], str(databin))
            self.assertEqual(call[call.index('--seed') + 1], '2')
            self.assertEqual(call[call.index('--fixed-validation-seed') + 1], '7')
            self.assertIn('--no-epoch-checkpoints', call)
            self.assertNotIn('--restore-file', call)
            self.assertEqual(alias.read_bytes(), b'original alias; not a real model')
            self.assertFalse(alias.is_symlink())
            self.assertIn('/fixed/checkpoints', call[call.index('--save-dir') + 1])

    def test_override_cannot_prepare_different_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, _, _ = self.fixture(root)
            env['SKIP_PREPARE'] = '0'
            result = subprocess.run(['bash', str(ROOT / 'stage1_retrosynthesis/scripts/run_base_train.sh'), str(root)],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('DATA_BIN override requires', result.stderr)
            self.assertFalse((root / 'calls.json').exists())


if __name__ == '__main__':
    unittest.main()
