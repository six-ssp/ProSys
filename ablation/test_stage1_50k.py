import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.run_stage1_50k_from_scratch import filter_indices, identities, upstream_splits

ROOT = Path(__file__).resolve().parents[1]


class FiftyKTests(unittest.TestCase):
    def test_upstream_classwise_nonshuffled_split(self):
        rows = [{'class': str(i % 2 + 1)} for i in range(20)]
        split = upstream_splits(rows)
        self.assertEqual(split['train'], list(range(16)))
        self.assertEqual(split['val'], [16, 17])
        self.assertEqual(split['test'], [18, 19])
        self.assertEqual(sorted(sum(split.values(), [])), list(range(20)))

    def test_raw_and_transformed_identities_both_protected(self):
        a = identities('[CH3:1][OH:2].[CH3:3][Cl:4]>>[CH2:1]=[O:2].[CH4:3]')
        b = identities('[CH3:1][OH:2]>>[CH2:1]=[O:2]')
        self.assertTrue(a & b)
        kept, removed, _ = filter_indices([0], [a], b, set(), set())
        self.assertEqual(kept, [])
        self.assertIn('reaxys_heldout_identity', removed[0]['reasons'])

    def test_test_and_validation_and_duplicates_excluded(self):
        keys = [{b'a'}, {b'b'}, {b'c'}, {b'c'}, set()]
        kept, removed, seen = filter_indices(range(5), keys, set(), {b'a'}, {b'b'})
        self.assertEqual(kept, [2])
        self.assertEqual(seen, {b'c'})
        self.assertEqual([r['source_csv_data_index'] for r in removed], [0, 1, 3, 4])

    def test_default_family_entrypoints_never_fall_back_to_full(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            old = root / 'stage1_retrosynthesis/checkpoints/checkpoint_USPTO_STAGE2_FILTERED_best.pt'
            old.parent.mkdir(parents=True)
            old.write_bytes(b'old FULL fixture')
            env = dict(os.environ, BASE_CKPT='', BASE_DATASET='USPTO_50K_FILTERED')
            for script in ('run_family_finetune_one.sh', 'run_family_finetune_batch.sh'):
                result = subprocess.run(['bash', str(ROOT / 'stage1_retrosynthesis/scripts' / script),
                                         str(root), 'REAXYS_F'], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('no FULL fallback', result.stderr)

    def test_50k_restore_weights_forbidden(self):
        with tempfile.TemporaryDirectory() as name:
            env = dict(os.environ, RESTORE_CKPT='/old/full/model.pt')
            result = subprocess.run(['bash', str(ROOT / 'stage1_retrosynthesis/scripts/run_base_train.sh'), name],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('from scratch', result.stderr)
            self.assertEqual(list(Path(name).iterdir()), [])

    def test_legacy_reset_wrapper_refuses_50k_before_work(self):
        with tempfile.TemporaryDirectory() as name:
            env = dict(os.environ, BASE_DATASET='USPTO_50K_FILTERED', SKIP_PREPROCESS='0')
            result = subprocess.run(['bash', str(ROOT / 'scripts/reproduce_mainline_from_raw.sh'), name],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('run_stage1_50k_from_scratch.py', result.stderr)
            self.assertEqual(list(Path(name).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
