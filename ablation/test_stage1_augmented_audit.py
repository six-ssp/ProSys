"""Synthetic identities and source-transformation tests, not corpus evidence."""

import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from scripts.audit_stage1_augmented_splits import base_membership
from scripts.audit_stage1_base_augmented import ZERO, reaction_key
from scripts.trace_stage1_augmented_collisions import transformed_keys
from scripts.prepare_stage1_expert_split_repair import exclusion_reasons


class AugmentedIdentityTests(unittest.TestCase):
    def base_fixture(self, *, row_count=1, invalid=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        data = bytes(32) if invalid else reaction_key('CO', 'C=O')
        splits = {}
        for split in ('train', 'val'):
            (folder / (split + '.keys.bin')).write_bytes(data)
            splits[split] = {'keys_sha256': hashlib.sha256(data).hexdigest(), 'augmented_rows': row_count}
        (folder / 'audit.json').write_text(json.dumps({'splits': splits}))
        base_membership.cache_clear()
        self.addCleanup(base_membership.cache_clear)
        return folder, data

    def test_base_membership_reads_full_exact_digests(self):
        folder, data = self.base_fixture()
        self.assertEqual(base_membership(str(folder)), {data})

    def test_base_membership_count_cannot_be_shorter_than_audit(self):
        folder, _ = self.base_fixture(row_count=2)
        with self.assertRaisesRegex(ValueError, 'row count'):
            base_membership(str(folder))

    def test_base_membership_rejects_invalid_sentinel(self):
        folder, _ = self.base_fixture(invalid=True)
        with self.assertRaisesRegex(ValueError, 'Invalid reaction'):
            base_membership(str(folder))

    def test_exclusion_policy_protects_heldout_without_dropping_tests(self):
        key = reaction_key('CO', 'C=O')
        condition = {'val': {key}, 'test': set()}
        augmented = {'val': set(), 'test': {key}}
        self.assertEqual(exclusion_reasons('train', key, condition, augmented, set()),
                         ['condition_val', 'augmented_test'])
        self.assertEqual(exclusion_reasons('val', key, condition, augmented, set()), ['augmented_test'])
        self.assertEqual(exclusion_reasons('test', key, condition, augmented, {key}), [])

    def test_base_membership_exclusion_applies_to_expert_validation(self):
        key = reaction_key('CO', 'C=O')
        heldout = {'val': set(), 'test': set()}
        self.assertEqual(exclusion_reasons('val', key, heldout, heldout, {key}),
                         ['base_train_or_validation_membership'])
        self.assertEqual(exclusion_reasons('train', key, heldout, heldout, {key}), [])

    def test_atom_maps_and_fragment_order_do_not_change_identity(self):
        self.assertEqual(reaction_key('[CH3:1][OH:2].[Na+:3]', '[CH2:1]=[O:2]'),
                         reaction_key('[Na+].CO', 'C=O'))

    def test_same_product_different_precursors_are_not_overlap(self):
        self.assertNotEqual(reaction_key('CCO', 'CC=O'), reaction_key('CCCl', 'CC=O'))

    def test_invalid_whole_side_is_not_salvaged(self):
        self.assertEqual(reaction_key('CCO.invalid', 'CC=O'), ZERO)

    def test_cross_dot_ring_closure_is_parsed_as_whole(self):
        self.assertEqual(reaction_key('C1CC.CC1', 'CCO'), reaction_key('CCCCC', 'CCO'))

    def test_stereo_preserved(self):
        self.assertNotEqual(reaction_key('F[C@H](Cl)Br', 'CO'), reaction_key('F[C@@H](Cl)Br', 'CO'))

    def test_product_splitting_and_precursor_selection_can_create_collision(self):
        source = '[CH3:1][OH:2].[CH3:3][Cl:4]>>[CH2:1]=[O:2].[CH4:3]'
        transformed = transformed_keys(source)
        target = reaction_key('CO', 'C=O')
        self.assertIn((target, 2, 1), transformed)
        self.assertNotEqual(reaction_key('CO.CCl', 'C=O.C'), target)


if __name__ == '__main__':
    unittest.main()
