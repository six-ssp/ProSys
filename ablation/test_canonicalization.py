"""Whole-side parsing, including valid cross-dot ring closures."""

import unittest
from rdkit import Chem
from prosys_shared.features import canonicalize_reaction_side


class CanonicalizationTests(unittest.TestCase):
    def test_dot_ring_closure_is_a_connected_molecule(self):
        self.assertEqual(canonicalize_reaction_side('C1.C1'), 'CC')

    def test_actual_training_record_preserves_every_atom(self):
        raw = '[K+].C=CC(=C)[Si-]123OC4=C(O1)C=CC=C4.O2C1=C(O3)C=CC=C1.O=C1C=CC(=O)N1C1=CC=CC=C1'
        result = canonicalize_reaction_side(raw)
        original_mol, result_mol = Chem.MolFromSmiles(raw), Chem.MolFromSmiles(result)
        self.assertEqual(original_mol.GetNumAtoms(), result_mol.GetNumAtoms())
        self.assertEqual(Chem.MolToSmiles(original_mol), Chem.MolToSmiles(result_mol))
        self.assertEqual(result, canonicalize_reaction_side(result))

    def test_components_are_order_invariant_and_keep_multiplicity(self):
        self.assertEqual(canonicalize_reaction_side('O.CC.O'), 'CC.O.O')
        self.assertEqual(canonicalize_reaction_side('CC.O.O'), 'CC.O.O')

    def test_invalid_whole_side_does_not_become_a_partial_route(self):
        self.assertEqual(canonicalize_reaction_side('CC.C1'), '')
        self.assertEqual(canonicalize_reaction_side(''), '')

    def test_stereochemistry_is_not_removed(self):
        left = canonicalize_reaction_side('F[C@H](Cl)Br.O')
        right = canonicalize_reaction_side('F[C@@H](Cl)Br.O')
        self.assertNotEqual(left, right)

    def test_preprocessing_uses_the_same_complete_keys(self):
        from data_preprocess.preprocess import make_canonical_reaction_key
        self.assertEqual(make_canonical_reaction_key('C1.C1', 'CCO'), 'CC>>CCO')
        self.assertEqual(make_canonical_reaction_key('CC.C1', 'CCO'), '')


if __name__ == '__main__':
    unittest.main()
