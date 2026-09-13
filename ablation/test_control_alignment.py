import unittest

import pandas as pd

from scripts.summarize_mainline_evidence import align_control_rows


class ControlAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            dict(sample_index=0, reaction_id="x", product="CCO", reactants="CC=O", reagent_norm=r, solvent_norm="water", score=s)
            for r, s in (("A", 1.0), ("B", 2.0), ("C", 3.0))])

    def test_permuted_control_scores_align_by_identity(self):
        shuffled = self.frame.iloc[[2, 0, 1]].copy()
        aligned = align_control_rows(self.frame, shuffled)
        self.assertEqual(aligned.score.tolist(), [1.0, 2.0, 3.0])

    def test_equal_count_different_candidate_rejected(self):
        different = self.frame.copy()
        different.loc[0, "reagent_norm"] = "D"
        with self.assertRaises(AssertionError):
            align_control_rows(self.frame, different)

    def test_ambiguous_duplicate_is_not_arbitrarily_paired(self):
        duplicate = pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            align_control_rows(duplicate, duplicate.iloc[::-1])


if __name__ == "__main__":
    unittest.main()
