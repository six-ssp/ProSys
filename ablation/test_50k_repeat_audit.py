import unittest

import numpy as np
import pandas as pd
import torch

from scripts.audit_50k_repeat import compare_frame, compare_state


class RepeatAuditTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(sample_index=0, reaction_id="q", product="CCO", reactants="CC=O",
            reagent_norm=r, solvent_norm="water", xgb_score=s, xgb_temperature_pred=25., optional=np.nan)
            for r, s in (("a", .9), ("b", .2))])

    def test_row_permutation_is_not_a_repeat_difference(self):
        frame = self.frame()
        self.assertTrue(compare_frame(frame, frame.iloc[::-1])["exact"])

    def test_even_small_score_change_is_reported(self):
        frame, other = self.frame(), self.frame()
        other.loc[0, "xgb_score"] += 1e-10
        result = compare_frame(frame, other)
        self.assertFalse(result["exact"])
        self.assertEqual(result["differences"]["xgb_score"]["rows"], 1)
        self.assertGreater(result["differences"]["xgb_score"]["max_absolute_difference"], 0)

    def test_temperature_missingness_is_not_ignored(self):
        frame, other = self.frame(), self.frame()
        other.loc[0, "xgb_temperature_pred"] = np.nan
        self.assertFalse(compare_frame(frame, other)["exact"])

    def test_different_candidate_pool_is_rejected(self):
        with self.assertRaises(AssertionError):
            compare_frame(self.frame(), self.frame().iloc[:1])

    def test_all_neural_state_and_scaler_values_are_compared(self):
        a = {"model_state": {"w": torch.tensor([1., 2.])}, "scaler": np.array([3., 4.]), "vocab": ["a", "b"]}
        b = {"model_state": {"w": torch.tensor([1., 2.])}, "scaler": np.array([3., 4.]), "vocab": ["a", "b"]}
        self.assertEqual(compare_state(a, b), [])
        b["scaler"][0] += 1
        b["model_state"]["w"][1] += 1
        self.assertEqual(compare_state(a, b), ["/model_state/w", "/scaler"])


if __name__ == "__main__":
    unittest.main()
