import unittest

import pandas as pd

from prosys_shared.evidence import require_same_candidates, require_same_temperature_support, temperature_support


def example():
    return pd.DataFrame({"sample_index": [0, 0], "reaction_id": ["q", "q"],
        "product": ["CC", "CC"], "reactants": ["C", "O"], "reagent_norm": ["a", "b"],
        "solvent_norm": ["c", "d"], "xgb_score": [2., 1.], "label": [1., 1.],
        "temperature_gold": [30., 40.], "temperature_pred": [31., 42.]})


class EvidenceTests(unittest.TestCase):
    def test_candidate_order_does_not_change_membership(self):
        frame = example()
        require_same_candidates(frame, frame.iloc[::-1])

    def test_same_count_different_identity_is_rejected(self):
        a = example()
        b = a.copy()
        b.loc[0, "reactants"] = "N"
        with self.assertRaises(AssertionError):
            require_same_candidates(a, b)

    def test_same_count_different_selected_system_is_rejected(self):
        a = example()
        b = a.copy()
        b["xgb_score"] = [1., 2.]
        with self.assertRaises(AssertionError):
            require_same_temperature_support(a, b)

    def test_different_temperature_labels_rejected(self):
        a = example()
        b = a.copy()
        b.loc[0, "temperature_gold"] = 35.
        with self.assertRaises(AssertionError):
            require_same_temperature_support(a, b)

    def test_predictions_may_differ_on_identical_support(self):
        a = example()
        b = a.copy()
        b["temperature_pred"] = [10., 15.]
        self.assertEqual(require_same_temperature_support(a, b)["count"], 1)

    def test_support_not_limited_to_top10(self):
        frame = pd.concat([example().iloc[[0]]] * 11, ignore_index=True)
        frame["xgb_score"] = range(11, 0, -1)
        frame["label"] = [0.] * 10 + [1.]
        self.assertEqual(int(temperature_support(frame)["final_rank"].iloc[0]), 11)


if __name__ == "__main__":
    unittest.main()
