import json
from pathlib import Path
import tempfile
import unittest

from baseline.run_direct_product_condition_baselines import ProductQuery
from scripts.audit_50k_baseline_evidence import load_direct_predictions, check_selected_fusion


class BaselineAuditTests(unittest.TestCase):
    def test_prediction_identity_not_only_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            row = dict(sample_index=0, reaction_id="r", product="CCO", contexts=[])
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(load_direct_predictions(path, [ProductQuery("A", 0, "r", "CCO")]), {0: []})
            with self.assertRaises(ValueError):
                load_direct_predictions(path, [ProductQuery("A", 0, "r", "CCN")])

    def test_duplicate_prediction_rows_are_not_collapsed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            row = dict(sample_index=0, reaction_id="r", product="CCO", contexts=[])
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                load_direct_predictions(path, [ProductQuery("A", 0, "r", "CCO")])

    def test_changed_selected_weight_rejected(self):
        with self.assertRaises(ValueError):
            check_selected_fusion({"selected": {"route_weight": 0.}}, {"selected": {"route_weight": 1.}})


if __name__ == "__main__":
    unittest.main()
