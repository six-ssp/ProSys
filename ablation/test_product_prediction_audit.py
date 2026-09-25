import unittest

import pandas as pd

from scripts.audit_product_prediction import align, compare


class ProductPredictionAuditTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([dict(sample_index=0, reaction_id="query", product="CCO", reactants="CC=O",
            reagent_norm=reagent, solvent_norm="water", feature=1., graph_feature=2.,
            retro_rank=1, retro_probability=1., xgb_score=score, xgb_temperature_pred=25.)
            for reagent, score in (("acid", 0.8), ("base", 0.2))])

    def test_identity_not_row_order_and_ids_drives_replay(self):
        expected = self.rows()
        observed = expected.iloc[::-1].copy()
        observed["reaction_id"], observed["sample_index"] = "new_query_id", 42
        result = compare(expected, observed, ["feature"], ["feature", "graph_feature"])
        self.assertTrue(result["pass"])
        self.assertEqual(result["candidate_count"], 2)

    def test_duplicate_systems_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-unique"):
            align(pd.concat([self.rows(), self.rows()]))

    def test_changed_features_scores_and_temperatures_rejected(self):
        expected = self.rows()
        for column in ("feature", "graph_feature", "xgb_score", "xgb_temperature_pred"):
            observed = expected.copy()
            observed.loc[0, column] += 1
            with self.subTest(column=column), self.assertRaises(AssertionError):
                compare(expected, observed, ["feature"], ["feature", "graph_feature"])

    def test_gold_columns_rejected(self):
        expected = self.rows()
        for column in ("label", "temperature_gold", "route_match", "rank_relevance"):
            observed = expected.assign(**{column: 1})
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Gold information"):
                compare(expected, observed, ["feature"], ["feature", "graph_feature"])


if __name__ == "__main__":
    unittest.main()
