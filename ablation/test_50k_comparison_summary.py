import unittest

import numpy as np
import pandas as pd

from scripts.collect_50k_comparisons import aggregate, format_stat, SEEDS, TEMP_KEYS


def rows():
    output = []
    for family, n, value in (("A", 10, .2), ("B", 90, .6)):
        for model, seeds in SEEDS.items():
            for seed in seeds:
                temp = model in ("ProSys", "B3_SequentialFNN", "Without_RGNN_temperature")
                output.append(dict(family=family, model=model, seed=seed, n_queries=n,
                    route10=.9, cover=.8, sys1=.1, sys3=.1, sys5=.1, sys10=value + seed * .01,
                    mrr=.1, ndcg10=.1, temp_n=n // 2 if temp else 0,
                    temp_mae=10. + seed if temp else None, temp_within_5c=.2 if temp else None,
                    temp_within_10c=.4 if temp else None, temp_within_20c=.6 if temp else None))
    return output


class ComparisonSummaryTests(unittest.TestCase):
    def test_equal_family_macro_and_total_denominator(self):
        frame, macro, family, summary = aggregate(rows(), ["A", "B"])
        self.assertEqual(len(frame), 50)
        item = macro[(macro.model == "ProSys") & (macro.seed == 0)].iloc[0]
        self.assertAlmostEqual(item.sys10, .4)
        self.assertEqual(item.n_queries, 100)
        self.assertEqual(item.temp_n, 50)
        self.assertEqual(item.temperature_family_count, 2)
        self.assertAlmostEqual(summary.loc["ProSys", ("sys10", "std")], .01)

    def test_deterministic_baseline_has_one_observation_and_no_sd(self):
        _, macro, family, summary = aggregate(rows(), ["A", "B"])
        self.assertEqual(len(macro[macro.model == "B1_ProductNB"]), 1)
        self.assertEqual(summary.loc["B1_ProductNB", ("sys10", "count")], 1)
        self.assertTrue(np.isnan(summary.loc["B1_ProductNB", ("sys10", "std")]))

    def test_missing_seed_or_family_and_duplicate_b1_are_rejected(self):
        for values in (rows()[:-1], rows() + [rows()[3]], [r for r in rows() if r["family"] != "B"]):
            with self.assertRaisesRegex(ValueError, "grid"):
                aggregate(values, ["A", "B"])

    def test_different_query_support_is_rejected(self):
        values = rows()
        values[0]["n_queries"] += 1
        with self.assertRaisesRegex(ValueError, "query support"):
            aggregate(values, ["A", "B"])

    def test_temperature_control_must_keep_system_metrics_and_support(self):
        for key, delta in (("temp_n", 1), ("sys10", -.01)):
            values = rows()
            target = next(r for r in values if r["model"] == "Without_RGNN_temperature")
            target[key] += delta
            with self.assertRaisesRegex(ValueError, "Temperature-only control"):
                aggregate(values, ["A", "B"])

    def test_missing_temperature_support_stays_na_and_has_family_count(self):
        values = rows()
        for r in values:
            if r["family"] == "B":
                r["temp_n"] = 0
                r.update({k: None for k in TEMP_KEYS})
        _, macro, _, _ = aggregate(values, ["A", "B"])
        item = macro[(macro.model == "ProSys") & (macro.seed == 0)].iloc[0]
        self.assertEqual(item.temperature_family_count, 1)
        self.assertEqual(item.temp_mae, 10.)
        nb = macro[macro.model == "B1_ProductNB"].iloc[0]
        self.assertTrue(np.isnan(nb.temp_mae))

    def test_zero_support_cannot_be_reported_as_zero_error(self):
        values = rows()
        target = next(r for r in values if r["model"] == "B1_ProductNB")
        target["temp_mae"] = 0.
        with self.assertRaisesRegex(ValueError, "must be NA"):
            aggregate(values, ["A", "B"])

    def test_sys_above_coverage_is_rejected(self):
        values = rows()
        values[0]["sys10"] = .85
        with self.assertRaisesRegex(ValueError, "ordering"):
            aggregate(values, ["A", "B"])

    def test_one_valid_temperature_seed_is_not_labeled_one_training_run(self):
        self.assertEqual(format_stat(pd.Series([None, 10., None]), "ProSys"), "10.00 (1 valid seed; SD NA)")
        self.assertEqual(format_stat(pd.Series([8.94]), "B1_ProductNB"), "8.94")
        self.assertEqual(format_stat(pd.Series([np.nan]), "B1_ProductNB"), "NA")


if __name__ == "__main__":
    unittest.main()
