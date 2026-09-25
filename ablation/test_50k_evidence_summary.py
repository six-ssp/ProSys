import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import summarize_50k_downstream_evidence as summary


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for family, offset, n in (("A", 0., 1), ("B", .8, 99)):
            for seed in (0, 1, 2):
                for arm in summary.ARMS:
                    self.rows.append(dict(family=family, seed=seed, arm=arm, n_queries=n,
                        cover=1., sys1=offset+.1*seed, sys3=offset+.1*seed,
                        sys5=offset+.1*seed, sys10=offset+.1*seed,
                        mrr=.5, ndcg10=.5, temp_n=0 if arm == "no_ltr" else 1,
                        temp_mae=None if arm == "no_ltr" else 10.,
                        temp_within_5c=None if arm == "no_ltr" else .2,
                        temp_within_10c=None if arm == "no_ltr" else .5,
                        temp_within_20c=None if arm == "no_ltr" else .9))

    def aggregate(self, rows=None):
        return summary.aggregate(self.rows if rows is None else rows, families=("A", "B"))

    def test_equal_family_mean_and_sample_sd(self):
        _, macro, families, stats = self.aggregate()
        np.testing.assert_allclose(macro[macro.arm == "full"].sys10, [.4, .5, .6])
        self.assertAlmostEqual(stats.loc["full", ("sys10", "mean")], .5)
        self.assertAlmostEqual(stats.loc["full", ("sys10", "std")], .1)
        self.assertTrue((macro.n_queries == 100).all())
        self.assertEqual(stats.loc["full", ("sys10", "count")], 3)

    def test_missing_temperature_support_is_not_zero(self):
        _, macro, _, stats = self.aggregate()
        self.assertTrue(macro[macro.arm == "no_ltr"].temp_mae.isna().all())
        self.assertEqual(stats.loc["no_ltr", ("temp_mae", "count")], 0)
        self.assertTrue((macro[macro.arm == "full"].temp_n == 2).all())

    def test_missing_or_duplicated_run_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                self.aggregate(rows)

    def test_changed_denominator_rejected(self):
        self.rows[0]["n_queries"] = 2
        with self.assertRaisesRegex(ValueError, "support differs"):
            self.aggregate()

    def test_nonfinite_and_nonmonotonic_rejected(self):
        for value in (float("nan"), float("inf"), 1.1, -.1):
            rows = copy.deepcopy(self.rows)
            rows[0]["sys10"] = value
            with self.assertRaises(ValueError):
                self.aggregate(rows)
        self.rows[0]["sys1"] = .9
        with self.assertRaisesRegex(ValueError, "monotonic"):
            self.aggregate()

    def test_missing_artifacts_withholds_report(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with patch("sys.argv", ["summary", "--study-root", temp, "--route-study", str(path / "routes")]):
                with self.assertRaisesRegex(RuntimeError, "withheld"):
                    summary.main()
            self.assertFalse((path / "RESULTS.md").exists())
            self.assertFalse((path / "replay_verification.json").exists())


if __name__ == "__main__":
    unittest.main()
