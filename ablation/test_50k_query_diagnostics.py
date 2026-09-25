import unittest

import pandas as pd

from prosys_shared.mainline import GoldConditionIndex
from scripts.audit_50k_query_diagnostics import diagnostic_tables, replay_queries


class QueryDiagnosticsTest(unittest.TestCase):
    def fixture(self):
        bucket = GoldConditionIndex(route_keys={"C"}, context_keys={("", "O")},
                                    exact_keys={("C", "", "O")})
        cache = {"reactions": [dict(sample_index=i, reaction_id=str(i), product="CC",
                    routes=[] if i == 0 else [dict(reactants="C")]) for i in range(4)]}
        gold = {(str(i), "CC"): bucket for i in range(4)}
        train = [dict(product="CC", reagent_norm="", solvent_norm="O")]
        rows = []
        for i in (1, 2, 3):
            # Query 1: pool miss. Query 2: first exact rank 11. Query 3: rank 1.
            for rank in range(1, 12):
                hit = (i == 2 and rank == 11) or (i == 3 and rank == 1)
                rows.append(dict(sample_index=i, reaction_id=str(i), product="CC",
                    reactants="C", reagent_norm="", solvent_norm="O" if hit else "N",
                    label=int(hit), xgb_score=12-rank))
        return cache, pd.DataFrame(rows), gold, train

    def replay(self, args):
        return replay_queries(*args, family="Test", seed=1)

    def test_four_categories_and_empty_slate_denominator(self):
        q = self.replay(self.fixture())
        self.assertEqual(q.failure_category.tolist(),
                         ["route_miss", "pool_miss", "ranking_miss", "hit"])
        self.assertEqual(q.sys10.mean(), .25)
        self.assertEqual(q.candidate_hit.mean(), .5)
        self.assertEqual(q.route_hit.mean(), .75)
        failures, groups = diagnostic_tables(q)
        self.assertEqual(failures.iloc[0][["route_miss", "pool_miss", "ranking_miss", "hit"]].tolist(), [1]*4)
        empty = groups[groups.n == 0]
        self.assertEqual(len(empty), 2)
        self.assertTrue(empty.sys10.isna().all())

    def test_fabricated_label_rejected(self):
        args = self.fixture()
        args[1].loc[0, "label"] = 1
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            self.replay(args)

    def test_candidate_outside_route_pool_rejected(self):
        args = self.fixture()
        args[1].loc[0, "reactants"] = "N"
        with self.assertRaisesRegex(ValueError, "predicted routes"):
            self.replay(args)

    def test_query_identity_mismatch_rejected(self):
        args = self.fixture()
        args[1].loc[0, "reaction_id"] = "other"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.replay(args)

    def test_foreign_candidate_query_rejected(self):
        args = self.fixture()
        args[1].loc[0, "sample_index"] = 99
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.replay(args)

    def test_duplicate_query_manifest_rejected(self):
        args = self.fixture()
        args[0]["reactions"].append(args[0]["reactions"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.replay(args)

    def test_context_absence_is_annotation_not_filter(self):
        args = self.fixture()
        q = self.replay((*args[:3], []))
        self.assertFalse(q.seen_product.any())
        self.assertFalse(q.context_in_train_library.any())
        self.assertEqual(q.sys10.mean(), .25)

    def test_duplicate_diagnostic_rows_rejected(self):
        q = self.replay(self.fixture())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            diagnostic_tables(pd.concat([q, q.iloc[:1]]))

    def test_inconsistent_failure_category_rejected(self):
        q = self.replay(self.fixture())
        q.loc[0, "failure_category"] = "hit"
        with self.assertRaisesRegex(ValueError, "disagree"):
            diagnostic_tables(q)


if __name__ == "__main__":
    unittest.main()
