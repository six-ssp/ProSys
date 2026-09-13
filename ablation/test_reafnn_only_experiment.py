"""Fast no-data tests of the strict KNN-removal boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ablation.reafnn_only_experiment import KNN_EVIDENCE_COLUMNS, ReaFNNOnlyPoolBuilder


class ReaFNNOnlyTests(unittest.TestCase):
    def builder(self):
        builder = object.__new__(ReaFNNOnlyPoolBuilder)
        builder.max_contexts = 1
        builder.prefilter_contexts = 64
        builder.reaffn_selector = SimpleNamespace(config=SimpleNamespace(independent_contexts=64))
        return builder

    def test_never_retrieves_or_falls_back_to_knn(self):
        builder = self.builder()
        self.assertEqual(builder._aggregate_knn_contexts(None, limit=64), [])
        with self.assertRaises(AssertionError):
            builder._route_similarities(None)

    def test_removes_knn_named_global_yield_fallback(self):
        builder = self.builder()
        context = {"reagent_norm": "A", "solvent_norm": "B", "mean_yield": 95,
                   "stage2_reafnn_check_score": 1.0, "stage2_reafnn_rank": 1}
        with patch.object(builder, "_independent_reafnn_contexts",
                          return_value=({("A", "B"): context}, [context])):
            rows = builder._independent_post_fusion_state(None, leave_one_reaction_out=True)
        self.assertEqual(len(rows), 1)
        self.assertTrue(all(rows[0][field] == 0 for field in KNN_EVIDENCE_COLUMNS))
        self.assertEqual(rows[0]["from_reafnn_generated"], 1)
        selected = builder._select_independent_post_fusion_contexts(rows, knn_weight=0)
        self.assertEqual(selected[0]["stage2_initial_score"], 1.0)

    def test_rejects_nonzero_knn_weight(self):
        with self.assertRaises(ValueError):
            self.builder()._select_independent_post_fusion_contexts([], knn_weight=0.1)


if __name__ == "__main__":
    unittest.main()
