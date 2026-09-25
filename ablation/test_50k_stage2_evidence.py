from pathlib import Path
import unittest

import pandas as pd

from scripts.audit_reafnn_only_evidence import resolve_route_root
from scripts.audit_50k_knn_only_evidence import verify_intervention, library_statistics, verify_library_statistics


class Stage2EvidenceTests(unittest.TestCase):
    def test_recorded_routes_are_used_instead_of_historical_default(self):
        root = Path("/tmp/new_route_test")
        self.assertEqual(resolve_route_root({"route_root": str(root)}), root)

    def test_different_explicit_route_root_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_route_root({"route_root": "/tmp/new"}, Path("/tmp/old"))

    def test_unbound_legacy_route_root_is_not_inferred(self):
        with self.assertRaises(ValueError):
            resolve_route_root({})

    def test_explicit_legacy_root_allowed_for_hash_checked_replay(self):
        self.assertEqual(resolve_route_root({}, Path("/tmp/legacy")), Path("/tmp/legacy"))

    def frame(self, count=20):
        return pd.DataFrame(dict(sample_index=[0]*count, reactants=["CCO"]*count,
            reafnn_token_score=[0.]*count, from_reafnn_generated=[0.]*count))

    def test_zero_neural_evidence_passes(self):
        self.assertEqual(len(verify_intervention(self.frame())), 2)

    def test_nonzero_or_missing_neural_evidence_is_rejected(self):
        frame = self.frame()
        frame.loc[0, "reafnn_token_score"] = .1
        with self.assertRaises(ValueError):
            verify_intervention(frame)
        with self.assertRaises(ValueError):
            verify_intervention(frame[["sample_index", "reactants"]])

    def test_context_cap_checked_per_route(self):
        with self.assertRaises(ValueError):
            verify_intervention(self.frame(21))
        frame = self.frame(40)
        frame.loc[20:, "reactants"] = "CCN"
        verify_intervention(frame)

    def test_legacy_named_library_statistics_are_not_neural_scores(self):
        frame = self.frame()
        frame["reafnn_context_count"] = 3.
        frame["reafnn_is_historical"] = 1
        verify_intervention(frame)

    def test_library_statistics_use_training_only_with_reaction_exclusion(self):
        rows = [dict(reactants="CCO", product="CC=O", reagent_norm="A", solvent_norm="water", **{"yield": 40}),
                dict(reactants="CCN", product="CC#N", reagent_norm="A", solvent_norm="water", **{"yield": 80})]
        stats = library_statistics(rows)
        frame = pd.DataFrame([dict(rows[0], reafnn_context_count=1., reafnn_context_support=1.,
                                   reafnn_mean_yield=80., reafnn_is_historical=1)])
        verify_library_statistics(frame, stats, training=True)
        with self.assertRaises(AssertionError):
            verify_library_statistics(frame, stats, training=False)
        frame["reafnn_context_count"] = 2.
        frame["reafnn_mean_yield"] = 60.
        verify_library_statistics(frame, stats, training=False)


if __name__ == "__main__":
    unittest.main()
