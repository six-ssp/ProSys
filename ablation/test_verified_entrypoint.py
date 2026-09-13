import tempfile
import json
import fcntl
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_verified_mainline import guard_family, verify_route_manifest
from prosys_shared.cache_integrity import CacheMismatchError


class VerifiedEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.calls = []
        def run(output_root, family, force_rebuild=False, reaffn_force_retrain=False,
                gnn_force_retrain=False, reuse_candidate_tables_root=None):
            self.calls.append((force_rebuild, reaffn_force_retrain, gnn_force_retrain))
            folder = output_root / family
            folder.mkdir(exist_ok=True)
            (folder / "result.json").write_text("{}")
            return {"ok": True}
        self.run = guard_family(run)
        self.binding = {"schema": 1, "inputs": {"data": "abc"}, "config": {"seed": 0}}

    def test_fresh_and_exact_reuse(self):
        with patch("scripts.run_verified_mainline.bind_run", return_value=self.binding):
            self.assertTrue(self.run(self.output, "F")["ok"])
            self.assertTrue(self.run(self.output, "F")["ok"])
        self.assertEqual(len(self.calls), 2)

    def test_changed_input_stops_before_pipeline(self):
        with patch("scripts.run_verified_mainline.bind_run", return_value=self.binding):
            self.run(self.output, "F")
        with patch("scripts.run_verified_mainline.bind_run", return_value={**self.binding, "inputs": {"data": "xyz"}}):
            with self.assertRaises(CacheMismatchError):
                self.run(self.output, "F")
        self.assertEqual(len(self.calls), 1)

    def test_force_retrains_all_dependent_models(self):
        with patch("scripts.run_verified_mainline.bind_run", return_value=self.binding):
            self.run(self.output, "F", gnn_force_retrain=True)
        self.assertEqual(self.calls, [(True, True, True)])

    def test_partial_or_legacy_cache_refused(self):
        folder = self.output / "F"
        folder.mkdir()
        (folder / "old_model.pt").write_bytes(b"old")
        with patch("scripts.run_verified_mainline.bind_run", return_value=self.binding):
            with self.assertRaises(CacheMismatchError):
                self.run(self.output, "F")
        self.assertEqual(self.calls, [])

    def test_concurrent_writer_refused(self):
        with (self.output / ".F.cache.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(CacheMismatchError):
                self.run(self.output, "F")
        self.assertEqual(self.calls, [])

    def test_route_manifest_rejects_stale_split(self):
        split = self.output / "test.tsv"
        split.write_text("id\tCC=O\tCCO\t80\ta\tb\t25\n")
        cache = self.output / "routes.json"
        value = {"family": "F", "reactions": [dict(sample_index=0, reaction_id="id", product="CCO", gold_reactants="CC=O", routes=[])]}
        cache.write_text(json.dumps(value))
        verify_route_manifest(cache, split, "F")
        split.write_text("id\tC=CO\tCCO\t80\ta\tb\t25\n")
        with self.assertRaises(CacheMismatchError):
            verify_route_manifest(cache, split, "F")


if __name__ == "__main__":
    unittest.main()
