"""Lightweight safety tests; these do not certify real model performance."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from prosys_shared.cache_integrity import CacheMismatchError
from scripts import run_50k_downstream_evidence as runner


class EvidenceHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(route_study=self.root / "routes",
            study_root=self.root / "study", scratch_root=self.root / "scratch",
            family="Beckmann", seed=0)
        self.binding = {"config": {"seed": 0}, "inputs": {"test": "new_routes"}}
        self.dest = self.args.study_root / "compact/seed_0/Beckmann"
        self.scratch = self.args.scratch_root / "seed_0/Beckmann"

    def fake_child(self, family, seed, scratch, destination, **kwargs):
        self.assertTrue(kwargs["verified"])
        self.assertEqual(kwargs["route_root"], self.args.route_study / "test")
        self.assertEqual(kwargs["validation_route_root"], self.args.route_study / "val")
        (scratch / family).mkdir(parents=True)
        runner.write_json({"binding": self.binding}, scratch / family / "cache_manifest.json")
        runner.write_json({"status": "complete"}, destination / "completion.json")
        runner.write_json({"metric": 0.5}, destination / "result.json")

    def run_fake(self, binding=None):
        with patch.object(runner, "binding_for", return_value=binding or self.binding), \
                patch.object(runner, "child", side_effect=self.fake_child) as child, \
                patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
            runner.execute(self.args)
            return child.call_count

    def test_completed_evidence_reuses_only_with_verified_hashes(self):
        self.assertEqual(self.run_fake(), 1)
        self.assertFalse(self.scratch.exists())
        self.assertTrue((self.dest / "pipeline_cache_manifest.json").is_file())
        self.assertEqual(self.run_fake(), 0)

    def test_changed_inputs_cannot_reuse(self):
        self.run_fake()
        with self.assertRaises(CacheMismatchError):
            self.run_fake({"config": {"seed": 0}, "inputs": {"test": "old_routes"}})

    def test_changed_retained_results_cannot_reuse(self):
        self.run_fake()
        runner.write_json({"metric": 1.0}, self.dest / "result.json")
        with self.assertRaises(CacheMismatchError):
            self.run_fake()

    def test_completion_without_manifest_is_not_reused(self):
        runner.write_json({"status": "complete"}, self.dest / "completion.json")
        with self.assertRaises(CacheMismatchError):
            self.run_fake()

    def test_partial_scratch_not_deleted_or_resumed(self):
        self.scratch.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Existing scratch"):
            self.run_fake()
        self.assertTrue(self.scratch.exists())

    def test_midrun_input_change_preserves_scratch_without_certification(self):
        with patch.object(runner, "binding_for", side_effect=[self.binding, {}]), \
                patch.object(runner, "child", side_effect=self.fake_child), \
                patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
            with self.assertRaisesRegex(ValueError, "changed during"):
                runner.execute(self.args)
        self.assertTrue(self.scratch.exists())
        self.assertFalse((self.dest / "evidence_manifest.json").exists())

    def test_low_space_refuses_fitting(self):
        with patch.object(runner, "binding_for", return_value=self.binding), \
                patch.object(runner, "child") as child, \
                patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=1024**3)):
            with self.assertRaisesRegex(RuntimeError, "8 GiB"):
                runner.execute(self.args)
            child.assert_not_called()

    def test_protected_path_overlap_rejected(self):
        routes, study = self.args.route_study, self.args.study_root
        for scratch in (study, study / "tmp", routes, routes / "tmp", runner.ROOT / "data/tmp"):
            with self.subTest(scratch=scratch), self.assertRaises(ValueError):
                runner.require_separate_paths(routes, study, scratch)


class RouteAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.routes = self.root / "routes"
        self.experts = self.root / "experts"
        self.completion = self.experts / "seed_1/Beckmann/completion.json"
        runner.write_json({"pass": True}, self.completion)
        self.checkpoint = self.root / "best.pt"
        runner.write_json({"fake_checkpoint": True}, self.checkpoint)
        self.splits = {s: self.root / (s + ".json") for s in ("train", "val", "test")}
        for s, path in self.splits.items():
            runner.write_json({"split": s}, path)
        self.verified = {"family": "Beckmann", "seed": 1, "best_sha256": runner.sha(self.checkpoint)}
        self.guards = {s: {"test_fixture": s} for s in ("test", "val")}
        for s in self.guards:
            runner.write_json({"checkpoint": str(self.checkpoint)}, self.routes / s / "Beckmann/route_cache.json")
        self.protocol = {"seed": 1, "families": ["Beckmann"], "expert_study": str(self.experts),
            "source_sha256": {"scripts/run_50k_downstream_evidence.py": runner.sha(Path(runner.__file__))}}
        self.receipt = {"seed": 1, "family": "Beckmann", "expert_verification": self.verified,
            "expert_completion_sha256": runner.sha(self.completion),
            "checkpoint_sha256": runner.sha(self.checkpoint), "guarded_routes": self.guards,
            "split_sha256": {s: runner.sha(p) for s, p in self.splits.items()}}

    def admit(self):
        runner.write_json(self.protocol, self.routes / "protocol.json")
        runner.write_json(self.receipt, self.routes / "receipts/Beckmann.json")
        with ExitStack() as stack:
            stack.enter_context(patch("prosys_shared.mainline.split_file_for_family",
                                      side_effect=lambda root, family, split: self.splits[split]))
            stack.enter_context(patch("scripts.run_verified_mainline.verify_route_manifest"))
            stack.enter_context(patch("scripts.summarize_stage1_multiseed.verify_job", return_value=self.verified))
            stack.enter_context(patch("scripts.stage1_route_admission.verify_guard",
                                      side_effect=lambda path: self.guards[path.parent.parent.name]))
            stack.enter_context(patch("scripts.stage1_route_admission.require_paired_checkpoints",
                                      return_value=runner.sha(self.checkpoint)))
            return runner.admit_routes(self.routes, "Beckmann")

    def test_valid_pair_returns_explicit_new_inputs(self):
        inputs = self.admit()
        self.assertEqual(inputs["checkpoint"], self.checkpoint)
        self.assertEqual(inputs["test_routes"], self.routes / "test/Beckmann/route_cache.json")
        self.assertIn("val_routes", inputs)

    def test_other_expert_seed_rejected(self):
        self.protocol["seed"] = 0
        with self.assertRaisesRegex(ValueError, "seed-1"):
            self.admit()

    def test_changed_publication_source_rejected(self):
        self.protocol["source_sha256"]["scripts/run_50k_downstream_evidence.py"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.admit()

    def test_changed_split_rejected(self):
        self.receipt["split_sha256"]["test"] = "wrong"
        with self.assertRaisesRegex(ValueError, "split changed"):
            self.admit()

    def test_changed_expert_receipt_rejected(self):
        self.receipt["expert_completion_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "completion changed"):
            self.admit()

    def test_wrong_checkpoint_rejected(self):
        self.receipt["checkpoint_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "admitted expert checkpoint"):
            self.admit()

    def test_changed_guarded_routes_rejected(self):
        self.receipt["guarded_routes"] = {"test": {}, "val": {}}
        with self.assertRaisesRegex(ValueError, "Guarded routes changed"):
            self.admit()


if __name__ == "__main__":
    unittest.main()
