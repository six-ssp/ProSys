import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prosys_shared.cache_integrity import file_sha256 as sha, save_manifest
from scripts.product_route_inference import (bind_routes, cached_routes, generate_routes,
                                              option, validate_single_guard)


class ProductRouteBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "bundle"
        self.artifact.mkdir()
        self.checkpoint = self.root / "expert.pt"
        self.checkpoint.write_bytes(b"fresh 50k expert")
        self.argv = ["--user-dir", "editretro", "/retained/data-bin", "--input", "/old/products.txt",
                     "--path", str(self.checkpoint), "--aug", "10", "--TOPK", "10",
                     "--repos-beam", "5", "--token-beam", "2", "--mask-beam", "1"]
        self.guard = {"pass": True, "protocol": "product_identity_fallback_v1",
            "source_sha256": {"guard.py": "digest"}, "query_count": 1, "augmented_input_count": 10,
            "replaced_variant_count": 1, "normalized_input_count": 0,
            "events": [{"query_index": 0, "augmentation_count": 10, "input_normalized": False,
                        "replaced_slots": [4]}], "argv": self.argv}
        self.cache = {"family": "Beckmann", "checkpoint": str(self.checkpoint),
            "aug": 10, "topk": 10, "n_best": 10,
            "reactions": [{"product": "CCO", "gold_reactants": "SHOULD_NOT_BE_USED",
                "routes": [{"reactants": "CC=O", "retro_rank": 1, "retro_score": 2., "retro_probability": 1.}]}]}
        inputs, provenance = {"checkpoint": sha(self.checkpoint)}, {"family": "Beckmann", "route_paths": {}}
        self.cache_paths = {}
        for split, key in (("val", "route_validation"), ("test", "route_test")):
            directory = self.root / split
            directory.mkdir()
            cache = directory / "route_cache.json"
            cache.write_text(json.dumps(self.cache))
            guard = directory / "augmentation_guard.json"
            guard.write_text(json.dumps(self.guard))
            inputs[split + "_routes"], inputs[split + "_guard"] = sha(cache), sha(guard)
            provenance[key], provenance["route_paths"][split] = sha(cache), str(cache)
            self.cache_paths[split] = cache
        self.binding = {"config": {"protocol": "50k_fixed_routes_exact_controls_v1",
                                   "family": "Beckmann", "expert_seed": 1}, "inputs": inputs}
        (self.artifact / "input_binding.json").write_text(json.dumps(self.binding))
        (self.artifact / "provenance.json").write_text(json.dumps(provenance))
        self.refresh_manifest()

    def refresh_manifest(self):
        save_manifest(self.artifact / "evidence_manifest.json", self.binding,
            {p.name: p for p in self.artifact.iterdir() if p.name != "evidence_manifest.json"})

    def admit(self, checkpoint=None):
        with patch("scripts.product_route_inference.verify_guard") as verifier:
            result = bind_routes(self.root, self.artifact, "Beckmann", checkpoint)
            self.assertEqual(verifier.call_count, 2)
            return result

    def test_admitted_model_resolves_its_own_expert(self):
        bound = self.admit()
        self.assertEqual(bound["checkpoint"], str(self.checkpoint))
        self.assertEqual(bound["argv"], self.argv)

    def test_old_bundle_without_input_binding_is_rejected(self):
        (self.artifact / "input_binding.json").unlink()
        with self.assertRaisesRegex(ValueError, "historical bundle"):
            self.admit()

    def test_checkpoint_relocation_allowed_but_other_model_rejected(self):
        replacement = self.root / "relocated.pt"
        replacement.write_bytes(self.checkpoint.read_bytes())
        self.assertEqual(self.admit(replacement)["checkpoint"], str(replacement))
        replacement.write_bytes(b"old full model")
        with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
            self.admit(replacement)

    def test_changed_bound_cache_is_rejected(self):
        self.cache_paths["test"].write_text("{}")
        with self.assertRaisesRegex(ValueError, "Stage 1 cache changed"):
            self.admit()

    def test_changed_guard_is_rejected(self):
        (self.cache_paths["test"].parent / "augmentation_guard.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Stage 1 guard changed"):
            self.admit()

    def test_missing_guard_evidence_cannot_be_bypassed(self):
        with patch("scripts.product_route_inference.verify_guard", side_effect=ValueError("guard failed")):
            with self.assertRaisesRegex(ValueError, "guard failed"):
                bind_routes(self.root, self.artifact, "Beckmann")

    def test_cached_routes_ignore_gold_and_reject_an_unbound_cache(self):
        bound = self.admit()
        routes, report = cached_routes(self.cache_paths["test"], "CCO", "Beckmann", bound)
        self.assertEqual(routes, self.cache["reactions"][0]["routes"])
        self.assertNotIn("SHOULD_NOT_BE_USED", json.dumps((routes, report)))
        unrelated = self.root / "old_cache.json"
        unrelated.write_text(json.dumps({**self.cache, "old": True}))
        with self.assertRaisesRegex(ValueError, "bound caches"):
            cached_routes(unrelated, "CCO", "Beckmann", bound)

    def test_ambiguous_product_caches_are_rejected(self):
        cache = copy.deepcopy(self.cache)
        cache["reactions"].append({"product": "CCO", "routes": []})
        path = self.root / "ambiguous.json"
        path.write_text(json.dumps(cache))
        bound = self.admit()
        bound["cache_sha256"] = {"test": sha(path)}
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            cached_routes(path, "CCO", "Beckmann", bound)

    def test_single_query_guard_checks_sources_counts_and_slots(self):
        validate_single_guard(self.guard, self.guard["source_sha256"])
        for key, value in (("pass", False), ("query_count", 2), ("augmented_input_count", 9),
                           ("source_sha256", {}), ("replaced_variant_count", 2)):
            invalid = {**self.guard, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_single_guard(invalid, self.guard["source_sha256"])
        invalid = copy.deepcopy(self.guard)
        invalid["events"][0]["replaced_slots"] = [0]
        with self.assertRaises(ValueError):
            validate_single_guard(invalid, self.guard["source_sha256"])

    def test_fresh_generation_replays_guarded_command_with_only_product_input(self):
        bound = self.admit()
        output = self.root / "prediction"

        def run(command, **kwargs):
            self.assertTrue(command[1].endswith("stage1_interactive_guarded.py"))
            argv = command[2:]
            self.assertEqual(Path(option(argv, "--input")).read_text(), "CCO\n")
            self.assertEqual(option(argv, "--path"), str(self.checkpoint))
            self.assertNotIn("gold", " ".join(command))
            receipt = Path(kwargs["env"]["PROSYS_AUGMENTATION_GUARD_RECEIPT"])
            receipt.write_text(json.dumps(self.guard))

        with patch("scripts.product_route_inference.subprocess.run", side_effect=run), \
             patch("stage1_retrosynthesis.build_route_cache.aggregate_routes", return_value=[[("CC=O", 2.)]]):
            routes, report = generate_routes(self.root, "CCO", output, "-1", bound)
        self.assertEqual(routes[0]["reactants"], "CC=O")
        self.assertEqual(report["mode"], "fresh_guarded_editretro_decoding")
        self.assertEqual(bound["argv"], self.argv)


if __name__ == "__main__":
    unittest.main()
