import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.predict_product import inference_features, readonly_builder, validate_feature_sources, ROOT
from prosys_shared.mainline import build_candidate_training_table


class ProductInferenceTests(unittest.TestCase):
    def test_label_free_feature_parity(self):
        rows = pd.DataFrame([dict(family="Beckmann", sample_index=0, reaction_id="x",
            product="CC(N)=O", reactants="CC=NO", reagent_norm="acid", solvent_norm="water",
            retro_rank=1, retro_score=2.0, retro_probability=0.8)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows.to_csv(root / "candidates.csv", index=False)
            (root / "gold.txt").write_text("x\tCC=NO\tCC(N)=O\t80\tacid\twater\t25\n")
            labelled = build_candidate_training_table(root / "candidates.csv", root / "gold.txt")
            inferred = inference_features(pd.read_csv(root / "candidates.csv"))
            pd.testing.assert_frame_equal(inferred, labelled[inferred.columns])
            self.assertNotIn("label", inferred)
            self.assertNotIn("temperature_gold", inferred)

    def test_no_training_or_validation_selection(self):
        artifact = ROOT / "Experiment/project_completion_20260913/canonical_sensitivity/corrected/seed_0"
        if not (artifact / "completion.json").exists():
            self.skipTest("Retained evidence bundle unavailable")
        with patch("stage2_ReaFNN.knn_condition_selector.train_reafnn_selector", side_effect=AssertionError("Training forbidden")), \
             patch("stage2_ReaFNN.knn_condition_selector.KNNContextPoolBuilder._load_or_select_post_fusion_weight", side_effect=AssertionError("Validation forbidden")):
            builder = readonly_builder(ROOT, "DielsAlder", artifact, "cpu")
            calibration = json.loads((artifact / "bundle/reafnn/post_fusion_calibration.json").read_text())
            self.assertEqual(builder._post_fusion_selected_weight, calibration["selected_knn_weight"])

    def test_missing_and_changed_feature_sources_fail_closed(self):
        from prosys_shared.cache_integrity import file_sha256
        files = ('prosys_shared/features.py', 'prosys_shared/mainline.py',
                 'prosys_shared/condition_modeling.py', 'prosys_shared/product_memory.py',
                 'stage2_ReaFNN/knn_condition_selector.py', 'stage2_ReaFNN/reafnn_selector.py',
                 'stage3_XGBoost/reaction_gnn_features.py', 'stage3_XGBoost/xgb_reranker.py')
        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name)
            with self.assertRaisesRegex(ValueError, 'lacks a feature-source'):
                validate_feature_sources(ROOT, artifact)
            manifest = {'source_sha256': {p: file_sha256(ROOT / p) for p in files}}
            (artifact / 'runtime_manifest.json').write_text(json.dumps(manifest))
            validate_feature_sources(ROOT, artifact)
            manifest['source_sha256']['prosys_shared/features.py'] = 'changed'
            (artifact / 'runtime_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'feature-source mismatch'):
                validate_feature_sources(ROOT, artifact)

    def test_legacy_override_is_not_accepted_as_corrected_model(self):
        artifact = ROOT / 'Experiment/project_completion_20260913/canonical_sensitivity/legacy/seed_0'
        if not (artifact / 'completion.json').exists():
            self.skipTest('Legacy sensitivity artifact unavailable')
        with self.assertRaisesRegex(ValueError, 'legacy feature override'):
            validate_feature_sources(ROOT, artifact)

    def test_late_positive_not_mislabeled_as_first_hit(self):
        from scripts.extract_current_case_examples import select_cases
        frame = pd.DataFrame([dict(sample_index=i, final_rank=r, label=int(r in hits), route_match=1)
            for i, hits in enumerate(({1, 10}, {3}, {10}, {5})) for r in range(1, 11)])
        selected = select_cases(frame, 3)
        self.assertEqual([int(row.sample_index) for _, row in selected], [0, 1, 2])
        self.assertEqual([int(row.final_rank) for _, row in selected], [1, 3, 10])


if __name__ == "__main__":
    unittest.main()
