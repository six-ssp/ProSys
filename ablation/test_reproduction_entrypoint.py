"""Execute shell orchestration against a temporary repo and recording stubs.

This tests command wiring, not neural training or regenerated data quality.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionEntrypointTests(unittest.TestCase):
    def test_full_workflow_wiring_without_real_deletions_or_training(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "scripts").mkdir()
            for script in ("reproduce_mainline_from_raw.sh", "run_stage23_non_oracle_suite.sh"):
                shutil.copy2(ROOT / "scripts" / script, root / "scripts" / script)
            calls = root / "calls.jsonl"
            python = root / "record_python"
            python.write_text("#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ['RECORD_CALLS'],'a') as h: h.write(json.dumps(sys.argv[1:])+'\\n')\n")
            python.chmod(0o755)
            for script in ("scripts/setup_prosys_env.sh", "stage1_retrosynthesis/scripts/run_base_train.sh",
                           "stage1_retrosynthesis/scripts/run_family_finetune_batch.sh"):
                path = root / script
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#!/bin/bash\n"$PYTHON_BIN" "' + script + '" "$@"\n')
            env = dict(os.environ, PYTHON_BIN=str(python), RECORD_CALLS=str(calls), RUN_BASE_TRAIN="1",
                RESET_PROCESSED="0", CLEAN_LEGACY="0", RESET_STAGE1_RESULTS="0", RESET_BASE_RESULTS="0",
                SKIP_PREPROCESS="0", SKIP_STAGE1_FINETUNE="0", SKIP_AUDIT="0", FAMILIES="all")
            result = subprocess.run(["bash", str(root / "scripts/reproduce_mainline_from_raw.sh"), str(root)],
                env=env, capture_output=True, text=True, check=True)
            records = [json.loads(line) for line in calls.read_text().splitlines()]
            preprocess = next(r for r in records if r[0] == "data_preprocess/preprocess.py")
            self.assertEqual(preprocess[preprocess.index("--min_label_freq") + 1], "6")
            self.assertEqual(preprocess[preprocess.index("--min_yield") + 1], "25")
            self.assertTrue(any("run_base_train.sh" in r[0] for r in records))
            self.assertTrue(any("run_family_finetune_batch.sh" in r[0] for r in records))
            routes = [r for r in records if r[0] == "stage1_retrosynthesis/build_route_cache.py"]
            self.assertEqual(len(routes), 18)
            validation = [r for r in routes if "--gold_split" in r]
            self.assertEqual(len(validation), 6)
            self.assertTrue(all("validate_labels_processed.txt" in r[r.index("--gold_split") + 1] for r in validation))
            self.assertTrue(any(r[0] == "scripts/run_verified_mainline.py" for r in records))
            self.assertTrue(any(r[0] == "scripts/collect_checklist_stats.py" for r in records))
            self.assertIn("[reproduce] done", result.stdout)

    def test_reset_plus_skip_is_rejected_before_work(self):
        env = dict(os.environ, SKIP_PREPROCESS="1", RESET_PROCESSED="1")
        result = subprocess.run(["bash", str(ROOT / "scripts/reproduce_mainline_from_raw.sh"), str(ROOT)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing to delete", result.stderr)


if __name__ == "__main__":
    unittest.main()
