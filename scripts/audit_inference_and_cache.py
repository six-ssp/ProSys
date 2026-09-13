#!/usr/bin/env python3
"""Real smoke replay: label-free inference parity and cache rejection/reuse."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "Experiment/mainline_evidence_completion_20260913"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from prosys_shared.cache_integrity import file_sha256
    from scripts.run_verified_mainline import output_files
    artifact = STUDY / "compact/seed_0/Beckmann"
    retained = pd.read_csv(artifact / "full_candidates.csv.gz").query("sample_index == 0")
    inferred = pd.read_csv(STUDY / "inference_smoke_cached/ranked_candidates.csv.gz")
    features = json.loads((artifact / "bundle/ranker/xgb_ranker_meta.json").read_text())["feature_columns"]
    key = ["reactants", "reagent_norm", "solvent_norm"]
    a = retained.fillna({k: "" for k in key}).sort_values(key)
    b = inferred.fillna({k: "" for k in key}).sort_values(key)
    assert a[key].values.tolist() == b[key].values.tolist()
    np.testing.assert_allclose(a[features], b[features], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(a[["xgb_score", "xgb_temperature_pred"]],
                               b[["xgb_score", "xgb_temperature_pred"]], rtol=1e-5, atol=1e-5)
    fresh = json.loads((STUDY / "inference_smoke_fresh_v2/prediction.json").read_text())
    assert fresh["route_source"]["mode"] == "fresh_editretro_decoding"
    assert fresh["num_routes"] == 10 and fresh["num_candidates"] == 200
    assert len(fresh["top10"]) == 10
    assert all(np.isfinite(row["xgb_temperature_pred"]) for row in fresh["top10"])
    output = ROOT / "outputs/verified_entrypoint_smoke_20260913"
    cmd = [sys.executable, str(ROOT / "scripts/run_verified_mainline.py"), "--families", "Beckmann",
        "--max_train_routes", "20", "--max_val_routes", "10", "--skip_temperature",
        "--reafnn_device", "cuda:0", "--output_root", str(output)]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    receipts = []
    if args.rebuild:
        started = time.perf_counter()
        run = subprocess.run(cmd + ["--force_rebuild"], cwd=ROOT, env=env, text=True, capture_output=True)
        (STUDY / "cache_smoke_build.log").write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stderr
        receipts.append({"command": "forced_rebuild", "seconds": time.perf_counter() - started, "returncode": run.returncode})
    files = output_files(output / "Beckmann")
    models = {k: (file_sha256(p), p.stat().st_mtime_ns) for k, p in files.items()
              if p.suffix == ".pt" or p.name in {"xgb_ranker.json", "xgb_temperature.json"}}
    assert models
    for description, extra, expected in (("identical_cache_reuse", [], 0), ("changed_seed_rejected", ["--seed", "1"], 1)):
        started = time.perf_counter()
        run = subprocess.run(cmd + extra, cwd=ROOT, env=env, text=True, capture_output=True)
        (STUDY / ("cache_smoke_" + description + ".log")).write_text(run.stdout + run.stderr)
        assert (run.returncode == 0) == (expected == 0), run.stderr
        if expected:
            assert "fingerprint changed" in run.stderr, run.stderr
        assert models == {k: (file_sha256(files[k]), files[k].stat().st_mtime_ns) for k in models}
        receipts.append({"command": description, "seconds": time.perf_counter() - started,
                         "returncode": run.returncode, "model_bytes_and_mtimes_unchanged": True})
    result = {"label_free_inference": {"candidate_identity_count": len(a), "feature_count": len(features),
        "features_equal": True, "ranking_and_temperature_equal": True,
        "fresh_stage1_to_temperature": True, "fresh_prediction_sha256": file_sha256(STUDY / "inference_smoke_fresh_v2/prediction.json")},
        "cache_entrypoint": receipts,
        "scope": "cache smoke limits labelled ranker training/validation routes to 20/10; not a reportable accuracy experiment"}
    (STUDY / "inference_cache_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
