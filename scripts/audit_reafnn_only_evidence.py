#!/usr/bin/env python3
"""Independently replay retained no-KNN candidate metrics without model fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prosys_shared.mainline import evaluate_scored_frame_with_manifest
from prosys_shared.route_cache import load_route_cache_sample_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "Experiment/stage2_reafnn_only_multiseed_20260913")
    args = parser.parse_args()
    rows = []
    for path in sorted((args.study / "compact").glob("seed_*/*/result.json")):
        record = json.loads(path.read_text())
        audit = json.loads((path.parent / "candidate_audit.json").read_text())
        frame = pd.read_csv(path.parent / "test_candidates.csv.gz", keep_default_na=False)
        keys = audit["candidate_identity_columns"]
        identities = frame[keys].astype(str).sort_values(keys, kind="mergesort")
        digest = hashlib.sha256(identities.to_csv(index=False).encode()).hexdigest()
        assert digest == audit["candidate_identity_sha256"], path
        cache = ROOT / "outputs/stage1_routes" / record["family"] / "route_cache.json"
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == record["input_sha256"]["stage1_test"]
        metrics = evaluate_scored_frame_with_manifest(frame,
            expected_sample_indices=load_route_cache_sample_indices(cache), score_column="xgb_score")
        fields = ["pool_coverage", "system_mrr", "system_ndcg10", "candidate_slates", "missing_candidate_slates"]
        fields += [f"system_top{k}_all" for k in (1, 3, 5, 10)]
        for field in fields:
            assert abs(float(metrics[field]) - float(record["metrics"][field])) < 1e-10, (path, field)
        rows.append({"family": record["family"], "seed": record["seed"],
                     "identity_hash_verified": True, "metrics_replayed": True})
    manifest = json.loads((args.study / "run_manifest.json").read_text())
    expected = len(manifest["seeds"]) * len(manifest["families"])
    status = {"completed_records_audited": len(rows), "expected_records": expected,
              "complete": len(rows) == expected, "rows": rows}
    (args.study / "independent_evidence_audit.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Verified candidate hashes and replayed metrics for {len(rows)}/{expected} records.")


if __name__ == "__main__":
    main()
