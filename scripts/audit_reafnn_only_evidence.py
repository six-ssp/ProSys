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


def resolve_route_root(manifest, explicit=None):
    recorded = manifest.get("route_root")
    if explicit is None and recorded is None:
        raise ValueError("No recorded route root; supply --route-root explicitly")
    chosen = Path(explicit if explicit is not None else recorded).resolve()
    if recorded is not None and chosen != Path(recorded).resolve():
        raise ValueError("Explicit route root differs from the study manifest")
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "Experiment/stage2_reafnn_only_multiseed_20260913")
    parser.add_argument("--route-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.study / "run_manifest.json").read_text())
    route_root = resolve_route_root(manifest, args.route_root)
    expected_pairs = {(family, seed) for family in manifest["families"] for seed in manifest["seeds"]}
    if not expected_pairs:
        raise ValueError("Cannot certify an empty ablation study")
    from scripts.run_reafnn_only_ablation import input_bindings, require_cached_evidence, sha
    bindings = None
    if manifest.get("family_inputs"):
        bindings = input_bindings(manifest["families"], route_root, Path(manifest["validation_route_root"]))
        if bindings != manifest["family_inputs"]:
            raise ValueError("Original ablation input bindings changed")
        for name, digest in manifest["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("Ablation source changed: " + name)
    rows = []
    observed = set()
    for path in sorted((args.study / "compact").glob("seed_*/*/result.json")):
        record = json.loads(path.read_text())
        pair = (record["family"], record["seed"])
        if pair in observed or pair not in expected_pairs:
            raise ValueError("Duplicate or unexpected family/seed record")
        observed.add(pair)
        if bindings is not None:
            require_cached_evidence(record, path.parent, bindings[record["family"]])
        audit = json.loads((path.parent / "candidate_audit.json").read_text())
        frame = pd.read_csv(path.parent / "test_candidates.csv.gz", keep_default_na=False)
        keys = audit["candidate_identity_columns"]
        identities = frame[keys].astype(str).sort_values(keys, kind="mergesort")
        digest = hashlib.sha256(identities.to_csv(index=False).encode()).hexdigest()
        assert digest == audit["candidate_identity_sha256"], path
        cache = route_root / record["family"] / "route_cache.json"
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == record["input_sha256"]["stage1_test"]
        metrics = evaluate_scored_frame_with_manifest(frame,
            expected_sample_indices=load_route_cache_sample_indices(cache), score_column="xgb_score")
        fields = ["pool_coverage", "system_mrr", "system_ndcg10", "candidate_slates", "missing_candidate_slates"]
        fields += [f"system_top{k}_all" for k in (1, 3, 5, 10)]
        for field in fields:
            assert abs(float(metrics[field]) - float(record["metrics"][field])) < 1e-10, (path, field)
        rows.append({"family": record["family"], "seed": record["seed"],
                     "identity_hash_verified": True, "metrics_replayed": True})
    expected = len(expected_pairs)
    status = {"completed_records_audited": len(rows), "expected_records": expected,
              "complete": observed == expected_pairs, "rows": rows,
              "route_root": str(route_root), "source_sha256": sha(Path(__file__))}
    (args.study / "independent_evidence_audit.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Verified candidate hashes and replayed metrics for {len(rows)}/{expected} records.")


if __name__ == "__main__":
    main()
