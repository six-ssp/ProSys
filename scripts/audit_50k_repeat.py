#!/usr/bin/env python3
"""Audit a fresh same-seed full Stage 2/3 repeat, without score selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from prosys_shared.cache_integrity import file_sha256 as sha
from prosys_shared.evidence import identity_hash, ranked_frame
from scripts.summarize_mainline_evidence import align_control_rows


def compare_frame(left, right):
    if set(left.columns) != set(right.columns):
        return {"exact": False, "reason": "column schema differs"}
    right = align_control_rows(left, right)
    left = left.reset_index(drop=True)
    differences = {}
    for column in left:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            mask = ~(np.equal(av, bv) | (np.isnan(av) & np.isnan(bv)))
            if mask.any():
                finite = np.isfinite(av) & np.isfinite(bv)
                differences[column] = {"rows": int(mask.sum()), "max_absolute_difference":
                    float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else None}
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            differences[column] = {"rows": int((a.fillna("").astype(str) != b.fillna("").astype(str)).sum())}
    return {"exact": not differences, "rows": len(left), "differences": differences,
            "candidate_identity_sha256": identity_hash(left)}


def compare_state(left, right, prefix=""):
    import torch
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [prefix + "/keys"]
        return [difference for key in left for difference in compare_state(left[key], right[key], prefix + "/" + str(key))]
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [prefix + "/length"]
        return [difference for i, (a, b) in enumerate(zip(left, right))
                for difference in compare_state(a, b, prefix + "/" + str(i))]
    if torch.is_tensor(left) and torch.is_tensor(right):
        return [] if left.dtype == right.dtype and torch.equal(left, right) else [prefix]
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return [] if left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True) else [prefix]
    return [] if type(left) is type(right) and left == right else [prefix]


def main():
    import torch
    from scripts.summarize_50k_downstream_evidence import verify_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-study", type=Path, required=True)
    parser.add_argument("--repeat-study", type=Path, required=True)
    parser.add_argument("--route-study", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new repeat-audit receipt")
    studies = [args.original_study.resolve(), args.repeat_study.resolve()]
    if studies[0] == studies[1]:
        raise ValueError("A repeat needs a distinct fitted study, not the same files")
    folders = [s / "compact" / f"seed_{args.seed}" / args.family for s in studies]
    rows, receipts = zip(*(verify_run(s, args.route_study.resolve(), args.family, args.seed) for s in studies))
    bindings = [json.loads((f / "input_binding.json").read_text()) for f in folders]
    if bindings[0] != bindings[1]:
        raise ValueError("Repeat has different inputs, sources, packages or seed")
    paths = ("full_candidates.csv.gz", "no_ltr_scores.csv.gz", "no_graph_predictions.csv.gz")
    frames, evidence = {}, {}
    for name in paths:
        left, right = [pd.read_csv(f / name, float_precision="round_trip") for f in folders]
        frames[name] = compare_frame(left, right)
        evidence[name] = [sha(f / name) for f in folders]
        if name == "full_candidates.csv.gz":
            same_ranking = identity_hash(ranked_frame(left), ordered=True) == identity_hash(ranked_frame(right), ordered=True)
    embeddings = [pd.read_csv(f / "route_embeddings.csv.gz", float_precision="round_trip") for f in folders]
    for frame in embeddings:
        for key, value in (("sample_index", 0), ("reaction_id", "route"), ("reagent_norm", ""), ("solvent_norm", "")):
            frame[key] = value
    frames["route_embeddings.csv.gz"] = compare_frame(*embeddings)
    model_checks = {}
    for name in ("reafnn/reafnn_model.pt", "rgnn/reaction_gnn.pt"):
        models = [torch.load(f / "bundle" / name, map_location="cpu", weights_only=False) for f in folders]
        differences = compare_state(*models)
        model_checks[name] = {"state_exact": not differences, "differing_keys": differences,
                              "sha256": [sha(f / "bundle" / name) for f in folders]}
    for name in ("ranker/xgb_ranker.json", "temperature/xgb_temperature.json"):
        hashes = [sha(f / "bundle" / name) for f in folders]
        model_checks[name] = {"state_exact": hashes[0] == hashes[1], "sha256": hashes}
    calibrations = [json.loads((f / "bundle/reafnn/post_fusion_calibration.json").read_text()) for f in folders]
    same_weight = calibrations[0]["selected_knn_weight"] == calibrations[1]["selected_knn_weight"]
    deterministic = (all(item["exact"] for item in frames.values()) and
                     all(item["state_exact"] for item in model_checks.values()) and same_ranking and same_weight)
    result = {"audit_completed": True, "same_seed_exact_reproduction": deterministic,
        "family": args.family, "seed": args.seed, "studies": [str(s) for s in studies],
        "inputs_and_sources_identical": True, "individual_replay_receipts": receipts,
        "same_complete_ranking": same_ranking, "same_validation_selected_weight": same_weight,
        "frames": frames, "models": model_checks, "evidence_sha256": evidence,
        "per_run_metrics": rows, "audit_source_sha256": sha(Path(__file__)),
        "scope": "One family's same-seed full Stage 2/3 repeat on fixed 50K expert routes; not cross-hardware or all-family determinism"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "same_seed_exact_reproduction": deterministic}))


if __name__ == "__main__":
    main()
