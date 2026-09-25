#!/usr/bin/env python3
"""Replay fresh 50K-route evidence; never read historical result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from prosys_shared.cache_integrity import require_manifest
from prosys_shared.evidence import identity_hash, ranked_frame, require_same_temperature_support
from prosys_shared.mainline import FAMILY_ORDER, evaluate_scored_frame_with_manifest
from scripts.run_50k_downstream_evidence import binding_for, evidence_files
from scripts.run_mainline_evidence import read_json, sha, write_json
from scripts.summarize_mainline_evidence import align_control_rows, flat

ARMS = ("full", "no_ltr", "no_rgnn_temperature")


def compare_metrics(actual, expected, keys=None):
    actual, expected = flat(actual), flat(expected)
    for key in keys or actual:
        a, b = actual[key], expected[key]
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError("Missing metric disagrees: " + key)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8, err_msg=key)


def verify_run(study, route_study, family, seed):
    directory = study / "compact" / f"seed_{seed}" / family
    binding = binding_for(route_study, family, seed)
    require_manifest(directory / "evidence_manifest.json", binding, evidence_files(directory))
    if read_json(directory / "completion.json").get("status") != "complete":
        raise ValueError("Run not complete: " + str(directory))
    full = read_json(directory / "result.json")
    controls = read_json(directory / "control_metrics.json")
    audit = read_json(directory / "exact_control_audit.json")
    provenance = read_json(directory / "provenance.json")
    for name, digest in provenance["models"].items():
        if sha(directory / "bundle" / name) != digest:
            raise ValueError("Retained model changed: " + name)
    cache_path = route_study / "test" / family / "route_cache.json"
    if provenance["route_test"] != sha(cache_path):
        raise ValueError("Different test route cache")
    if provenance["route_validation"] != sha(route_study / "val" / family / "route_cache.json"):
        raise ValueError("Different validation route cache")
    queries = read_json(cache_path)["reactions"]
    expected = [int(q["sample_index"]) for q in queries]
    frame = pd.read_csv(directory / "full_candidates.csv.gz", float_precision="round_trip")
    if identity_hash(frame) != audit["candidate_identity_sha256"]:
        raise ValueError("Candidate pool differs from exact-control receipt")
    if not audit["same_pool_no_ltr"] or not audit["same_ranked_identities_no_graph"]:
        raise ValueError("Exact control audit did not pass")
    replay = evaluate_scored_frame_with_manifest(frame, expected_sample_indices=expected,
        score_column="xgb_score", temperature_column="xgb_temperature_pred")
    compare_metrics(replay, full["metrics"])
    no_ltr = align_control_rows(frame, pd.read_csv(directory / "no_ltr_scores.csv.gz", float_precision="round_trip"))
    alternate = frame.copy()
    alternate["stage2_prior_score"] = no_ltr.stage2_prior_score.to_numpy()
    replay_no_ltr = evaluate_scored_frame_with_manifest(alternate,
        expected_sample_indices=expected, score_column="stage2_prior_score")
    compare_metrics(replay_no_ltr, controls["no_ltr"], ("cover", "sys1", "sys3", "sys5", "sys10", "mrr", "ndcg10"))
    no_graph = align_control_rows(frame, pd.read_csv(directory / "no_graph_predictions.csv.gz", float_precision="round_trip"))
    np.testing.assert_allclose(frame.xgb_score, no_graph.xgb_score, rtol=1e-6, atol=1e-7)
    alternate = frame.copy()
    alternate["xgb_temperature_pred"] = no_graph.xgb_temperature_pred.to_numpy()
    support = require_same_temperature_support(frame, alternate,
        left_column="xgb_temperature_pred", right_column="xgb_temperature_pred")
    if support != audit["temperature_support"]:
        raise ValueError("Conditional temperature support changed")
    if identity_hash(ranked_frame(frame), ordered=True) != identity_hash(ranked_frame(alternate), ordered=True):
        raise ValueError("No-R-GNN control changed system order")
    replay_no_graph = evaluate_scored_frame_with_manifest(alternate, expected_sample_indices=expected,
        score_column="xgb_score", temperature_column="xgb_temperature_pred")
    compare_metrics(replay_no_graph, controls["no_rgnn_temperature"])
    q = pd.read_csv(directory / "queries.csv")
    if q.sample_index.tolist() != expected:
        raise ValueError("Query denominator or order changed")
    ordered = ranked_frame(frame)
    first = ordered[ordered.label > 0.5].groupby("sample_index").final_rank.min()
    for k in (1, 3, 5, 10):
        hits = q.sample_index.map(first).le(k)
        np.testing.assert_array_equal(hits.to_numpy(), q[f"sys{k}"].to_numpy())
        np.testing.assert_allclose(hits.mean(), replay[f"system_top{k}_all"], atol=1e-10)
    rows = [{"family": family, "seed": seed, "arm": arm, "n_queries": len(expected), **flat(metric)}
            for arm, metric in zip(ARMS, (replay, replay_no_ltr, replay_no_graph))]
    receipt = {"family": family, "seed": seed, "n_queries": len(expected),
        "evidence_manifest_sha256": sha(directory / "evidence_manifest.json"),
        "candidate_sha256": identity_hash(frame), "metric_replay": True,
        "temperature_support": support}
    return rows, receipt


def aggregate(rows, families=FAMILY_ORDER):
    frame = pd.DataFrame(rows)
    expected = {(f, s, a) for f in families for s in (0, 1, 2) for a in ARMS}
    actual = list(zip(frame.family, frame.seed, frame.arm))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Need every family, seed and arm exactly once")
    if (frame.groupby("family").n_queries.nunique() != 1).any():
        raise ValueError("Query support differs between arms/seeds")
    if (frame.n_queries <= 0).any() or (frame.n_queries % 1 != 0).any():
        raise ValueError("Query support must be positive integers")
    for key in ("cover", "sys1", "sys3", "sys5", "sys10", "mrr", "ndcg10"):
        if not np.isfinite(frame[key]).all() or not frame[key].between(0, 1).all():
            raise ValueError("Nonfinite or out-of-range metric: " + key)
    if (frame[["sys1", "sys3", "sys5", "sys10"]].diff(axis=1).iloc[:, 1:] < 0).any().any():
        raise ValueError("Sys@k must be monotonic")
    metrics = [c for c in frame if c not in ("family", "seed", "arm", "n_queries")]
    macro = frame.groupby(["arm", "seed"])[metrics].mean()
    macro["n_queries"] = frame.groupby(["arm", "seed"]).n_queries.sum()
    macro["temp_n"] = frame.groupby(["arm", "seed"]).temp_n.sum()
    macro["temperature_families"] = frame.groupby(["arm", "seed"]).temp_mae.count()
    family_summary = frame.groupby(["family", "arm"])[metrics].agg(["mean", "std", "count"])
    macro_summary = macro.groupby("arm")[[k for k in metrics if k != "temp_n"]].agg(["mean", "std", "count"])
    return frame, macro.reset_index(), family_summary, macro_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--route-study", type=Path, required=True)
    args = parser.parse_args()
    study, routes = args.study_root.resolve(), args.route_study.resolve()
    missing = [(family, seed) for family in FAMILY_ORDER for seed in (0, 1, 2)
        if not (study / "compact" / f"seed_{seed}" / family / "evidence_manifest.json").exists()]
    if missing:
        raise RuntimeError("Full-study report withheld; missing certified runs: " + str(missing))
    rows, receipts = [], []
    for family in FAMILY_ORDER:
        for seed in (0, 1, 2):
            values, receipt = verify_run(study, routes, family, seed)
            rows.extend(values)
            receipts.append(receipt)
    frame, macro, family_summary, macro_summary = aggregate(rows)
    frame.to_csv(study / "per_family_seed_controls.csv", index=False)
    macro.to_csv(study / "macro_controls_by_seed.csv", index=False)
    family_summary.to_csv(study / "per_family_controls_mean_std.csv")
    macro_summary.to_csv(study / "macro_controls_mean_std.csv")
    lines = ["# Fresh 50K-Route Downstream Evidence", "",
        "All 18 family/downstream-seed runs passed retained-row replay. All downstream seeds use "
        "the fixed family expert seed 1. These are not three independent base-pretraining runs.", "",
        "Rates are percentages; mean +/- sample SD across downstream seeds 0/1/2. "
        "Macro averages weight families equally, not queries. Temperature is evaluated separately "
        "on the first ranked exact system with a finite reference/prediction, without a top-10 restriction. "
        "Missing temperature support is NA, never zero; support counts and available-family counts are in CSV.", "",
        "| Arm | Sys@1 | Sys@3 | Sys@5 | Sys@10 | Temp MAE (C) | Within 10 C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        part = macro[macro.arm == arm]
        cells = []
        for key in ("sys1", "sys3", "sys5", "sys10", "temp_mae", "temp_within_10c"):
            scale = 1 if key == "temp_mae" else 100
            values = part[key].dropna() * scale
            cells.append("NA" if values.empty else f"{values.mean():.2f} +/- {values.std(ddof=1):.2f}")
        lines.append("| " + arm + " | " + " | ".join(cells) + " |")
    lines += ["", "No-LTR reuses exactly the full candidate pool. No-R-GNN changes only the "
        "temperature regressor and preserves system order and conditional temperature support. "
        "These controls do not replace Stage 2 removal experiments or the four baseline models.", ""]
    (study / "RESULTS.md").write_text("\n".join(lines))
    write_json({"pass": True, "complete": True, "verified_runs": receipts,
        "summary_source_sha256": sha(Path(__file__)), "scope": "mainline and exact Stage 3 controls only"},
        study / "replay_verification.json")


if __name__ == "__main__":
    main()
