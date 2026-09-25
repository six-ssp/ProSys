#!/usr/bin/env python3
"""Replay and compact fresh KNN-only runs before explicitly pruning scratch."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from prosys_shared.cache_integrity import require_manifest, save_manifest
from prosys_shared.evidence import identity_hash
from prosys_shared.mainline import (FAMILY_ORDER, evaluate_scored_frame_with_manifest,
    load_split_rows, split_file_for_family, normalize_condition_labels,
    canonicalize_reaction_side, canonicalize_smiles)
from scripts.run_50k_downstream_evidence import binding_for
from scripts.run_mainline_evidence import sha, write_json


LIBRARY_COLUMNS = {"reafnn_context_count", "reafnn_context_support",
                   "reafnn_mean_yield", "reafnn_is_historical"}


def verify_intervention(frame):
    # The historical schema aliases four train-library statistics under this
    # prefix; verify their source independently rather than treating them as NN output.
    neural = [c for c in frame if c not in LIBRARY_COLUMNS and
              (c.startswith("reafnn_") or c.startswith("from_reafnn_") or c.startswith("stage2_reafnn_"))]
    if not neural:
        raise ValueError("No neural-feature columns available to verify removal")
    if not frame[neural].apply(pd.to_numeric, errors="raise").eq(0).all().all():
        raise ValueError("Nonzero ReaFNN evidence in the KNN-only control")
    if frame.groupby(["sample_index", "reactants"], dropna=False).size().max() > 20:
        raise ValueError("More than 20 contexts on a route")
    if "stage2_post_fusion_enabled" in frame and frame.stage2_post_fusion_enabled.ne(0).any():
        raise ValueError("Post-fusion unexpectedly enabled in KNN-only control")
    return neural


@lru_cache(maxsize=100000)
def reaction_key(reactants, product):
    r, p = canonicalize_reaction_side(reactants), canonicalize_smiles(product)
    return (r, p) if r and p else ("raw:" + reactants, "raw:" + product)


def library_statistics(rows):
    global_counts = defaultdict(lambda: np.zeros(3, dtype=float))
    per_reaction = defaultdict(lambda: defaultdict(lambda: np.zeros(3, dtype=float)))
    for row in rows:
        context = (normalize_condition_labels(row["reagent_norm"]), normalize_condition_labels(row["solvent_norm"]))
        key = reaction_key(row["reactants"], row["product"])
        try:
            y = float(row["yield"])
        except (TypeError, ValueError):
            y = float("nan")
        contribution = np.array([1., y if np.isfinite(y) else 0., float(np.isfinite(y))])
        global_counts[context] += contribution
        per_reaction[key][context] += contribution
    return global_counts, per_reaction, len(rows)


def verify_library_statistics(frame, statistics, *, training):
    global_counts, per_reaction, total = statistics
    expected = []
    for row in frame.itertuples(index=False):
        context = tuple(normalize_condition_labels("" if pd.isna(value) else str(value))
                        for value in (row.reagent_norm, row.solvent_norm))
        if context not in global_counts:
            raise ValueError("KNN context absent from training library")
        removed = per_reaction.get(reaction_key(row.reactants, row.product), {}) if training else {}
        count, yield_sum, yield_n = global_counts[context] - removed.get(context, np.zeros(3))
        remaining = total - sum(v[0] for v in removed.values())
        if count <= 0:
            raise ValueError("No support after leave-one-reaction-out exclusion")
        expected.append([count, count / max(remaining, 1.), yield_sum / yield_n if yield_n > 0 else 0., 1.])
    columns = ["reafnn_context_count", "reafnn_context_support", "reafnn_mean_yield", "reafnn_is_historical"]
    np.testing.assert_allclose(frame[columns].to_numpy(float), np.asarray(expected), rtol=1e-10, atol=1e-10)
    return {"columns": columns, "rows_verified": len(frame), "train_library_only": True,
            "leave_one_canonical_reaction_out": training}


def current_binding(routes, family):
    binding = binding_for(routes, family, 0)
    binding["config"].update(arm="knn_only", seeds=[0, 1, 2])
    name = "scripts/run_current_mainline_matched_ablations.py"
    binding["inputs"]["source/" + name] = sha(ROOT / name)
    return binding


def audit_family(study, scratch, routes, family, prune=False):
    binding = current_binding(routes, family)
    if json.loads((study / "preflight_inputs.json").read_text()) != binding:
        raise ValueError("KNN-only sources/inputs changed since pre-training admission")
    config = json.loads((study / "run_manifest.json").read_text())
    if (config["families"] != [family] or config["seeds"] != [0, 1, 2] or
            set(config["arms"]) != {"knn_only"} or Path(config["route_root"]).resolve() != routes / "test"):
        raise ValueError("Unexpected KNN-only experiment configuration")
    cache = routes / "test" / family / "route_cache.json"
    expected = [int(q["sample_index"]) for q in json.loads(cache.read_text())["reactions"]]
    statistics = library_statistics(load_split_rows(split_file_for_family(ROOT, family, "train")))
    records = []
    for seed in (0, 1, 2):
        job = scratch / "knn_only" / f"seed_{seed}" / family
        raw_path = job / "knn_xgb/non_oracle/result.json"
        raw = json.loads(raw_path.read_text())
        if raw["family"] != family or raw["seed"] != seed:
            raise ValueError("KNN-only family/seed mismatch")
        protocol = raw["model"]["stage2_protocol"]
        if protocol["architecture"] != "knn_only" or protocol["reafnn_enabled"]:
            raise ValueError("Expected ReaFNN to be disabled")
        destination = study / "compact/knn_only" / f"seed_{seed}" / family
        compact = json.loads((destination / "result.json").read_text())
        if compact["metrics"] != raw["metrics"]:
            raise ValueError("Compacted result differs from raw metrics")
        table_root = Path(raw["candidate_table"]).parent
        tables = {}
        for split in ("train", "val", "test"):
            path = table_root / (split + ".csv")
            frame = pd.read_csv(path, float_precision="round_trip")
            columns = verify_intervention(frame)
            library_check = verify_library_statistics(frame, statistics, training=split == "train")
            tables[split] = {"sha256": sha(path), "rows": len(frame),
                "zero_neural_columns": columns, "identity_sha256": identity_hash(frame),
                "legacy_named_library_statistics": library_check}
        original = pd.read_csv(raw["scored_test_file"], float_precision="round_trip")
        replay = evaluate_scored_frame_with_manifest(original, expected_sample_indices=expected, score_column="xgb_score")
        fields = ["pool_coverage", "system_mrr", "system_ndcg10", "candidate_slates", "missing_candidate_slates"]
        fields += [f"system_top{k}_all" for k in (1, 3, 5, 10)]
        for field in fields:
            np.testing.assert_allclose(replay[field], raw["metrics"][field], atol=1e-10, rtol=0)
        retained = destination / "test_candidates.csv.gz"
        original.to_csv(retained, index=False, compression="gzip")
        roundtrip = pd.read_csv(retained, float_precision="round_trip")
        pd.testing.assert_frame_equal(original, roundtrip, check_exact=False, rtol=1e-12, atol=1e-12)
        if identity_hash(roundtrip) != identity_hash(original):
            raise ValueError("Candidate identity changed during compression")
        ranker = Path(raw["model"]["model_file"])
        copied = destination / ranker.name
        shutil.copy2(ranker, copied)
        if sha(ranker) != sha(copied):
            raise ValueError("Ranker retention changed model bytes")
        receipt = {"family": family, "seed": seed, "n_queries": len(expected), "metric_replay": True,
            "tables": tables, "candidate_sha256": identity_hash(original),
            "raw_result_sha256": sha(raw_path), "metrics": replay,
            "source_sha256": sha(Path(__file__))}
        write_json(receipt, destination / "independent_replay.json")
        outputs = {str(p.relative_to(destination)): p for p in destination.rglob("*")
                   if p.is_file() and p.name != "evidence_manifest.json"}
        save_manifest(destination / "evidence_manifest.json", binding, outputs)
        require_manifest(destination / "evidence_manifest.json", binding, outputs)
        records.append(receipt)
    if current_binding(routes, family) != binding:
        raise ValueError("Inputs changed while retaining evidence")
    # No raw directory is deleted until all three independent replays passed.
    if prune:
        for seed in (0, 1, 2):
            shutil.rmtree(scratch / "knn_only" / f"seed_{seed}" / family)
    report = {"pass": True, "complete": True, "family": family,
        "scope": "one-family KNN-only three-seed control", "records": records,
        "scratch_pruned": prune, "binding": binding}
    write_json(report, study / "independent_evidence_audit.json")
    print(json.dumps({"family": family, "verified_runs": len(records), "scratch_pruned": prune,
        "sys10": [r["metrics"]["system_top10_all"] for r in records]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--route-study", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--prune-scratch", action="store_true")
    args = parser.parse_args()
    study, scratch, routes = [p.resolve() for p in (args.study, args.scratch, args.route_study)]
    scratch.relative_to(ROOT / "outputs")
    if (scratch == ROOT / "outputs" or scratch == study or scratch in study.parents or
            study in scratch.parents or scratch in routes.parents or routes in scratch.parents):
        raise ValueError("Scratch must be isolated below outputs")
    if args.family not in FAMILY_ORDER:
        raise ValueError("Unsupported family")
    audit_family(study, scratch, routes, args.family, args.prune_scratch)


if __name__ == "__main__":
    main()
