#!/usr/bin/env python3
"""Reconstruct baseline systems and validation fusion from retained predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from baseline import run_direct_product_condition_baselines as direct
from baseline.experiment_integrity import (source_files, family_inputs, verify_export,
                                           reuse_completed)
from baseline.external_adapters.run_baselines23 import _select_fusion
from prosys_shared.cache_integrity import fingerprint
from prosys_shared.mainline import build_candidate_training_table, evaluate_scored_frame_with_manifest, split_file_for_family
from scripts.run_50k_downstream_evidence import admit_routes
from scripts.run_mainline_evidence import read_json, write_json, sha
from scripts.summarize_50k_downstream_evidence import compare_metrics


def load_direct_predictions(path, queries):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    actual = [(r["sample_index"], str(r["reaction_id"]), r["product"]) for r in records]
    expected = [(q.sample_index, q.reaction_id, q.product) for q in queries]
    if actual != expected:
        raise ValueError("Direct prediction identities differ from current routes")
    return {r["sample_index"]: [direct.ContextPrediction(
        reagent_norm=c["reagent_norm"], solvent_norm=c["solvent_norm"],
        score=c["condition_score"], rank=c["context_rank"], support=c["knn_support"],
        max_similarity=c["knn_max_similarity"], neighbor_count=c["knn_neighbor_count"])
        for c in r["contexts"]] for r in records}


def check_selected_fusion(replayed, recorded):
    if replayed["selected"]["route_weight"] != recorded["selected"]["route_weight"]:
        raise ValueError("Validation-selected route weight differs")
    if [r["route_weight"] for r in replayed["candidates"]] != [r["route_weight"] for r in recorded["candidates"]]:
        raise ValueError("Fusion grid differs")
    for a, b in zip(replayed["candidates"], recorded["candidates"]):
        compare_metrics(a["metrics"], b["metrics"])


def audit_direct(directory, family, method, routes, scratch):
    metadata = read_json(directory / "run_metadata.json")
    recorded = read_json(directory / "fusion_selection.json")
    frames, indices = {}, {}
    for split, name in (("val", "validation"), ("test", "test")):
        queries, proposals = direct._cache_queries_and_routes(routes / split / family / "route_cache.json", family)
        direct._assert_cache_matches_split(queries=queries, split_file=split_file_for_family(ROOT, family, split), family=family)
        predictions = load_direct_predictions(directory / (name + "_condition_predictions.jsonl"), queries)
        raw = direct.build_joint_candidate_frame(queries, proposals, predictions, method=method)
        path = scratch / (name + ".csv")
        raw.to_csv(path, index=False)
        frames[split] = build_candidate_training_table(path, split_file_for_family(ROOT, family, split))
        indices[split] = [q.sample_index for q in queries]
    selection = direct.select_route_weight(frames["val"], indices["val"],
        route_weights=tuple(r["route_weight"] for r in recorded["candidates"]))
    check_selected_fusion(selection, recorded)
    scored = direct.fuse_candidate_scores(frames["test"], route_weight=selection["selected"]["route_weight"])
    metrics = evaluate_scored_frame_with_manifest(scored,
        expected_sample_indices=indices["test"], score_column="system_score")
    compare_metrics(metrics, metadata["test_metrics"])
    if len(scored) != metadata["test_candidate_rows"]:
        raise ValueError("Direct baseline joint candidate count differs")
    return metrics, selection["selected"]["route_weight"]


def audit_external(directory, family, routes):
    metadata = read_json(directory / "run_metadata.json")
    recorded = read_json(directory / "fusion_selection.json")
    indices = {}
    for split, name in (("val", "val"), ("test", "test")):
        query_records = read_json(routes / split / family / "route_cache.json")["reactions"]
        manifest = [json.loads(line) for line in (directory / (name + "_manifest.jsonl")).read_text().splitlines() if line.strip()]
        expected = [(int(q["sample_index"]), str(q["reaction_id"]), q["product"]) for q in query_records]
        actual = [(int(q["sample_index"]), str(q["reaction_id"]), q["product"]) for q in manifest]
        if actual != expected:
            raise ValueError("External baseline query manifest differs")
        indices[split] = [q[0] for q in expected]
    validation = build_candidate_training_table(directory / "validation_candidates.csv.gz",
        split_file_for_family(ROOT, family, "val"))
    selection = _select_fusion(validation, directory / "val_manifest.jsonl",
        route_weights=tuple(r["route_weight"] for r in recorded["candidates"]),
        temperature_column=metadata["temperature_column"])
    check_selected_fusion(selection, recorded)
    test = build_candidate_training_table(directory / "test_candidates.csv.gz",
        split_file_for_family(ROOT, family, "test"))
    stored = pd.read_csv(directory / "test_labeled_candidates.csv.gz", float_precision="round_trip")
    if len(test) != len(stored):
        raise ValueError("Relabeled external candidates changed row count")
    for column in ("label", "route_match", "context_match", "temperature_gold"):
        np.testing.assert_allclose(test[column], stored[column], equal_nan=True, rtol=1e-10, atol=1e-10)
    metrics = evaluate_scored_frame_with_manifest(test, expected_sample_indices=indices["test"],
        score_column="system_score", temperature_column=metadata["temperature_column"])
    compare_metrics(metrics, metadata["test_metrics"])
    return metrics, selection["selected"]["route_weight"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--route-study", type=Path, required=True)
    args = parser.parse_args()
    study, routes = args.study.resolve(), args.route_study.resolve()
    config = read_json(study / "experiment_config.json")
    binding = read_json(study / "experiment_binding.json")
    inputs = source_files(ROOT)
    export_root = Path(config["input_root"])
    for family in config["families"]:
        admit_routes(routes, family)
        inputs.update({family + "/" + k: v for k, v in family_inputs(ROOT, routes / "test", routes / "val", family).items()})
        for method in ("sequential_fnn", "reaction_gcnn"):
            verify_export(export_root, ROOT, routes / "test", routes / "val", family, method)
            directory = export_root / method / family
            inputs.update({"export/" + method + "/" + family + "/" + str(p.relative_to(directory)): p
                           for p in directory.rglob("*") if p.is_file()})
    if fingerprint(binding["config"], inputs) != binding:
        raise ValueError("Baseline source/data/export binding changed")
    for folder in [study / "deterministic_b1"] + [study / f"seed_{s}" for s in config["seeds"]]:
        if not reuse_completed(folder, binding):
            raise ValueError("Missing certified baseline run: " + str(folder))
    records = []
    with tempfile.TemporaryDirectory(prefix="baseline_replay_", dir=ROOT / "outputs") as temporary:
        scratch = Path(temporary)
        for family in config["families"]:
            jobs = [("product_naive_bayes", 0, study / "deterministic_b1/product_naive_bayes" / family)]
            for seed in config["seeds"]:
                jobs += [("product_gnn", seed, study / f"seed_{seed}/product_gnn/product_gnn" / family)]
                jobs += [(m, seed, study / f"seed_{seed}/external_compact" / m / family)
                         for m in ("sequential_fnn", "reaction_gcnn")]
            for method, seed, directory in jobs:
                if method in ("product_naive_bayes", "product_gnn"):
                    metrics, weight = audit_direct(directory, family, method, routes, scratch)
                else:
                    metrics, weight = audit_external(directory, family, routes)
                records.append({"method": method, "family": family, "seed": seed,
                    "validation_fusion_replayed": True, "selected_route_weight": weight,
                    "test_metrics_replayed": True, "metrics": metrics})
                print("Verified", family, method, seed, flush=True)
    if fingerprint(binding["config"], inputs) != binding:
        raise ValueError("Inputs changed during baseline replay")
    write_json({"pass": True, "complete": True, "families": config["families"],
        "scope": "configured families only; deterministic B1 once and neural baselines across configured seeds",
        "records": records, "source_sha256": sha(Path(__file__)),
        "experiment_binding_sha256": sha(study / "experiment_binding.json")}, study / "independent_replay.json")


if __name__ == "__main__":
    main()
