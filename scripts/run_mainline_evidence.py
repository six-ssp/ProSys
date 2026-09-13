#!/usr/bin/env python3
"""Reconstruct frozen mainline runs with exact controls and retained evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "Experiment/mainline_evidence_completion_20260913"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def child(family, seed, scratch, destination):
    import numpy as np
    import pandas as pd
    import torch
    from prosys_shared.evidence import (identity_hash, ranked_frame,
        require_same_candidates, require_same_temperature_support, temperature_support)
    from prosys_shared.mainline import (evaluate_scored_frame_with_manifest,
        split_file_for_family, load_split_rows, load_gold_condition_index,
        canonicalize_smiles, canonicalize_reaction_side, normalize_condition_labels)
    from scripts import run_stage23_mainline_non_oracle as pipeline
    from stage3_XGBoost.xgb_reranker import score_table_with_xgb, train_xgb_temperature_regressor

    destination.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), ROOT / "scripts/run_stage23_mainline_non_oracle.py"]
    for module in ("prosys_shared", "stage2_KNN", "stage2_ReaFNN", "stage3_XGBoost"):
        source_files.extend((ROOT / module).rglob("*.py"))
    write_json({"pid": os.getpid(), "cpu_threads": torch.get_num_threads(),
        "thread_environment": {k: os.environ.get(k) for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "device": torch.cuda.get_device_name(0),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(source_files))}},
        destination / "runtime_manifest.json")
    times = {}

    def timed(name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                times.setdefault(name, []).append(time.perf_counter() - started)
                write_json(times, destination / "stage_timings_live.json")
        return measured

    for name in ("_ensure_knn_tables", "_ensure_gnn_augmented_tables",
                 "train_xgb_ranker_and_temperature", "train_xgb_temperature_regressor",
                 "score_table_with_xgb"):
        setattr(pipeline, name, timed(name, getattr(pipeline, name)))
    started = time.perf_counter()
    sys.argv = [str(Path(__file__)), "--repo_root", str(ROOT), "--families", family,
        "--seed", str(seed), "--output_root", str(scratch),
        "--reafnn_device", "cuda:0", "--gnn_device", "cuda:0"]
    pipeline.main()
    full = read_json(scratch / family / "knn_xgb/non_oracle/result.json")
    full_table = Path(full["candidate_table"])
    graph_root = full_table.parent.parent
    stage2_root = graph_root.parent
    original = pd.read_csv(full["scored_test_file"])
    rank_dir = Path(full["model"]["model_file"]).parent
    tabular_tables = stage2_root / "training_tables"
    assert all((tabular_tables / (s + ".csv")).is_file() for s in ("train", "val", "test"))
    cache = read_json(ROOT / "outputs/stage1_routes" / family / "route_cache.json")
    expected = [int(r["sample_index"]) for r in cache["reactions"]]

    # Both controls consume the exact same retained rows as this full run.
    no_ltr = pipeline._score_table_with_stage2_heuristic(full_table)
    pool_hash = require_same_candidates(original, no_ltr)
    no_ltr_metrics = evaluate_scored_frame_with_manifest(no_ltr,
        expected_sample_indices=expected, score_column="stage2_prior_score")
    controls_start = time.perf_counter()
    no_graph_model = train_xgb_temperature_regressor(
        tabular_tables / "train.csv", tabular_tables / "val.csv",
        scratch / family / "no_graph_temperature", random_state=seed)
    assert len(no_graph_model["feature_columns"]) == 52
    no_graph = score_table_with_xgb(full_table, model_file=full["model"]["model_file"],
        metadata_file=full["model"]["metadata_file"],
        temperature_model_file=no_graph_model["model_file"],
        temperature_metadata_file=no_graph_model["metadata_file"])
    require_same_candidates(original, no_graph)
    a, b = ranked_frame(original), ranked_frame(no_graph)
    assert identity_hash(a, ordered=True) == identity_hash(b, ordered=True)
    np.testing.assert_allclose(a["xgb_score"], b["xgb_score"], rtol=1e-6, atol=1e-7)
    support = require_same_temperature_support(original, no_graph,
        left_column="xgb_temperature_pred", right_column="xgb_temperature_pred")
    no_graph_metrics = evaluate_scored_frame_with_manifest(no_graph,
        expected_sample_indices=expected, score_column="xgb_score", temperature_column="xgb_temperature_pred")
    controls_seconds = time.perf_counter() - controls_start

    # Assert graph augmentation preserved all shared candidate feature values.
    table_checks = {}
    for split in ("train", "val", "test"):
        base = pd.read_csv(tabular_tables / (split + ".csv"))
        augmented = pd.read_csv(graph_root / "training_tables" / (split + ".csv"))
        require_same_candidates(base, augmented)
        common = list(base.columns)
        pd.testing.assert_frame_equal(base[common], augmented[common], check_dtype=False,
                                      check_exact=False, rtol=1e-6, atol=1e-7)
        table_checks[split] = {"candidate_identity_sha256": identity_hash(base),
                              "rows": len(base), "shared_feature_values_equal": True}
    audit = {"candidate_identity_sha256": pool_hash, "same_pool_no_ltr": True,
        "same_ranked_identities_no_graph": True, "temperature_support": support,
        "training_validation_pairing": table_checks}
    write_json(audit, destination / "exact_control_audit.json")
    write_json({"no_ltr": no_ltr_metrics, "no_rgnn_temperature": no_graph_metrics},
               destination / "control_metrics.json")

    # Retain tabular intermediates, not repeated per-context graph embeddings.
    kept = [c for c in original if not c.startswith("route_gnn_feat_")]
    original[kept].to_csv(destination / "full_candidates.csv.gz", index=False, compression="gzip")
    ids = ["sample_index", "reaction_id", "product", "reactants", "reagent_norm", "solvent_norm"]
    no_ltr[ids + ["stage2_prior_score"]].to_csv(destination / "no_ltr_scores.csv.gz", index=False, compression="gzip")
    no_graph[ids + ["xgb_score", "xgb_temperature_pred"]].to_csv(destination / "no_graph_predictions.csv.gz", index=False, compression="gzip")
    temperature_support(original, "xgb_temperature_pred").to_csv(
        destination / "temperature_support.csv.gz", index=False, compression="gzip")
    embeddings = original[["product", "reactants"] +
        [c for c in original if c.startswith("route_gnn_feat_")]].drop_duplicates(["product", "reactants"])
    embeddings.to_csv(destination / "route_embeddings.csv.gz", index=False, compression="gzip")

    train_rows = load_split_rows(split_file_for_family(ROOT, family, "train"))
    train_products = {canonicalize_smiles(r["product"]) for r in train_rows}
    train_contexts = {(normalize_condition_labels(r["reagent_norm"]), normalize_condition_labels(r["solvent_norm"])) for r in train_rows}
    gold = load_gold_condition_index(split_file_for_family(ROOT, family, "test"))
    groups = {int(i): g for i, g in a.groupby("sample_index", sort=False)}
    query_rows, cases = [], []
    categories_seen = set()
    for query in cache["reactions"]:
        index = int(query["sample_index"])
        product = canonicalize_smiles(query["product"])
        bucket = gold[(str(query["reaction_id"]), product)]
        group = groups.get(index, a.iloc[:0])
        hits = group[group["label"] > 0.5]
        first = int(hits["final_rank"].min()) if len(hits) else None
        route_hit = any(canonicalize_reaction_side(r["reactants"]) in bucket.route_keys for r in query["routes"])
        category = "route_miss" if not route_hit else "pool_miss" if first is None else "ranking_miss" if first > 10 else "hit"
        if first is not None:
            assert route_hit, "Exact candidate without a matching Stage 1 route"
        row = {"family": family, "seed": seed, "sample_index": index,
            "reaction_id": query["reaction_id"], "product": product,
            "seen_product": product in train_products,
            "context_in_train_library": bool(bucket.context_keys & train_contexts),
            "route_hit": route_hit, "candidate_hit": first is not None,
            "first_exact_rank": first, "failure_category": category,
            **{f"sys{k}": first is not None and first <= k for k in (1, 3, 5, 10)}}
        query_rows.append(row)
        stratum = category if category != "hit" else "first_hit_" + (
            "1" if first == 1 else "2_3" if first <= 3 else "4_5" if first <= 5 else "6_10")
        if stratum not in categories_seen:
            categories_seen.add(stratum)
            cases.append({"stratum": stratum, "query": row, "stage1": query,
                "gold_exact_systems": sorted(bucket.exact_keys),
                "top10": json.loads(group.head(10)[kept + ["final_rank"]].to_json(orient="records")),
                "first_exact": None if first is None else json.loads(hits.head(1)[kept + ["final_rank"]].to_json(orient="records"))[0]})
    queries = pd.DataFrame(query_rows)
    assert len(queries) == len(expected)
    for k in (1, 3, 5, 10):
        assert abs(queries[f"sys{k}"].mean() - full["metrics"][f"system_top{k}_all"]) < 1e-10
    queries.to_csv(destination / "queries.csv", index=False)
    write_json(cases, destination / "cases.json")

    bundle = destination / "bundle"
    for source, name in ((stage2_root / "reafnn", "reafnn"), (graph_root / "model", "rgnn"),
                         (rank_dir, "ranker"), (Path(full["model"]["temperature_model_file"]).parent, "temperature")):
        shutil.copytree(source, bundle / name, dirs_exist_ok=True)
    provenance = {"family": family, "seed": seed,
        "splits": {s: sha(split_file_for_family(ROOT, family, s)) for s in ("train", "val", "test")},
        "route_test": sha(ROOT / "outputs/stage1_routes" / family / "route_cache.json"),
        "route_validation": sha(ROOT / "outputs/stage1_routes_validation" / family / "route_cache.json"),
        "models": {str(p.relative_to(bundle)): sha(p) for p in bundle.rglob("*") if p.is_file()}}
    write_json(provenance, destination / "provenance.json")
    write_json(full, destination / "result.json")
    write_json({"wall_seconds": time.perf_counter() - started, "stages": times,
        "controls_seconds": controls_seconds, "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_torch_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_initialized() else 0,
        "scope": "concurrent Stage 2/3 reconstruction; no Stage 1 decoding; timings are not isolated inference latency"},
        destination / "cost.json")
    write_json({"status": "complete", "exact_controls_pass": True}, destination / "completion.json")


def run_job(family, seed):
    destination = STUDY / "compact" / f"seed_{seed}" / family
    if (destination / "completion.json").exists():
        return {"family": family, "seed": seed, "reused": True}
    scratch = ROOT / "outputs/mainline_evidence_scratch_20260913" / f"seed_{seed}" / family
    if scratch.exists():
        raise RuntimeError(f"Incomplete scratch exists; inspect before restarting: {scratch}")
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free on data disk")
    scratch.mkdir(parents=True)
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "TMPDIR": str(scratch), "PYTHONUNBUFFERED": "1"})
    print(f"START full evidence seed={seed} family={family}", flush=True)
    with (destination / "run.log").open("w") as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", "--family", family,
                        "--seed", str(seed)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert (destination / "completion.json").exists()
    shutil.rmtree(scratch)
    print(f"DONE full evidence seed={seed} family={family}", flush=True)
    return {"family": family, "seed": seed, "reused": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--family", default="Beckmann")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot_only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.child:
        return child(args.family, args.seed,
            ROOT / "outputs/mainline_evidence_scratch_20260913" / f"seed_{args.seed}" / args.family,
            STUDY / "compact" / f"seed_{args.seed}" / args.family)
    from prosys_shared.mainline import FAMILY_ORDER
    STUDY.mkdir(parents=True, exist_ok=True)
    write_json({"workers": args.workers, "threads_per_child": 2,
        "pid": os.getpid(), "minimum_free_disk_gib": 8,
        "training_hyperparameters_changed": False,
        "scope": "independent family/seed processes; not distributed model training"},
        STUDY / "scheduler.json")
    tasks = [(family, seed) for seed in (0, 1, 2) for family in FAMILY_ORDER]
    if args.pilot_only:
        tasks = [(args.family, args.seed)]
    run_job(*tasks[0])
    completed = [tasks[0]]
    write_json({"status": "running", "completed": completed, "expected": len(tasks)}, STUDY / "status.json")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, *task): task for task in tasks[1:]}
        for future in as_completed(futures):
            future.result()
            completed.append(futures[future])
            write_json({"status": "running", "completed": completed, "expected": len(tasks)}, STUDY / "status.json")
    write_json({"status": "complete", "completed": completed, "expected": len(tasks)}, STUDY / "status.json")


if __name__ == "__main__":
    main()
