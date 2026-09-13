#!/usr/bin/env python3
"""Requirement-level audit of the completed six-item evidence plan."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "Experiment/mainline_evidence_completion_20260913"


def read(path):
    return json.loads(path.read_text())


def main():
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from prosys_shared.mainline import (FAMILY_ORDER, split_file_for_family, load_split_rows,
        load_gold_condition_index, canonicalize_smiles, normalize_condition_labels)
    from prosys_shared.condition_modeling import route_feature_vector
    from prosys_shared.cache_integrity import file_sha256
    assert read(STUDY / "finalization.json")["complete"]
    pairs = {(f, s) for f in FAMILY_ORDER for s in (0, 1, 2)}
    rows = read(STUDY / "replay_audit.json")
    assert {(r["family"], r["seed"]) for r in rows} == pairs and len(rows) == 18
    assert all(r["metric_replay"] and r["same_pool_no_ltr"] for r in rows)
    neural = read(ROOT / "Experiment/stage2_reafnn_only_multiseed_20260913/independent_evidence_audit.json")
    assert neural["complete"] and neural["completed_records_audited"] == 18
    assert {(r["family"], r["seed"]) for r in neural["rows"]} == pairs
    q = pd.read_csv(STUDY / "queries_all.csv.gz", dtype={"reaction_id": str})
    assert q.groupby("seed").size().tolist() == [3860, 3860, 3860]
    assert not q.duplicated(["family", "seed", "sample_index"]).any()
    for k in (1, 3, 5, 10):
        np.testing.assert_array_equal(q[f"sys{k}"], q.first_exact_rank <= k)
    classification = np.where(~q.route_hit, "route_miss", np.where(q.first_exact_rank.isna(), "pool_miss",
        np.where(q.first_exact_rank > 10, "ranking_miss", "hit")))
    np.testing.assert_array_equal(q.failure_category, classification)
    for family in FAMILY_ORDER:
        train = load_split_rows(split_file_for_family(ROOT, family, "train"))
        products = {canonicalize_smiles(r["product"]) for r in train}
        contexts = {(normalize_condition_labels(r["reagent_norm"]), normalize_condition_labels(r["solvent_norm"])) for r in train}
        gold = load_gold_condition_index(split_file_for_family(ROOT, family, "test"))
        part = q[q.family == family]
        np.testing.assert_array_equal(part.seen_product, part["product"].isin(products))
        context_available = [bool(gold[(str(r["reaction_id"]), canonicalize_smiles(r["product"]))].context_keys & contexts)
                             for _, r in part.iterrows()]
        np.testing.assert_array_equal(part.context_in_train_library, context_available)
    index = read(STUDY / "examples/index.json")
    assert len(index) == 18
    assert pd.DataFrame(index).groupby("family").size().to_dict() == {f: 3 for f in FAMILY_ORDER}
    first_ranks = set()
    for entry in index:
        with gzip.open(ROOT / entry["file"], "rt") as handle:
            case = json.load(handle)
        selected = case["selected"]
        assert Chem.MolFromSmiles(selected["product"]) is not None
        assert selected["product"] == case["stage1"]["product"]
        candidates = pd.DataFrame(case["all_candidates"])
        positive = candidates[candidates.label > 0.5]
        first = None if positive.empty else int(positive.final_rank.min())
        assert first == case["first_exact_rank"] == entry["first_exact_rank"]
        if selected["label"] > 0.5:
            assert int(selected["final_rank"]) == first
        first_ranks.add(first)
        feature = case["reafnn_input"]
        sparse = np.zeros(feature["dimension"], dtype=np.float32)
        sparse[feature["raw_nonzero_indices"]] = feature["raw_nonzero_values"]
        correct = route_feature_vector(selected["reactants"], selected["product"], fpsize=4096, radius=2)
        np.testing.assert_array_equal(sparse, correct)
        assert np.isfinite(feature["scaled_dense"]).all()
        assert any(sparse[:4096])
        assert len(case["knn_neighbors_replayed"]) <= 64
        assert len(case["branch_union_replayed_before_weight"]) <= 128
        w = selected["stage2_post_fusion_weight"]
        np.testing.assert_allclose(selected["stage2_post_fusion_score"],
            w * selected["stage2_knn_score"] + (1-w) * selected["stage2_reafnn_score"], atol=1e-7)
        for key in ("reagent_probabilities", "solvent_probabilities"):
            assert case[key] and all(0 <= v <= 1 for v in case[key].values())
        assert len([k for k in selected if k.startswith("route_gnn_feat_")]) == 128
    assert {1, 3, 5, 10}.issubset(first_ranks), first_ranks
    assert "<bound method" not in (ROOT / "example.md").read_text()
    cold = pd.read_csv(STUDY / "cold_inference_cost.csv")
    assert set(cold.family) == set(FAMILY_ORDER) and len(cold) == 6
    assert (cold.external_wall_seconds_including_imports > 0).all()
    assert (cold.gpu_device_peak_observed_mib >= cold.gpu_device_baseline_mib).all()
    for family in FAMILY_ORDER:
        folder = STUDY / "cold_inference" / family
        prediction = read(folder / "prediction.json")
        assert prediction["route_source"]["mode"] == "fresh_editretro_decoding"
        table = pd.read_csv(folder / "ranked_candidates.csv.gz", nrows=2)
        assert not {"label", "temperature_gold", "yield_gold", "rank_relevance", "route_match", "context_match"} & set(table)
    cache = read(STUDY / "inference_cache_audit.json")
    assert all(r.get("model_bytes_and_mtimes_unchanged", True) for r in cache["cache_entrypoint"])
    assert {r["command"] for r in cache["cache_entrypoint"]} >= {"identical_cache_reuse", "changed_seed_rejected"}
    tests = ["ablation.test_control_alignment", "ablation.test_verified_entrypoint", "ablation.test_cache_integrity",
        "ablation.test_reproduction_entrypoint", "ablation.test_product_inference", "ablation.test_evidence", "ablation.test_reafnn_only_experiment"]
    run = subprocess.run([sys.executable, "-m", "unittest", *tests], cwd=ROOT, text=True, capture_output=True)
    (STUDY / "completion_tests.log").write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stderr
    files = ["PLAN.md", "SUMMARY.md", "COMPLETION_REPORT.md", "replay_audit.json", "per_family_seed_controls.csv",
        "subgroups_per_family_seed.csv", "failures_per_family_seed.csv", "cold_inference_cost.csv",
        "cost_per_family_seed.csv", "inference_cache_audit.json", "canonical_fragment_audit.json", "examples/index.json"]
    record = {"six_item_plan_verified": True, "stage2_control_records": 18, "identity_control_records": 18,
        "fixed_queries_per_seed": 3860, "case_count": 18, "cold_inference_families": 6,
        "test_process_exit_code": run.returncode, "artifact_sha256": {f: file_sha256(STUDY / f) for f in files},
        "scope_limits": ["Not an independent external dataset", "Stage 1 is fixed, not retrained in this study",
            "One known training-only dot-splitting edge case remains in frozen models",
            "Pilot runtime source manifest was not captured", "Cold latency has one query per family, not a throughput estimate"]}
    (STUDY / "completion_gate.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
