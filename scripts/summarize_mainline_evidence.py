#!/usr/bin/env python3
"""Replay retained rows and report subgroup diagnostics without test selection."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "Experiment/mainline_evidence_completion_20260913"


def read(path):
    return json.loads(path.read_text())


def flat(metrics):
    return {"cover": metrics["pool_coverage"],
        **{f"sys{k}": metrics[f"system_top{k}_all"] for k in (1, 3, 5, 10)},
        "mrr": metrics["system_mrr"], "ndcg10": metrics["system_ndcg10"],
        **{"temp_" + k: metrics["temperature"][k] for k in ("n", "mae", "within_5c", "within_10c", "within_20c")}}


def align_control_rows(reference, control):
    import pandas as pd
    from prosys_shared.evidence import identity_table, require_same_candidates
    require_same_candidates(reference, control)
    left = pd.MultiIndex.from_frame(identity_table(reference, ordered=True))
    right = pd.MultiIndex.from_frame(identity_table(control, ordered=True))
    if not left.is_unique or not right.is_unique:
        raise ValueError("Ambiguous duplicate candidate identity; retain a finer identity before pairing scores")
    positions = right.get_indexer(left)
    assert (positions >= 0).all()
    return control.iloc[positions].reset_index(drop=True)


def main():
    import numpy as np
    import pandas as pd
    from prosys_shared.mainline import FAMILY_ORDER, evaluate_scored_frame_with_manifest
    from prosys_shared.evidence import identity_hash, require_same_candidates, require_same_temperature_support
    from prosys_shared.cache_integrity import file_sha256

    expected = [(family, seed) for seed in (0, 1, 2) for family in FAMILY_ORDER]
    missing = [(f, s) for f, s in expected if not (STUDY / "compact" / f"seed_{s}" / f / "completion.json").exists()]
    if missing:
        raise RuntimeError(f"Complete 18-run report withheld; missing: {missing}")
    metrics, queries, groups, failures, costs, receipts = [], [], [], [], [], []
    identifiers = ["sample_index", "reaction_id", "product", "reactants", "reagent_norm", "solvent_norm"]
    for family, seed in expected:
        directory = STUDY / "compact" / f"seed_{seed}" / family
        result = read(directory / "result.json")
        controls = read(directory / "control_metrics.json")
        audit = read(directory / "exact_control_audit.json")
        provenance = read(directory / "provenance.json")
        historical = read(ROOT / "Experiment/stage23_parallel_post_fusion_multiseed_20260903/compact/prosys" / f"seed_{seed}" / family / "result.json")
        protocol_matches = {key: historical["model"][key] == result["model"][key]
                            for key in ("temperature_protocol", "feature_columns")}
        stage2_fixed = lambda model: {k: v for k, v in model["stage2_protocol"].items()
                                     if k != "reafnn_post_fusion_calibration"}
        protocol_matches["stage2_fixed_protocol"] = stage2_fixed(historical["model"]) == stage2_fixed(result["model"])
        assert all(protocol_matches.values()), protocol_matches
        for name, sha in provenance["models"].items():
            assert file_sha256(directory / "bundle" / name) == sha
        frame = pd.read_csv(directory / "full_candidates.csv.gz", float_precision="round_trip")
        assert identity_hash(frame) == audit["candidate_identity_sha256"]
        q = pd.read_csv(directory / "queries.csv")
        cache_path = ROOT / "outputs/stage1_routes" / family / "route_cache.json"
        assert file_sha256(cache_path) == provenance["route_test"]
        manifest = [int(r["sample_index"]) for r in read(cache_path)["reactions"]]
        assert sorted(q.sample_index.tolist()) == sorted(manifest)
        replay = evaluate_scored_frame_with_manifest(frame, expected_sample_indices=manifest,
            score_column="xgb_score", temperature_column="xgb_temperature_pred")
        for k, value in flat(replay).items():
            np.testing.assert_allclose(value, flat(result["metrics"])[k], rtol=1e-6, atol=1e-6)
        no_ltr = align_control_rows(frame, pd.read_csv(directory / "no_ltr_scores.csv.gz", float_precision="round_trip"))
        alternate = frame.copy()
        alternate["stage2_prior_score"] = no_ltr.stage2_prior_score.to_numpy()
        replay_no_ltr = evaluate_scored_frame_with_manifest(alternate, expected_sample_indices=manifest,
            score_column="stage2_prior_score")
        for k in ("cover", "sys1", "sys3", "sys5", "sys10", "mrr", "ndcg10"):
            np.testing.assert_allclose(flat(replay_no_ltr)[k], flat(controls["no_ltr"])[k], atol=1e-10)
        no_graph = align_control_rows(frame, pd.read_csv(directory / "no_graph_predictions.csv.gz", float_precision="round_trip"))
        np.testing.assert_allclose(frame.xgb_score, no_graph.xgb_score, rtol=1e-6, atol=1e-7)
        alternate = frame.copy()
        alternate["xgb_temperature_pred"] = no_graph.xgb_temperature_pred.to_numpy()
        support = require_same_temperature_support(frame, alternate,
            left_column="xgb_temperature_pred", right_column="xgb_temperature_pred")
        assert support == audit["temperature_support"]
        replay_no_graph = evaluate_scored_frame_with_manifest(alternate, expected_sample_indices=manifest,
            score_column="xgb_score", temperature_column="xgb_temperature_pred")
        for k, value in flat(replay_no_graph).items():
            np.testing.assert_allclose(value, flat(controls["no_rgnn_temperature"])[k], rtol=1e-6, atol=1e-6)
        for arm, metric in (("full", replay), ("no_ltr", replay_no_ltr), ("no_rgnn_temperature", replay_no_graph)):
            metrics.append({"family": family, "seed": seed, "arm": arm, **flat(metric)})
        for column in ("seen_product", "context_in_train_library"):
            for flag in (False, True):
                subgroup = q[q[column] == flag]
                groups.append({"family": family, "seed": seed, "stratifier": column, "value": flag,
                    "n": len(subgroup), "route10": subgroup.route_hit.mean(), "cover": subgroup.candidate_hit.mean(),
                    **{f"sys{k}": subgroup[f"sys{k}"].mean() for k in (1, 3, 5, 10)}})
        assert q.failure_category.isin(["route_miss", "pool_miss", "ranking_miss", "hit"]).all()
        assert q.loc[q.failure_category == "hit", "sys10"].all()
        assert not q.loc[q.failure_category != "hit", "sys10"].any()
        counts = q.failure_category.value_counts()
        failures.append({"family": family, "seed": seed, "n": len(q),
            **{c: int(counts.get(c, 0)) for c in ("route_miss", "pool_miss", "ranking_miss", "hit")}})
        queries.append(q)
        cost = read(directory / "cost.json")
        costs.append({"family": family, "seed": seed,
            **{k: v for k, v in cost.items() if k != "stages"},
            **{k: sum(v) for k, v in cost["stages"].items()}})
        receipts.append({"family": family, "seed": seed, "candidate_sha256": identity_hash(frame),
            "full_file_sha256": file_sha256(directory / "full_candidates.csv.gz"),
            "metric_replay": True, "same_pool_no_ltr": True, "same_temperature_support": support,
            "promoted_protocol_matches": protocol_matches,
            "fitted_stage2_calibration_equal": historical["model"]["stage2_protocol"].get("reafnn_post_fusion_calibration") == result["model"]["stage2_protocol"].get("reafnn_post_fusion_calibration"),
            "runtime_source_manifest_available": (directory / "runtime_manifest.json").exists()})
    all_queries = pd.concat(queries, ignore_index=True)
    assert all_queries.groupby("seed").size().tolist() == [3860] * 3
    m = pd.DataFrame(metrics)
    m.to_csv(STUDY / "per_family_seed_controls.csv", index=False)
    macro = m.groupby(["arm", "seed"]).mean(numeric_only=True)
    macro["temp_n"] = m.groupby(["arm", "seed"]).temp_n.sum()
    macro = macro.reset_index()
    macro.to_csv(STUDY / "macro_controls_by_seed.csv", index=False)
    macro.groupby("arm").agg({k: ["mean", "std"] for k in flat(replay) if k != "temp_n"}).to_csv(STUDY / "macro_controls_mean_std.csv")
    subgroup_frame = pd.DataFrame(groups)
    subgroup_frame.to_csv(STUDY / "subgroups_per_family_seed.csv", index=False)
    subgroup_keys = ["seed", "stratifier", "value"]
    subgroup_macro = subgroup_frame.groupby(subgroup_keys).mean(numeric_only=True)
    subgroup_macro["n"] = subgroup_frame.groupby(subgroup_keys).n.sum()
    subgroup_macro["nonempty_families"] = subgroup_frame[subgroup_frame.n > 0].groupby(subgroup_keys).size()
    subgroup_macro.to_csv(STUDY / "subgroups_macro_by_seed.csv")
    pd.DataFrame(failures).to_csv(STUDY / "failures_per_family_seed.csv", index=False)
    pd.DataFrame(costs).to_csv(STUDY / "cost_per_family_seed.csv", index=False)
    all_queries.to_csv(STUDY / "queries_all.csv.gz", index=False, compression="gzip")
    old = pd.read_csv(ROOT / "Experiment/stage23_parallel_post_fusion_multiseed_20260903/per_family_seed_metrics.csv")
    paired = m[m.arm == "full"].merge(old, on=["family", "seed"], suffixes=("_reconstructed", "_promoted"), validate="one_to_one")
    differences = paired[["family", "seed"]].copy()
    for key in ("cover", "sys1", "sys3", "sys5", "sys10", "mrr", "ndcg10", "temp_mae", "temp_within_5c", "temp_within_10c", "temp_within_20c"):
        differences[key + "_difference"] = paired[key + "_reconstructed"] - paired[key + "_promoted"]
    differences.to_csv(STUDY / "reconstructed_minus_promoted.csv", index=False)
    lines = ["# Mainline Evidence Reconstruction", "",
        "All 18 family/seed jobs completed. Retained candidate files were replayed independently; "
        "no-LTR uses identical candidate identities, and no-R-GNN uses identical ranked identities, "
        "temperature support and reference temperatures. Graph augmentation preserves the shared "
        "training/validation feature values. These controls pair to the reconstruction, not deleted historical rows.", "",
        "## Reconstructed Controls", "", "Equal-family macro mean +/- sample SD over fixed-split downstream seeds 0/1/2. Rates in percent.", "",
        "| Arm | Sys@1 | Sys@3 | Sys@5 | Sys@10 | Temp MAE (C) | Within 10 C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ("full", "no_ltr", "no_rgnn_temperature"):
        values = macro[macro.arm == arm]
        cells = []
        for key in ("sys1", "sys3", "sys5", "sys10", "temp_mae", "temp_within_10c"):
            factor = 1 if key == "temp_mae" else 100
            cells.append("NA" if values[key].isna().all() else f"{values[key].mean()*factor:.2f} +/- {values[key].std()*factor:.2f}")
        lines.append("| " + arm + " | " + " | ".join(cells) + " |")
    sys_delta = differences[[c for c in differences if c.startswith("sys")]].abs().to_numpy().max()
    temp_delta = differences.temp_mae_difference.abs().max()
    lines += ["", f"Maximum absolute per-family/seed Sys@k difference from promoted results: {sys_delta*100:.8f} pp. "
        f"Maximum absolute temperature-MAE difference: {temp_delta:.6f} C. "
        "See reconstructed_minus_promoted.csv for every difference. Promoted figures are not silently overwritten.", "",
        "## Seen/Unseen Product Diagnostics", "",
        "Seen means canonical product occurs in this family's condition-training split. The existing split groups "
        "canonical reactions, not products. This is a descriptive subgroup analysis, not an independent external benchmark. "
        "Denominators are query identities, not unique molecules; the same fixed identities recur across seeds. "
        "Unseen here means absent from the condition-training memory only; Stage 1 pretraining or expert-training data may still contain these products.", "",
        "| Family | Seen n | Seen Sys@10 | Unseen n | Unseen Sys@10 | Train-context available n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for family in FAMILY_ORDER:
        q = all_queries[all_queries.family == family]
        cells = []
        for flag in (True, False):
            part = q[q.seen_product == flag]
            by_seed = part.groupby("seed").sys10.mean() * 100
            cells.extend([str(len(part) // 3), "NA" if part.empty else f"{by_seed.mean():.2f} +/- {by_seed.std():.2f}"])
        lines.append("| " + family + " | " + " | ".join(cells) + f" | {int(q.context_in_train_library.sum())//3} |")
    lines += ["", "Training-context availability asks whether at least one gold reagent/solvent pair belongs to the "
        "train-only historical library; it is an evaluation annotation, not a model input. Empty subgroups report NA.", "",
        "## Exhaustive Failure Accounting", "", "Seed-0 counts below; all seeds are in failures_per_family_seed.csv. "
        "Route miss: no predicted reference-matching route; pool miss: route retained but no exact system; "
        "ranking miss: first exact system below rank 10; hit: first exact system within rank 10. "
        "No-slate queries remain in the denominator.", "",
        "| Family | n | Route miss | Pool miss | Ranking miss | Hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in failures:
        if r["seed"] == 0:
            lines.append("| " + r["family"] + " | " + " | ".join(str(r[k]) for k in ("n", "route_miss", "pool_miss", "ranking_miss", "hit")) + " |")
    lines += ["", "## Resource Scope", "",
        "cost_per_family_seed.csv records concurrent reconstruction wall time, nested stage times and per-process peak "
        "RSS/PyTorch allocated GPU memory. Stage 1 decoding is excluded; stage intervals include table work or model loading "
        "as their names indicate. Concurrent wall times are not isolated single-query latency or additive total GPU-hours. "
        "The product-query CLI records cold-load inference separately. No new hyperparameter selection used these test results."]
    lines += ["", "Source provenance note: the initial Beckmann seed-0 pilot predates runtime/source-manifest "
        "instrumentation; its model/data hashes and exact-control audit are retained, but its missing runtime "
        "source manifest is not fabricated retrospectively. The other jobs retain that manifest. Configuration "
        "and feature-schema comparisons against promoted compact records pass for every family/seed.", "",
        "Known implementation boundary: see CANONICALIZATION_NOTE.md for one training-only cross-dot ring-closure "
        "record. The independent whole-molecule split scan found no cross-split reaction overlap."]
    (STUDY / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (STUDY / "replay_audit.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print("PASS: 18 retained runs replayed; exact controls, diagnostics, differences and cost reports written")


if __name__ == "__main__":
    main()
