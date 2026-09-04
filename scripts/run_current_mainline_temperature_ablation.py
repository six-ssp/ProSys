#!/usr/bin/env python3
"""Run the three-seed matched no-R-GNN temperature ablation for ProSys.

The existing parallel mainline compact records provide the R-GNN arm. This
runner recomputes the same Stage 1/2 and tabular XGB-LTR pipeline per
family/seed, then substitutes only the temperature regressor's feature space:
the no-R-GNN control uses the 52-dimensional tabular candidate representation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prosys_shared.mainline import FAMILY_ORDER, display_family_name, parse_families_arg


FULL_ARM = "rgnn_temperature"
ABLATION_ARM = "no_rgnn_temperature"
TEMPERATURE_METRICS = (
    ("temperature_mae", "Temperature MAE (C)"),
    ("temperature_within_5c", "Within +/-5 C"),
    ("temperature_within_10c", "Within +/-10 C"),
    ("temperature_within_20c", "Within +/-20 C"),
)
RANKING_MATCH_FIELDS = (
    "pool_coverage",
    "pool_route_coverage",
    "pool_context_coverage",
    "candidate_slates",
    "missing_candidate_slates",
    "system_mrr",
    "system_ndcg10",
    "system_top1_all",
    "system_top3_all",
    "system_top5_all",
    "system_top10_all",
)


def _parse_values(value: str, *, label: str) -> list[str]:
    values = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    return values


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in _parse_values(value, label="--seeds")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    return seeds


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return FAMILY_ORDER.index(family), family
    except ValueError:
        return len(FAMILY_ORDER), family


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = json.loads(json.dumps(result))
    compact["candidate_table"] = "pruned_after_compaction"
    compact["scored_test_file"] = "pruned_after_compaction"
    compact["raw_artifacts_pruned"] = True
    compact["retention_note"] = (
        "Raw candidate tables, scored tables, and binary checkpoints were removed "
        "after compact result and metadata retention."
    )
    return compact


def _as_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} is missing or non-numeric")
    return float(value)


def _assert_equal(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        if abs(float(observed) - float(expected)) <= 1e-12:
            return
    elif observed == expected:
        return
    raise ValueError(f"{label}: observed={observed!r}, expected={expected!r}")


def _load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing result record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_full_reference(result: dict[str, Any], *, family: str, seed: int) -> None:
    if result.get("family") != family or int(result.get("seed")) != seed:
        raise ValueError(f"full reference {family}/seed_{seed}: wrong identity")
    model = result.get("model") or {}
    stage2 = model.get("stage2_protocol") or {}
    ranker = model.get("ranking_protocol") or {}
    temperature = model.get("temperature_protocol") or {}
    if result.get("baseline") != "knn_xgb_reaction_gnn_temperature":
        raise ValueError(f"full reference {family}/seed_{seed}: unexpected baseline")
    if stage2.get("architecture") != "knn_reafnn":
        raise ValueError(f"full reference {family}/seed_{seed}: unexpected Stage 2 architecture")
    if stage2.get("reafnn_candidate_policy") != "independent_knn_reafnn_post_fusion":
        raise ValueError(f"full reference {family}/seed_{seed}: unexpected Stage 2 policy")
    if ranker.get("architecture") != "xgb_ranker" or int(ranker.get("feature_count") or 0) != 52:
        raise ValueError(f"full reference {family}/seed_{seed}: expected 52-feature XGB-LTR")
    if temperature.get("architecture") != "reaction_gnn_augmented_xgboost_regressor":
        raise ValueError(f"full reference {family}/seed_{seed}: missing R-GNN temperature head")
    gnn_config = temperature.get("reaction_gnn_config") or {}
    if int(gnn_config.get("embedding_dim") or 0) != 128:
        raise ValueError(f"full reference {family}/seed_{seed}: expected 128D R-GNN embedding")
    temp = (result.get("metrics") or {}).get("temperature") or {}
    if int(temp.get("n") or 0) <= 0:
        raise ValueError(f"full reference {family}/seed_{seed}: no valid temperature support")


def _validate_no_rgnn_result(result: dict[str, Any], *, family: str, seed: int) -> None:
    if result.get("family") != family or int(result.get("seed")) != seed:
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: wrong identity")
    model = result.get("model") or {}
    stage2 = model.get("stage2_protocol") or {}
    ranker = model.get("ranking_protocol") or {}
    temperature = model.get("temperature_protocol") or {}
    temperature_model = model.get("temperature") or {}
    feature_columns = temperature_model.get("feature_columns") or []

    if result.get("baseline") != "knn_xgb_tabular_temperature_ablation":
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: unexpected baseline")
    if stage2.get("architecture") != "knn_reafnn":
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: unexpected Stage 2 architecture")
    if stage2.get("knn_retrieval_mode") != "product_morgan":
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: retrieval mode changed")
    if int(stage2.get("knn_top_k") or 0) != 64:
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: KNN K changed")
    if int(stage2.get("prefilter_contexts") or 0) != 64:
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: wide pool changed")
    if int(stage2.get("max_contexts") or 0) != 20:
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: candidate cap changed")
    if stage2.get("training_candidate_table_mode") != "reference_split_routes":
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: training table mode changed")
    if stage2.get("training_candidate_route_source") != "reference_split_routes":
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: training route source changed")
    if stage2.get("reafnn_candidate_policy") != "independent_knn_reafnn_post_fusion":
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: Stage 2 policy changed")
    if ranker.get("architecture") != "xgb_ranker" or int(ranker.get("feature_count") or 0) != 52:
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: expected unchanged 52-feature XGB-LTR")
    if temperature.get("architecture") != "tabular_xgboost_regressor_without_reaction_gnn":
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: incorrect temperature architecture")
    if temperature.get("always_enabled") is not True or temperature.get("reaction_gnn_enabled") is not False:
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: temperature protocol is not tabular-only")
    if len(feature_columns) != 52:
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: expected 52 tabular temperature features")
    if any(str(column).startswith("route_gnn_feat_") for column in feature_columns):
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: leaked R-GNN feature")
    temp = (result.get("metrics") or {}).get("temperature") or {}
    if int(temp.get("n") or 0) <= 0:
        raise ValueError(f"no-R-GNN {family}/seed_{seed}: no valid temperature support")


def _validate_matched_pair(
    no_rgnn: dict[str, Any],
    full: dict[str, Any],
    *,
    family: str,
    seed: int,
) -> dict[str, Any]:
    _validate_full_reference(full, family=family, seed=seed)
    _validate_no_rgnn_result(no_rgnn, family=family, seed=seed)

    no_model = no_rgnn.get("model") or {}
    full_model = full.get("model") or {}
    _assert_equal(
        no_model.get("stage2_protocol"),
        full_model.get("stage2_protocol"),
        label=f"{family}/seed_{seed}: Stage 2 protocol",
    )
    _assert_equal(
        no_rgnn.get("stage1_route_recall"),
        full.get("stage1_route_recall"),
        label=f"{family}/seed_{seed}: Stage 1 route recall",
    )

    no_metrics = no_rgnn.get("metrics") or {}
    full_metrics = full.get("metrics") or {}
    for field in RANKING_MATCH_FIELDS:
        _assert_equal(
            no_metrics.get(field),
            full_metrics.get(field),
            label=f"{family}/seed_{seed}: {field}",
        )

    no_temp = no_metrics.get("temperature") or {}
    full_temp = full_metrics.get("temperature") or {}
    _assert_equal(
        int(no_temp.get("n") or 0),
        int(full_temp.get("n") or 0),
        label=f"{family}/seed_{seed}: conditional temperature support",
    )
    return {
        "family": family,
        "seed": seed,
        "stage1_route_recall_exact": True,
        "stage2_protocol_exact": True,
        "candidate_pool_exact": True,
        "ranking_metrics_exact": True,
        "temperature_support_exact": True,
        "no_rgnn_temperature_feature_count": len(
            ((no_rgnn.get("model") or {}).get("temperature") or {}).get("feature_columns") or []
        ),
        "no_rgnn_has_route_gnn_features": False,
    }


def _metric_row(result: dict[str, Any], *, arm: str, seed: int) -> dict[str, Any]:
    metrics = result.get("metrics") or {}
    route = result.get("stage1_route_recall") or {}
    temp = metrics.get("temperature") or {}
    return {
        "arm": arm,
        "seed": seed,
        "family": result["family"],
        "n_test_manifest": int(route.get("n") or 0),
        "candidate_slates": int(metrics.get("candidate_slates") or 0),
        "missing_candidate_slates": int(metrics.get("missing_candidate_slates") or 0),
        "candidate_recall": _as_number(metrics.get("pool_coverage"), label="candidate recall"),
        "sys_at_1": _as_number(metrics.get("system_top1_all"), label="Sys@1"),
        "sys_at_3": _as_number(metrics.get("system_top3_all"), label="Sys@3"),
        "sys_at_5": _as_number(metrics.get("system_top5_all"), label="Sys@5"),
        "sys_at_10": _as_number(metrics.get("system_top10_all"), label="Sys@10"),
        "mrr": _as_number(metrics.get("system_mrr"), label="MRR"),
        "ndcg_at_10": _as_number(metrics.get("system_ndcg10"), label="nDCG@10"),
        "temperature_n": int(temp.get("n") or 0),
        "temperature_mae": _as_number(temp.get("mae"), label="temperature MAE"),
        "temperature_within_5c": _as_number(temp.get("within_5c"), label="temperature within 5C"),
        "temperature_within_10c": _as_number(temp.get("within_10c"), label="temperature within 10C"),
        "temperature_within_20c": _as_number(temp.get("within_20c"), label="temperature within 20C"),
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_by_seed(rows: list[dict[str, Any]], *, families: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in (FULL_ARM, ABLATION_ARM):
        for seed in seeds:
            members = [row for row in rows if row["arm"] == arm and int(row["seed"]) == seed]
            if len(members) != len(families):
                raise ValueError(f"{arm}/seed_{seed}: incomplete result set")
            summary = {
                "arm": arm,
                "seed": seed,
                "family_count": len(members),
                "n_test_manifest": sum(int(row["n_test_manifest"]) for row in members),
                "candidate_slates": sum(int(row["candidate_slates"]) for row in members),
                "missing_candidate_slates": sum(int(row["missing_candidate_slates"]) for row in members),
                "temperature_support": sum(int(row["temperature_n"]) for row in members),
            }
            for field in (
                "candidate_recall",
                "sys_at_1",
                "sys_at_3",
                "sys_at_5",
                "sys_at_10",
                "mrr",
                "ndcg_at_10",
                *(metric for metric, _ in TEMPERATURE_METRICS),
            ):
                summary[field] = _mean([float(row[field]) for row in members])
            output.append(summary)
    return output


def _aggregate_mean_std(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    value_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        summary = dict(zip(group_keys, key))
        summary["n_records"] = len(members)
        for field in value_fields:
            values = [float(member[field]) for member in members]
            summary[f"{field}_mean"] = _mean(values)
            summary[f"{field}_std"] = _std(values)
        output.append(summary)
    return output


def _format_rate(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} +/- {100.0 * std:.2f}"


def _format_number(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.2f}"


def _runner_command(
    *,
    python_bin: str,
    repo_root: Path,
    family: str,
    seed: int,
    scratch_root: Path,
    route_root: Path,
    post_fusion_validation_route_root: Path,
    reafnn_device: str,
) -> list[str]:
    return [
        python_bin,
        str(repo_root / "scripts" / "run_stage23_mainline_non_oracle.py"),
        "--repo_root",
        str(repo_root),
        "--families",
        family,
        "--output_root",
        str(scratch_root),
        "--route_root",
        str(route_root),
        "--fpsize",
        "4096",
        "--radius",
        "2",
        "--knn_retrieval_mode",
        "product_morgan",
        "--knn_top_k",
        "64",
        "--prefilter_contexts",
        "64",
        "--max_contexts",
        "20",
        "--reafnn_hidden_dim",
        "512",
        "--reafnn_hidden_layers",
        "2",
        "--reafnn_dropout",
        "0.10",
        "--reafnn_device",
        reafnn_device,
        "--reafnn_enable_independent_post_fusion",
        "--no-reafnn_enable_knn_wide_refinement",
        "--reafnn_independent_contexts",
        "64",
        "--reafnn_post_fusion_weights",
        "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        "--reafnn_post_fusion_validation_route_root",
        str(post_fusion_validation_route_root),
        "--gnn_device",
        "cpu",
        "--temperature_feature_mode",
        "no_rgnn",
        "--ranking_mode",
        "xgb_ltr",
        "--seed",
        str(seed),
    ]


def _compact_family_run(
    *,
    scratch_root: Path,
    compact_root: Path,
    family: str,
    seed: int,
) -> dict[str, Any]:
    result_dir = scratch_root / family / "knn_xgb" / "non_oracle"
    result = _load_result(result_dir / "result.json")
    _validate_no_rgnn_result(result, family=family, seed=seed)

    compact_dir = compact_root / ABLATION_ARM / f"seed_{seed}" / family
    _write_json(_compact_result(result), compact_dir / "result.json")
    candidate_table = Path(str(result["candidate_table"]))
    shared_root = candidate_table.parent.parent
    _copy_if_exists(
        shared_root / "reafnn" / "reafnn_meta.json",
        compact_dir / "metadata" / "reafnn_meta.json",
    )
    _copy_if_exists(
        result_dir / "model" / "xgb_ranker_meta.json",
        compact_dir / "metadata" / "xgb_ranker_meta.json",
    )
    _copy_if_exists(
        result_dir / "tabular_temperature_model" / "xgb_temperature_meta.json",
        compact_dir / "metadata" / "xgb_temperature_meta.json",
    )
    return result


def _read_compact_result(compact_root: Path, *, family: str, seed: int) -> dict[str, Any] | None:
    path = compact_root / ABLATION_ARM / f"seed_{seed}" / family / "result.json"
    if not path.exists():
        return None
    result = _load_result(path)
    _validate_no_rgnn_result(result, family=family, seed=seed)
    return result


def _write_report(
    *,
    output_root: Path,
    rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    families: list[str],
    seeds: list[int],
) -> None:
    rows = sorted(
        rows,
        key=lambda row: (str(row["arm"]), int(row["seed"]), _family_sort_key(str(row["family"]))),
    )
    _write_csv(output_root / "per_family_seed_metrics.csv", rows)

    by_seed = _aggregate_by_seed(rows, families=families, seeds=seeds)
    by_seed.sort(key=lambda row: (str(row["arm"]), int(row["seed"])))
    _write_csv(output_root / "macro_by_seed.csv", by_seed)

    macro_metric_fields = (
        "candidate_recall",
        "sys_at_1",
        "sys_at_3",
        "sys_at_5",
        "sys_at_10",
        "mrr",
        "ndcg_at_10",
        "temperature_support",
        *(metric for metric, _ in TEMPERATURE_METRICS),
    )
    macro = _aggregate_mean_std(
        by_seed,
        group_keys=("arm",),
        value_fields=macro_metric_fields,
    )
    _write_csv(output_root / "macro_mean_std.csv", macro)

    family_metric_fields = (
        "candidate_recall",
        "sys_at_1",
        "sys_at_3",
        "sys_at_5",
        "sys_at_10",
        "mrr",
        "ndcg_at_10",
        "temperature_n",
        *(metric for metric, _ in TEMPERATURE_METRICS),
    )
    family_summary = _aggregate_mean_std(
        rows,
        group_keys=("arm", "family"),
        value_fields=family_metric_fields,
    )
    family_summary.sort(key=lambda row: (str(row["arm"]), _family_sort_key(str(row["family"]))))
    _write_csv(output_root / "per_family_mean_std.csv", family_summary)
    _write_csv(output_root / "audit.csv", audit_rows)

    macro_by_arm = {str(row["arm"]): row for row in macro}
    full_macro = macro_by_arm[FULL_ARM]
    no_macro = macro_by_arm[ABLATION_ARM]
    family_by_key = {(str(row["arm"]), str(row["family"])): row for row in family_summary}

    lines = [
        "# Matched R-GNN Temperature Ablation",
        "",
        "## Scope",
        "",
        "This is a three-seed, six-family temperature-only ablation matched to the maintained parallel mainline. The R-GNN arm is read from the official current mainline compact records. The new no-R-GNN arm reruns the same fixed Stage 1 routes, product-Morgan KNN (radius 2, 4,096 bits, K=64), independent ReaFNN top-64 post-fusion pool, route-local top-20 cap, and 52-feature XGB-LTR.",
        "",
        "The only intentional difference is the temperature XGBoost input: the full arm uses the 52 tabular candidate features plus a 128D Reaction-GNN route embedding, whereas the control uses exactly the 52 tabular features and contains no route_gnn_feat_* column. Temperature never contributes candidates or ranking scores.",
        "",
        "Temperature is evaluated only on the highest-ranked exact full-system match with a valid temperature label. The support and all Stage 1/2/3 ranking metrics are checked to be identical within each family/seed pair, so any reported temperature difference is not attributable to a changed route, candidate pool, or ranked system.",
        "",
        "## Macro Results",
        "",
        "| Arm | Conditional support (mean +/- sd) | MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| R-GNN + XGBoost temperature | {_format_number(full_macro['temperature_support_mean'], full_macro['temperature_support_std'])} | {_format_number(full_macro['temperature_mae_mean'], full_macro['temperature_mae_std'])} | {_format_rate(full_macro['temperature_within_5c_mean'], full_macro['temperature_within_5c_std'])} | {_format_rate(full_macro['temperature_within_10c_mean'], full_macro['temperature_within_10c_std'])} | {_format_rate(full_macro['temperature_within_20c_mean'], full_macro['temperature_within_20c_std'])} |",
        f"| Tabular XGBoost, no R-GNN | {_format_number(no_macro['temperature_support_mean'], no_macro['temperature_support_std'])} | {_format_number(no_macro['temperature_mae_mean'], no_macro['temperature_mae_std'])} | {_format_rate(no_macro['temperature_within_5c_mean'], no_macro['temperature_within_5c_std'])} | {_format_rate(no_macro['temperature_within_10c_mean'], no_macro['temperature_within_10c_std'])} | {_format_rate(no_macro['temperature_within_20c_mean'], no_macro['temperature_within_20c_std'])} |",
        f"| R-GNN gain (full - no R-GNN) | 0.00 | {full_macro['temperature_mae_mean'] - no_macro['temperature_mae_mean']:.2f} | {100.0 * (full_macro['temperature_within_5c_mean'] - no_macro['temperature_within_5c_mean']):+.2f} pp | {100.0 * (full_macro['temperature_within_10c_mean'] - no_macro['temperature_within_10c_mean']):+.2f} pp | {100.0 * (full_macro['temperature_within_20c_mean'] - no_macro['temperature_within_20c_mean']):+.2f} pp |",
        "",
        "A negative MAE difference in the final row favors the R-GNN arm; a positive hit-rate difference favors the R-GNN arm.",
        "",
        "## Per-Family Results",
        "",
        "| Family | R-GNN MAE (C) | No-R-GNN MAE (C) | R-GNN MAE gain (C) | R-GNN +/-10 C | No-R-GNN +/-10 C | R-GNN +/-10 C gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in sorted(families, key=_family_sort_key):
        full = family_by_key[(FULL_ARM, family)]
        no_rgnn = family_by_key[(ABLATION_ARM, family)]
        lines.append(
            f"| {display_family_name(family)} | "
            f"{_format_number(full['temperature_mae_mean'], full['temperature_mae_std'])} | "
            f"{_format_number(no_rgnn['temperature_mae_mean'], no_rgnn['temperature_mae_std'])} | "
            f"{full['temperature_mae_mean'] - no_rgnn['temperature_mae_mean']:.2f} | "
            f"{_format_rate(full['temperature_within_10c_mean'], full['temperature_within_10c_std'])} | "
            f"{_format_rate(no_rgnn['temperature_within_10c_mean'], no_rgnn['temperature_within_10c_std'])} | "
            f"{100.0 * (full['temperature_within_10c_mean'] - no_rgnn['temperature_within_10c_mean']):+.2f} pp |"
        )

    lines.extend(
        [
            "",
            "## Audit Contract",
            "",
            f"- {len(audit_rows)} matched family/seed pairs passed exact Stage 1 route, Stage 2 protocol/pool, XGB-LTR ranking-metric, and conditional-temperature-support checks.",
            "- Every no-R-GNN temperature regressor used exactly 52 tabular features and zero route_gnn_feat_* dimensions.",
            "- Raw candidate tables, scored tables, and binary checkpoints were removed after compact retention to respect disk limits.",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit_lines = [
        "# Temperature-Ablation Audit",
        "",
        "Each row below is a paired current-mainline R-GNN versus rerun no-R-GNN comparison.",
        "",
        "| Family | Seed | Stage 1 | Stage 2 pool | Ranking metrics | Temperature support | No-R-GNN features | Route-GNN features |",
        "| --- | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in sorted(audit_rows, key=lambda row: (int(row["seed"]), _family_sort_key(str(row["family"])))):
        audit_lines.append(
            f"| {display_family_name(str(row['family']))} | {row['seed']} | PASS | PASS | PASS | PASS | "
            f"{row['no_rgnn_temperature_feature_count']} | absent |"
        )
    audit_lines.append("")
    audit_lines.append(
        f"PASS: {len(audit_rows)} compact no-R-GNN records; all paired control contracts hold."
    )
    (output_root / "audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--families", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--route_root", type=Path, default=Path("outputs/stage1_routes"))
    parser.add_argument(
        "--post_fusion_validation_route_root",
        type=Path,
        default=Path("outputs/stage1_routes_validation"),
    )
    parser.add_argument(
        "--mainline_compact_root",
        type=Path,
        default=Path("Experiment/stage23_parallel_post_fusion_multiseed_20260903/compact/prosys"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904"),
    )
    parser.add_argument(
        "--scratch_root",
        type=Path,
        default=Path("/tmp/prosys_stage3_temperature_no_rgnn_ablation_20260904"),
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--reafnn_device", default="cuda:0")
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--keep_scratch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = (repo_root / args.output_root).resolve()
    scratch_root = Path(args.scratch_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    post_fusion_validation_route_root = (
        repo_root / args.post_fusion_validation_route_root
    ).resolve()
    mainline_compact_root = (repo_root / args.mainline_compact_root).resolve()
    families = parse_families_arg(args.families)
    seeds = _parse_seeds(args.seeds)

    if output_root == scratch_root or scratch_root == repo_root:
        parser.error("--scratch_root must be separate from --output_root and repository root")
    if args.cpu_threads < 1:
        parser.error("--cpu_threads must be positive")
    for path, label in (
        (route_root, "--route_root"),
        (post_fusion_validation_route_root, "--post_fusion_validation_route_root"),
        (mainline_compact_root, "--mainline_compact_root"),
    ):
        if not path.is_dir():
            parser.error(f"{label} does not exist: {path}")

    manifest = {
        "purpose": "three-seed matched no-R-GNN temperature ablation",
        "families": families,
        "seeds": seeds,
        "full_reference_arm": FULL_ARM,
        "new_ablation_arm": ABLATION_ARM,
        "mainline_compact_root": str(mainline_compact_root),
        "route_root": str(route_root),
        "post_fusion_validation_route_root": str(post_fusion_validation_route_root),
        "fixed_configuration": {
            "knn_retrieval_mode": "product_morgan",
            "fpsize": 4096,
            "radius": 2,
            "knn_top_k": 64,
            "prefilter_contexts": 64,
            "max_contexts": 20,
            "training_candidate_table_mode": "reference_split_routes",
            "reafnn_hidden_dim": 512,
            "reafnn_hidden_layers": 2,
            "reafnn_dropout": 0.10,
            "reafnn_candidate_policy": "independent_knn_reafnn_post_fusion",
            "reafnn_independent_contexts": 64,
            "post_fusion_validation_source": "predicted_stage1_validation_routes",
            "post_fusion_weight_grid": [round(step / 10.0, 1) for step in range(11)],
            "xgb_ltr_feature_count": 52,
            "full_temperature_feature_space": "52_tabular_plus_128_rgnn",
            "ablation_temperature_feature_space": "52_tabular_only",
        },
        "scratch_retention": "kept" if args.keep_scratch else "removed after compact retention",
    }
    _write_json(manifest, output_root / "run_manifest.json")

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.cpu_threads)
    environment["MKL_NUM_THREADS"] = str(args.cpu_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)

    all_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for family in families:
            reference = _load_result(
                mainline_compact_root / f"seed_{seed}" / family / "result.json"
            )
            _validate_full_reference(reference, family=family, seed=seed)
            all_rows.append(_metric_row(reference, arm=FULL_ARM, seed=seed))

            compact = _read_compact_result(
                output_root / "compact",
                family=family,
                seed=seed,
            )
            if compact is not None and not args.force:
                audit_rows.append(
                    _validate_matched_pair(compact, reference, family=family, seed=seed)
                )
                all_rows.append(_metric_row(compact, arm=ABLATION_ARM, seed=seed))
                print(f"[temperature-ablation] reuse compact seed_{seed}/{family}", flush=True)
                continue

            family_scratch_root = scratch_root / f"seed_{seed}"
            command = _runner_command(
                python_bin=args.python_bin,
                repo_root=repo_root,
                family=family,
                seed=seed,
                scratch_root=family_scratch_root,
                route_root=route_root,
                post_fusion_validation_route_root=post_fusion_validation_route_root,
                reafnn_device=args.reafnn_device,
            )
            print(
                f"[temperature-ablation] run seed_{seed}/{family}: " + " ".join(command),
                flush=True,
            )
            if args.dry_run:
                continue

            subprocess.run(command, check=True, cwd=repo_root, env=environment)
            result = _compact_family_run(
                scratch_root=family_scratch_root,
                compact_root=output_root / "compact",
                family=family,
                seed=seed,
            )
            audit_rows.append(
                _validate_matched_pair(result, reference, family=family, seed=seed)
            )
            all_rows.append(_metric_row(result, arm=ABLATION_ARM, seed=seed))

            if not args.keep_scratch:
                scratch_family_dir = family_scratch_root / family
                if scratch_family_dir.exists():
                    shutil.rmtree(scratch_family_dir)
                    print(
                        f"[temperature-ablation] compacted and removed scratch seed_{seed}/{family}",
                        flush=True,
                    )

    if args.dry_run:
        return

    expected_rows = 2 * len(families) * len(seeds)
    expected_audits = len(families) * len(seeds)
    if len(all_rows) != expected_rows or len(audit_rows) != expected_audits:
        raise RuntimeError(
            f"Expected {expected_rows} metric rows and {expected_audits} audits; "
            f"got {len(all_rows)} rows and {len(audit_rows)} audits."
        )

    _write_report(
        output_root=output_root,
        rows=all_rows,
        audit_rows=audit_rows,
        families=families,
        seeds=seeds,
    )
    _write_json(
        {
            "status": "complete",
            "families": families,
            "seeds": seeds,
            "full_reference_arm": FULL_ARM,
            "new_ablation_arm": ABLATION_ARM,
            "result_count": len(all_rows),
            "audit_count": len(audit_rows),
        },
        output_root / "completion.json",
    )
    print(f"[temperature-ablation] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
