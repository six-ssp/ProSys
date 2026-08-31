#!/usr/bin/env python3
"""Run compact, three-seed ablations matched to the promoted Stage 2/3 mainline.

The runner processes one family at a time, retains only compact result JSON and
model metadata, then removes its own scratch directory. This keeps the study
reproducible on storage-constrained machines without deleting any pre-existing
project outputs.
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


ARMS: dict[str, dict[str, Any]] = {
    "knn_only": {
        "description": (
            "Remove ReaFNN while retaining product-Morgan KNN, the 20-context cap, "
            "and a separately retrained tabular XGB-LTR."
        ),
        "runner_args": (
            "--disable_reafnn",
            "--skip_temperature",
            "--ranking_mode",
            "xgb_ltr",
        ),
        "result_subdir": "knn_xgb",
        "expected_baseline": "knn_xgb_stage2_ablation_ranking_only",
        "expected_stage2_architecture": "knn_only",
        "expected_ranker": "xgb_ranker",
    },
    "no_xgb_ltr": {
        "description": (
            "Keep the full product-Morgan KNN plus ReaFNN Stage 2 pool, then replace "
            "XGB-LTR with a deterministic Stage 1/2 prior."
        ),
        "runner_args": (
            "--skip_temperature",
            "--ranking_mode",
            "stage2_heuristic",
        ),
        "result_subdir": "stage2_heuristic",
        "expected_baseline": "stage2_heuristic_no_xgb_ltr",
        "expected_stage2_architecture": "knn_reafnn",
        "expected_ranker": "deterministic_stage1_stage2_prior",
    },
}

METRIC_FIELDS = (
    ("route_at_10", "Route@10"),
    ("candidate_recall", "Candidate recall"),
    ("sys_at_1", "Sys@1"),
    ("sys_at_3", "Sys@3"),
    ("sys_at_5", "Sys@5"),
    ("sys_at_10", "Sys@10"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "nDCG@10"),
)


def _parse_csv_values(value: str, *, label: str) -> list[str]:
    values = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    return values


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in _parse_csv_values(value, label="--seeds")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    return seeds


def _parse_arms(value: str) -> list[str]:
    arms = _parse_csv_values(value, label="--arms")
    unknown = sorted(set(arms).difference(ARMS))
    if unknown:
        raise ValueError(f"Unknown ablation arm(s): {', '.join(unknown)}")
    if len(set(arms)) != len(arms):
        raise ValueError("--arms contains duplicates")
    return arms


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Mark raw paths as pruned while preserving metrics and protocol metadata."""

    compact = json.loads(json.dumps(result))
    compact["candidate_table"] = "pruned_after_compaction"
    compact["scored_test_file"] = "pruned_after_compaction"
    compact["raw_artifacts_pruned"] = True
    compact["retention_note"] = (
        "Raw candidate tables, scored tables, and binary checkpoints were removed "
        "after compact result and metadata retention."
    )
    return compact


def _metric_row(result: dict[str, Any], *, arm: str, seed: int) -> dict[str, Any]:
    metrics = result["metrics"]
    route = result["stage1_route_recall"]
    return {
        "arm": arm,
        "seed": seed,
        "family": result["family"],
        "n_test_manifest": int(route.get("n") or 0),
        "candidate_slates": int(metrics.get("candidate_slates") or 0),
        "missing_candidate_slates": int(metrics.get("missing_candidate_slates") or 0),
        "route_at_10": float(route.get("route_recall_top10") or 0.0),
        "candidate_recall": float(metrics.get("pool_coverage") or 0.0),
        "sys_at_1": float(metrics.get("system_top1_all") or 0.0),
        "sys_at_3": float(metrics.get("system_top3_all") or 0.0),
        "sys_at_5": float(metrics.get("system_top5_all") or 0.0),
        "sys_at_10": float(metrics.get("system_top10_all") or 0.0),
        "mrr": float(metrics.get("system_mrr") or 0.0),
        "ndcg_at_10": float(metrics.get("system_ndcg10") or 0.0),
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    outputs: list[dict[str, Any]] = []
    metric_keys = [field for field, _ in METRIC_FIELDS]
    for key, members in sorted(grouped.items()):
        output = dict(zip(group_keys, key))
        output["n_records"] = len(members)
        output["n_test_manifest"] = int(members[0]["n_test_manifest"])
        output["candidate_slates"] = int(members[0]["candidate_slates"])
        output["missing_candidate_slates"] = int(members[0]["missing_candidate_slates"])
        for metric in metric_keys:
            values = [float(member[metric]) for member in members]
            output[f"{metric}_mean"] = _mean(values)
            output[f"{metric}_std"] = _std(values)
        outputs.append(output)
    return outputs


def _format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}"


def _format_mean_std(row: dict[str, Any], metric: str) -> str:
    return f"{100.0 * float(row[f'{metric}_mean']):.2f} +/- {100.0 * float(row[f'{metric}_std']):.2f}"


def _load_mainline_reference(macro_file: Path) -> dict[str, Any] | None:
    if not macro_file.exists():
        return None
    with macro_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one current-mainline row in {macro_file}")
    row = rows[0]
    return {
        "arm": "full_mainline",
        "route_at_10_mean": 0.6320,
        "route_at_10_std": 0.0,
        "candidate_recall_mean": float(row["cover_mean"]),
        "candidate_recall_std": float(row["cover_std"]),
        "sys_at_1_mean": float(row["sys1_mean"]),
        "sys_at_1_std": float(row["sys1_std"]),
        "sys_at_3_mean": float(row["sys3_mean"]),
        "sys_at_3_std": float(row["sys3_std"]),
        "sys_at_5_mean": float(row["sys5_mean"]),
        "sys_at_5_std": float(row["sys5_std"]),
        "sys_at_10_mean": float(row["sys10_mean"]),
        "sys_at_10_std": float(row["sys10_std"]),
        "mrr_mean": float(row["mrr_mean"]),
        "mrr_std": float(row["mrr_std"]),
        "ndcg_at_10_mean": float(row["ndcg10_mean"]),
        "ndcg_at_10_std": float(row["ndcg10_std"]),
    }


def _validate_result(
    result: dict[str, Any],
    *,
    arm: str,
    family: str,
    seed: int,
) -> None:
    spec = ARMS[arm]
    if result.get("family") != family:
        raise ValueError(f"{arm}/{seed}/{family}: wrong family in result")
    if int(result.get("seed")) != seed:
        raise ValueError(f"{arm}/{seed}/{family}: wrong seed in result")
    if result.get("baseline") != spec["expected_baseline"]:
        raise ValueError(f"{arm}/{seed}/{family}: unexpected baseline identifier")

    model = result.get("model") or {}
    stage2 = model.get("stage2_protocol") or {}
    ranker = model.get("ranking_protocol") or {}
    temperature = model.get("temperature_protocol") or {}
    if stage2.get("architecture") != spec["expected_stage2_architecture"]:
        raise ValueError(f"{arm}/{seed}/{family}: unexpected Stage 2 architecture")
    if stage2.get("knn_retrieval_mode") != "product_morgan":
        raise ValueError(f"{arm}/{seed}/{family}: KNN is not product-Morgan")
    if int(stage2.get("knn_top_k") or 0) != 64:
        raise ValueError(f"{arm}/{seed}/{family}: KNN K is not 64")
    if int(stage2.get("prefilter_contexts") or 0) != 64:
        raise ValueError(f"{arm}/{seed}/{family}: wide pool is not 64 contexts")
    if int(stage2.get("max_contexts") or 0) != 20:
        raise ValueError(f"{arm}/{seed}/{family}: candidate cap is not 20")
    if stage2.get("training_candidate_table_mode") != "oracle":
        raise ValueError(f"{arm}/{seed}/{family}: training table mode changed")
    if ranker.get("architecture") != spec["expected_ranker"]:
        raise ValueError(f"{arm}/{seed}/{family}: unexpected Stage 3 ranking policy")
    if temperature.get("always_enabled") is not False:
        raise ValueError(f"{arm}/{seed}/{family}: temperature must be disabled for Sys@k ablation")
    if int((result.get("stage1_route_recall") or {}).get("n") or 0) <= 0:
        raise ValueError(f"{arm}/{seed}/{family}: missing Stage 1 manifest count")


def _validate_no_xgb_stage2_match(
    result: dict[str, Any],
    *,
    mainline_compact_root: Path,
    family: str,
    seed: int,
) -> None:
    """Verify that the no-XGB arm preserves the official Stage 2 pool."""

    reference_file = mainline_compact_root / f"seed_{seed}" / family / "result.json"
    if not reference_file.exists():
        raise FileNotFoundError(
            f"Missing official current-mainline compact result for Stage 2 match: {reference_file}"
        )
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    result_stage2 = (result.get("model") or {}).get("stage2_protocol") or {}
    reference_stage2 = (reference.get("model") or {}).get("stage2_protocol") or {}
    if result_stage2 != reference_stage2:
        raise ValueError(f"no_xgb_ltr/{seed}/{family}: Stage 2 protocol differs from the official mainline")

    result_metrics = result.get("metrics") or {}
    reference_metrics = reference.get("metrics") or {}
    for field in (
        "pool_coverage",
        "pool_route_coverage",
        "pool_context_coverage",
        "candidate_slates",
        "missing_candidate_slates",
    ):
        observed = result_metrics.get(field)
        expected = reference_metrics.get(field)
        if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
            if abs(float(observed) - float(expected)) > 1e-12:
                raise ValueError(
                    f"no_xgb_ltr/{seed}/{family}: {field} differs from the official Stage 2 pool"
                )
        elif observed != expected:
            raise ValueError(
                f"no_xgb_ltr/{seed}/{family}: {field} differs from the official Stage 2 pool"
            )


def _runner_command(
    *,
    python_bin: str,
    repo_root: Path,
    family: str,
    seed: int,
    arm: str,
    scratch_root: Path,
    route_root: Path,
    reafnn_device: str,
) -> list[str]:
    command = [
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
        "--train_table_mode",
        "oracle",
        "--reafnn_hidden_dim",
        "512",
        "--reafnn_hidden_layers",
        "2",
        "--reafnn_dropout",
        "0.10",
        "--reafnn_knn_anchor_contexts",
        "12",
        "--reafnn_correction_weight",
        "0.65",
        "--reafnn_correction_clip",
        "0.35",
        "--reafnn_enable_knn_wide_refinement",
        "--reafnn_device",
        reafnn_device if arm == "no_xgb_ltr" else "cpu",
        "--gnn_device",
        "cpu",
        "--seed",
        str(seed),
    ]
    command.extend(ARMS[arm]["runner_args"])
    return command


def _compact_family_run(
    *,
    scratch_root: Path,
    compact_root: Path,
    arm: str,
    seed: int,
    family: str,
) -> dict[str, Any]:
    spec = ARMS[arm]
    result_dir = scratch_root / family / str(spec["result_subdir"]) / "non_oracle"
    result_file = result_dir / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Missing result file: {result_file}")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    _validate_result(result, arm=arm, family=family, seed=seed)

    compact_dir = compact_root / arm / f"seed_{seed}" / family
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
    return result


def _read_compact_result(compact_root: Path, arm: str, seed: int, family: str) -> dict[str, Any] | None:
    result_file = compact_root / arm / f"seed_{seed}" / family / "result.json"
    if not result_file.exists():
        return None
    result = json.loads(result_file.read_text(encoding="utf-8"))
    _validate_result(result, arm=arm, family=family, seed=seed)
    return result


def _write_report(
    *,
    output_root: Path,
    rows: list[dict[str, Any]],
    families: list[str],
    seeds: list[int],
    mainline_reference: dict[str, Any] | None,
) -> None:
    per_family_seed = sorted(
        rows,
        key=lambda row: (str(row["arm"]), int(row["seed"]), _family_sort_key(str(row["family"]))),
    )
    _write_csv(output_root / "per_family_seed_metrics.csv", per_family_seed)

    macro_by_seed: list[dict[str, Any]] = []
    for arm in sorted({str(row["arm"]) for row in rows}):
        for seed in seeds:
            arm_rows = [row for row in rows if row["arm"] == arm and int(row["seed"]) == seed]
            if len(arm_rows) != len(families):
                raise ValueError(f"{arm}/seed_{seed}: incomplete compact result set")
            macro = {
                "arm": arm,
                "seed": seed,
                "family_count": len(arm_rows),
                "n_test_manifest": sum(int(row["n_test_manifest"]) for row in arm_rows),
                "candidate_slates": sum(int(row["candidate_slates"]) for row in arm_rows),
                "missing_candidate_slates": sum(int(row["missing_candidate_slates"]) for row in arm_rows),
            }
            for metric, _ in METRIC_FIELDS:
                macro[metric] = _mean([float(row[metric]) for row in arm_rows])
            macro_by_seed.append(macro)
    macro_by_seed.sort(key=lambda row: (str(row["arm"]), int(row["seed"])))
    _write_csv(output_root / "macro_by_seed.csv", macro_by_seed)

    macro_mean_std: list[dict[str, Any]] = []
    for arm in sorted({str(row["arm"]) for row in rows}):
        arm_rows = [row for row in macro_by_seed if row["arm"] == arm]
        summary = {
            "arm": arm,
            "seeds": ",".join(str(seed) for seed in seeds),
            "n_seeds": len(arm_rows),
            "family_count": len(families),
            "n_test_manifest": int(arm_rows[0]["n_test_manifest"]),
        }
        for metric, _ in METRIC_FIELDS:
            values = [float(row[metric]) for row in arm_rows]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_std"] = _std(values)
        macro_mean_std.append(summary)
    _write_csv(output_root / "macro_mean_std.csv", macro_mean_std)

    per_family_mean_std = _aggregate_rows(
        per_family_seed,
        ("arm", "family"),
    )
    per_family_mean_std.sort(key=lambda row: (str(row["arm"]), _family_sort_key(str(row["family"]))))
    _write_csv(output_root / "per_family_mean_std.csv", per_family_mean_std)

    full_official_scope = set(families) == set(FAMILY_ORDER) and set(seeds) == {0, 1, 2}
    reference_rows = [mainline_reference] if mainline_reference is not None and full_official_scope else []
    report_rows = reference_rows + macro_mean_std
    arm_labels = {
        "full_mainline": "Full current mainline",
        "knn_only": "KNN-only + XGB-LTR",
        "no_xgb_ltr": "Full Stage 2 + deterministic no-XGB-LTR",
    }

    lines = [
        "# Matched Current-Mainline Ablations",
        "",
        "## Scope",
        "",
        "This record is matched to the promoted product-Morgan mainline: fixed Stage 1 route caches, product-Morgan KNN (radius 2, 4,096 bits, K=64), a 64-context wide pool, and a 20-context cap.",
        "",
        f"Each ablation was evaluated over {len(families)} family/families and seeds {', '.join(str(seed) for seed in seeds)}. Each family/seed run was compacted immediately after validation, so raw candidate tables and binary checkpoints are intentionally absent.",
        "",
        "## Arms",
        "",
        "- **KNN-only + XGB-LTR:** removes ReaFNN but retrains a tabular 52-feature XGB-LTR on its changed candidate distribution.",
        "- **Full Stage 2 + deterministic no-XGB-LTR:** preserves the full ReaFNN candidate pool but sorts using only the fixed Stage 1/2 prior: route rank, route probability, Stage 2 initial score, KNN evidence, and stable reagent/solvent tie-breaks. It has no fitted ranking parameters.",
        "- Both ablation arms skip temperature because the temperature branch does not add candidates or contribute to system ranking; temperature omission cannot change Sys@k.",
        "",
        "## Macro Results",
        "",
        "| Arm | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report_rows:
        label = arm_labels.get(str(row["arm"]), str(row["arm"]))
        values = [
            _format_mean_std(row, metric)
            for metric in (
                "candidate_recall",
                "sys_at_1",
                "sys_at_3",
                "sys_at_5",
                "sys_at_10",
                "mrr",
                "ndcg_at_10",
            )
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Per-Family Sys@10",
            "",
            "| Family | KNN-only + XGB-LTR | Full Stage 2 + deterministic no-XGB-LTR |",
            "| --- | ---: | ---: |",
        ]
    )
    summary_by_key = {(row["arm"], row["family"]): row for row in per_family_mean_std}
    for family in sorted(families, key=_family_sort_key):
        knn = summary_by_key[("knn_only", family)]
        no_xgb = summary_by_key[("no_xgb_ltr", family)]
        lines.append(
            f"| {display_family_name(family)} | "
            f"{_format_mean_std(knn, 'sys_at_10')} | "
            f"{_format_mean_std(no_xgb, 'sys_at_10')} |"
        )

    lines.extend(
        [
            "",
            "## Audit Contracts",
            "",
            "- Every compact result is checked for the expected family, seed, fixed Stage 1 manifest, product-Morgan retrieval, K=64, 64-context wide pool, 20-context cap, and oracle-only train/validation tables.",
            "- The KNN-only arm must have a tabular non-graph XGB-LTR and ReaFNN disabled.",
            "- The no-XGB-LTR arm must have the deterministic Stage 1/2 prior, no fitted ranking parameters, and an exactly matching official Stage 2 protocol plus candidate-coverage record for the same family and seed.",
            "- The official full-mainline row is read from the retained three-seed compact artifact; it is not recomputed or numerically mixed with old direct-R-GNN snapshots.",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--families", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--arms", default="knn_only,no_xgb_ltr")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("Experiment/current_mainline_matched_ablation_multiseed_20260830"),
    )
    parser.add_argument(
        "--scratch_root",
        type=Path,
        default=Path("outputs/current_mainline_matched_ablation_scratch_20260830"),
    )
    parser.add_argument(
        "--mainline_macro_file",
        type=Path,
        default=Path("Experiment/stage23_product_morgan_reafnn_multiseed_20260830/macro_mean_std.csv"),
    )
    parser.add_argument(
        "--mainline_compact_root",
        type=Path,
        default=Path("Experiment/stage23_product_morgan_reafnn_multiseed_20260830/compact/prosys"),
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--reafnn_device", default="cuda")
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--keep_scratch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = (repo_root / args.output_root).resolve()
    scratch_root = (repo_root / args.scratch_root).resolve()
    mainline_macro_file = (repo_root / args.mainline_macro_file).resolve()
    mainline_compact_root = (repo_root / args.mainline_compact_root).resolve()
    families = parse_families_arg(args.families)
    seeds = _parse_seeds(args.seeds)
    arms = _parse_arms(args.arms)

    if output_root == scratch_root or scratch_root == repo_root:
        parser.error("--scratch_root must be separate from --output_root and the repository root")
    if args.cpu_threads < 1:
        parser.error("--cpu_threads must be positive")

    manifest = {
        "purpose": "matched current-mainline Stage 2 and Stage 3 ablations",
        "families": families,
        "seeds": seeds,
        "arms": {arm: ARMS[arm] for arm in arms},
        "mainline_reference": str(mainline_macro_file),
        "mainline_compact_root": str(mainline_compact_root),
        "fixed_configuration": {
            "knn_retrieval_mode": "product_morgan",
            "fpsize": 4096,
            "radius": 2,
            "knn_top_k": 64,
            "prefilter_contexts": 64,
            "max_contexts": 20,
            "train_table_mode": "oracle",
            "reafnn_hidden_dim": 512,
            "reafnn_hidden_layers": 2,
            "reafnn_dropout": 0.10,
            "reafnn_knn_anchor_contexts": 12,
            "reafnn_correction_weight": 0.65,
            "reafnn_correction_clip": 0.35,
        },
        "scratch_retention": "kept" if args.keep_scratch else "removed after compact retention",
    }
    _write_json(manifest, output_root / "run_manifest.json")

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.cpu_threads)
    environment["MKL_NUM_THREADS"] = str(args.cpu_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
    all_rows: list[dict[str, Any]] = []

    for arm in arms:
        for seed in seeds:
            for family in families:
                compact = _read_compact_result(output_root / "compact", arm, seed, family)
                if compact is not None and not args.force:
                    if arm == "no_xgb_ltr":
                        _validate_no_xgb_stage2_match(
                            compact,
                            mainline_compact_root=mainline_compact_root,
                            family=family,
                            seed=seed,
                        )
                    print(f"[matched-ablation] reuse compact {arm}/seed_{seed}/{family}", flush=True)
                    all_rows.append(_metric_row(compact, arm=arm, seed=seed))
                    continue

                family_scratch_root = scratch_root / arm / f"seed_{seed}"
                command = _runner_command(
                    python_bin=args.python_bin,
                    repo_root=repo_root,
                    family=family,
                    seed=seed,
                    arm=arm,
                    scratch_root=family_scratch_root,
                    route_root=(repo_root / "outputs/stage1_routes").resolve(),
                    reafnn_device=args.reafnn_device,
                )
                print(
                    f"[matched-ablation] run {arm}/seed_{seed}/{family}: "
                    + " ".join(command),
                    flush=True,
                )
                if args.dry_run:
                    continue

                subprocess.run(command, check=True, cwd=repo_root, env=environment)
                result = _compact_family_run(
                    scratch_root=family_scratch_root,
                    compact_root=output_root / "compact",
                    arm=arm,
                    seed=seed,
                    family=family,
                )
                if arm == "no_xgb_ltr":
                    _validate_no_xgb_stage2_match(
                        result,
                        mainline_compact_root=mainline_compact_root,
                        family=family,
                        seed=seed,
                    )
                all_rows.append(_metric_row(result, arm=arm, seed=seed))

                if not args.keep_scratch:
                    scratch_family_dir = family_scratch_root / family
                    if scratch_family_dir.exists():
                        shutil.rmtree(scratch_family_dir)
                        print(
                            f"[matched-ablation] compacted and removed scratch {arm}/seed_{seed}/{family}",
                            flush=True,
                        )

    if args.dry_run:
        return

    expected_count = len(arms) * len(seeds) * len(families)
    if len(all_rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} compact results, found {len(all_rows)}. "
            "Rerun the same command to resume."
        )
    _write_report(
        output_root=output_root,
        rows=all_rows,
        families=families,
        seeds=seeds,
        mainline_reference=_load_mainline_reference(mainline_macro_file),
    )
    _write_json(
        {
            "status": "complete",
            "result_count": len(all_rows),
            "families": families,
            "seeds": seeds,
            "arms": arms,
        },
        output_root / "completion.json",
    )
    print(f"[matched-ablation] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
