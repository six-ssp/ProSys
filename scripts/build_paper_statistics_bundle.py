"""Build a lightweight, paper-facing bundle from fixed experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FAMILIES = (
    "Beckmann",
    "Buchwald-HartwigCross-Coupling",
    "Chan_LamCoupling",
    "DielsAlder",
    "Friedel-CraftsAcylation",
    "Friedel-CraftsAlkylation",
)

DISPLAY_NAMES = {
    "Beckmann": "Beckmann",
    "Buchwald-HartwigCross-Coupling": "Buchwald-Hartwig",
    "Chan_LamCoupling": "Chan-Lam",
    "DielsAlder": "Diels-Alder",
    "Friedel-CraftsAcylation": "Friedel-Crafts Acylation",
    "Friedel-CraftsAlkylation": "Friedel-Crafts Alkylation",
}

ABLATION_GROUPS = {
    "full_mainline": "A2,A3,A4",
    "frequency_top20_xgb": "A2",
    "knn_only_xgb": "A2,A4",
    "no_gnn_xgb": "A3,A4",
    "no_stage3": "A3",
    "knn_only_no_gnn_xgb": "A4",
}

ABLATION_LABELS = {
    "full_mainline": "Historical KNN + ReaFNN + R-GNN + XGB-LTR",
    "frequency_top20_xgb": "Historical top-20 frequency + R-GNN + XGB-LTR",
    "knn_only_xgb": "Historical KNN only + R-GNN + XGB-LTR",
    "no_gnn_xgb": "Historical KNN + ReaFNN + XGB-LTR (without R-GNN)",
    "no_stage3": "Historical KNN + ReaFNN (without XGB-LTR)",
    "knn_only_no_gnn_xgb": "Historical KNN only + XGB-LTR (without ReaFNN and R-GNN)",
}


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result


def _pct(value: Any) -> float | None:
    number = _number(value)
    return None if number is None else 100.0 * number


def _mean(values: list[Any]) -> float | None:
    valid = [number for value in values if (number := _number(value)) is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _macro(rows: list[dict[str, Any]], metric_columns: tuple[str, ...]) -> dict[str, Any]:
    row: dict[str, Any] = {"family": "MACRO-AVG", "display_family": "Macro average"}
    for column in metric_columns:
        row[column] = _mean([item.get(column) for item in rows])
    return row


def _mainline_rows(repo_root: Path, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        source = root / family / "knn_xgb" / "non_oracle" / "result.json"
        if not source.exists():
            continue
        payload = _read_json(source)
        metrics = payload.get("metrics", {})
        temperature = metrics.get("temperature", {})
        route = payload.get("stage1_route_recall", {})
        model = payload.get("model", {})
        rows.append(
            {
                "family": family,
                "display_family": DISPLAY_NAMES[family],
                "route_at_1_pct": _pct(route.get("route_recall_top1")),
                "route_at_3_pct": _pct(route.get("route_recall_top3")),
                "route_at_5_pct": _pct(route.get("route_recall_top5")),
                "route_at_10_pct": _pct(route.get("route_recall_top10")),
                "route_pool_coverage_pct": _pct(metrics.get("pool_route_coverage")),
                "context_pool_coverage_pct": _pct(metrics.get("pool_context_coverage")),
                "cover_pct": _pct(metrics.get("pool_coverage")),
                "sys_at_1_pct": _pct(metrics.get("system_top1_all")),
                "sys_at_3_pct": _pct(metrics.get("system_top3_all")),
                "sys_at_5_pct": _pct(metrics.get("system_top5_all")),
                "sys_at_10_pct": _pct(metrics.get("system_top10_all")),
                "mrr_pct": _pct(metrics.get("system_mrr")),
                "ndcg_at_10_pct": _pct(metrics.get("system_ndcg10")),
                "temperature_support": temperature.get("n"),
                "temperature_mae_c": _number(temperature.get("mae")),
                "temperature_rmse_c": _number(temperature.get("rmse")),
                "temperature_within_5c_pct": _pct(temperature.get("within_5c")),
                "temperature_within_10c_pct": _pct(temperature.get("within_10c")),
                "temperature_within_20c_pct": _pct(temperature.get("within_20c")),
                "test_manifest_samples": metrics.get("num_slates"),
                "candidate_slates": metrics.get("candidate_slates"),
                "missing_candidate_slates": metrics.get("missing_candidate_slates"),
                "candidate_table": _relative(repo_root, Path(payload.get("candidate_table", ""))),
                "source_result_json": _relative(repo_root, source),
                "metric_denominator": metrics.get("denominator"),
            }
        )

    metric_columns = tuple(
        key
        for key in rows[0]
        if key.endswith("_pct") or key in {"temperature_mae_c", "temperature_rmse_c"}
    ) if rows else ()
    if rows:
        macro = _macro(rows, metric_columns)
        macro.update(
            {
                "temperature_support": int(sum(int(row["temperature_support"] or 0) for row in rows)),
                "test_manifest_samples": int(sum(int(row["test_manifest_samples"] or 0) for row in rows)),
                "candidate_slates": int(sum(int(row["candidate_slates"] or 0) for row in rows)),
                "missing_candidate_slates": int(sum(int(row["missing_candidate_slates"] or 0) for row in rows)),
                "candidate_table": "Per-family tables; see source_result_json",
                "source_result_json": "Per-family result.json files",
                "metric_denominator": "all_test_manifest_samples",
            }
        )
        rows.append(macro)
    return rows


def _direct_baseline_rows(repo_root: Path, source: Path, baseline_id: str, method_label: str) -> list[dict[str, Any]]:
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for item in pd.read_csv(source).to_dict(orient="records"):
        rows.append(
            {
                "baseline_id": baseline_id,
                "method": method_label,
                "information_setting": "target product only; route paired after condition prediction",
                "family": item.get("family"),
                "display_family": item.get("display_family"),
                "condition_at_1_pct": _pct(item.get("condition1")),
                "condition_at_3_pct": _pct(item.get("condition3")),
                "condition_at_5_pct": _pct(item.get("condition5")),
                "condition_at_10_pct": _pct(item.get("condition10")),
                "cover_pct": _pct(item.get("cover")),
                "sys_at_1_pct": _pct(item.get("sys1")),
                "sys_at_3_pct": _pct(item.get("sys3")),
                "sys_at_5_pct": _pct(item.get("sys5")),
                "sys_at_10_pct": _pct(item.get("sys10")),
                "mrr_pct": _pct(item.get("mrr")),
                "ndcg_at_10_pct": _pct(item.get("ndcg10")),
                "temperature_support": None,
                "temperature_mae_c": None,
                "temperature_within_10c_pct": None,
                "candidate_slates": _number(item.get("candidate_slates")),
                "missing_candidate_slates": _number(item.get("missing_candidate_slates")),
                "selected_route_weight": _number(item.get("route_weight")),
                "source_summary_csv": _relative(repo_root, source),
            }
        )
    return rows


def _external_baseline_rows(repo_root: Path, source: Path) -> list[dict[str, Any]]:
    if not source.exists():
        return []
    mapping = {
        "sequential_fnn": ("B3", "EditRetro + Sequential FNN"),
        "reaction_gcnn": ("B4", "EditRetro + Reaction-GCNN"),
    }
    rows: list[dict[str, Any]] = []
    for item in pd.read_csv(source).to_dict(orient="records"):
        method = str(item.get("method", ""))
        if method not in mapping:
            continue
        baseline_id, method_label = mapping[method]
        rows.append(
            {
                "baseline_id": baseline_id,
                "method": method_label,
                "information_setting": "predicted route plus product",
                "family": item.get("family"),
                "display_family": DISPLAY_NAMES.get(str(item.get("family")), item.get("family")),
                "condition_at_1_pct": None,
                "condition_at_3_pct": None,
                "condition_at_5_pct": None,
                "condition_at_10_pct": None,
                "cover_pct": _pct(item.get("cover")),
                "sys_at_1_pct": _pct(item.get("sys1")),
                "sys_at_3_pct": _pct(item.get("sys3")),
                "sys_at_5_pct": _pct(item.get("sys5")),
                "sys_at_10_pct": _pct(item.get("sys10")),
                "mrr_pct": _pct(item.get("mrr")),
                "ndcg_at_10_pct": _pct(item.get("ndcg10")),
                "temperature_support": _number(item.get("temperature_n")),
                "temperature_mae_c": _number(item.get("temperature_mae")),
                "temperature_within_10c_pct": _pct(item.get("temperature_within_10c")),
                "candidate_slates": _number(item.get("candidate_slates")),
                "missing_candidate_slates": _number(item.get("missing_candidate_slates")),
                "selected_route_weight": _number(item.get("route_weight")),
                "source_summary_csv": _relative(repo_root, source),
            }
        )
    return rows


def _ablation_rows(repo_root: Path, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for method, groups in ABLATION_GROUPS.items():
            source = root / family / method / "non_oracle" / "result.json"
            if not source.exists():
                continue
            payload = _read_json(source)
            metrics = payload.get("metrics", {})
            temperature = metrics.get("temperature", {})
            budget = payload.get("candidate_budget", {})
            rows.append(
                {
                    "arm": method,
                    "arm_label": ABLATION_LABELS[method],
                    "family": family,
                    "display_family": DISPLAY_NAMES[family],
                    "route_at_10_pct": _pct(payload.get("stage1_route_recall", {}).get("route_recall_top10")),
                    "route_pool_coverage_pct": _pct(metrics.get("pool_route_coverage")),
                    "context_pool_coverage_pct": _pct(metrics.get("pool_context_coverage")),
                    "cover_pct": _pct(metrics.get("pool_coverage")),
                    "sys_at_1_pct": _pct(metrics.get("system_top1_all")),
                    "sys_at_3_pct": _pct(metrics.get("system_top3_all")),
                    "sys_at_5_pct": _pct(metrics.get("system_top5_all")),
                    "sys_at_10_pct": _pct(metrics.get("system_top10_all")),
                    "mrr_pct": _pct(metrics.get("system_mrr")),
                    "ndcg_at_10_pct": _pct(metrics.get("system_ndcg10")),
                    "temperature_support": temperature.get("n"),
                    "temperature_mae_c": _number(temperature.get("mae")),
                    "temperature_within_10c_pct": _pct(temperature.get("within_10c")),
                    "test_manifest_samples": budget.get("test_manifest_samples"),
                    "candidate_slates": budget.get("candidate_slates"),
                    "missing_candidate_slates": budget.get("missing_candidate_slates"),
                    "candidate_rows": budget.get("candidate_rows"),
                    "mean_rows_per_candidate_slate": _number(budget.get("mean_rows_per_candidate_slate")),
                    "max_rows_per_candidate_slate": budget.get("max_rows_per_candidate_slate"),
                    "mean_contexts_per_route": _number(budget.get("mean_contexts_per_route")),
                    "max_contexts_per_route": budget.get("max_contexts_per_route"),
                    "source_result_json": _relative(repo_root, source),
                }
            )

    if not rows:
        return rows
    metrics = (
        "route_at_10_pct",
        "route_pool_coverage_pct",
        "context_pool_coverage_pct",
        "cover_pct",
        "sys_at_1_pct",
        "sys_at_3_pct",
        "sys_at_5_pct",
        "sys_at_10_pct",
        "mrr_pct",
        "ndcg_at_10_pct",
        "temperature_mae_c",
        "temperature_within_10c_pct",
        "mean_rows_per_candidate_slate",
        "max_rows_per_candidate_slate",
        "mean_contexts_per_route",
        "max_contexts_per_route",
    )
    macro_rows: list[dict[str, Any]] = []
    for method in ABLATION_GROUPS:
        members = [row for row in rows if row["arm"] == method]
        if not members:
            continue
        macro = _macro(members, metrics)
        macro.update(
            {
                "ablation_groups": ABLATION_GROUPS[method],
                "arm": method,
                "arm_label": ABLATION_LABELS[method],
                "temperature_support": int(sum(int(row["temperature_support"] or 0) for row in members)),
                "test_manifest_samples": int(sum(int(row["test_manifest_samples"] or 0) for row in members)),
                "candidate_slates": int(sum(int(row["candidate_slates"] or 0) for row in members)),
                "missing_candidate_slates": int(sum(int(row["missing_candidate_slates"] or 0) for row in members)),
                "candidate_rows": int(sum(int(row["candidate_rows"] or 0) for row in members)),
                "source_result_json": "Per-family result.json files",
            }
        )
        macro_rows.append(macro)
    return rows + macro_rows


def _xgb_metadata_rows(repo_root: Path, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        source = root / family / "knn_xgb" / "non_oracle" / "model" / "xgb_ranker_meta.json"
        if not source.exists():
            continue
        payload = _read_json(source)
        features = [str(item) for item in payload.get("feature_columns", [])]
        params = payload.get("params", {})
        fusion = payload.get("score_fusion", {})
        rows.append(
            {
                "family": family,
                "display_family": DISPLAY_NAMES[family],
                "best_iteration": payload.get("best_iteration"),
                "feature_count": len(features),
                "reaction_gnn_feature_count": sum(item.startswith("route_gnn_feat_") for item in features),
                "objective": params.get("objective"),
                "eval_metric": params.get("eval_metric"),
                "n_estimators_cap": params.get("n_estimators"),
                "learning_rate": params.get("learning_rate"),
                "max_depth": params.get("max_depth"),
                "subsample": params.get("subsample"),
                "colsample_bytree": params.get("colsample_bytree"),
                "reg_lambda": params.get("reg_lambda"),
                "seed": params.get("random_state"),
                "heuristic_weight": _number(fusion.get("heuristic_weight")),
                "validation_sys_at_10_pct": _pct(fusion.get("val_system_top10_all")),
                "validation_sys_at_1_pct": _pct(fusion.get("val_system_top1_all")),
                "score_definition": fusion.get("score_definition"),
                "source_metadata_json": _relative(repo_root, source),
            }
        )
    return rows


def _clean_to_split_reconciliation_rows(output_root: Path) -> list[dict[str, Any]]:
    """Record whether a code-current raw-clean replay matches persisted split rows."""

    clean_path = output_root / "01_raw_to_clean.csv"
    split_path = output_root / "03_split_stats.csv"
    if not clean_path.exists() or not split_path.exists():
        return []

    clean = pd.read_csv(clean_path).set_index("family")
    splits = pd.read_csv(split_path).set_index("family")
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        if family not in clean.index or family not in splits.index:
            continue
        recomputed = int(clean.at[family, "final_all_dataset"])
        persisted = int(
            splits.at[family, "train_reactions"]
            + splits.at[family, "valid_reactions"]
            + splits.at[family, "test_reactions"]
        )
        difference = persisted - recomputed
        rows.append(
            {
                "family": family,
                "display_family": DISPLAY_NAMES[family],
                "recomputed_raw_clean_records": recomputed,
                "persisted_fixed_split_records": persisted,
                "fixed_minus_recomputed": difference,
                "absolute_difference": abs(difference),
                "status": "match" if difference == 0 else "minor historical drift",
                "interpretation": (
                    "The persisted split is the formal evaluation source; the raw-clean replay is a current-code diagnostic."
                ),
            }
        )
    return rows


def _copy_source_documents(repo_root: Path, output_root: Path) -> list[dict[str, Any]]:
    source_paths = (
        "CURRENT_RESULTS.md",
        "baseline&ablation.md",
        "baseline/current_baseline_results_20260727.md",
        "ablation/current_mainline_ablation_results_20260727.md",
        "project_audit_20260727.md",
        "audit/split_audit_20260727.txt",
        "outputs/ablation_reafnn_gnn_20260726/audit.md",
        "outputs/ablation_reafnn_gnn_20260726/stage1_route_ablation.md",
        "outputs/ablation_reafnn_gnn_20260726/stage2_pool_ablation.md",
        "outputs/ablation_reafnn_gnn_20260726/stage3_reranking_ablation.md",
        "outputs/ablation_reafnn_gnn_20260726/stage23_interaction_ablation.md",
        "outputs/baselines/direct_product_condition_nb_20260727/RESULTS.md",
        "outputs/baselines/direct_product_condition_20260727/RESULTS.md",
        "outputs/baselines/non_oracle_external_b23_20260726/RESULTS.md",
        "example.md",
        "checklist.md",
        "outputs/robustness_multiseed_20260730/README.md",
        "outputs/checklist_supplement_20260730/README.md",
        "outputs/checklist_supplement_20260730/fixed_manifest_checks/README.md",
        "outputs/checklist_supplement_20260730/product_disjoint_direct_baselines/README.md",
    )
    rows: list[dict[str, Any]] = []
    destination_root = output_root / "source_documents"
    for source_string in source_paths:
        source = repo_root / source_string
        if not source.exists():
            rows.append({"source": source_string, "status": "missing", "bytes": None, "sha256": None})
            continue
        destination = destination_root / source_string
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows.append(
            {
                "source": source_string,
                "status": "included",
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    return rows


def _copy_supplemental_tables(
    output_root: Path,
    robustness_root: Path,
    supplement_root: Path,
) -> None:
    """Keep compact, paper-facing robustness tables in the evidence bundle."""

    artifacts = (
        ("27_fixed_stage1_multiseed_macro_by_seed.csv", robustness_root / "macro_by_seed.csv"),
        ("28_fixed_stage1_multiseed_macro_mean_std.csv", robustness_root / "macro_mean_std.csv"),
        (
            "29_fixed_manifest_bootstrap_sys_metrics.csv",
            supplement_root / "fixed_manifest_checks" / "bootstrap_sys_metrics.csv",
        ),
        (
            "30_mainline_error_decomposition.csv",
            supplement_root / "fixed_manifest_checks" / "mainline_error_decomposition.csv",
        ),
        (
            "31_product_disjoint_direct_baseline_macro.csv",
            supplement_root / "product_disjoint_direct_baselines" / "macro_results.csv",
        ),
        (
            "32_product_disjoint_direct_baseline_per_family.csv",
            supplement_root / "product_disjoint_direct_baselines" / "per_family_results.csv",
        ),
        (
            "33_product_disjoint_subset_counts.csv",
            supplement_root / "product_disjoint_direct_baselines" / "subset_counts.csv",
        ),
    )
    for filename, source in artifacts:
        if source.exists():
            shutil.copy2(source, output_root / filename)


def _gap_rows() -> list[dict[str, str]]:
    return [
        {
            "checklist_area": "Versioning",
            "item": "Immutable data version identifiers and a clean code commit",
            "status": "not fully recoverable",
            "note": "The current worktree is dirty. The bundle records the HEAD commit and hashes selected artifacts, but this is not a replacement for a clean tagged release.",
        },
        {
            "checklist_area": "Data cleaning",
            "item": "Raw-clean replay versus persisted fixed split",
            "status": "minor drift documented",
            "note": "See 26_cleaning_to_fixed_split_reconciliation.csv. Three families differ by -2, +5, and -7 records respectively; formal metrics use the persisted fixed split, whose canonical-route overlap audit passes.",
        },
        {
            "checklist_area": "Data cleaning",
            "item": "Separate RDKit failures for reactants and products",
            "status": "not collected",
            "note": "The reproducible cleaning table records combined RDKit failures only; the source audit does not preserve the two subcounts separately.",
        },
        {
            "checklist_area": "Data cleaning",
            "item": "Count of multi-temperature records resolved by taking the maximum temperature",
            "status": "not collected",
            "note": "The preprocessing rule is implemented, but the historical run did not persist this event count.",
        },
        {
            "checklist_area": "Temperature",
            "item": "Median absolute error, signed error, error standard deviation, and temperature-bin breakdown",
            "status": "not collected",
            "note": "The fixed results contain MAE, RMSE, and tolerance hits, but not all requested residual summaries.",
        },
        {
            "checklist_area": "Model cost",
            "item": "Training time, inference latency, peak memory, and hardware measurements",
            "status": "not recorded",
            "note": "No reliable run logs capture these values. They must not be inferred from file timestamps or current hardware.",
        },
        {
            "checklist_area": "Statistical robustness",
            "item": "Multi-seed uncertainty, bootstrap confidence intervals, and product-disjoint stress test",
            "status": "partially completed with explicit scope",
            "note": "Three seeds were evaluated for ProSys and B3 with fixed Stage-1 route caches; bootstrap CIs use the formal seed-0 fixed manifest; the product-disjoint stress test is a post-hoc subset evaluation of fixed B1/B2 models. These do not replace a fully product-disjoint retraining protocol or Stage-1 retraining uncertainty.",
        },
        {
            "checklist_area": "Error analysis",
            "item": "Mutually exclusive end-to-end error decomposition",
            "status": "completed",
            "note": "A seven-category exhaustive and mutually exclusive decomposition over the 3,860-record formal manifest is included in 30_mainline_error_decomposition.csv.",
        },
        {
            "checklist_area": "Case studies",
            "item": "Explicit Stage 1, retrieval, and ranking failure examples",
            "status": "partially available",
            "note": "example.md contains six representative successful examples; the requested failure-category examples remain to be selected.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("paper_statistics"))
    parser.add_argument(
        "--mainline-root",
        type=Path,
        default=Path("outputs/stage23_mainline_gnn_temperature_gated_20260803"),
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("outputs/ablation_reafnn_gnn_20260726"),
    )
    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=Path("outputs/robustness_multiseed_20260730"),
    )
    parser.add_argument(
        "--supplement-root",
        type=Path,
        default=Path("outputs/checklist_supplement_20260730"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    mainline_root = (repo_root / args.mainline_root).resolve()
    ablation_root = (repo_root / args.ablation_root).resolve()
    robustness_root = (repo_root / args.robustness_root).resolve()
    supplement_root = (repo_root / args.supplement_root).resolve()

    baseline_sources = {
        "B1": repo_root / "outputs/baselines/direct_product_condition_nb_20260727/summary.csv",
        "B2": repo_root / "outputs/baselines/direct_product_condition_20260727/summary.csv",
        "B3_B4": repo_root / "outputs/baselines/non_oracle_external_b23_20260726/summary.csv",
    }
    supplemental_sources = [
        robustness_root / "macro_by_seed.csv",
        robustness_root / "macro_mean_std.csv",
        supplement_root / "fixed_manifest_checks" / "bootstrap_sys_metrics.csv",
        supplement_root / "fixed_manifest_checks" / "mainline_error_decomposition.csv",
        supplement_root / "product_disjoint_direct_baselines" / "macro_results.csv",
    ]
    source_hashes = {
        _relative(repo_root, path): _sha256(path)
        for path in [
            repo_root / "CURRENT_RESULTS.md",
            repo_root / "baseline/current_baseline_results_20260727.md",
            repo_root / "ablation/current_mainline_ablation_results_20260727.md",
            *baseline_sources.values(),
            *supplemental_sources,
        ]
        if path.exists()
    }
    dirty = _git_value(repo_root, "status", "--porcelain")
    reconciliation_rows = _clean_to_split_reconciliation_rows(output_root)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_snapshot_date": "2026-08-03",
        "scope": "six-family target-product-driven fixed-manifest study",
        "test_manifest_records": 3860,
        "stage1_missing_route_records": 27,
        "git_head": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_worktree_dirty": bool(dirty),
        "git_status_note": "The worktree is dirty; HEAD alone does not fully identify all local code changes.",
        "mainline_root": _relative(repo_root, mainline_root),
        "ablation_root": _relative(repo_root, ablation_root),
        "robustness_root": _relative(repo_root, robustness_root),
        "supplement_root": _relative(repo_root, supplement_root),
        "raw_clean_to_fixed_split_drift_families": [row["family"] for row in reconciliation_rows if row["fixed_minus_recomputed"] != 0],
        "artifact_sha256": source_hashes,
    }
    _write_json(manifest, output_root / "00_version_manifest.json")

    mainline_rows = _mainline_rows(repo_root, mainline_root)
    _write_csv(mainline_rows, output_root / "20_official_mainline_metrics.csv")

    baseline_rows = []
    baseline_rows.extend(_direct_baseline_rows(repo_root, baseline_sources["B1"], "B1", "Product-Bernoulli Naive Bayes"))
    baseline_rows.extend(_direct_baseline_rows(repo_root, baseline_sources["B2"], "B2", "Product-GNN"))
    baseline_rows.extend(_external_baseline_rows(repo_root, baseline_sources["B3_B4"]))
    _write_csv(baseline_rows, output_root / "21_official_baseline_metrics.csv")

    _write_csv(_ablation_rows(repo_root, ablation_root), output_root / "22_official_ablation_metrics.csv")
    _write_csv(_xgb_metadata_rows(repo_root, mainline_root), output_root / "23_xgboost_metadata.csv")
    _write_csv(_gap_rows(), output_root / "24_data_gaps_and_notes.csv")
    _write_csv(reconciliation_rows, output_root / "26_cleaning_to_fixed_split_reconciliation.csv")
    _copy_supplemental_tables(output_root, robustness_root, supplement_root)
    _write_csv(_copy_source_documents(repo_root, output_root), output_root / "25_source_document_inventory.csv")
    print(f"[paper-statistics] wrote {output_root}")


if __name__ == "__main__":
    main()
