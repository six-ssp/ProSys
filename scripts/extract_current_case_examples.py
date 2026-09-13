#!/usr/bin/env python3
"""Create compact, traceable current-mainline case records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FAMILIES = (
    ("Beckmann", "Beckmann"),
    ("Buchwald-HartwigCross-Coupling", "Buchwald-Hartwig"),
    ("Chan_LamCoupling", "Chan-Lam"),
    ("DielsAlder", "Diels-Alder"),
    ("Friedel-CraftsAcylation", "Friedel-Crafts Acyl."),
    ("Friedel-CraftsAlkylation", "Friedel-Crafts Alkyl."),
)

MIDDLE_RANK = {
    "Beckmann": 3,
    "Buchwald-HartwigCross-Coupling": 5,
    "Chan_LamCoupling": 3,
    "DielsAlder": 5,
    "Friedel-CraftsAcylation": 3,
    "Friedel-CraftsAlkylation": 5,
}

FINAL_ORDER = (
    ("sample_index", True),
    ("xgb_score", False),
    ("retro_rank", True),
    ("retro_probability", False),
    ("stage2_initial_score", False),
    ("knn_similarity_sum", False),
    ("reagent_norm", True),
    ("solvent_norm", True),
)
STAGE2_ORDER = (
    ("stage2_post_fusion_score", False),
    ("stage2_initial_score", False),
    ("stage2_knn_score", False),
    ("stage2_reafnn_score", False),
    ("knn_similarity_sum", False),
    ("reagent_norm", True),
    ("solvent_norm", True),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--seed-record", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=Path("example.md"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def value(row: pd.Series, column: str, default: float = 0.0) -> float:
    result = number(row.get(column))
    return default if result is None else result


def serialise(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return number(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or pd.isna(value):
        return None
    return str(value)


def row_dict(row: pd.Series) -> dict[str, Any]:
    return {str(key): serialise(item) for key, item in row.items()}


def sort_frame(frame: pd.DataFrame, order: tuple[tuple[str, bool], ...]) -> pd.DataFrame:
    columns = [column for column, _ascending in order if column in frame.columns]
    ascending = [is_ascending for column, is_ascending in order if column in frame.columns]
    if not columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(columns, ascending=ascending, kind="mergesort").reset_index(drop=True)


def rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = sort_frame(frame, FINAL_ORDER).copy()
    result["final_rank"] = result.groupby("sample_index", sort=False).cumcount() + 1
    return result


def route_cache(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["sample_index"]): row for row in payload.get("reactions", [])}


def first_row(group: pd.DataFrame, predicate) -> pd.Series | None:
    found = group.loc[predicate(group)]
    return None if found.empty else found.sort_values("final_rank", kind="mergesort").iloc[0]


def select_cases(frame: pd.DataFrame, middle_rank: int) -> list[tuple[str, pd.Series]]:
    groups = [group.sort_values("final_rank", kind="mergesort") for _, group in frame.groupby("sample_index", sort=True)]
    used: set[int] = set()
    selected: list[tuple[str, pd.Series]] = []

    def add(title: str, predicate) -> None:
        if len(selected) >= 3:
            return
        for group in groups:
            sample = int(group["sample_index"].iloc[0])
            if sample in used:
                continue
            candidate = first_row(group, predicate)
            if candidate is not None and float(candidate.get("label", 0)) > 0.5:
                first_exact_rank = group.loc[group["label"] > 0.5, "final_rank"].min()
                if int(candidate["final_rank"]) != int(first_exact_rank):
                    continue
            if candidate is not None:
                used.add(sample)
                selected.append((title, candidate))
                return

    add("Top-1 exact system hit", lambda group: (group["label"] > 0.5) & (group["final_rank"] == 1))
    add("First exact system hit at rank " + str(middle_rank) + " (Top-" + str(middle_rank) + ")", lambda group: (group["label"] > 0.5) & (group["final_rank"] == middle_rank))
    add("First exact system hit at rank 10 (Top-10)", lambda group: (group["label"] > 0.5) & (group["final_rank"] == 10))
    add("Exact system hit within Top-5", lambda group: (group["label"] > 0.5) & (group["final_rank"] >= 2) & (group["final_rank"] <= 5))
    add("Exact system hit within Top-10", lambda group: (group["label"] > 0.5) & (group["final_rank"] >= 6) & (group["final_rank"] <= 10))
    add("Ranking miss: exact system remains below Top-10", lambda group: (group["label"] > 0.5) & (group["final_rank"] > 10))
    add("Stage-2 pool miss: gold route retained, exact system absent", lambda group: (group["route_match"] > 0.5) & ~(group["label"] > 0.5).any())
    add("Stage-1 route miss: no candidate uses the gold route", lambda group: ~(group["route_match"] > 0.5).any())
    return selected


def text(value: Any, limit: int = 72) -> str:
    result = str(value if value is not None else "").replace("|", chr(92) + "|")
    return result if len(result) <= limit else result[:limit - 3] + "..."


def show(value: Any, digits: int = 4) -> str:
    result = number(value)
    return "NA" if result is None else f"{result:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def gnn_vector(row: pd.Series) -> dict[str, Any]:
    columns = sorted(
        (column for column in row.index if column.startswith("route_gnn_feat_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    values = np.asarray([value(row, column) for column in columns], dtype=np.float64)
    if not len(values):
        return {"dimension": 0, "values": [], "l2_norm": None, "mean": None, "min": None, "max": None}
    return {
        "dimension": int(len(values)),
        "values": [float(item) for item in values],
        "l2_norm": float(np.linalg.norm(values)),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def temperature(group: pd.DataFrame) -> dict[str, Any]:
    positives = group.loc[group["label"] > 0.5].sort_values("final_rank", kind="mergesort")
    if positives.empty or "temperature_gold" not in positives:
        return {"official_support": False, "reason": "no_exact_candidate_with_valid_temperature"}
    valid = positives.loc[positives["temperature_gold"].map(number).notna()]
    if valid.empty:
        return {"official_support": False, "reason": "no_exact_candidate_with_valid_temperature"}
    chosen = valid.iloc[0]
    gold = value(chosen, "temperature_gold")
    predicted = number(chosen.get("xgb_temperature_pred"))
    error = None if predicted is None else abs(predicted - gold)
    return {
        "official_support": True,
        "definition": "highest-ranked exact full-system candidate with a valid temperature label",
        "candidate_final_rank": int(chosen["final_rank"]),
        "predicted_temperature_c": predicted,
        "gold_temperature_c": gold,
        "absolute_error_c": error,
        "within_5c": None if error is None else bool(error <= 5.0),
        "within_10c": None if error is None else bool(error <= 10.0),
        "within_20c": None if error is None else bool(error <= 20.0),
    }


def case_record(
    family: str,
    display: str,
    case_id: str,
    selection: str,
    chosen: pd.Series,
    group: pd.DataFrame,
    routes: dict[str, Any],
    rebuilt: dict[str, Any],
    retained: dict[str, Any],
) -> dict[str, Any]:
    route_rank = int(chosen["retro_rank"])
    stage2 = sort_frame(group.loc[group["retro_rank"] == route_rank], STAGE2_ORDER).head(8)
    top10 = group.sort_values("final_rank", kind="mergesort").head(10)
    exact = group.loc[group["label"] > 0.5].sort_values("final_rank", kind="mergesort")
    plotting = pd.concat([top10, exact], ignore_index=True).drop_duplicates()
    return {
        "case_id": case_id,
        "family": family,
        "family_display": display,
        "selection": selection,
        "identity": {
            "sample_index": int(chosen["sample_index"]),
            "reaction_id": str(chosen["reaction_id"]),
            "product": str(chosen["product"]),
            "selected_route_rank": route_rank,
            "selected_final_rank": int(chosen["final_rank"]),
        },
        "selected_candidate_all_columns": row_dict(chosen),
        "stage1": {
            "gold_reactants_evaluation_only": str(routes.get("gold_reactants", "")),
            "predicted_top10_routes": routes.get("routes", []),
        },
        "stage2": {
            "slate_candidate_count": int(len(group)),
            "selected_route_candidate_count": int((group["retro_rank"] == route_rank).sum()),
            "route_covered": bool((group["route_match"] > 0.5).any()),
            "context_covered": bool((group["context_match"] > 0.5).any()),
            "exact_system_in_pool": bool((group["label"] > 0.5).any()),
            "selected_route_top8_all_columns": [row_dict(row) for _, row in stage2.iterrows()],
        },
        "stage3": {
            "top10_all_columns": [row_dict(row) for _, row in top10.iterrows()],
            "all_exact_candidates_all_columns": [row_dict(row) for _, row in exact.iterrows()],
            "top10_plus_exact_for_plotting": [row_dict(row) for _, row in plotting.iterrows()],
        },
        "temperature": temperature(group),
        "route_gnn_temperature_only": gnn_vector(chosen),
        "rebuilt_model_record": rebuilt.get("model", {}),
        "retained_seed0_model_record": retained.get("model", {}),
        "guardrail": "Gold route, condition, yield, and temperature fields are post-hoc evaluation annotations only; they are not model inputs.",
    }


def render_case(record: dict[str, Any], group: pd.DataFrame, routes: dict[str, Any], rebuilt: dict[str, Any]) -> list[str]:
    q = chr(96)
    chosen_rank = record["identity"]["selected_final_rank"]
    chosen = group.loc[group["final_rank"] == chosen_rank].iloc[0]
    positives = group.loc[group["label"] > 0.5].sort_values("final_rank", kind="mergesort")
    calibration = rebuilt.get("model", {}).get("stage2_protocol", {}).get("reafnn_post_fusion_calibration", {})
    lines = [
        "### " + record["case_id"] + " - " + record["selection"],
        "",
        "- Identity: " + q + "sample_index=" + str(record["identity"]["sample_index"]) + q
        + ", " + q + "reaction_id=" + record["identity"]["reaction_id"] + q
        + ", selected Stage-1 route rank " + q + str(record["identity"]["selected_route_rank"]) + q
        + ", final system rank " + q + str(chosen_rank) + q + ".",
        "- Target product:",
        "> " + record["identity"]["product"],
        "- Reference route, exact-system, yield, and temperature values below are post-hoc evaluation annotations only.",
    ]
    if positives.empty:
        lines.append("- No exact full-system candidate is in this candidate slate.")
    else:
        lines.append("- Exact full-system reference candidate(s) in this slate:")
        lines.extend(markdown_table(
            ["final", "route", "reagent", "solvent", "yield", "temperature"],
            [[
                str(int(row["final_rank"])), str(int(row["retro_rank"])),
                text(row.get("reagent_norm"), 46), text(row.get("solvent_norm"), 46),
                show(row.get("yield_gold"), 2), show(row.get("temperature_gold"), 2),
            ] for _, row in positives.iterrows()],
        ))

    lines.extend(["", "#### Stage 1: predicted route slate", ""])
    route_labels = group.groupby("retro_rank", sort=True)["route_match"].max().to_dict()
    lines.extend(markdown_table(
        ["route", "retro score", "probability", "gold-route annotation", "predicted reactants"],
        [[
            str(item.get("retro_rank", "")), show(item.get("retro_score"), 5), show(item.get("retro_probability"), 5),
            "yes" if float(route_labels.get(item.get("retro_rank"), 0.0)) > 0.5 else "no",
            text(item.get("reactants"), 115),
        ] for item in sorted(routes.get("routes", []), key=lambda item: item.get("retro_rank", 0))],
    ))
    lines.extend([
        "",
        "- Evaluation-only reference reactants: " + q + str(routes.get("gold_reactants", "")) + q + ".",
        "",
        "#### Stage 2: parallel KNN/ReaFNN candidate pool",
        "",
        "- Candidate slate size " + q + str(len(group)) + q
        + "; route covered " + q + str(bool((group["route_match"] > 0.5).any())) + q
        + "; exact system in pool " + q + str(bool((group["label"] > 0.5).any())) + q + ".",
        "- Validation-selected KNN/ReaFNN post-fusion weights: " + q
        + show(calibration.get("selected_knn_weight"), 1) + "/" + show(calibration.get("selected_reafnn_weight"), 1) + q + ".",
        "- Highest Stage-2 contexts on the selected route:",
    ])
    stage2 = sort_frame(group.loc[group["retro_rank"] == int(chosen["retro_rank"])], STAGE2_ORDER).head(8)
    lines.extend(markdown_table(
        ["fusion", "KNN", "ReaFNN", "max sim.", "ReaFNN ctx.", "source/label", "reagent", "solvent"],
        [[
            show(row.get("stage2_post_fusion_score")), show(row.get("stage2_knn_score")),
            show(row.get("stage2_reafnn_score")), show(row.get("knn_similarity_max")),
            show(row.get("reafnn_context_score")),
            ",".join(name for name, active in (
                ("KNN", value(row, "from_baseline_knn") > 0.5),
                ("ReaFNN-gen", value(row, "from_reafnn_generated") > 0.5),
                ("novel", value(row, "from_reafnn_novel") > 0.5),
                ("exact", value(row, "label") > 0.5),
            ) if active) or "-",
            text(row.get("reagent_norm"), 36), text(row.get("solvent_norm"), 36),
        ] for _, row in stage2.iterrows()],
    ))

    lines.extend(["", "#### Stage 3: final XGB-LTR ranking", ""])
    lines.append("- Stable tie order: final score, route rank, route probability, Stage-2 initial score, KNN similarity, reagent, then solvent.")
    lines.extend(markdown_table(
        ["final", "route", "final score", "Stage-2 fusion", "evaluation", "reagent", "solvent"],
        [[
            str(int(row["final_rank"])), str(int(row["retro_rank"])), show(row.get("xgb_score"), 5),
            show(row.get("stage2_post_fusion_score")),
            "exact" if value(row, "label") > 0.5 else str(row.get("label_type", "-")),
            text(row.get("reagent_norm"), 34), text(row.get("solvent_norm"), 34),
        ] for _, row in group.sort_values("final_rank", kind="mergesort").head(10).iterrows()],
    ))

    temp = record["temperature"]
    vector = record["route_gnn_temperature_only"]
    lines.extend(["", "#### Temperature and plotting record", ""])
    if temp["official_support"]:
        lines.append(
            "- Temperature-support row: final rank " + q + str(temp["candidate_final_rank"]) + q
            + "; predicted/gold/absolute error = " + q + show(temp["predicted_temperature_c"], 2)
            + " C / " + show(temp["gold_temperature_c"], 2) + " C / " + show(temp["absolute_error_c"], 2) + " C" + q
            + "; within +/-5/10/20 C = " + q + str(temp["within_5c"]) + "/"
            + str(temp["within_10c"]) + "/" + str(temp["within_20c"]) + q + "."
        )
    else:
        lines.append("- No valid exact-system temperature row in this slate; it does not contribute to temperature metrics.")
    lines.append(
        "- Selected route R-GNN vector is temperature-only, not an XGB-LTR feature: dimension " + q
        + str(vector["dimension"]) + q + ", L2 norm " + q + show(vector["l2_norm"]) + q
        + ", mean/min/max " + q + show(vector["mean"]) + "/" + show(vector["min"]) + "/" + show(vector["max"]) + q + "."
    )
    lines.append("- Untruncated rows and the full 128D graph vector are retained in Experiment/current_mainline_case_examples_seed0_20260905/cases.json.")
    lines.append("")
    return lines


def main() -> None:
    args = arguments()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    body: list[str] = []

    for family, display in FAMILIES:
        scored = args.run_root.resolve() / family / "knn_xgb" / "non_oracle" / "test_scored.csv"
        rebuilt_file = args.run_root.resolve() / family / "knn_xgb" / "non_oracle" / "result.json"
        routes_file = args.route_root.resolve() / family / "route_cache.json"
        retained_file = args.seed_record.resolve() / "compact" / "prosys" / "seed_0" / family / "result.json"
        for path in (scored, rebuilt_file, routes_file, retained_file):
            if not path.exists():
                raise FileNotFoundError(path)

        frame = rank_frame(pd.read_csv(scored))
        rebuilt = json.loads(rebuilt_file.read_text(encoding="utf-8"))
        retained = json.loads(retained_file.read_text(encoding="utf-8"))
        routes = route_cache(routes_file)
        eligible_samples = {index for index, record in routes.items() if len(record.get("routes", [])) == 10}
        chosen_cases = select_cases(frame.loc[frame["sample_index"].isin(eligible_samples)].copy(), MIDDLE_RANK[family])
        if len(chosen_cases) != 3:
            raise RuntimeError(family + " did not yield three cases")

        body.extend(["## " + display, ""])
        calibration = rebuilt.get("model", {}).get("stage2_protocol", {}).get("reafnn_post_fusion_calibration", {})
        body.append("- Rebuilt Seed-0 KNN/ReaFNN fusion: " + str(calibration.get("selected_knn_weight")) + "/" + str(calibration.get("selected_reafnn_weight")) + ".")
        body.append("")
        for position, (selection, chosen) in enumerate(chosen_cases, start=1):
            sample = int(chosen["sample_index"])
            group = frame.loc[frame["sample_index"] == sample].copy()
            route = routes.get(sample)
            if route is None:
                raise KeyError(family + " missing route sample " + str(sample))
            case_id = display + "-C" + str(position)
            record = case_record(family, display, case_id, selection, chosen, group, route, rebuilt, retained)
            cases.append(record)
            index.append({
                "case_id": case_id, "family": family, "selection": selection, "sample_index": sample,
                "reaction_id": str(chosen["reaction_id"]), "retro_rank": int(chosen["retro_rank"]),
                "final_rank": int(chosen["final_rank"]),
                "route_covered": int((group["route_match"] > 0.5).any()),
                "exact_system_in_pool": int((group["label"] > 0.5).any()),
                "selected_exact": int(value(chosen, "label") > 0.5),
            })
            body.extend(render_case(record, group, route, rebuilt))

    q = chr(96)
    header = [
        "# ProSys Representative Current-Mainline Cases",
        "",
        "## Scope and provenance",
        "",
        "This Seed-0 reproduction snapshot of the maintained parallel KNN/ReaFNN plus XGB-LTR mainline "
        "was generated on 2026-09-05 for traceability, case analysis, and figure preparation. It does not replace "
        "the separately retained three-seed headline aggregate.",
        "",
        "- Six maintained reaction families are included, with three selected test slates per family.",
        "- Stage 1 uses persisted family-specific top-10 routes only.",
        "- Stage 2 uses parallel product-Morgan KNN and ReaFNN generation with family validation-selected post-fusion.",
        "- Stage 3 ranks through fixed 52D tabular XGB-LTR. The 128D R-GNN vector belongs only to the separate temperature XGBoost branch.",
        "- Cases use exactly ten persisted Stage-1 routes for consistent graphics; they target exact hits at rank 1 and 10 per family, with rank 3 or 5 distributed as the middle stratum.",
        "- Gold route, condition, yield, and temperature values are evaluation annotations, never inference inputs.",
        "",
        "## Case index",
        "",
    ]
    header.extend(markdown_table(
        ["case", "family", "selection", "route", "final", "route covered", "exact in pool"],
        [[
            row["case_id"], dict(FAMILIES)[row["family"]], row["selection"], str(row["retro_rank"]),
            str(row["final_rank"]), str(bool(row["route_covered"])), str(bool(row["exact_system_in_pool"])),
        ] for row in index],
    ))
    header.extend([
        "",
        "Durable plotting sidecars: " + q + "Experiment/current_mainline_case_examples_seed0_20260905/cases.json" + q
        + " and " + q + "Experiment/current_mainline_case_examples_seed0_20260905/selection.csv" + q + ".",
        "",
    ])
    provenance = {
        "schema_version": 1,
        "run_root": str(args.run_root.resolve()),
        "route_root": str(args.route_root.resolve()),
        "retained_seed0_record": str(args.seed_record.resolve()),
        "selection_policy": {
            "primary": ["Exact rank 1", "Exact rank 3 or 5", "Exact rank 10"],
            "fallback": ["ranking miss", "Stage-2 pool miss", "Stage-1 route miss"],
            "final_rank_order": FINAL_ORDER,
        },
        "warning": "Case snapshot only; headline performance remains the retained three-seed aggregate. "
        "The final tie sort is deterministic, whereas GPU graph-network training can vary slightly in temperature outputs.",
    }
    (output_dir / "cases.json").write_text(json.dumps({"provenance": provenance, "cases": cases}, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    pd.DataFrame(index).to_csv(output_dir / "selection.csv", index=False)
    (output_dir / "README.md").write_text(
        "# Current-mainline case records" + chr(10) + chr(10)
        + "Compact plotting-ready Seed-0 trace for 18 test slates. The temporary full scoring tables may be removed after validation." + chr(10),
        encoding="utf-8",
    )
    args.output_md.resolve().write_text(chr(10).join(header + body).rstrip() + chr(10), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "output_md": str(args.output_md.resolve()), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
