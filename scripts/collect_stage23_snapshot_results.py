#!/usr/bin/env python3
"""Aggregate compact per-family Stage 2/3 result snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prosys_shared.mainline import FAMILY_ORDER, display_family_name


SYSTEM_FIELDS = (
    ("route_at_10", "Route@10"),
    ("candidate_recall", "Candidate recall"),
    ("sys_at_1", "Sys@1"),
    ("sys_at_3", "Sys@3"),
    ("sys_at_5", "Sys@5"),
    ("sys_at_10", "Sys@10"),
)
TEMPERATURE_FIELDS = (
    ("temperature_mae_c", "MAE (deg C)"),
    ("temperature_within_5c", "Within +/-5 C"),
    ("temperature_within_10c", "Within +/-10 C"),
    ("temperature_within_20c", "Within +/-20 C"),
)


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_as_float(row.get(field)) for row in rows]
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _weighted_mean(
    rows: list[dict[str, Any]],
    field: str,
    weight_field: str,
) -> float | None:
    pairs = [
        (_as_float(row.get(field)), int(row.get(weight_field) or 0))
        for row in rows
    ]
    pairs = [(value, weight) for value, weight in pairs if value is not None and weight > 0]
    total_weight = sum(weight for _, weight in pairs)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def _result_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    temperature = metrics.get("temperature") or {}
    route = result["stage1_route_recall"]
    return {
        "family": result["family"],
        "route_at_10": route.get("route_recall_top10"),
        "candidate_recall": metrics.get("pool_coverage"),
        "sys_at_1": metrics.get("system_top1_all"),
        "sys_at_3": metrics.get("system_top3_all"),
        "sys_at_5": metrics.get("system_top5_all"),
        "sys_at_10": metrics.get("system_top10_all"),
        "temperature_n": int(temperature.get("n") or 0),
        "temperature_mae_c": temperature.get("mae"),
        "temperature_within_5c": temperature.get("within_5c"),
        "temperature_within_10c": temperature.get("within_10c"),
        "temperature_within_20c": temperature.get("within_20c"),
    }


def _sort_key(row: dict[str, Any]) -> tuple[int, str]:
    family = str(row["family"])
    try:
        return FAMILY_ORDER.index(family), family
    except ValueError:
        return len(FAMILY_ORDER), family


def _percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}"


def _number(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def _write_overview(
    output_file: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    stage2 = summary["stage2_protocol"]
    lines = [
        "# Stage 2/3 Snapshot Summary",
        "",
        "Generated only from retained result snapshots; large candidate tables are not required.",
        "",
        f"- Families: {summary['family_count']}",
        f"- Baseline identifier: {summary['baseline']}",
        f"- Shared seed: {summary['seed']}",
        f"- KNN representation: {stage2['knn_feature_space']}",
        f"- Stage 2 policy: {stage2['reafnn_candidate_policy']}",
        "",
        "## System Metrics",
        "",
        "| Family | Route@10 | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = " | ".join(_percent(_as_float(row[field])) for field, _ in SYSTEM_FIELDS)
        lines.append(f"| {display_family_name(str(row['family']))} | {values} |")
    macro = summary["macro_average"]
    values = " | ".join(_percent(macro[field]) for field, _ in SYSTEM_FIELDS)
    lines.append(f"| MACRO-AVG | {values} |")
    lines.extend(
        [
            "",
            "## Conditional Temperature Metrics",
            "",
            "Temperature is evaluated on highest-ranked full-system matches with valid labels.",
            "",
            "| Family | N | MAE (deg C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        values = [
            _number(_as_float(row["temperature_mae_c"])),
            _percent(_as_float(row["temperature_within_5c"])),
            _percent(_as_float(row["temperature_within_10c"])),
            _percent(_as_float(row["temperature_within_20c"])),
        ]
        lines.append(
            f"| {display_family_name(str(row['family']))} | {row['temperature_n']} | "
            + " | ".join(values)
            + " |"
        )
    temperature_macro = summary["temperature_macro_average"]
    values = [
        _number(temperature_macro["temperature_mae_c"]),
        _percent(temperature_macro["temperature_within_5c"]),
        _percent(temperature_macro["temperature_within_10c"]),
        _percent(temperature_macro["temperature_within_20c"]),
    ]
    lines.append("| MACRO-AVG | NA | " + " | ".join(values) + " |")
    temperature_pooled = summary["temperature_pooled"]
    values = [
        _number(temperature_pooled["temperature_mae_c"]),
        _percent(temperature_pooled["temperature_within_5c"]),
        _percent(temperature_pooled["temperature_within_10c"]),
        _percent(temperature_pooled["temperature_within_20c"]),
    ]
    lines.append(
        f"| POOLED | {temperature_pooled['temperature_n']} | "
        + " | ".join(values)
        + " |"
    )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_prefix", default="mainline")
    args = parser.parse_args()

    root = args.snapshot_root.resolve()
    output_dir = (args.output_dir or root).resolve()
    output_prefix = str(args.output_prefix).strip()
    if not output_prefix or Path(output_prefix).name != output_prefix:
        parser.error(
            "--output_prefix must be a non-empty filename prefix without path separators"
        )
    result_files = sorted(root.glob("*/result.json"))
    if not result_files:
        raise FileNotFoundError(f"No direct result.json files found under {root}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    baselines = {str(result.get("baseline")) for result in results}
    seeds = {int(result.get("seed")) for result in results}
    protocols = {
        json.dumps(result["model"]["stage2_protocol"], sort_keys=True)
        for result in results
    }
    if len(baselines) != 1 or len(seeds) != 1 or len(protocols) != 1:
        raise ValueError("All snapshots must share the same baseline, seed, and Stage 2 protocol.")

    rows = sorted((_result_row(result) for result in results), key=_sort_key)
    summary = {
        "family_count": len(rows),
        "baseline": baselines.pop(),
        "seed": seeds.pop(),
        "stage2_protocol": json.loads(protocols.pop()),
        "macro_average": {field: _mean(rows, field) for field, _ in SYSTEM_FIELDS},
        "temperature_macro_average": {
            field: _mean(rows, field) for field, _ in TEMPERATURE_FIELDS
        },
        "temperature_pooled": {
            "temperature_n": sum(row["temperature_n"] for row in rows),
            **{
                field: _weighted_mean(rows, field, "temperature_n")
                for field, _ in TEMPERATURE_FIELDS
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{output_prefix}_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_overview(output_dir / f"{output_prefix}_overview.md", rows, summary)


if __name__ == "__main__":
    main()
