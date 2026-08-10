"""Run stochastic ProSys baselines across seeds with disk-safe retention.

The runner keeps Product-GNN artifacts because they are small. Sequential FNN
and Reaction-GCNN full candidate tables are copied into a compact audit record
and then pruned only after the compact record has been verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from prosys_shared.mainline import parse_families_arg


METHODS = ("product_gnn", "sequential_fnn", "reaction_gcnn")
EXTERNAL_METHODS = ("sequential_fnn", "reaction_gcnn")
DISPLAY_NAMES = {
    "product_naive_bayes": "B1 Product-Bernoulli Naive Bayes",
    "product_gnn": "B2 Product-GNN",
    "sequential_fnn": "B3 EditRetro + Sequential FNN",
    "reaction_gcnn": "B4 EditRetro + Reaction-GCNN",
}
RATE_FIELDS = ("cover", "sys1", "sys3", "sys5", "sys10", "mrr", "ndcg10")
TEMPERATURE_RATE_FIELDS = (
    "temperature_within_5c",
    "temperature_within_10c",
    "temperature_within_20c",
)
MEAN_FIELDS = RATE_FIELDS + ("temperature_mae",) + TEMPERATURE_RATE_FIELDS
COUNT_FIELDS = ("n_test_manifest", "candidate_slates", "missing_candidate_slates", "temperature_n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_cache_hashes(route_root: Path, families: list[str]) -> dict[str, str]:
    return {
        family: _sha256(route_root / family / "route_cache.json")
        for family in families
    }


def _run(command: list[str], *, repo_root: Path) -> None:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    print("[multiseed-baseline] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def _copy_external_compact(work_root: Path, compact_root: Path, families: list[str]) -> list[str]:
    required = ("run_metadata.json", "fusion_selection.json")
    copied: list[str] = []
    for relative in ("run_config.json", "summary.csv", "summary.json"):
        source = work_root / relative
        if not source.exists():
            raise FileNotFoundError(f"Expected external baseline output is missing: {source}")
        destination = compact_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(compact_root)))

    for method in EXTERNAL_METHODS:
        for family in families:
            source_root = work_root / method / family
            destination_root = compact_root / method / family
            for relative in required:
                source = source_root / relative
                if not source.exists():
                    raise FileNotFoundError(f"Expected external baseline output is missing: {source}")
                destination = destination_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.append(str(destination.relative_to(compact_root)))
            for source in sorted((source_root / "artifacts").glob("*.json")):
                destination = destination_root / "artifacts" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.append(str(destination.relative_to(compact_root)))

    for relative in required:
        if not all((compact_root / method / family / relative).exists() for method in EXTERNAL_METHODS for family in families):
            raise RuntimeError("Compact external-baseline retention verification failed.")
    return copied


def _metric_row(record: dict[str, Any], *, method: str, seed: int, family: str) -> dict[str, Any]:
    metrics = dict(record.get("test_metrics") or {})
    temperature = dict(metrics.get("temperature") or {})
    row: dict[str, Any] = {
        "method": method,
        "display_method": DISPLAY_NAMES[method],
        "seed": seed,
        "family": family,
        "n_test_manifest": metrics.get("num_slates", metrics.get("candidate_slates")),
        "candidate_slates": metrics.get("candidate_slates"),
        "missing_candidate_slates": metrics.get("missing_candidate_slates"),
        "cover": metrics.get("pool_coverage"),
        "sys1": metrics.get("system_top1_all"),
        "sys3": metrics.get("system_top3_all"),
        "sys5": metrics.get("system_top5_all"),
        "sys10": metrics.get("system_top10_all"),
        "mrr": metrics.get("system_mrr"),
        "ndcg10": metrics.get("system_ndcg10"),
        "temperature_n": temperature.get("n"),
        "temperature_mae": temperature.get("mae"),
        "temperature_within_5c": temperature.get("within_5c"),
        "temperature_within_10c": temperature.get("within_10c"),
        "temperature_within_20c": temperature.get("within_20c"),
        "selected_route_weight": (record.get("fusion") or {}).get("selected", {}).get("route_weight"),
    }
    return row


def _load_seed_rows(seed_root: Path, *, seed: int, families: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = {
        "product_gnn": seed_root / "product_gnn" / "product_gnn",
        "sequential_fnn": seed_root / "external_compact" / "sequential_fnn",
        "reaction_gcnn": seed_root / "external_compact" / "reaction_gcnn",
    }
    for method, method_root in roots.items():
        for family in families:
            metadata_file = method_root / family / "run_metadata.json"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Missing retained metadata for {method}/{family}: {metadata_file}")
            record = json.loads(metadata_file.read_text(encoding="utf-8"))
            if record.get("method") != method or record.get("family") != family:
                raise ValueError(f"Metadata identity mismatch in {metadata_file}")
            rows.append(_metric_row(record, method=method, seed=seed, family=family))
    return rows


def _macro_row(rows: list[dict[str, Any]], *, method: str, seed: int | None) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method"] == method and (seed is None or row["seed"] == seed)]
    if not method_rows:
        raise ValueError(f"No rows available for {method}, seed={seed}")
    result: dict[str, Any] = {
        "method": method,
        "display_method": DISPLAY_NAMES[method],
        "seed": seed,
        "family": "MACRO-AVG",
        "n_families": len(method_rows),
    }
    for field in MEAN_FIELDS:
        values = [float(row[field]) for row in method_rows if row.get(field) is not None]
        result[field] = sum(values) / len(values) if values else None
    for field in COUNT_FIELDS:
        values = [float(row[field]) for row in method_rows if row.get(field) is not None]
        result[field] = int(sum(values)) if values else 0
    return result


def _mean_std_rows(macro_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in macro_rows:
        by_method[str(row["method"])].append(row)
    for method, method_rows in sorted(by_method.items()):
        row: dict[str, Any] = {
            "method": method,
            "display_method": DISPLAY_NAMES[method],
            "seeds": ",".join(str(item["seed"]) for item in sorted(method_rows, key=lambda item: int(item["seed"]))),
            "n_seeds": len(method_rows),
        }
        for field in MEAN_FIELDS + COUNT_FIELDS:
            values = [float(item[field]) for item in method_rows if item.get(field) is not None]
            row[f"{field}_mean"] = sum(values) / len(values) if values else None
            row[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_rate(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "N/A"
    if std is None:
        return f"{100.0 * mean:.2f}"
    return f"{100.0 * mean:.2f} +/- {100.0 * std:.2f}"


def _format_scalar(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} +/- {std:.2f}"


def _load_deterministic_nb_rows(repo_root: Path, families: list[str]) -> list[dict[str, Any]]:
    root = repo_root / "outputs" / "baselines" / "direct_product_condition_nb_20260727" / "product_naive_bayes"
    rows: list[dict[str, Any]] = []
    for family in families:
        metadata_file = root / family / "run_metadata.json"
        if not metadata_file.exists():
            return []
        record = json.loads(metadata_file.read_text(encoding="utf-8"))
        rows.append(_metric_row(record, method="product_naive_bayes", seed=0, family=family))
    return rows


def _write_report(
    output_root: Path,
    *,
    mean_std_rows: list[dict[str, Any]],
    deterministic_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Multi-Seed Baseline Robustness",
        "",
        "B2 Product-GNN, B3 EditRetro + Sequential FNN, and B4 EditRetro + Reaction-GCNN were independently retrained at seeds 0, 1, and 2 with fixed Stage 1 route caches, formal family splits, validation-only fusion selection, and the fixed full test manifest.",
        "",
        "B1 Product-Bernoulli Naive Bayes is deterministic under its fixed training data and hyperparameters, so it is retained as one deterministic reference rather than pseudo-replicated.",
        "",
        "| Method | Seeds | Candidate recall | Full-system Top-1 | Full-system Top-10 | MRR | nDCG@10 | Conditional temp. MAE (C) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if deterministic_rows:
        nb = _macro_row(deterministic_rows, method="product_naive_bayes", seed=0)
        lines.append(
            "| {method} | deterministic | {cover} | {sys1} | {sys10} | {mrr} | {ndcg} | N/A |".format(
                method=DISPLAY_NAMES["product_naive_bayes"],
                cover=_format_rate(nb["cover"], None),
                sys1=_format_rate(nb["sys1"], None),
                sys10=_format_rate(nb["sys10"], None),
                mrr=_format_rate(nb["mrr"], None),
                ndcg=_format_rate(nb["ndcg10"], None),
            )
        )
    for row in mean_std_rows:
        lines.append(
            "| {method} | {seeds} | {cover} | {sys1} | {sys10} | {mrr} | {ndcg} | {temp} |".format(
                method=row["display_method"],
                seeds=row["seeds"],
                cover=_format_rate(row["cover_mean"], row["cover_std"]),
                sys1=_format_rate(row["sys1_mean"], row["sys1_std"]),
                sys10=_format_rate(row["sys10_mean"], row["sys10_std"]),
                mrr=_format_rate(row["mrr_mean"], row["mrr_std"]),
                ndcg=_format_rate(row["ndcg10_mean"], row["ndcg10_std"]),
                temp=_format_scalar(row["temperature_mae_mean"], row["temperature_mae_std"]),
            )
        )
    lines.extend(
        [
            "",
            "External B3/B4 work directories were pruned only after their per-family run metadata, validation fusion selections, and model metadata had been copied into `seed_<n>/external_compact/`. Product-GNN retains its compact model and top-10 audit outputs directly.",
        ]
    )
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_seed(
    *,
    repo_root: Path,
    input_root: Path,
    output_root: Path,
    route_root: Path,
    validation_route_root: Path,
    families: list[str],
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    top_contexts: int,
    resume: bool,
    keep_external_work: bool,
) -> None:
    seed_root = output_root / f"seed_{seed}"
    complete_file = seed_root / "complete.json"
    if complete_file.exists():
        print(f"[multiseed-baseline] seed={seed} is already complete; skipping.", flush=True)
        return

    direct_root = seed_root / "product_gnn"
    external_work = seed_root / "external_work"
    external_compact = seed_root / "external_compact"
    resume_args = ["--resume"] if resume else []

    _run(
        [
            sys.executable,
            "-B",
            "-m",
            "baseline.run_direct_product_condition_baselines",
            "--families",
            ",".join(families),
            "--methods",
            "product_gnn",
            "--output-root",
            str(direct_root),
            "--route-root",
            str(route_root),
            "--validation-route-root",
            str(validation_route_root),
            "--device",
            device,
            "--max-epochs",
            str(max_epochs),
            "--patience",
            str(patience),
            "--seed",
            str(seed),
            "--top-contexts",
            str(top_contexts),
            *resume_args,
        ],
        repo_root=repo_root,
    )
    _run(
        [
            sys.executable,
            "-B",
            "-m",
            "baseline.external_adapters.run_baselines23",
            "--repo-root",
            str(repo_root),
            "--input-root",
            str(input_root),
            "--output-root",
            str(external_work),
            "--families",
            ",".join(families),
            "--methods",
            "all",
            "--device",
            device,
            "--max-epochs",
            str(max_epochs),
            "--patience",
            str(patience),
            "--seed",
            str(seed),
            "--top-contexts",
            str(top_contexts),
            *resume_args,
        ],
        repo_root=repo_root,
    )

    copied = _copy_external_compact(external_work, external_compact, families)
    _load_seed_rows(seed_root, seed=seed, families=families)
    retention = {
        "seed": seed,
        "external_work_root": str(external_work),
        "retained_compact_root": str(external_compact),
        "retained_files": copied,
        "external_work_pruned": not keep_external_work,
    }
    if not keep_external_work:
        shutil.rmtree(external_work)
    _write_json(seed_root / "retention.json", retention)
    _write_json(seed_root / "complete.json", {"seed": seed, "status": "complete"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, default=Path("outputs/stage1_routes"))
    parser.add_argument("--validation-route-root", type=Path, default=Path("outputs/stage1_routes_validation"))
    parser.add_argument("--families", default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--top-contexts", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-external-work", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    input_root = args.input_root if args.input_root.is_absolute() else repo_root / args.input_root
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    route_root = args.route_root if args.route_root.is_absolute() else repo_root / args.route_root
    validation_route_root = (
        args.validation_route_root
        if args.validation_route_root.is_absolute()
        else repo_root / args.validation_route_root
    )
    families = parse_families_arg(args.families)
    seeds = sorted(set(int(seed) for seed in args.seeds))
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for a multi-seed baseline report.")
    if not input_root.exists():
        raise FileNotFoundError(f"Input package does not exist: {input_root}")
    for root in (route_root, validation_route_root):
        for family in families:
            if not (root / family / "route_cache.json").exists():
                raise FileNotFoundError(f"Missing route cache: {root / family / 'route_cache.json'}")

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "experiment_config.json",
        {
            "protocol": "fixed_stage1_route_cache_multiseed_baselines",
            "methods": list(METHODS),
            "seeds": seeds,
            "families": families,
            "input_root": str(input_root),
            "route_cache_sha256": _route_cache_hashes(route_root, families),
            "validation_route_cache_sha256": _route_cache_hashes(validation_route_root, families),
            "device": args.device,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "top_contexts": args.top_contexts,
            "external_work_retention": "compact JSON audit only unless --keep-external-work is set",
        },
    )

    for seed in seeds:
        _run_seed(
            repo_root=repo_root,
            input_root=input_root,
            output_root=output_root,
            route_root=route_root,
            validation_route_root=validation_route_root,
            families=families,
            seed=seed,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            top_contexts=args.top_contexts,
            resume=args.resume,
            keep_external_work=args.keep_external_work,
        )

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(_load_seed_rows(output_root / f"seed_{seed}", seed=seed, families=families))
    macro_rows = [_macro_row(rows, method=method, seed=seed) for method in METHODS for seed in seeds]
    mean_std_rows = _mean_std_rows(macro_rows)
    deterministic_rows = _load_deterministic_nb_rows(repo_root, families)
    _write_csv(output_root / "per_family_seed_metrics.csv", rows)
    _write_csv(output_root / "macro_by_seed.csv", macro_rows)
    _write_csv(output_root / "macro_mean_std.csv", mean_std_rows)
    if deterministic_rows:
        _write_csv(output_root / "deterministic_b1_per_family.csv", deterministic_rows)
    _write_report(output_root, mean_std_rows=mean_std_rows, deterministic_rows=deterministic_rows)
    print(f"[multiseed-baseline] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
