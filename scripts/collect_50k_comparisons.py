#!/usr/bin/env python3
"""Collect explicitly scoped 50K comparisons after retained-evidence replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from prosys_shared.cache_integrity import file_sha256 as sha, require_manifest
from prosys_shared.mainline import FAMILY_ORDER, evaluate_scored_frame_with_manifest, stage1_route_recall
from scripts.summarize_mainline_evidence import flat

SEEDS = {"ProSys": (0, 1, 2), "B1_ProductNB": (0,), "B2_ProductGNN": (0, 1, 2),
    "B3_SequentialFNN": (0, 1, 2), "B4_ReactionGCNN": (0, 1, 2),
    "Without_KNN": (0, 1, 2), "Without_ReaFNN": (0, 1, 2),
    "Without_LTR": (0, 1, 2), "Without_RGNN_temperature": (0, 1, 2)}
METHODS = dict(zip(("product_naive_bayes", "product_gnn", "sequential_fnn", "reaction_gcnn"),
                  ("B1_ProductNB", "B2_ProductGNN", "B3_SequentialFNN", "B4_ReactionGCNN")))
RATE_KEYS = ("route10", "cover", "sys1", "sys3", "sys5", "sys10", "mrr", "ndcg10")
TEMP_KEYS = ("temp_mae", "temp_within_5c", "temp_within_10c", "temp_within_20c")
METRICS = RATE_KEYS + TEMP_KEYS


def read(path):
    return json.loads(Path(path).read_text())


def format_stat(series, model):
    values = series.dropna()
    if values.empty:
        return "NA"
    mean = f"{values.mean():.2f}"
    if len(values) == 1:
        return mean if model == "B1_ProductNB" else mean + " (1 valid seed; SD NA)"
    return mean + f" +/- {values.std(ddof=1):.2f}"


def aggregate(rows, families):
    """B1 has one fit; never manufacture seed variability by duplicating it."""
    if not families or len(families) != len(set(families)):
        raise ValueError("Need unique, nonempty requested families")
    frame = pd.DataFrame(rows)
    expected = {(f, model, seed) for f in families for model, seeds in SEEDS.items() for seed in seeds}
    actual = list(zip(frame.family, frame.model, frame.seed)) if len(frame) else []
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete, duplicated or unexpected family/model/seed grid")
    if (frame.groupby("family").n_queries.nunique() != 1).any() or (frame.n_queries <= 0).any() or (frame.n_queries % 1 != 0).any():
        raise ValueError("Different or invalid query support")
    if (frame.groupby("family").route10.nunique() != 1).any():
        raise ValueError("Different frozen Stage 1 route support")
    if not np.isfinite(frame[list(RATE_KEYS)]).all().all() or not frame[list(RATE_KEYS)].map(lambda v: 0 <= v <= 1).all().all():
        raise ValueError("Invalid system or route metric")
    if (frame[["sys1", "sys3", "sys5", "sys10", "cover", "route10"]].diff(axis=1).iloc[:, 1:] < -1e-12).any().any():
        raise ValueError("Sys/Cover/Route metric ordering is inconsistent")
    if (frame.temp_n < 0).any() or (frame.temp_n > frame.n_queries).any() or (frame.temp_n % 1 != 0).any():
        raise ValueError("Invalid conditional temperature support")
    for _, row in frame.iterrows():
        values = pd.to_numeric(row[list(TEMP_KEYS)], errors="coerce").to_numpy(float)
        if row.temp_n == 0:
            if not np.isnan(values).all():
                raise ValueError("Unsupported temperature metrics must be NA")
        elif (not np.isfinite(values).all() or values[0] < 0 or
              not (0 <= values[1] <= values[2] <= values[3] <= 1)):
            raise ValueError("Invalid supported temperature metrics")
    for family in families:
        for seed in (0, 1, 2):
            pair = frame[(frame.family == family) & (frame.seed == seed)]
            full = pair[pair.model == "ProSys"].iloc[0]
            control = pair[pair.model == "Without_RGNN_temperature"].iloc[0]
            if any(full[k] != control[k] for k in RATE_KEYS + ("temp_n",)):
                raise ValueError("Temperature-only control changed ranking or support")
    macro = frame.groupby(["model", "seed"], sort=False)[list(METRICS)].mean()
    macro["n_queries"] = frame.groupby(["model", "seed"]).n_queries.sum()
    macro["temp_n"] = frame.groupby(["model", "seed"]).temp_n.sum()
    macro["temperature_family_count"] = frame.groupby(["model", "seed"]).temp_mae.count()
    macro["family_count"] = len(families)
    family = frame.groupby(["family", "model"], sort=False)[list(METRICS) + ["temp_n"]].agg(["mean", "std", "count"])
    summary = macro.groupby("model", sort=False)[list(METRICS)].agg(["mean", "std", "count"])
    return frame, macro.reset_index(), family, summary


def row(family, model, seed, metrics, route10, n):
    if metrics["num_slates"] != n:
        raise ValueError("Retained metrics use the wrong query denominator")
    return {"family": family, "model": model, "seed": seed, "n_queries": n,
            "route10": route10, **flat(metrics)}


def collect_family(args, family, output):
    from scripts.summarize_50k_downstream_evidence import compare_metrics, verify_run
    from scripts.audit_50k_knn_only_evidence import current_binding, verify_intervention, verify_library_statistics, library_statistics
    from prosys_shared.mainline import load_split_rows, split_file_for_family

    cache_path = args.routes / "test" / family / "route_cache.json"
    queries = read(cache_path)["reactions"]
    expected = [q["sample_index"] for q in queries]
    route10 = stage1_route_recall(cache_path)["route_recall_top10"]
    rows, sources = [], {}
    for seed in (0, 1, 2):
        values, _ = verify_run(args.mainline, args.routes, family, seed)
        for value in values:
            model = {"full": "ProSys", "no_ltr": "Without_LTR", "no_rgnn_temperature": "Without_RGNN_temperature"}[value.pop("arm")]
            rows.append({**value, "model": model, "route10": route10})
        path = args.mainline / "compact" / f"seed_{seed}" / family / "evidence_manifest.json"
        sources[str(path)] = sha(path)

    # Reuse the maintained independent auditors, not a receipt-presence shortcut.
    for script, command in (("audit_50k_baseline_evidence.py", ["--study", args.baselines / family, "--route-study", args.routes]),
            ("audit_reafnn_only_evidence.py", ["--study", args.no_knn / family, "--route-root", args.routes / "test"])):
        path = ROOT / "scripts" / script
        with (output / (family + "_" + script + ".log")).open("w") as handle:
            subprocess.run([sys.executable, "-B", str(path), *map(str, command)], cwd=ROOT, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        sources[str(path)] = sha(path)
    baseline_receipt = args.baselines / family / "independent_replay.json"
    baseline = read(baseline_receipt)
    if not baseline["pass"] or not baseline["complete"] or baseline["families"] != [family]:
        raise ValueError("Baseline replay did not complete the selected family")
    sources[str(baseline_receipt)] = sha(baseline_receipt)
    for record in baseline["records"]:
        if record["family"] != family:
            raise ValueError("Unexpected baseline family")
        rows.append(row(family, METHODS[record["method"]], record["seed"], record["metrics"], route10, len(expected)))
    no_knn_receipt = args.no_knn / family / "independent_evidence_audit.json"
    if read(no_knn_receipt).get("complete") is not True:
        raise ValueError("No-KNN replay incomplete")
    sources[str(no_knn_receipt)] = sha(no_knn_receipt)
    for seed in (0, 1, 2):
        path = args.no_knn / family / "compact" / f"seed_{seed}" / family / "result.json"
        record = read(path)
        if (record["family"], record["seed"]) != (family, seed):
            raise ValueError("No-KNN result does not match its family/seed path")
        rows.append(row(family, "Without_KNN", seed, record["metrics"], route10, len(expected)))
        sources[str(path)] = sha(path)

    # Raw KNN-only scratch was safely pruned; replay its content-certified
    # retained candidates instead of invoking the original compaction step.
    study = args.knn_only / family
    binding = current_binding(args.routes, family)
    original_audit = read(study / "independent_evidence_audit.json")
    if not original_audit["pass"] or not original_audit["complete"] or original_audit["binding"] != binding:
        raise ValueError("KNN-only original intervention audit does not match current inputs")
    stats = library_statistics(load_split_rows(split_file_for_family(ROOT, family, "train")))
    for seed in (0, 1, 2):
        directory = study / "compact/knn_only" / f"seed_{seed}" / family
        outputs = {str(p.relative_to(directory)): p for p in directory.rglob("*")
                   if p.is_file() and p.name != "evidence_manifest.json"}
        require_manifest(directory / "evidence_manifest.json", binding, outputs)
        frame = pd.read_csv(directory / "test_candidates.csv.gz", float_precision="round_trip")
        verify_intervention(frame)
        verify_library_statistics(frame, stats, training=False)
        replay = evaluate_scored_frame_with_manifest(frame, expected_sample_indices=expected, score_column="xgb_score")
        compare_metrics(replay, read(directory / "result.json")["metrics"], ("cover", "sys1", "sys3", "sys5", "sys10", "mrr", "ndcg10"))
        rows.append(row(family, "Without_ReaFNN", seed, replay, route10, len(expected)))
        sources[str(directory / "evidence_manifest.json")] = sha(directory / "evidence_manifest.json")
    return rows, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mainline", "baselines", "no-knn", "knn-only", "routes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=FAMILY_ORDER)
    args = parser.parse_args()
    for key in ("mainline", "baselines", "no_knn", "knn_only", "routes", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.exists() or any(args.output == p or p in args.output.parents or args.output in p.parents
                                  for p in (args.mainline, args.baselines, args.no_knn, args.knn_only, args.routes)):
        raise ValueError("Use a new output directory separate from evidence inputs")
    if not args.families or len(args.families) != len(set(args.families)):
        raise ValueError("Need unique requested families")
    required = []
    for family in args.families:
        required += [args.routes / "receipts" / (family + ".json"),
            args.baselines / family / "independent_replay.json",
            args.no_knn / family / "independent_evidence_audit.json",
            args.knn_only / family / "independent_evidence_audit.json"]
        required += [args.mainline / "compact" / f"seed_{s}" / family / "evidence_manifest.json" for s in (0, 1, 2)]
    if any(not p.is_file() for p in required):
        raise RuntimeError("Requested-scope report withheld; missing evidence: " + str([str(p) for p in required if not p.is_file()]))
    args.output.mkdir(parents=True)
    rows, sources = [], {str(Path(__file__)): sha(Path(__file__))}
    for family in args.families:
        values, bindings = collect_family(args, family, args.output)
        rows.extend(values)
        sources.update(bindings)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError("Report sources changed during aggregation")
    frame, macro, family_summary, summary = aggregate(rows, args.families)
    frame.to_csv(args.output / "per_family_model_seed.csv", index=False)
    macro.to_csv(args.output / "macro_by_model_seed.csv", index=False)
    family_summary.to_csv(args.output / "per_family_mean_std.csv")
    summary.to_csv(args.output / "macro_mean_std.csv")
    complete = set(args.families) == set(FAMILY_ORDER)
    lines = ["# Verified 50K Model Comparisons", "", "Scope: " + ", ".join(args.families) + ".",
        "Full six-family study: " + str(complete) + ". Partial exports do not replace the full study.", "",
        "Rates are percentages; neural-model uncertainty is mean +/- sample SD across downstream seeds 0/1/2. "
        "B1 is deterministic and fitted once per family: its SD is NA, not zero and not three duplicated runs. "
        "Macro rates weight families equally before seed averaging. All use the fixed expert seed 1.", "",
        "Temperature uses the highest-ranked exact system with finite reference/prediction over the entire slate, "
        "not just Top-10. Missing support is NA. Available-family and per-seed support counts are in the CSVs. "
        "The R-GNN removal is paired on identical support; B3 temperature errors have their own support and are not a paired comparison.", "",
        "| Model | Sys@1 | Sys@3 | Sys@5 | Sys@10 | Temp MAE (C) | Within 10 C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in SEEDS:
        values = macro[macro.model == model]
        cells = []
        for metric in ("sys1", "sys3", "sys5", "sys10", "temp_mae", "temp_within_10c"):
            series = values[metric].dropna() * (1 if metric == "temp_mae" else 100)
            cells.append(format_stat(series, model))
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    receipt = {"requested_scope_complete": True, "six_family_scope": complete, "families": args.families,
        "rows": len(frame), "source_sha256": sources,
        "output_sha256": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        "scope": "downstream mainline, four baselines and four removal controls; not Stage 1 seed uncertainty or manuscript completion"}
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(frame), "six_family_scope": complete}))


if __name__ == "__main__":
    main()
