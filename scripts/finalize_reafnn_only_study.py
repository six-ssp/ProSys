#!/usr/bin/env python3
"""Wait for the isolated study, replay evidence, and write a three-way summary."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "Experiment/stage2_reafnn_only_multiseed_20260913"
FULL = ROOT / "Experiment/stage23_parallel_post_fusion_multiseed_20260903/compact/prosys"
KNN = ROOT / "Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/compact/knn_only"


def values(root, family):
    return [float(json.loads((root / f"seed_{s}" / family / "result.json").read_text())["metrics"]["system_top10_all"])
            for s in (0, 1, 2)]


def fmt(xs):
    return f"{100*statistics.mean(xs):.2f} +/- {100*statistics.stdev(xs):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait_pid", type=int)
    args = parser.parse_args()
    deadline = time.monotonic() + 6 * 3600
    previous = None
    while True:
        state = json.loads((STUDY / "status.json").read_text())
        if state["status"] == "complete":
            break
        current = (state.get("seed"), state.get("family"))
        if current != previous:
            print("Waiting for", current, flush=True)
            previous = current
        if not args.wait_pid or not Path(f"/proc/{args.wait_pid}").exists():
            raise RuntimeError("Study runner stopped before completion; inspect launcher.log")
        if time.monotonic() > deadline:
            raise RuntimeError("Six-hour finalization wait exceeded; training was not terminated")
        time.sleep(15)
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_reafnn_only_evidence.py")], check=True, cwd=ROOT)
    audit = json.loads((STUDY / "independent_evidence_audit.json").read_text())
    assert audit["complete"], "Evidence audit must cover all planned records"
    manifest = json.loads((STUDY / "run_manifest.json").read_text())
    families = manifest["families"]
    assert set(manifest["seeds"]) == {0, 1, 2} and len(families) == 6
    lines = ["# Strict Stage 2 Three-Way Control", "",
        "All six families and seeds 0/1/2 are included. Values are equal-family macro means and sample SD across training seeds, in percent. Stage 1 and data splits are fixed. No mainline promotion is performed.", "",
        "| Family | KNN-only + XGB-LTR | ReaFNN-only + XGB-LTR | Full parallel ProSys | Full minus ReaFNN-only (pp) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    full_family, knn_family, neural_family = [], [], []
    for family in families:
        full, knn, neural = values(FULL, family), values(KNN, family), values(STUDY / "compact", family)
        full_family.append(full)
        knn_family.append(knn)
        neural_family.append(neural)
        delta = 100 * (statistics.mean(full) - statistics.mean(neural))
        lines.append(f"| {family} | {fmt(knn)} | {fmt(neural)} | {fmt(full)} | {delta:+.2f} |")
    full = [statistics.mean(xs[s] for xs in full_family) for s in range(3)]
    knn = [statistics.mean(xs[s] for xs in knn_family) for s in range(3)]
    neural = [statistics.mean(xs[s] for xs in neural_family) for s in range(3)]
    delta = 100 * (statistics.mean(full) - statistics.mean(neural))
    lines += [f"| MACRO-AVG | {fmt(knn)} | {fmt(neural)} | {fmt(full)} | {delta:+.2f} |", "",
        "The ReaFNN-only intervention removes KNN proposals and KNN evidence, then retrains its ranker. Historical context priors and neural features remain. Thus this is a downstream branch-removal comparison, not a pure same-candidate reranking test.", "",
        "Retained ReaFNN-only candidate hashes and replayed evaluation agree for all 18 records. Historical full/KNN-only controls provide compact results, not matching per-candidate files; this summary does not claim a per-query paired bootstrap or identity-matched pool across changed Stage 2 arms.", "",
        "The final outcome is retained regardless of which arm performs best. Detailed Top-1/3/5/10, candidate recall, MRR and nDCG tables are in the study CSVs."]
    (STUDY / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (STUDY / "finalization.json").write_text(json.dumps({"status": "complete", "audited_records": 18,
        "full_minus_reafnn_only_sys10_pp": delta, "mainline_changed": False}, indent=2) + "\n")
    print("Evidence audit and summary complete:", STUDY / "SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
