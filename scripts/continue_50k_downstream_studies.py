#!/usr/bin/env python3
"""Consume newly admitted family routes without retries or historical fallbacks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.experiment_integrity import source_files
from prosys_shared.mainline import parse_families_arg, stage1_route_recall
from scripts.continue_stage1_50k_experts import process_identity, waiting_state
from scripts.run_mainline_evidence import read_json, write_json, sha
from scripts.run_50k_downstream_evidence import admit_routes
from scripts.audit_50k_knn_only_evidence import current_binding
from scripts.summarize_50k_downstream_evidence import verify_run, aggregate


def family_reference(study, routes, family):
    import pandas as pd
    rows, receipts = [], []
    for seed in (0, 1, 2):
        values, receipt = verify_run(study, routes, family, seed)
        rows.extend(values)
        receipts.append(receipt)
    frame, _, family_stats, _ = aggregate(rows, families=[family])
    output = study / "family_reports" / family
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_seed_controls.csv", index=False)
    family_stats.to_csv(output / "controls_mean_std.csv")
    full = frame[frame.arm == "full"]
    reference = {}
    for name in ("cover", "sys1", "sys3", "sys5", "sys10", "mrr", "ndcg10"):
        reference[name + "_mean"] = full[name].mean()
        reference[name + "_std"] = full[name].std(ddof=1)
    reference["route_at_10_mean"] = stage1_route_recall(routes / "test" / family / "route_cache.json")["route_recall_top10"]
    reference["route_at_10_std"] = 0.
    path = output / "mainline_reference.csv"
    pd.DataFrame([reference]).to_csv(path, index=False)
    write_json({"pass": True, "family": family, "scope": "one family, not six-family macro",
        "expert_seed": 1, "downstream_seeds": [0, 1, 2], "verified_runs": receipts}, output / "verification.json")
    return path


def frozen_sources():
    sources = source_files(ROOT)
    for folder in ("stage2_KNN", "stage2_ReaFNN"):
        for p in (ROOT / folder).rglob("*.py"):
            sources[str(p.relative_to(ROOT))] = p
    names = ("continue_50k_downstream_studies.py", "run_mainline_evidence.py", "run_50k_downstream_evidence.py",
        "summarize_50k_downstream_evidence.py", "summarize_mainline_evidence.py", "run_verified_mainline.py",
        "run_stage23_mainline_non_oracle.py", "audit_50k_knn_only_evidence.py", "audit_reafnn_only_evidence.py",
        "audit_50k_baseline_evidence.py", "run_reafnn_only_ablation.py", "run_current_mainline_matched_ablations.py",
        "stage1_route_admission.py", "summarize_stage1_multiseed.py", "continue_stage1_50k_experts.py")
    for name in names:
        sources["scripts/" + name] = ROOT / "scripts" / name
    sources["ablation/reafnn_only_experiment.py"] = ROOT / "ablation/reafnn_only_experiment.py"
    return {str(p.relative_to(ROOT)): sha(p) for p in set(sources.values())}


def require_frozen(bindings):
    if any(sha(ROOT / name) != digest for name, digest in bindings.items()):
        raise ValueError("Downstream sources changed; no automatic continuation")


def run(command, log, environment, bindings, minimum_free_gib=8):
    require_frozen(bindings)
    if shutil.disk_usage(ROOT).free < minimum_free_gib * 1024**3:
        raise RuntimeError(f"Less than {minimum_free_gib} GiB free before downstream command")
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError("No silent command retry: " + str(log))
    with log.open("x") as handle:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=environment,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    require_frozen(bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-study", type=Path, required=True)
    parser.add_argument("--route-controller-pid", type=int, required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--controller-study", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    routes, controller = args.route_study.resolve(), args.controller_study.resolve()
    families = parse_families_arg(args.families)
    if not families or len(set(families)) != len(families):
        raise ValueError("Nonempty unique family list required")
    observed = process_identity(args.route_controller_pid)
    if observed is None:
        raise RuntimeError("Route publication controller must be live at admission")
    command = (Path('/proc') / str(args.route_controller_pid) / 'cmdline').read_bytes().split(b'\0')
    if (not any(Path(os.fsdecode(token)).name == 'prepare_stage1_downstream_routes.py' for token in command if token)
            or str(routes).encode() not in command):
        raise ValueError("PID does not identify the selected route publication controller")
    if controller.exists():
        raise FileExistsError("Use a new controller study; no silent partial resume")
    controller.mkdir(parents=True)
    lock = (controller / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bindings = frozen_sources()
    bindings[str((routes / "protocol.json").relative_to(ROOT))] = sha(routes / "protocol.json")
    write_json({"families": families, "route_study": str(routes), "route_controller_identity": observed,
        "pid": os.getpid(), "workers": args.workers, "threads_per_child": 2,
        "source_sha256": bindings, "scope": "selected families; fixed expert seed 1 and downstream seeds 0/1/2",
        "retry_policy": "none; preserve failed artifacts and require explicit review"}, controller / "protocol.json")
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    python = [sys.executable, "-B", "-u"]
    mainline = ROOT / "Experiment/stage23_50k_evidence_20260924"
    main_scratch = ROOT / "outputs/stage23_50k_evidence_scratch_20260924"
    exports = ROOT / "outputs/baseline_inputs_50k_20260924"
    completed = []
    try:
        for family in families:
            while True:
                require_frozen(bindings)
                status = read_json(routes / "status.json")
                ready = (routes / "receipts" / (family + ".json")).is_file()
                state = waiting_state(ready, status.get("phase") == "failed",
                    process_identity(args.route_controller_pid), observed)
                if state == "ready_for_verification":
                    break
                write_json({"phase": "waiting_for_admitted_routes", "family": family,
                    "observed_route_controller": process_identity(args.route_controller_pid),
                    "checked_at": time.time(), "completed": completed}, controller / "status.json")
                time.sleep(30)
            admit_routes(routes, family)
            write_json({"phase": "mainline", "family": family, "completed": completed}, controller / "status.json")
            def main_job(seed):
                run(python + [ROOT / "scripts/run_50k_downstream_evidence.py", "--route-study", routes,
                    "--study-root", mainline, "--scratch-root", main_scratch, "--family", family, "--seed", str(seed)],
                    controller / "logs" / family / f"mainline_{seed}.log", env, bindings)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(main_job, (0, 1, 2)))
            reference = family_reference(mainline, routes, family)
            run(python + ["-m", "baseline.external_adapters.build_datasets", "--output-root", exports,
                "--route-root", routes / "test", "--validation-route-root", routes / "val",
                "--families", family, "--models", "sequential_fnn,reaction_gcnn"],
                controller / "logs" / family / "baseline_export.log", env, bindings)
            baseline = ROOT / "Experiment/baseline_50k_multiseed_20260924" / family
            no_knn = ROOT / "Experiment/stage2_50k_reafnn_only_20260924" / family
            knn = ROOT / "Experiment/stage2_50k_knn_only_20260924" / family
            knn_scratch = ROOT / "outputs/stage2_50k_knn_only_scratch_20260924" / family
            if any(p.exists() for p in (baseline, no_knn, knn, knn_scratch)):
                raise FileExistsError("Expected new baseline/control roots for " + family)
            write_json(current_binding(routes, family), knn / "preflight_inputs.json")
            jobs = [
                ("baselines", python + ["-m", "baseline.run_multiseed_baselines", "--input-root", exports,
                    "--output-root", baseline, "--route-root", routes / "test", "--validation-route-root", routes / "val",
                    "--families", family, "--seeds", "0", "1", "2", "--device", "cuda",
                    "--max-epochs", "50", "--patience", "7", "--top-contexts", "20"]),
                ("no_knn", python + [ROOT / "scripts/run_reafnn_only_ablation.py", "--families", family,
                    "--seeds", "0,1,2", "--device", "cuda:0", "--cpu_threads", "2", "--output_root", no_knn,
                    "--scratch_root", ROOT / "outputs/stage2_50k_reafnn_only_scratch_20260924" / family,
                    "--route_root", routes / "test", "--post_fusion_validation_route_root", routes / "val",
                    "--mainline_reference", reference, "--mainline_compact_root", mainline / "compact"]),
                ("knn_only", python + [ROOT / "scripts/run_current_mainline_matched_ablations.py", "--families", family,
                    "--seeds", "0,1,2", "--arms", "knn_only", "--route_root", routes / "test",
                    "--post_fusion_validation_route_root", routes / "val", "--output_root", knn,
                    "--scratch_root", knn_scratch, "--mainline_macro_file", reference,
                    "--mainline_compact_root", mainline / "compact", "--cpu_threads", "2", "--keep_scratch"])]
            write_json({"phase": "baselines_and_stage2", "family": family, "completed": completed}, controller / "status.json")
            def comparison_job(item):
                name, command = item
                run(command, controller / "logs" / family / (name + ".log"), env, bindings)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(comparison_job, jobs))
            audits = [
                ("baseline_audit", [ROOT / "scripts/audit_50k_baseline_evidence.py", "--study", baseline, "--route-study", routes]),
                ("no_knn_audit", [ROOT / "scripts/audit_reafnn_only_evidence.py", "--study", no_knn, "--route-root", routes / "test"]),
                ("knn_audit", [ROOT / "scripts/audit_50k_knn_only_evidence.py", "--study", knn, "--scratch", knn_scratch,
                    "--route-study", routes, "--family", family, "--prune-scratch"])]
            for name, command in audits:
                run(python + command, controller / "logs" / family / (name + ".log"), env, bindings,
                    minimum_free_gib=1)
            require_frozen(bindings)
            completed.append(family)
            write_json({"phase": "family_complete", "family": family, "completed": completed}, controller / "status.json")
            print("Verified complete downstream family:", family, flush=True)
        write_json({"phase": "complete", "completed": completed,
            "scope": "configured downstream families only; manuscript/global final audit remains separate"}, controller / "status.json")
    except Exception as exc:
        write_json({"phase": "failed", "completed": completed, "error": str(exc), "pid": os.getpid()}, controller / "status.json")
        raise


if __name__ == "__main__":
    main()
