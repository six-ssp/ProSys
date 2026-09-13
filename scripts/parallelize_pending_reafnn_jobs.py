#!/usr/bin/env python3
"""Shard pending family/seed jobs without editing the frozen experiment code."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "Experiment/stage2_reafnn_only_multiseed_20260913"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_shard(task):
    seed, family = task
    shard = STUDY / "parallel_shards" / f"seed_{seed}" / family
    scratch = ROOT / "outputs/reafnn_only_parallel_scratch_20260913" / f"seed_{seed}" / family
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free; declining a new parallel task")
    shard.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "scripts/run_reafnn_only_ablation.py"),
        "--families", family, "--seeds", str(seed), "--cpu_threads", "2",
        "--output_root", str(shard), "--scratch_root", str(scratch)]
    started = time.time()
    print(f"START parallel {seed}/{family}", flush=True)
    with (shard / "launcher.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_reafnn_only_evidence.py"),
                    "--study", str(shard)], cwd=ROOT, check=True)
    source = shard / "compact" / f"seed_{seed}" / family
    destination = STUDY / "compact" / f"seed_{seed}" / family
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite concurrently written result: {destination}")
    record = json.loads((source / "result.json").read_text())
    original = json.loads((STUDY / "run_manifest.json").read_text())
    shard_manifest = json.loads((shard / "run_manifest.json").read_text())
    assert shard_manifest["source_sha256"] == original["source_sha256"]
    assert record["family"] == family and record["seed"] == seed
    assert all(sha(ROOT / name) == value for name, value in original["source_sha256"].items())
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    receipt = {"seed": seed, "family": family, "elapsed_seconds": time.time() - started,
               "source_hashes_match_frozen_study": True, "cpu_threads": 2,
               "batch_size_and_model_hyperparameters_changed": False}
    (destination / "parallel_execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"MERGED parallel {seed}/{family}", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator_pid", type=int, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cmd = Path(f"/proc/{args.coordinator_pid}/cmdline").read_bytes().replace(b"\0", b" ")
    if b"run_reafnn_only_ablation.py" not in cmd or b"--child" in cmd:
        raise RuntimeError("Coordinator PID is not the expected parent runner")
    if not 1 <= args.workers <= 3:
        parser.error("Use 1-3 additional family workers on the current resource budget")
    paused = False
    try:
        # Pause scheduling, not the already-running child. The original
        # coordinator later compacts that child and skips merged shard results.
        os.kill(args.coordinator_pid, signal.SIGSTOP)
        paused = True
        state = json.loads((STUDY / "status.json").read_text())
        manifest = json.loads((STUDY / "run_manifest.json").read_text())
        active = (state.get("seed"), state.get("family"))
        tasks = [(seed, family) for seed in manifest["seeds"] for family in manifest["families"]
                 if (seed, family) != active and not
                 (STUDY / "compact" / f"seed_{seed}" / family / "result.json").exists()]
        print(f"Coordinator paused; active child preserved: {active}; pending shards: {tasks}", flush=True)
        receipts = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_shard, task) for task in tasks]):
                receipts.append(future.result())
        (STUDY / "parallel_dispatch.json").write_text(json.dumps({
            "status": "complete", "workers": args.workers, "preserved_active_job": active,
            "receipts": receipts}, indent=2) + "\n")
    finally:
        if paused:
            os.kill(args.coordinator_pid, signal.SIGCONT)
            print("Original coordinator resumed", flush=True)


if __name__ == "__main__":
    main()
