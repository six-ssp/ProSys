#!/usr/bin/env python3
"""One cold full-product query per family after reconstruction workers exit.

This is a small measured deployment smoke, not a throughput or population
latency benchmark. Each deterministic query is the first frozen test identity.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "Experiment/mainline_evidence_completion_20260913"


def main():
    from prosys_shared.mainline import FAMILY_ORDER
    import pandas as pd
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes()
        except OSError:
            continue
        if b"run_mainline_evidence.py" in command and b"--child" in command:
            raise RuntimeError("Reconstruction worker is still active; do not label concurrent latency isolated")
    rows = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    for family in FAMILY_ORDER:
        product = json.loads((ROOT / "outputs/stage1_routes" / family / "route_cache.json").read_text())["reactions"][0]["product"]
        output = STUDY / "cold_inference" / family
        prediction = output / "prediction.json"
        if not prediction.exists():
            def gpu_memory_mib():
                return float(subprocess.check_output(["nvidia-smi", "--id=0",
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
            baseline_gpu = gpu_memory_mib()
            peak_gpu = baseline_gpu
            output.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with output.with_suffix(".log").open("w") as handle:
                process = subprocess.Popen([sys.executable, str(ROOT / "scripts/predict_product.py"), "--product", product,
                "--family", family, "--artifact_root", str(STUDY / "compact/seed_0" / family),
                    "--output", str(output), "--device", "cuda:0"], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    peak_gpu = max(peak_gpu, gpu_memory_mib())
                    time.sleep(0.2)
                if process.returncode:
                    raise RuntimeError(f"Product inference failed; inspect {output.with_suffix('.log')}")
            resources = {"external_wall_seconds_including_imports": time.perf_counter() - started,
                "gpu_device_baseline_mib": baseline_gpu, "gpu_device_peak_observed_mib": peak_gpu,
                "gpu_monitor_interval_seconds": 0.2,
                "scope": "sampled whole-device memory, including Stage 1; not allocator-exact peak"}
            (output / "invocation_resources.json").write_text(json.dumps(resources, indent=2) + "\n")
        data = json.loads(prediction.read_text())
        resources = json.loads((output / "invocation_resources.json").read_text())
        assert data["route_source"]["mode"] == "fresh_editretro_decoding"
        rows.append({"family": family, "query_rule": "first frozen test identity; product only",
            "routes": data["num_routes"], "candidates": data["num_candidates"],
            "wall_seconds_after_imports": data["wall_seconds_after_imports"],
            **{k: v for k, v in resources.items() if k != "scope"},
            "peak_self_rss_mib": data["peak_self_rss_mib"], "peak_child_rss_mib": data["peak_child_rss_mib"],
            "peak_torch_allocated_mib_stage23": data["peak_torch_allocated_mib"], **data["timings_seconds"]})
    pd.DataFrame(rows).to_csv(STUDY / "cold_inference_cost.csv", index=False)
    hardware = {"scope": "one cold query per family; no reconstruction workers; model loads included, Python imports excluded",
        "stage1_gpu_peak": "included in sampled whole-device peak; not allocator-exact. Do not sum self/child RSS peaks",
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(), "cpu_threads_per_query": 2}
    (STUDY / "cold_inference_hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    print("Cold full-pipeline inference measured for all six families")


if __name__ == "__main__":
    main()
