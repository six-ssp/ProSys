#!/usr/bin/env python3
"""One fresh-process product query per family after training queues exit.

This is a small measured deployment smoke, not a throughput or population
latency benchmark. Each deterministic query is the first frozen test identity.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require_idle():
    names = (b"run_stage1_multiseed.py", b"continue_50k_downstream_studies.py",
             b"run_50k_downstream_evidence.py", b"run_mainline_evidence.py",
             b"fairseq_cli/train.py")
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes()
        except OSError:
            continue
        if any(name in command for name in names):
            raise RuntimeError("Training/reconstruction queue is active; defer inference cost measurement")
    active = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"], text=True).strip()
    if active:
        raise RuntimeError("GPU compute processes are active; defer inference cost measurement")


def main():
    from prosys_shared.mainline import FAMILY_ORDER
    from prosys_shared.cache_integrity import file_sha256 as sha
    from scripts.product_route_inference import bind_routes
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="One completed 50K downstream seed directory containing family bundles")
    parser.add_argument("--output-root", type=Path, required=True, help="New directory; no historical result reuse")
    parser.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=FAMILY_ORDER)
    args = parser.parse_args()
    artifact_root, study = args.artifact_root.resolve(), args.output_root.resolve()
    if study.exists() or study == artifact_root or artifact_root in study.parents or study in artifact_root.parents:
        raise ValueError("Use a new, separate output directory; never overwrite model evidence")
    if len(set(args.families)) != len(args.families):
        raise ValueError("Duplicate requested family")
    require_idle()
    products, bindings = {}, {}
    for family in args.families:
        artifact = artifact_root / family
        bindings[family] = bind_routes(ROOT, artifact, family)
        provenance = json.loads((artifact / "provenance.json").read_text())
        cache = json.loads(Path(provenance["route_paths"]["test"]).read_text())
        products[family] = cache["reactions"][0]["product"]
    study.mkdir(parents=True)
    rows = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    for family in args.families:
        require_idle()
        product = products[family]
        output = study / "cold_inference" / family
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
                "--family", family, "--artifact_root", str(artifact_root / family),
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
        assert data["route_source"]["mode"] == "fresh_guarded_editretro_decoding"
        assert data["route_source"]["checkpoint_sha256"] == bindings[family]["checkpoint_sha256"]
        rows.append({"family": family, "query_rule": "first frozen test identity; product only",
            "routes": data["num_routes"], "candidates": data["num_candidates"],
            "wall_seconds_after_imports": data["wall_seconds_after_imports"],
            **{k: v for k, v in resources.items() if k != "scope"},
            "peak_self_rss_mib": data["peak_self_rss_mib"], "peak_child_rss_mib": data["peak_child_rss_mib"],
            "peak_torch_allocated_mib_stage23": data["peak_torch_allocated_mib"], **data["timings_seconds"]})
    pd.DataFrame(rows).to_csv(study / "cold_inference_cost.csv", index=False)
    hardware = {"scope": "one fresh-process query per requested family; idle-GPU and queue preflight; model loads included; internal component times exclude imports/provenance checks; external wall time includes them; not a throughput benchmark",
        "families": args.families, "artifact_root": str(artifact_root),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
            (Path(__file__), ROOT / "scripts/predict_product.py", ROOT / "scripts/product_route_inference.py")},
        "stage1_gpu_peak": "included in sampled whole-device peak; not allocator-exact. Do not sum self/child RSS peaks",
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(), "cpu_threads_per_query": 2}
    (study / "cold_inference_hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    print("Fresh full-pipeline inference measured for", len(args.families), "families")


if __name__ == "__main__":
    main()
