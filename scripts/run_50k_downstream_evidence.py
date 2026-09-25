#!/usr/bin/env python3
"""Run fresh downstream evidence on admitted, fixed-seed 50K expert routes.

One family/seed per invocation; no implicit historical inputs or partial resume.
The expensive scratch tables are removed only after compact evidence is hashed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prosys_shared.cache_integrity import fingerprint, require_manifest, save_manifest
from scripts.run_mainline_evidence import child, sha, write_json


def admit_routes(route_study, family):
    from prosys_shared.mainline import split_file_for_family
    from scripts.run_verified_mainline import verify_route_manifest
    from scripts.stage1_route_admission import require_paired_checkpoints, verify_guard
    from scripts.summarize_stage1_multiseed import verify_job

    protocol_path = route_study / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    receipt_path = route_study / "receipts" / (family + ".json")
    receipt = json.loads(receipt_path.read_text())
    if (protocol.get("seed") != 1 or family not in protocol.get("families", []) or
            receipt.get("seed") != 1 or receipt.get("family") != family):
        raise ValueError("Expected an admitted fixed-seed-1 family route pair")
    if not protocol.get("source_sha256"):
        raise ValueError("Missing route-publication source bindings")
    for name, digest in protocol["source_sha256"].items():
        source = (ROOT / name).resolve()
        source.relative_to(ROOT)
        if sha(source) != digest:
            raise ValueError("Route-publication source changed: " + name)
    expert_study = Path(protocol["expert_study"])
    expert = verify_job(expert_study, family, 1)
    if receipt.get("expert_verification") != expert:
        raise ValueError("Expert verification differs from published receipt")
    completion = expert_study / "seed_1" / family / "completion.json"
    if sha(completion) != receipt["expert_completion_sha256"]:
        raise ValueError("Expert completion changed after route publication")
    inputs = {"route_protocol": protocol_path, "route_receipt": receipt_path,
              "expert_completion": completion}
    caches = {}
    for split in ("train", "val", "test"):
        inputs["split_" + split] = split_file_for_family(ROOT, family, split)
        if sha(inputs["split_" + split]) != receipt["split_sha256"][split]:
            raise ValueError("Condition split changed: " + split)
        if split == "train":
            continue
        cache = route_study / split / family / "route_cache.json"
        verify_route_manifest(cache, inputs["split_" + split], family)
        if verify_guard(cache) != receipt["guarded_routes"][split]:
            raise ValueError("Guarded routes changed: " + split)
        caches[split] = cache
        inputs[split + "_routes"] = cache
        inputs[split + "_guard"] = cache.parent / "augmentation_guard.json"
    checkpoint_sha = require_paired_checkpoints(caches["test"], caches["val"])
    if checkpoint_sha != receipt["checkpoint_sha256"] or checkpoint_sha != expert["best_sha256"]:
        raise ValueError("Paired routes do not use the admitted expert checkpoint")
    inputs["checkpoint"] = Path(json.loads(caches["test"].read_text())["checkpoint"])
    return inputs


def binding_for(route_study, family, seed):
    inputs = admit_routes(route_study, family)
    for directory in ("prosys_shared", "stage2_KNN", "stage2_ReaFNN", "stage3_XGBoost"):
        for source in (ROOT / directory).rglob("*.py"):
            inputs["source/" + str(source.relative_to(ROOT))] = source
    for name in ("run_50k_downstream_evidence.py", "run_mainline_evidence.py",
                 "run_verified_mainline.py", "run_stage23_mainline_non_oracle.py"):
        inputs["source/scripts/" + name] = ROOT / "scripts" / name
    from importlib.metadata import version
    config = {"family": family, "seed": seed, "expert_seed": 1,
              "protocol": "50k_fixed_routes_exact_controls_v1",
              "packages": {p: version(p) for p in ("numpy", "pandas", "torch", "xgboost")}}
    return fingerprint(config, inputs)


def evidence_files(destination):
    return {str(p.relative_to(destination)): p for p in destination.rglob("*")
            if p.is_file() and p.name not in {"evidence_manifest.json", "evidence_manifest.json.tmp", "run.log"}}


def require_separate_paths(route_study, study, scratch):
    protected = [ROOT / "data", ROOT / "stage1_retrosynthesis", route_study]
    paths = [study, scratch]
    for output in paths:
        for other in protected + [p for p in paths if p != output]:
            if output == other or output in other.parents or other in output.parents:
                raise ValueError("Study, scratch and protected input paths must not overlap")
    if study == scratch:
        raise ValueError("Study and scratch paths must differ")


def execute(args):
    route_study, study, scratch_root = (Path(p).resolve() for p in
                                      (args.route_study, args.study_root, args.scratch_root))
    require_separate_paths(route_study, study, scratch_root)
    destination = study / "compact" / f"seed_{args.seed}" / args.family
    scratch = scratch_root / f"seed_{args.seed}" / args.family
    study.mkdir(parents=True, exist_ok=True)
    with (study / f".{args.family}.seed_{args.seed}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = binding_for(route_study, args.family, args.seed)
        manifest = destination / "evidence_manifest.json"
        if destination.exists() and any(destination.iterdir()):
            require_manifest(manifest, binding, evidence_files(destination))
            print("Verified existing completed evidence:", destination, flush=True)
            return
        if scratch.exists():
            raise RuntimeError("Existing scratch must be inspected, not silently resumed: " + str(scratch))
        if shutil.disk_usage(ROOT).free < 8 * 1024**3:
            raise RuntimeError("Less than 8 GiB free before downstream training")
        destination.mkdir(parents=True, exist_ok=True)
        write_json(binding, destination / "input_binding.json")
        child(args.family, args.seed, scratch, destination,
              route_root=route_study / "test", validation_route_root=route_study / "val", verified=True)
        if binding_for(route_study, args.family, args.seed) != binding:
            raise ValueError("Inputs or sources changed during downstream fitting")
        source_manifest = scratch / args.family / "cache_manifest.json"
        shutil.copy2(source_manifest, destination / "pipeline_cache_manifest.json")
        save_manifest(manifest, binding, evidence_files(destination))
        require_manifest(manifest, binding, evidence_files(destination))
        shutil.rmtree(scratch)
        print("Completed and verified fresh downstream evidence:", destination, flush=True)


def main():
    from prosys_shared.mainline import FAMILY_ORDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-study", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILY_ORDER, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
