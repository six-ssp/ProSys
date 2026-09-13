#!/usr/bin/env python3
"""Maintained mainline entrypoint with fail-closed content-bound family caches.

Accepts the same arguments as run_stage23_mainline_non_oracle.py. Legacy or
partial artifacts cannot be silently reused: choose a new output directory or
explicitly request --force_rebuild (which also retrains both neural models).
"""

from __future__ import annotations

from functools import wraps
import fcntl
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prosys_shared.cache_integrity import CacheMismatchError, fingerprint, require_manifest, save_manifest


def output_files(directory):
    return {str(p.relative_to(directory)): p for p in directory.rglob("*")
            if p.is_file() and p.name not in {"cache_manifest.json", "cache_manifest.json.tmp"}}


def verify_route_manifest(cache_path, split_path, family):
    cache = json.loads(Path(cache_path).read_text())
    expected = []
    seen = set()
    with Path(split_path).open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            reaction_id, reactants, product = fields[:3]
            if (reaction_id, product) not in seen:
                seen.add((reaction_id, product))
                expected.append((len(expected), reaction_id, product, reactants))
    actual = [(r["sample_index"], str(r["reaction_id"]), r["product"], r.get("gold_reactants"))
              for r in cache.get("reactions", [])]
    if cache.get("family") != family or not expected or actual != expected:
        raise CacheMismatchError(f"Route cache is not aligned to the current {family} split: {cache_path}")


def bind_run(arguments):
    from prosys_shared.mainline import split_file_for_family
    root = Path(arguments["repo_root"])
    family = arguments["family"]
    inputs = {"split_" + s: split_file_for_family(root, family, s) for s in ("train", "val", "test")}
    inputs["test_routes"] = Path(arguments["route_cache"])
    validation = arguments["post_fusion_validation_route_root"]
    if validation is not None:
        inputs["validation_routes"] = Path(validation) / family / "route_cache.json"
    verify_route_manifest(inputs["test_routes"], inputs["split_test"], family)
    if "validation_routes" in inputs:
        verify_route_manifest(inputs["validation_routes"], inputs["split_val"], family)
    for kind in ("test_routes", "validation_routes"):
        if kind in inputs:
            checkpoint = Path(json.loads(inputs[kind].read_text())["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = root / checkpoint
            inputs[kind + "_checkpoint"] = checkpoint
    for directory in ("prosys_shared", "stage2_KNN", "stage2_ReaFNN", "stage3_XGBoost"):
        for path in (root / directory).glob("*.py"):
            inputs["code/" + str(path.relative_to(root))] = path
    for filename in ("run_verified_mainline.py", "run_stage23_mainline_non_oracle.py"):
        inputs["code/scripts/" + filename] = root / "scripts" / filename
    ignored = {"repo_root", "output_root", "route_cache", "post_fusion_validation_route_root",
               "force_rebuild", "reaffn_force_retrain", "gnn_force_retrain", "reuse_candidate_tables_root"}
    config = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in arguments.items() if k not in ignored}
    config["packages"] = {package: version(package) for package in ("numpy", "pandas", "torch", "xgboost")}
    config["reused_stage2_tables"] = arguments["reuse_candidate_tables_root"] is not None
    return fingerprint(config, inputs)


def verify_reused_stage2(arguments, binding):
    source = Path(arguments["reuse_candidate_tables_root"]) / arguments["family"]
    manifest = source / "cache_manifest.json"
    if not manifest.is_file():
        raise CacheMismatchError("Reusable Stage 2 source has no content manifest; rebuild it with the verified entrypoint")
    stored = json.loads(manifest.read_text())["binding"]
    # Same current inputs/code/checkpoints, but source may use another Stage 3 arm.
    require_manifest(manifest, {**stored, "inputs": binding["inputs"]}, output_files(source))
    stage2_keys = ("family", "top_k", "max_contexts", "prefilter_contexts", "fpsize", "radius",
                  "knn_retrieval_mode", "max_train_routes", "max_val_routes", "reafnn_config", "use_reafnn", "seed", "packages")
    for key in stage2_keys:
        if stored["config"][key] != binding["config"][key]:
            raise CacheMismatchError(f"Reusable Stage 2 configuration differs: {key}")


def guard_family(function):
    signature = inspect.signature(function)

    @wraps(function)
    def guarded(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        output_root = Path(bound.arguments["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        lock_path = output_root / ("." + bound.arguments["family"] + ".cache.lock")
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CacheMismatchError(f"Another process is writing this family cache: {lock_path}") from exc
            return execute(bound)

    def execute(bound):
        arguments = bound.arguments
        family_dir = Path(arguments["output_root"]) / arguments["family"]
        manifest = family_dir / "cache_manifest.json"
        binding = bind_run(arguments)
        force = any(arguments[k] for k in ("force_rebuild", "reaffn_force_retrain", "gnn_force_retrain"))
        if family_dir.exists() and any(family_dir.iterdir()) and not force:
            try:
                require_manifest(manifest, binding, output_files(family_dir))
            except CacheMismatchError as exc:
                raise CacheMismatchError(f"Unsafe cache reuse at {family_dir}. Use a NEW output root or --force_rebuild. {exc}") from exc
        if arguments["reuse_candidate_tables_root"] is not None:
            verify_reused_stage2(arguments, binding)
        if force:
            for key in ("force_rebuild", "reaffn_force_retrain", "gnn_force_retrain"):
                arguments[key] = True
        if manifest.exists():
            # Interrupted refresh must not retain an apparently complete manifest.
            manifest.unlink()
        result = function(*bound.args, **bound.kwargs)
        save_manifest(manifest, binding, output_files(family_dir))
        return result

    return guarded


def main():
    from scripts import run_stage23_mainline_non_oracle as pipeline
    pipeline._run_family = guard_family(pipeline._run_family)
    pipeline.main()


if __name__ == "__main__":
    main()
