"""Read-only, checkpoint-bound Stage 1 routing for deployed 50K bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from prosys_shared.cache_integrity import file_sha256 as sha, require_manifest
from scripts.stage1_route_admission import PROTOCOL, verify_guard


def option(argv, name):
    if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
        raise ValueError("Missing or repeated decoding option: " + name)
    return argv[argv.index(name) + 1]


def bind_routes(root, artifact, family, checkpoint=None):
    """Resolve the expert from the fitted bundle, never historical aliases."""
    root, artifact = Path(root).resolve(), Path(artifact).resolve()
    binding_path = artifact / "input_binding.json"
    if not binding_path.is_file():
        raise ValueError("Deploy a completed 50K evidence bundle, not a historical bundle")
    binding = json.loads(binding_path.read_text())
    config = binding.get("config", {})
    if (config.get("protocol") != "50k_fixed_routes_exact_controls_v1" or
            config.get("family") != family or config.get("expert_seed") != 1):
        raise ValueError("Model bundle has an incompatible Stage 1 binding")
    outputs = {str(p.relative_to(artifact)): p for p in artifact.rglob("*")
               if p.is_file() and p.name not in {"evidence_manifest.json", "evidence_manifest.json.tmp", "run.log"}}
    require_manifest(artifact / "evidence_manifest.json", binding, outputs)
    inputs = binding["inputs"]
    for name, digest in inputs.items():
        if name.startswith("source/"):
            path = (root / name.removeprefix("source/")).resolve()
            path.relative_to(root)
            if sha(path) != digest:
                raise ValueError("Fitted model source changed: " + name)
    provenance = json.loads((artifact / "provenance.json").read_text())
    if provenance["family"] != family:
        raise ValueError("Requested family does not match model bundle")
    caches, guards = {}, {}
    for split, key in (("val", "route_validation"), ("test", "route_test")):
        path = Path(provenance["route_paths"][split])
        digest = sha(path)
        if digest != provenance[key] or digest != inputs[split + "_routes"]:
            raise ValueError("Fitted Stage 1 cache changed: " + split)
        verify_guard(path, root)
        guard_path = path.parent / "augmentation_guard.json"
        if sha(guard_path) != inputs[split + "_guard"]:
            raise ValueError("Fitted Stage 1 guard changed: " + split)
        caches[split] = json.loads(path.read_text())
        guards[split] = json.loads(guard_path.read_text())
        if caches[split]["family"] != family or any(caches[split][k] != 10 for k in ("aug", "topk", "n_best")):
            raise ValueError("Incompatible family or Stage 1 decoding settings")
    trained_checkpoint = Path(caches["val"]["checkpoint"]).resolve()
    if trained_checkpoint != Path(caches["test"]["checkpoint"]).resolve():
        raise ValueError("Validation/test caches use different experts")
    selected = Path(checkpoint).resolve() if checkpoint is not None else trained_checkpoint
    if sha(selected) != inputs["checkpoint"]:
        raise ValueError("Checkpoint differs from the expert used to fit this model bundle")
    # Replay the retained, guarded command, replacing only the product file and
    # an optional byte-identical checkpoint relocation. No gold data is passed.
    argv = guards["val"]["argv"]
    if Path(option(argv, "--path")).resolve() != trained_checkpoint:
        raise ValueError("Guard command does not identify the fitted expert")
    for name, expected in (("--aug", "10"), ("--TOPK", "10"), ("--repos-beam", "5"),
                           ("--token-beam", "2"), ("--mask-beam", "1")):
        if option(argv, name) != expected:
            raise ValueError("Unexpected retained decoder setting: " + name)
    return {"checkpoint": str(selected), "checkpoint_sha256": inputs["checkpoint"],
            "argv": argv, "guard_sources": guards["val"]["source_sha256"],
            "cache_sha256": {s: inputs[s + "_routes"] for s in ("val", "test")}}


def cached_routes(path, product, family, binding):
    from prosys_shared.mainline import canonicalize_smiles

    path = Path(path)
    digest = sha(path)
    if digest not in binding["cache_sha256"].values():
        raise ValueError("Route cache is not one of this fitted bundle's bound caches")
    cache = json.loads(path.read_text())
    if cache.get("family") != family:
        raise ValueError("Stage 1 route cache family mismatch")
    matching = [r for r in cache["reactions"] if canonicalize_smiles(r["product"]) == product]
    if not matching:
        raise ValueError("Product not present in the supplied Stage 1 cache")
    routes = matching[0]["routes"]
    if any(r["routes"] != routes for r in matching[1:]):
        raise ValueError("Ambiguous cached routes for this product; use fresh decoding")
    return routes, {"mode": "frozen_stage1_predictions", "sha256": digest,
                    "checkpoint_sha256": binding["checkpoint_sha256"], "guard_protocol": PROTOCOL}


def validate_single_guard(record, sources):
    events = record.get("events", [])
    if (record.get("pass") is not True or record.get("protocol") != PROTOCOL or
            record.get("source_sha256") != sources or record.get("query_count") != 1 or
            record.get("augmented_input_count") != 10 or len(events) != 1):
        raise ValueError("Fresh product decoding lacks valid identity protection")
    event = events[0]
    slots = event.get("replaced_slots", [])
    if (event.get("query_index") != 0 or event.get("augmentation_count") != 10 or
            type(event.get("input_normalized")) is not bool or
            any(type(s) is not int or not 1 <= s < 10 for s in slots) or slots != sorted(set(slots)) or
            record.get("replaced_variant_count") != len(slots) or
            record.get("normalized_input_count") != int(event["input_normalized"])):
        raise ValueError("Fresh product guard has inconsistent per-query evidence")


def generate_routes(root, product, output, device, binding):
    from stage1_retrosynthesis.build_route_cache import aggregate_routes

    root, output = Path(root).resolve(), Path(output).resolve()
    if sha(binding["checkpoint"]) != binding["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed before decoding")
    output.mkdir(parents=True, exist_ok=False)
    input_file = output / "input_products.txt"
    input_file.write_text(product + "\n")
    argv = list(binding["argv"])
    for name, value in (("--input", str(input_file)), ("--path", binding["checkpoint"])):
        option(argv, name)
        argv[argv.index(name) + 1] = value
    receipt = output / "augmentation_guard.json"
    stage1 = root / "stage1_retrosynthesis"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PROSYS_AUGMENTATION_GUARD_RECEIPT=str(receipt))
    env["PYTHONPATH"] = str(stage1 / "fairseq") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "2")
    generation = output / "generation.txt"
    with generation.open("w") as handle:
        subprocess.run([sys.executable, str(root / "scripts/stage1_interactive_guarded.py"), *argv],
                       cwd=stage1, env=env, stdout=handle, check=True)
    guard = json.loads(receipt.read_text())
    validate_single_guard(guard, binding["guard_sources"])
    if sha(binding["checkpoint"]) != binding["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during decoding")
    ranked = aggregate_routes(generation, num_reactions=1, aug=10, beam_size=10,
                              n_best=10, score_alpha=0.1, processes=2)[0]
    total = sum(score for _, score in ranked) or 1.0
    routes = [{"reactants": reactants, "retro_rank": i + 1,
               "retro_score": score, "retro_probability": score / total}
              for i, (reactants, score) in enumerate(ranked)]
    return routes, {"mode": "fresh_guarded_editretro_decoding", "checkpoint": binding["checkpoint"],
        "checkpoint_sha256": binding["checkpoint_sha256"], "aug": 10, "topk": 10,
        "guard_protocol": guard["protocol"], "guard_sha256": sha(receipt),
        "replaced_variant_count": guard["replaced_variant_count"],
        "scope": "Fresh per-product augmentation; not necessarily bit-identical to batched test decoding"}
