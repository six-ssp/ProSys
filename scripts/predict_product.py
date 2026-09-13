#!/usr/bin/env python3
"""Read-only product-to-system inference using a retained evidence model bundle.

Family selects deployed artifacts; it is not a learned input feature. No gold
route, condition, yield or temperature is accepted by this command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inference_features(frame):
    """Label-free equivalent of the training table's molecular feature block."""
    import numpy as np
    from prosys_shared.mainline import (
        SUPPORT_FEATURE_COLUMNS_V2, PRODUCT_DESCRIPTOR_COLUMNS_V2,
        _ensure_text_series, _map_unique_strings, _canonical_from_existing_or_source,
        normalize_condition_labels, canonicalize_smiles, canonicalize_reaction_side,
        count_condition_tokens, molecule_graph_descriptors, stable_sort_candidate_frame,
    )
    frame = frame.copy()
    if frame.empty:
        return frame
    for column in SUPPORT_FEATURE_COLUMNS_V2:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0).astype(np.float32)
    frame["reaction_id"] = _ensure_text_series(frame, "reaction_id")
    for column in ("reagent_norm", "solvent_norm"):
        frame[column] = _map_unique_strings(_ensure_text_series(frame, column), normalize_condition_labels)
    for column, source, canonicalizer in (
        ("product_canonical", "product", canonicalize_smiles),
        ("route_canonical", "reactants", canonicalize_reaction_side),
    ):
        frame[column] = _canonical_from_existing_or_source(frame, source_column=source,
            existing_column=column, canonicalizer=canonicalizer)
    frame["num_reagents"] = frame.reagent_norm.map(count_condition_tokens).astype(np.int32)
    frame["num_solvents"] = frame.solvent_norm.map(count_condition_tokens).astype(np.int32)
    frame["route_component_count"] = frame.route_canonical.map(lambda s: len(s.split('.')) if s else 0).astype(np.int32)
    frame["reactants_length"] = _ensure_text_series(frame, "reactants").str.len().astype(np.int32)
    descriptors = {p: molecule_graph_descriptors(p) for p in frame.product_canonical.unique()}
    matrix = np.vstack([descriptors[p] for p in frame.product_canonical]).astype(np.float32)
    for i, column in enumerate(PRODUCT_DESCRIPTOR_COLUMNS_V2):
        frame[column] = matrix[:, i]
    return stable_sort_candidate_frame(frame)


def validate_feature_sources(repo_root, artifact_root):
    from prosys_shared.cache_integrity import file_sha256
    artifact_root, repo_root = Path(artifact_root), Path(repo_root)
    manifest_path = artifact_root / 'runtime_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Model bundle lacks a feature-source manifest; use a verified compatible bundle')
    manifest = json.loads(manifest_path.read_text())
    files = ('prosys_shared/features.py', 'prosys_shared/mainline.py',
             'prosys_shared/condition_modeling.py', 'prosys_shared/product_memory.py',
             'stage2_ReaFNN/knn_condition_selector.py', 'stage2_ReaFNN/reafnn_selector.py',
             'stage3_XGBoost/reaction_gnn_features.py', 'stage3_XGBoost/xgb_reranker.py')
    for name in files:
        if manifest.get('source_sha256', {}).get(name) != file_sha256(repo_root / name):
            raise ValueError(f'Model feature-source mismatch: {name}; use matching code or a refitted bundle')
    intervention = artifact_root / 'intervention.json'
    if intervention.exists() and json.loads(intervention.read_text()).get('mode') != 'corrected':
        raise ValueError('Experimental legacy feature override is not a deployable corrected bundle')


def readonly_builder(repo_root, family, artifact_root, device):
    from prosys_shared.cache_integrity import file_sha256
    from prosys_shared.mainline import split_file_for_family
    from stage2_ReaFNN.knn_condition_selector import KNNContextPoolBuilder
    from stage2_ReaFNN.reafnn_selector import ReaFNNSelector, build_default_context_library

    artifact_root = Path(artifact_root)
    validate_feature_sources(repo_root, artifact_root)
    provenance = json.loads((artifact_root / "provenance.json").read_text())
    if provenance["family"] != family:
        raise ValueError("Requested family does not match model bundle")
    train = split_file_for_family(repo_root, family, "train")
    if file_sha256(train) != provenance["splits"]["train"]:
        raise ValueError("Training condition library differs from the fitted model")
    bundle = artifact_root / "bundle"
    for relative, expected in provenance["models"].items():
        if file_sha256(bundle / relative) != expected:
            raise ValueError(f"Model artifact checksum mismatch: {relative}")
    calibration = json.loads((bundle / "reafnn/post_fusion_calibration.json").read_text())
    protocol = calibration["protocol"]
    # No neural artifact path is passed to the training-oriented constructor.
    builder = KNNContextPoolBuilder(repo_root=repo_root, family=family,
        top_k=protocol["knn_top_k"], max_contexts=protocol["max_contexts"],
        prefilter_contexts=protocol["knn_contexts"], retrieval_mode=protocol["retrieval_mode"],
        fpsize=4096, radius=2, sparse_similarity=True)
    builder.reaffn_selector = ReaFNNSelector(artifact_dir=bundle / "reafnn",
        context_library=build_default_context_library(train), device=device)
    config = builder.reaffn_selector.config
    if not config.enable_independent_post_fusion or config.fpsize != builder.fpsize or config.radius != builder.radius:
        raise ValueError("Only the maintained parallel fingerprint configuration is supported")
    builder.post_fusion_calibration = calibration
    builder._post_fusion_selected_weight = float(calibration["selected_knn_weight"])
    return builder


def generate_routes(product, family, output, device, checkpoint=None):
    from stage1_retrosynthesis.build_route_cache import (
        resolve_checkpoint, dataset_name, run_interactive, aggregate_routes,
    )
    from prosys_shared.cache_integrity import file_sha256
    output = Path(output).resolve()
    checkpoint = resolve_checkpoint(ROOT, family, checkpoint).resolve()
    databin = ROOT / "data/editretro/datasets" / dataset_name(family) / "aug10/data-bin"
    output.mkdir(parents=True, exist_ok=True)
    input_file = output / "input_products.txt"
    input_file.write_text(product + "\n")
    generation = output / "generation.txt"
    run_interactive(repo_root=ROOT, databin=databin, checkpoint=checkpoint,
        input_file=input_file, output_file=generation, aug=10, topk=10,
        repos_beam=5, token_beam=2, mask_beam=1, device=device,
        batch_size=64, buffer_size=2000, max_tokens=4000)
    ranked = aggregate_routes(generation, num_reactions=1, aug=10, beam_size=10,
        n_best=10, score_alpha=0.1, processes=2)[0]
    total = sum(score for _, score in ranked) or 1.0
    routes = [{"reactants": reactants, "retro_rank": i + 1,
               "retro_score": score, "retro_probability": score / total}
              for i, (reactants, score) in enumerate(ranked)]
    return routes, {"mode": "fresh_editretro_decoding", "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint), "aug": 10, "topk": 10}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, help="Target product SMILES only")
    parser.add_argument("--family", required=True, help="Select a deployed family expert")
    parser.add_argument("--artifact_root", required=True, type=Path, help="Directory containing bundle/ and provenance.json")
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing output is never overwritten")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--route_cache", type=Path, help="Optional frozen Stage 1 predictions; skips decoding")
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.output = args.output.resolve()
    import pandas as pd
    import torch
    from rdkit import Chem
    from prosys_shared.mainline import canonicalize_smiles, FAMILY_ORDER
    from prosys_shared.route_cache import RouteRecord
    from prosys_shared.cache_integrity import file_sha256
    from prosys_shared.evidence import ranked_frame
    from stage3_XGBoost.reaction_gnn_features import augment_table_with_reaction_gnn_features
    from stage3_XGBoost.xgb_reranker import score_table_with_xgb
    if args.family not in FAMILY_ORDER:
        parser.error("Unknown deployed family")
    if Chem.MolFromSmiles(args.product) is None:
        parser.error("Invalid product SMILES")
    product = canonicalize_smiles(args.product)
    if not product:
        parser.error("Empty product")
    if args.output.exists():
        parser.error("Output directory already exists; use a new directory")
    validate_feature_sources(ROOT, args.artifact_root)
    args.output.mkdir(parents=True)
    timings = {}

    def measured(name, function):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        start = time.perf_counter()
        value = function()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        timings[name] = time.perf_counter() - start
        return value

    started = time.perf_counter()
    if args.route_cache:
        cache = json.loads(args.route_cache.read_text())
        if cache.get("family") != args.family:
            raise ValueError("Stage 1 route cache family mismatch")
        matching = [r for r in cache["reactions"] if canonicalize_smiles(r["product"]) == product]
        if not matching:
            raise ValueError("Product not present in the supplied Stage 1 cache")
        routes = matching[0]["routes"]
        if any(r["routes"] != routes for r in matching[1:]):
            raise ValueError("Ambiguous cached routes for this product; use fresh decoding")
        route_provenance = {"mode": "frozen_stage1_predictions", "sha256": file_sha256(args.route_cache)}
    else:
        routes, route_provenance = measured("stage1_decode_and_aggregate", lambda: generate_routes(
            product, args.family, args.output / "stage1", "-1" if args.device == "cpu" else args.device.split(":")[-1], args.checkpoint))
    (args.output / "routes.json").write_text(json.dumps(routes, indent=2) + "\n")
    builder = measured("stage2_load_models_and_train_library", lambda: readonly_builder(ROOT, args.family, args.artifact_root, args.device))
    records = [RouteRecord(sample_index=0, reaction_id="query", product=product, family=args.family,
               **{k: r[k] for k in ("reactants", "retro_rank", "retro_score", "retro_probability")}) for r in routes]
    frame = measured("stage2_candidates_and_features", lambda: inference_features(builder._records_to_frame(records, allow_novel=False)))
    prediction = []
    if not frame.empty:
        forbidden = {"label", "temperature_gold", "yield_gold", "route_match", "context_match", "rank_relevance"}
        assert not forbidden.intersection(frame.columns)
        table = args.output / "candidates.csv"
        frame.to_csv(table, index=False)
        bundle = args.artifact_root / "bundle"
        augmented = args.output / "graph_candidates.csv"
        measured("stage3_graph_load_and_encode", lambda: augment_table_with_reaction_gnn_features(
            table_file=table, artifact_dir=bundle / "rgnn", output_file=augmented, device=args.device))
        scored = measured("stage3_xgb_load_rank_and_regress", lambda: score_table_with_xgb(
            augmented, model_file=bundle / "ranker/xgb_ranker.json",
            metadata_file=bundle / "ranker/xgb_ranker_meta.json",
            temperature_model_file=bundle / "temperature/xgb_temperature.json",
            temperature_metadata_file=bundle / "temperature/xgb_temperature_meta.json"))
        scored = ranked_frame(scored)
        scored.to_csv(args.output / "ranked_candidates.csv.gz", index=False, compression="gzip")
        prediction = json.loads(scored.head(10).to_json(orient="records"))
        augmented.unlink()
    report = {"product": product, "family_artifact": args.family, "route_source": route_provenance,
        "artifact_provenance_sha256": file_sha256(args.artifact_root / "provenance.json"),
        "num_routes": len(routes), "num_candidates": len(frame), "top10": prediction,
        "timings_seconds": timings, "wall_seconds_after_imports": time.perf_counter() - started,
        "peak_self_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_child_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_initialized() else 0,
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_initialized() else None,
        "threads": torch.get_num_threads(), "timing_scope": "cold per-query model loading included; imports excluded; no training"}
    (args.output / "prediction.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"prediction": str(args.output / "prediction.json"), "candidates": len(frame), "timings_seconds": timings}))


if __name__ == "__main__":
    main()
