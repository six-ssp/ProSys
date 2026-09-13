#!/usr/bin/env python3
"""Export three first-hit-correct cases per family from reconstructed slates."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "Experiment/mainline_evidence_completion_20260913"


def escaped(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def main():
    import numpy as np
    import pandas as pd
    from prosys_shared.mainline import FAMILY_ORDER
    from prosys_shared.evidence import ranked_frame
    from prosys_shared.condition_modeling import route_feature_vector
    from prosys_shared.route_cache import RouteRecord
    from scripts.extract_current_case_examples import select_cases
    from scripts.predict_product import readonly_builder
    if not (STUDY / "replay_audit.json").exists():
        raise RuntimeError("Run the complete retained-row replay before publishing examples")
    output = STUDY / "examples"
    output.mkdir(exist_ok=True)
    lines = ["# Current Parallel ProSys: Detailed Cases", "",
        "Source: the 2026-09-13 evidence reconstruction, seed 0, fixed Stage 1 caches. "
        "Ranking metrics are checked against promoted results in the study report. Temperature predictions "
        "belong to the reconstructed regressors and must not be substituted for historical predictions.", "",
        "Each family has three deterministically selected queries. Top-k labels refer to the FIRST exact "
        "system hit, not any later positive row. These are descriptive cases, not evidence of prospective "
        "chemical feasibility. Full candidate slates, neural token probabilities, sparse fingerprint inputs "
        "and 128-dimensional graph features are retained in the linked compressed JSON files.", "",
        "## Reading the Trace", "",
        "The external molecular input is product SMILES; the deployment configuration chooses the family expert. "
        "Stage 1 produces ranked reactant sets. Product-Morgan KNN (4096 bits, radius 2, cosine, K=64) and "
        "route-conditioned ReaFNN (8218 features, two 512-unit hidden layers, reagent/solvent heads) independently "
        "propose training-library contexts. Validation-selected w mixes their rank priors. The fused top-20 "
        "contexts per route enter 52-feature XGB-LTR and a Stage 1/2 score prior. A separate 52+128-feature "
        "regressor predicts temperature. Gold fields are evaluation-only.", ""]
    index = []
    for fi, family in enumerate(FAMILY_ORDER):
        folder = STUDY / "compact/seed_0" / family
        frame = ranked_frame(pd.read_csv(folder / "full_candidates.csv.gz"))
        embeddings = pd.read_csv(folder / "route_embeddings.csv.gz")
        graph_cols = [f"route_gnn_feat_{i}" for i in range(128)]
        frame = frame.merge(embeddings, on=["product", "reactants"], how="left", validate="many_to_one")
        assert frame[graph_cols].notna().all().all()
        metadata = json.loads((folder / "bundle/ranker/xgb_ranker_meta.json").read_text())
        features = metadata["feature_columns"]
        cache = json.loads((ROOT / "outputs/stage1_routes" / family / "route_cache.json").read_text())
        routes = {r["sample_index"]: r for r in cache["reactions"]}
        builder = readonly_builder(ROOT, family, folder, "cpu")
        selector = builder.reaffn_selector
        selected = select_cases(frame, 3 if fi % 2 == 0 else 5)
        assert len(selected) == 3
        lines += [f"## {family}", ""]
        for ci, (title, chosen) in enumerate(selected, 1):
            sample = int(chosen.sample_index)
            group = frame[frame.sample_index == sample].sort_values("final_rank")
            exact = group[group.label > 0.5]
            first = None if exact.empty else int(exact.final_rank.min())
            if float(chosen.label) > 0.5:
                assert first == int(chosen.final_rank)
            product = str(chosen["product"])
            reactants = str(chosen["reactants"])
            reagent, solvent = selector.predict_token_probabilities(reactants, product)
            raw = route_feature_vector(reactants, product, fpsize=4096, radius=2)
            scaled = selector.scaler.transform(raw[None, :])[0]
            route = RouteRecord(sample_index=sample, reaction_id=str(chosen.reaction_id),
                reactants=reactants, product=product, family=family,
                retro_rank=int(chosen.retro_rank), retro_score=float(chosen.retro_score),
                retro_probability=float(chosen.retro_probability))
            knn_fp = builder._retrieval_fp(route.reactants, route.product)
            similarities = builder._similarities_for_record(route)
            neighbor_indices = [int(i) for i in np.argsort(similarities)[::-1] if similarities[i] > 0][:builder.top_k]
            neighbors = [{"memory_index": i, "training_route_identity": builder.route_keys[i],
                "cosine_similarity": float(similarities[i]), "training_conditions": builder.route_contexts[i]}
                for i in neighbor_indices]
            union = builder._independent_post_fusion_state(route, leave_one_reaction_out=False)
            case = {"family": family, "seed": 0, "sample_index": sample,
                "selection": title, "first_exact_rank": first, "stage1": routes[sample],
                "selected": json.loads(chosen.to_json()),
                "stage2_fusion_calibration": builder.post_fusion_calibration,
                "stage3_ranking_metadata": metadata,
                "reafnn_input": {"dimension": int(raw.size), "raw_nonzero_indices": np.flatnonzero(raw).tolist(),
                    "raw_nonzero_values": raw[raw != 0].tolist(),
                    "scaled_dense": scaled.tolist(), "scaler": "training-fitted; checkpoint stores mean/std",
                    "block_slices_half_open": {"product_fp": [0, 4096], "product_minus_reactant_fp": [4096, 8192],
                        "reactant_descriptors": [8192, 8200], "product_descriptors": [8200, 8208],
                        "descriptor_difference": [8208, 8216], "component_count_and_string_length": [8216, 8218]},
                    "descriptor_order": ["atoms", "bonds", "rings", "aromatic_atoms", "hetero_atoms", "exact_molecular_weight", "TPSA", "fraction_sp3"]},
                "replay_note": "Token probabilities and branch/neighbor traces are read-only CPU replay of retained weights; original candidate scores/graph vectors are retained separately in all_candidates",
                "knn_product_fp": {"dimension": len(knn_fp), "nonzero_indices": np.flatnonzero(knn_fp).tolist(),
                    "nonzero_values_l2_normalized": knn_fp[knn_fp != 0].tolist()},
                "knn_neighbors_replayed": neighbors,
                "branch_union_replayed_before_weight": json.loads(pd.DataFrame(union).to_json(orient="records")),
                "reagent_probabilities": dict(zip(selector.reagent_vocab, reagent.tolist())),
                "solvent_probabilities": dict(zip(selector.solvent_vocab, solvent.tolist())),
                "all_candidates": json.loads(group.to_json(orient="records"))}
            filename = f"{family}_case_{ci}.json.gz"
            with gzip.open(output / filename, "wt") as handle:
                json.dump(case, handle, allow_nan=False)
            rel = (output / filename).relative_to(ROOT)
            index.append({"family": family, "case": ci, "sample_index": sample, "first_exact_rank": first, "file": str(rel)})
            lines += [f"### Case {ci}: {title}", "",
                f"Query identity: sample_index={sample}, reaction_id={chosen.reaction_id}. These IDs are not model features.", "",
                f"Product: `{product}`", "",
                f"Reference reactants (evaluation only): `{routes[sample].get('gold_reactants', '')}`", "",
                f"First exact hit rank: **{first if first is not None else 'absent'}**; candidate count: {len(group)}. "
                f"Selected candidate: rank {int(chosen.final_rank)}, Stage 1 route rank {int(chosen.retro_rank)}.", "",
                f"[Complete machine-readable intermediate values]({rel})", "",
                "#### Stage 1 Routes", "", "| Route rank | Reactants | Score | Normalized score |",
                "| ---: | --- | ---: | ---: |"]
            for route in routes[sample]["routes"]:
                lines.append(f"| {route['retro_rank']} | `{escaped(route['reactants'])}` | {route['retro_score']:.6f} | {route['retro_probability']:.6f} |")
            lines += ["", "#### Stage 2 And Final Ranking", "",
                f"Validation-selected KNN weight: {builder._post_fusion_selected_weight:.2f}. "
                "A and B below are branch rank priors, not calibrated chemical-success probabilities. "
                "LTR is the fused final ordering score, not a yield prediction.", "",
                "| Final rank | Route | Reagents | Solvents | KNN A | Neural B | wA+(1-w)B | LTR | Exact | T (C) |",
                "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for _, r in group.head(10).iterrows():
                lines.append(f"| {int(r.final_rank)} | {int(r.retro_rank)} | {escaped(r.reagent_norm) if pd.notna(r.reagent_norm) else '(none)'} | "
                    f"{escaped(r.solvent_norm)} | {r.stage2_knn_score:.5f} | {r.stage2_reafnn_score:.5f} | "
                    f"{r.stage2_post_fusion_score:.5f} | {r.xgb_score:.5f} | {int(r.label)} | {r.xgb_temperature_pred:.2f} |")
            lines += ["", "#### Selected Candidate Features", "",
                "The following is the complete 52-column LTR input schema, including constant compatibility fields. "
                "The 128 graph coordinates are in the JSON selected row and are used only by the temperature regressor.", "",
                "| Feature | Value |", "| --- | ---: |"]
            for feature in features:
                lines.append(f"| `{feature}` | {float(chosen[feature]):.8g} |")
            lines += ["", f"Selected temperature: {float(chosen.xgb_temperature_pred):.4f} C. "
                f"Reference temperature for this exact system: {chosen.temperature_gold} C. "
                "Official temperature support selects the first exact row with finite reference and prediction over "
                "the ENTIRE slate, not necessarily this illustrative row and not restricted to Top-10.", ""]
    old = ROOT / "example.md"
    archive = ROOT / "Experiment/document_archive_20260913/example_before_first_hit_fix.md"
    if old.exists() and not archive.exists():
        shutil.copy2(old, archive)
    old.write_text("\n".join(lines) + "\n")
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Wrote {len(index)} fully traced cases to example.md; archived previous text")


if __name__ == "__main__":
    main()
