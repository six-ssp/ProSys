#!/usr/bin/env python3
"""Compare label-free product inference with retained fixed-route evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def align(frame):
    from prosys_shared.mainline import canonicalize_smiles, canonicalize_reaction_side
    result = frame.copy()
    result["product"] = result["product"].map(canonicalize_smiles)
    result["reactants"] = result["reactants"].map(canonicalize_reaction_side)
    keys = ["product", "reactants", "reagent_norm", "solvent_norm"]
    result[keys] = result[keys].fillna("")
    if result.duplicated(keys).any():
        raise ValueError("Non-unique system identities in product replay")
    return result.set_index(keys).sort_index()


def compare(expected, observed, features, temperature_features):
    import numpy as np
    import pandas as pd
    from prosys_shared.evidence import ranked_frame

    forbidden = {"label", "temperature_gold", "yield_gold", "route_match", "context_match", "rank_relevance"}
    if forbidden.intersection(observed):
        raise ValueError("Gold information found in deployment candidate outputs")
    expected, observed = align(ranked_frame(expected)), align(ranked_frame(observed))
    pd.testing.assert_index_equal(expected.index, observed.index)
    pd.testing.assert_series_equal(expected.final_rank, observed.final_rank, check_dtype=False)
    errors = {}
    for label, columns, tolerance in (("ranking_features", features, 1e-6),
            ("temperature_features", temperature_features, 1e-5),
            ("predictions", ["xgb_score", "xgb_temperature_pred"], 1e-5)):
        a, b = expected[columns].to_numpy(float), observed[columns].to_numpy(float)
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance, equal_nan=True,
                                   err_msg="Deployment mismatch: " + label)
        difference = np.abs(a - b)
        errors[label] = float(np.nanmax(difference)) if difference.size else 0.0
    return {"pass": True, "candidate_count": len(expected), "same_complete_ranking": True,
            "ranker_feature_count": len(features), "temperature_feature_count": len(temperature_features),
            "max_absolute_difference": errors, "gold_columns_absent": True}


def main():
    import pandas as pd
    from prosys_shared.cache_integrity import file_sha256 as sha
    from prosys_shared.mainline import canonicalize_smiles
    from scripts.product_route_inference import bind_routes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite an audit receipt")
    report = json.loads((args.prediction / "prediction.json").read_text())
    bound = bind_routes(ROOT, args.artifact, report["family_artifact"])
    if (report["route_source"]["mode"] != "frozen_stage1_predictions" or
            report["route_source"]["sha256"] != bound["cache_sha256"]["test"]):
        raise ValueError("Exact replay requires the bundle's frozen test routes")
    expected = pd.read_csv(args.artifact / "full_candidates.csv.gz")
    expected = expected.loc[expected.sample_index == args.sample_index].copy()
    if expected.empty or set(expected["product"].map(canonicalize_smiles)) != {report["product"]}:
        raise ValueError("Wrong or empty product evidence")
    embeddings = pd.read_csv(args.artifact / "route_embeddings.csv.gz")
    expected = expected.merge(embeddings, on=["product", "reactants"], how="left", validate="many_to_one")
    observed = pd.read_csv(args.prediction / "ranked_candidates.csv.gz")
    rank = json.loads((args.artifact / "bundle/ranker/xgb_ranker_meta.json").read_text())["feature_columns"]
    temperature = json.loads((args.artifact / "bundle/temperature/xgb_temperature_meta.json").read_text())["feature_columns"]
    result = compare(expected, observed, rank, temperature)
    sources = [Path(__file__), ROOT / "scripts/predict_product.py", ROOT / "scripts/product_route_inference.py"]
    artifacts = [args.artifact / name for name in ("full_candidates.csv.gz", "route_embeddings.csv.gz", "evidence_manifest.json")]
    artifacts += [args.prediction / name for name in ("prediction.json", "ranked_candidates.csv.gz")]
    result.update({"sample_index": args.sample_index, "family": report["family_artifact"],
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "evidence_sha256": {str(p.resolve()): sha(p) for p in artifacts},
        "scope": "One fixed-query deployment replay; not a family performance estimate or fresh-decoding equality claim"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
