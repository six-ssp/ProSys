#!/usr/bin/env python3
"""Read-only fresh-route failure/subgroup audit; no fitting or test selection."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from prosys_shared.evidence import ranked_frame
from prosys_shared.mainline import (FAMILY_ORDER, canonicalize_reaction_side,
    canonicalize_smiles, load_gold_condition_index, load_split_rows,
    normalize_condition_labels, parse_families_arg, split_file_for_family)
from scripts.run_mainline_evidence import read_json, sha, write_json
from scripts.summarize_50k_downstream_evidence import verify_run

CATEGORIES = ("route_miss", "pool_miss", "ranking_miss", "hit")


def replay_queries(cache, frame, gold, train_rows, family, seed):
    product_key = lru_cache(None)(canonicalize_smiles)
    route_key = lru_cache(None)(canonicalize_reaction_side)
    condition_key = lru_cache(None)(normalize_condition_labels)
    expected = [int(q["sample_index"]) for q in cache["reactions"]]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Empty or duplicate query manifest")
    if not set(frame.sample_index).issubset(expected):
        raise ValueError("Candidate outside query manifest")
    products = {product_key(r["product"]) for r in train_rows}
    contexts = {(condition_key(r["reagent_norm"]), condition_key(r["solvent_norm"]))
                for r in train_rows}
    ordered = ranked_frame(frame)
    groups = {int(i): g for i, g in ordered.groupby("sample_index", sort=False)}
    rows = []
    for query in cache["reactions"]:
        index = int(query["sample_index"])
        product = product_key(query["product"])
        bucket = gold[(str(query["reaction_id"]), product)]
        routes = {route_key(r["reactants"]) for r in query["routes"]}
        group = groups.get(index, ordered.iloc[:0])
        exact = []
        for candidate in group.itertuples(index=False):
            if (str(candidate.reaction_id) != str(query["reaction_id"])
                    or product_key(candidate.product) != product):
                raise ValueError("Candidate query identity mismatch")
            route = route_key(candidate.reactants)
            if route not in routes:
                raise ValueError("Candidate outside predicted routes")
            key = (route, condition_key(candidate.reagent_norm),
                   condition_key(candidate.solvent_norm))
            hit = key in bucket.exact_keys
            if float(candidate.label) != int(hit):
                raise ValueError("Candidate exact-system label mismatch")
            if hit:
                exact.append(int(candidate.final_rank))
        first = min(exact) if exact else None
        route_hit = bool(routes & bucket.route_keys)
        category = ("route_miss" if not route_hit else "pool_miss" if first is None
                    else "ranking_miss" if first > 10 else "hit")
        rows.append(dict(family=family, seed=seed, sample_index=index,
            reaction_id=str(query["reaction_id"]), product=product,
            seen_product=product in products,
            context_in_train_library=bool(bucket.context_keys & contexts),
            route_hit=route_hit, candidate_hit=first is not None,
            first_exact_rank=first, failure_category=category,
            **{f"sys{k}": first is not None and first <= k for k in (1, 3, 5, 10)}))
    return pd.DataFrame(rows)


def diagnostic_tables(queries):
    failures, subgroups = [], []
    for (family, seed), q in queries.groupby(["family", "seed"], sort=False):
        if q.sample_index.duplicated().any():
            raise ValueError("Duplicate query rows")
        if not q.failure_category.isin(CATEGORIES).all():
            raise ValueError("Unknown failure category")
        if not (q.failure_category.eq("hit") == q.sys10).all():
            raise ValueError("Failure and Sys@10 disagree")
        counts = q.failure_category.value_counts()
        failures.append(dict(family=family, seed=seed, n=len(q),
                             **{k: int(counts.get(k, 0)) for k in CATEGORIES}))
        for column in ("seen_product", "context_in_train_library"):
            for flag in (False, True):
                group = q[q[column] == flag]
                subgroups.append(dict(family=family, seed=seed, stratifier=column,
                    value=flag, n=len(group), route10=group.route_hit.mean(),
                    cover=group.candidate_hit.mean(),
                    **{f"sys{k}": group[f"sys{k}"].mean() for k in (1, 3, 5, 10)}))
    return pd.DataFrame(failures), pd.DataFrame(subgroups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--route-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", default="all")
    args = parser.parse_args()
    study, routes, output = (p.resolve() for p in
                             (args.study_root, args.route_study, args.output))
    families = parse_families_arg(args.families)
    if not families or len(families) != len(set(families)):
        raise ValueError("Need distinct families")
    if output.exists():
        raise ValueError("Use a new output directory")
    expected = [(f, s) for f in families for s in (0, 1, 2)]
    missing = [(f, s) for f, s in expected if not
        (study / "compact" / f"seed_{s}" / f / "evidence_manifest.json").exists()]
    if missing:
        raise ValueError("Requested report withheld; incomplete runs: " + str(missing))
    inputs, rows, receipts = {}, [], []
    for family in families:
        split_paths = {s: split_file_for_family(ROOT, family, s) for s in ("train", "test")}
        gold = load_gold_condition_index(split_paths["test"])
        train = load_split_rows(split_paths["train"])
        route_path = routes / "test" / family / "route_cache.json"
        cache = read_json(route_path)
        for path in [*split_paths.values(), route_path]:
            inputs[str(path)] = sha(path)
        for seed in (0, 1, 2):
            metrics, receipt = verify_run(study, routes, family, seed)
            directory = study / "compact" / f"seed_{seed}" / family
            paths = [directory / name for name in
                     ("full_candidates.csv.gz", "queries.csv", "evidence_manifest.json")]
            for path in paths:
                inputs[str(path)] = sha(path)
            frame = pd.read_csv(paths[0], float_precision="round_trip",
                                dtype={"reaction_id": str})
            q = replay_queries(cache, frame, gold, train, family, seed)
            saved = pd.read_csv(paths[1], dtype={"reaction_id": str})
            pd.testing.assert_frame_equal(q, saved[q.columns], check_dtype=False)
            for k in (1, 3, 5, 10):
                np.testing.assert_allclose(q[f"sys{k}"].mean(), metrics[0][f"sys{k}"],
                                           rtol=0, atol=1e-10)
            rows.append(q)
            receipts.append(receipt)
            print(f"Verified diagnostics: {family} seed {seed}, {len(q)} queries", flush=True)
    queries = pd.concat(rows, ignore_index=True)
    failures, subgroups = diagnostic_tables(queries)
    # Detect changes during the read-only audit before publishing any report.
    for name, digest in inputs.items():
        if sha(Path(name)) != digest:
            raise ValueError("Input changed during replay: " + name)
    output.mkdir(parents=True)
    queries.to_csv(output / "queries.csv.gz", index=False, compression="gzip")
    failures.to_csv(output / "failures_per_family_seed.csv", index=False)
    subgroups.to_csv(output / "subgroups_per_family_seed.csv", index=False)
    lines = ["# Fresh 50K Query Diagnostics", "",
        "Scope: " + ", ".join(families) + ". All downstream seeds use fixed expert seed 1.",
        "No fitting, filtering of failed queries, or test-driven selection was performed.", "",
        "Each candidate's exact-system label was reconstructed from the persisted test gold index; "
        "query identities and route membership were checked. Query annotations were regenerated "
        "and compared with retained records. Exact-control replay also passed.", "",
        "## Exhaustive Failure Counts", "", "Counts below use downstream seed 1; CSV contains all seeds.", "",
        "| Family | Queries | Route miss | Pool miss | Ranking miss | Hit@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in failures[failures.seed == 1].to_dict("records"):
        lines.append("| " + r["family"] + " | " + " | ".join(
            str(r[k]) for k in ("n", *CATEGORIES)) + " |")
    lines += ["", "Route miss: no reference-matching predicted route. Pool miss: matching route "
        "exists but no exact route-condition candidate. Ranking miss: exact system exists only "
        "below rank 10. Hit: exact system is within top 10. Empty slates remain in the denominator.",
        "", "## Subgroup Scope", "",
        "Seen product means present in this family's condition-training memory, not necessarily "
        "Stage 1 pretraining/fine-tuning. Context availability means at least one reference "
        "reagent-solvent pair is in the training library. Both are evaluation annotations only. "
        "Empty subgroups report NA; no prospective or product-disjoint benchmark is claimed. "
        "Seeds reuse identical queries and are not additional independent test samples.", "",
        "Partial-family reports are not six-family macro estimates. This audit uses maintained "
        "canonicalization and ranking primitives, not an independent chemistry implementation.", ""]
    (output / "README.md").write_text("\n".join(lines))
    outputs = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write_json(dict(requested_scope_complete=True,
        six_family_scope=set(families) == set(FAMILY_ORDER), families=families,
        seeds=[0, 1, 2], verified_runs=receipts, input_sha256=inputs,
        source_sha256={str(Path(__file__).resolve()): sha(Path(__file__))},
        output_sha256=outputs), output / "verification.json")


if __name__ == "__main__":
    main()
