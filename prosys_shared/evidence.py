"""Candidate-level identity, ordering and temperature-support audit helpers."""

import hashlib
import json

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = (
    "sample_index", "reaction_id", "product", "reactants", "reagent_norm", "solvent_norm",
)
TIE_ORDER = (
    ("retro_rank", True), ("retro_probability", False),
    ("stage2_initial_score", False), ("knn_similarity_sum", False),
    ("reagent_norm", True), ("solvent_norm", True),
)


def ranked_frame(frame, score_column="xgb_score"):
    specs = [("sample_index", True), (score_column, False)] + list(TIE_ORDER)
    specs = [(key, ascending) for key, ascending in specs if key in frame]
    result = frame.sort_values([k for k, _ in specs],
                               ascending=[v for _, v in specs], kind="mergesort").copy()
    result["final_rank"] = result.groupby("sample_index", sort=False).cumcount() + 1
    return result


def identity_table(frame, ordered=False):
    missing = set(IDENTITY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing identity fields: {sorted(missing)}")
    table = frame[list(IDENTITY_COLUMNS)].fillna("").astype(str)
    if not ordered:
        table = table.sort_values(list(IDENTITY_COLUMNS), kind="mergesort")
    return table.reset_index(drop=True)


def identity_hash(frame, ordered=False):
    return hashlib.sha256(identity_table(frame, ordered).to_csv(index=False).encode()).hexdigest()


def require_same_candidates(left, right):
    pd.testing.assert_frame_equal(identity_table(left), identity_table(right))
    return identity_hash(left)


def temperature_support(frame, column="temperature_pred", score_column="xgb_score"):
    ordered = ranked_frame(frame, score_column)
    valid = (ordered["label"].astype(float) > 0.5)
    valid &= np.isfinite(pd.to_numeric(ordered["temperature_gold"], errors="coerce"))
    valid &= np.isfinite(pd.to_numeric(ordered[column], errors="coerce"))
    selected = ordered.loc[valid].groupby("sample_index", sort=False).head(1).copy()
    return selected.sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def require_same_temperature_support(left, right, left_column="temperature_pred", right_column="temperature_pred"):
    a = temperature_support(left, left_column)
    b = temperature_support(right, right_column)
    pd.testing.assert_frame_equal(identity_table(a, ordered=True), identity_table(b, ordered=True))
    np.testing.assert_array_equal(a["temperature_gold"].to_numpy(float), b["temperature_gold"].to_numpy(float))
    return {"count": len(a), "identity_sha256": identity_hash(a, ordered=True),
            "reference_temperature_sha256": hashlib.sha256(
                json.dumps(a["temperature_gold"].tolist(), allow_nan=False).encode()).hexdigest()}
