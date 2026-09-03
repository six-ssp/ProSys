# Candidate-aware R-GNN residual: implementation and status

Updated: `2026-08-03`

## Status

This module is an implemented Stage 3 research branch, not part of the
maintained headline pipeline. The official pipeline remains:

```text
Stage 2 candidate pool
  -> no-graph XGBRanker for system ranking
  -> fixed graph-enhanced XGBRegressor for temperature only
```

The candidate-aware residual is evaluated with validation-only fusion selection.
The available Beckmann interaction-model pilot did not meet the required
validation full-system Top-10 accuracy gain, so its selected residual weight was zero. A related
six-family auxiliary-token residual probe also selected no nonzero residual.
Neither probe changes the fixed mainline results or any headline full-system Top-k value.

## Purpose

The original R-GNN produces one 64-dimensional structural vector per
route. Every reagent-solvent candidate attached to the same route consequently
shares that vector. `condition_aware_gnn.py` tests a more discriminative
route-context score: it combines the frozen route-GNN vector with learned
reagent and solvent token representations, so different contexts on one route
can receive different GNN residual scores.

Source files:

- `Experiment/legacy_stage3/condition_aware_gnn.py`
- `Experiment/legacy_stage3/probes/probe_condition_aware_gnn_residual.py`
- `outputs/condition_aware_gnn_probe_20260803/`

## Inputs and vocabulary

For each candidate row, the model reads only:

- the frozen `route_gnn_feat_0` through `route_gnn_feat_63` columns;
- normalized `reagent_norm` and `solvent_norm` tokens;
- the strict binary training label `label`.

The reagent and solvent vocabularies are built from the Stage 3 training table
only. Token index 0 is padding and index 1 is the train-time unknown-token
slot. Multi-token reagent and solvent fields are represented as padded token
lists and pooled by a masked mean. The route-feature mean and standard
deviation are also fitted on the training table only and reused unchanged for
validation and test scoring.

## Network

The default configuration is intentionally lightweight:

- route input: 64 frozen route-GNN features;
- reagent token embedding: 32 dimensions;
- solvent token embedding: 32 dimensions;
- condition projection: `Linear(32 + 32 + 2, 64) -> ReLU -> Dropout(0.15)`;
- route projection: `Linear(64, 64) -> ReLU -> LayerNorm(64)`;
- interaction vector: `concat(route, condition, route * condition, abs(route - condition))`;
- score head: `Linear(256, 96) -> ReLU -> Dropout(0.15) -> Linear(96, 1)`.

The two extra condition inputs are the normalized counts of non-padding reagent
and solvent tokens. The output is one uncalibrated scalar score per
route-context candidate, named `condition_aware_gnn_score_raw`.

## Training

The residual network is trained only on one family's Stage 3 training table.
It uses binary `label` with `BCEWithLogitsLoss`. The positive class weight is
the negative-to-positive ratio, clipped to a maximum of 12.0 to avoid an
unstable long-tail correction. The optimizer is AdamW with learning rate
`1e-3`, weight decay `1e-4`, batch size `1024`, dropout `0.15`, and at most
12 epochs. The current implementation fixes a random seed per run.

This GNN score is never written back as an in-sample XGBoost training feature.
The XGBRanker is trained independently on its no-graph candidate features. This
prevents the residual branch from leaking scores trained on the same target into
the ranker's feature learning.

## Validation-only fusion and gate

For a validation or test candidate slate, raw GNN residuals are standardized
within each `sample_index`:

```text
z_gnn = (gnn_score - mean_in_slate) / std_in_slate
fused_score = xgb_score + alpha * z_gnn
```

`alpha` is searched only on validation data over
`0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30`. The selected nonzero alpha must
improve validation full-system Top-10 accuracy by at least `0.01` over the fixed no-graph XGB-LTR
baseline. full-system Top-1 accuracy and MRR break ties. Only after that decision is the selected
alpha evaluated on the test table. The test labels therefore do not determine
the residual weight or acceptance decision.

## Current evidence and interpretation

The completed interaction-model pilot covers Beckmann. Its best validation
full-system Top-10 accuracy did not exceed the fixed XGB-LTR baseline by the pre-specified margin,
so `alpha = 0` was selected. Its test full-system Top-10 accuracy consequently remained
`29.79%`, identical to the no-graph mainline for that family. The separate
six-family auxiliary-token residual probe also rejected all six nonzero
residuals under the same 0.01 validation full-system Top-10 accuracy gain criterion.

The appropriate current conclusion is therefore:

> Candidate-aware route-context GNN scoring is implemented and evaluated under
> a validation gate, but it has not yet demonstrated a robust ranking gain and
> is not used in the maintained mainline.

Promotion to the official ranker would require a completed six-family run,
pre-registered validation selection, a frozen test evaluation, and an updated
ablation/audit package. Until then, the graph representation is retained only
for the independently selected temperature branch.
