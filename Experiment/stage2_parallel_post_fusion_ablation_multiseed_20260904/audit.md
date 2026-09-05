# Post-Run Protocol Audit

Date: `2026-09-05`

## Completion

- Result records: `18 / 18` (`6` families x seeds `0, 1, 2`).
- Completion marker: `completion.json` reports `status = complete`.
- Full-system denominator: `3,860` product identities per seed, with `3,833`
  candidate slates and `27` retained no-slate failures.

## KNN-Only Contract

Every compact result passed the following checks:

- `baseline = knn_xgb_stage2_ablation_ranking_only`;
- `stage2_protocol.architecture = knn_only`, `reafnn_enabled = false`, and
  `reafnn_candidate_policy = not_used`;
- product-Morgan KNN with `K=64`, a 64-context prefilter, and a top-20
  route-local context cap;
- reference-split route construction for both training candidate-table fields;
- `ranking_protocol.architecture = xgb_ranker` with exactly 52 non-target-named
  feature columns;
- temperature disabled for this Sys@k-only comparison.

## Pairing Audit

Each KNN-only family-seed result was compared to the same family-seed current
parallel mainline compact record. All 18 pairs exactly match on:

- Stage-1 route-recall object;
- test-manifest size, candidate-slate count, missing-slate count, and
  denominator;
- shared KNN retrieval mode and numerical settings;
- reference-split candidate-table provenance.

Candidate coverage is intentionally not required to match, because ReaFNN is
removed in the control. No temperature metric is interpreted for this arm.

## Result Boundary

The generated `README.md`, CSV tables, and compact `result.json` files are the
machine-readable sources. The paper-ready interpretation is maintained in
[`ablation/current_parallel_stage2_ablation_results_20260905.md`](../../ablation/current_parallel_stage2_ablation_results_20260905.md).
