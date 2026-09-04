# Audit: Parallel Stage-3 Matched Ablation

## Status

**PASS**. This audit covers the completed current-parallel Stage-3 control in
this directory: six families x seeds `0, 1, 2` = 18 compact result records.
The paired official full-mainline records are in
[`../stage23_parallel_post_fusion_multiseed_20260903/`](../stage23_parallel_post_fusion_multiseed_20260903/README.md).

## Contracts Checked

For every family and seed, the audit confirmed:

- family and seed identifiers are correct;
- the fixed Stage-1 route recall record and 3,860-product test manifest are
  unchanged; each seed has 3,833 candidate slates and 27 retained no-slate
  identities;
- Stage 2 is `knn_reafnn` with product-Morgan KNN, `K=64`, a 64-context
  prefilter, a 20-context cap, and reference-split train/validation tables;
- the full `stage2_protocol` object is exactly equal to the corresponding
  official parallel-mainline record, including independent KNN/ReaFNN
  post-fusion settings and the validation-selected fusion calibration;
- `pool_coverage`, `pool_route_coverage`, `pool_context_coverage`,
  `candidate_slates`, and `missing_candidate_slates` are exactly equal to the
  corresponding full-mainline record;
- Stage 3 is `deterministic_stage1_stage2_prior`, with `learned_parameters=false`;
- temperature is marked `always_enabled=false` because it does not influence
  candidate generation or system ranking in this Sys@k-only control.

## Consequence

Candidate recall is identical to the full parallel mainline for every matched
family-seed pair. The observed macro Sys@10 difference (`43.77 +/- 0.60%` with
XGB-LTR versus `36.03 +/- 0.16%` without it) is therefore a within-pool
reranking effect, not a change in candidate availability.

Raw candidate tables and binary checkpoints were pruned after compact
validation to respect disk limits. The retained `result.json` files, metrics
CSVs, `run_manifest.json`, and `completion.json` are sufficient to recheck the
reported protocol and aggregate results.
