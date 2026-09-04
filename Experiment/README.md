# Experiment Records

This directory stores compact, auditable records. The promoted implementation
is the parallel KNN + ReaFNN Stage-2 workflow; older serial and exploratory
artifacts are retained only for traceability.

## Current Maintained Record

- `stage23_parallel_post_fusion_multiseed_20260903/`
  - Fixed-Stage-1, six-family parallel evaluation repeated at seeds 0, 1, and 2.
  - The current reportable record: `54.26 +/- 0.15%` candidate recall and
    `43.77 +/- 0.60%` macro Sys@10.
  - Contains compact per-family metrics, per-seed summaries, route-cache hashes,
    and small source-run records; large model/table intermediates were pruned.

- `stage3_parallel_post_fusion_ablation_multiseed_20260904/`
  - Paired Stage-3-only control for the current parallel candidate pool.
  - All 18 family-seed records preserve the full mainline Stage-2 protocol and
    candidate availability exactly, but use deterministic no-XGB-LTR ranking.
  - Macro Sys@10 is `36.03 +/- 0.16%` versus `43.77 +/- 0.60%` for the full
    mainline; the `7.74 pp` gap is therefore a within-pool reranking effect.

- `stage3_temperature_no_rgnn_ablation_multiseed_20260904/`
  - Paired current-mainline temperature-representation control over the same
    six families and seeds 0/1/2.
  - Full temperature XGBoost uses 52 tabular plus 128 R-GNN features; the
    control uses exactly the same 52 tabular fields and no graph features.
  - All 18 pairs match Stage 1/2/3 system metrics and temperature support
    exactly. R-GNN reduces conditional MAE from `13.93 +/- 0.38 C` to
    `11.49 +/- 0.26 C` and improves within-10 C accuracy by `7.53 pp`.

- `stage2_parallel_post_fusion_20260901.md`
  - Fixed-Stage-1, six-family, seed-0 evaluation of the maintained parallel
    KNN/ReaFNN post-fusion procedure.
  - KNN and ReaFNN independently propose train-only historical contexts; a
    family-specific mixture weight is selected on predicted validation routes.
  - Reports 54.10% candidate recall and 43.44% macro Sys@10.
  - This is a verified seed-0 development record; the three-seed directory
    above is the current headline record.

## Historical Serial References

- `stage23_product_morgan_reafnn_multiseed_20260830/`
  - Former serial wide-pool, fixed-Stage-1 three-seed record.
  - Its 44.62 +/- 0.42% Sys@10 result is retained for traceability only and
    must not be mixed with the parallel mainline.

- `current_mainline_matched_ablation_multiseed_20260830/`
  - Former serial KNN-only and no-XGB-LTR component controls.
  - These controls are not paired ablations of the parallel candidate pool.

## Scope Exclusions

Joint Stage-2/Stage-3 optimization, wrong-route negative-sample training, and
route-validity auxiliary supervision were explored locally and then retired.
They are not maintained code, are not current experiments, and are not sources
of reportable mainline results.

## Archived Contents

- `notebooks/`: exploratory notebooks and analysis assets.
- `legacy_outputs/`: old baseline result trees.
- `route_budget/`: route-budget sensitivity experiments.
- `legacy_stage2/`: older Stage-2 code preserved for historical reproduction.
- `legacy_stage3/`: retired ranking probes preserved for historical reproduction.
- `legacy_tools/`: one-off rendering and helper utilities.
- `local_archive/`: ignored local-only exploratory records and scripts.
