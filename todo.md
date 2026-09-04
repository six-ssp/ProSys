# ProSys TODO

Updated: `2026-09-04`

## Active

- Keep the six-family parallel KNN + ReaFNN post-fusion workflow as the only
  maintained Stage 2/3 mainline.
- Treat `Experiment/stage23_parallel_post_fusion_multiseed_20260903/`,
  `Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/`,
  `Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/`, and the
  current section of `CURRENT_RESULTS.md` as the authoritative promoted record.
- Keep the 2026-09-01 seed-0 file as a development record. Do not borrow
  historical serial multi-seed values for current-parallel claims.
- Run only explicitly paired parallel baseline/ablation controls when a new
  numerical component claim is needed.
- Keep Stage 1 test route caches fixed for controlled Stage 2/3 comparisons.

## Maintenance

- Keep `README.md`, `CURRENT_RESULTS.md`, `MAINTENANCE.md`, `todo.md`, and
  `log.md` synchronized whenever the maintained protocol changes.
- Place new non-mainline experiments under `Experiment/local_archive/` locally
  or a named tracked `Experiment/` record if they become reportable.
- Do not reintroduce joint Stage-2/Stage-3 training, wrong-route negatives, or
  candidate-aware GNN ranking into the maintained runner without a new audit.
- Preserve `stage2_KNN/` as a compatibility shim while active code lives in
  `stage2_ReaFNN/`.

## Completed

- Promoted the parallel KNN + ReaFNN post-fusion Stage 2 implementation.
- Completed the matched fixed-Stage-1 parallel seeds-0/1/2 study, including
  conditional temperature aggregation and compact source provenance.
- Kept XGB-LTR in a fixed 52-feature non-graph ranking space and R-GNN features
  in the separate temperature regressor only.
- Completed the paired current-parallel Stage-3 three-seed control: it preserves
  the official Stage-2 pool exactly and measures a `+7.74 pp` Sys@10 effect
  from XGB-LTR reranking.
- Completed the paired current-parallel temperature three-seed control: it
  preserves all ranked systems and conditional support exactly, and shows that
  the 128D R-GNN route representation reduces conditional temperature MAE by
  `2.43 C` and improves within-10 C accuracy by `7.53 pp`.
- Archived retired candidate-aware Stage 3 ranking probes under
  `Experiment/legacy_stage3/`.
- Segregated local-only exploratory records and scripts from maintained files.
