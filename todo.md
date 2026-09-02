# ProSys TODO

Updated: `2026-09-02`

## Active

- Keep the six-family parallel KNN + ReaFNN post-fusion workflow as the only
  maintained Stage 2/3 mainline.
- Treat `Experiment/stage2_parallel_post_fusion_20260901.md` and the current
  section of `CURRENT_RESULTS.md` as the authoritative promoted record.
- Run a matched parallel multi-seed study, temperature aggregate, and paired
  baseline/ablation controls when compute is available; do not borrow the
  historical serial multi-seed values for those claims.
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
- Kept XGB-LTR in a fixed 52-feature non-graph ranking space and R-GNN features
  in the separate temperature regressor only.
- Archived retired candidate-aware Stage 3 ranking probes under
  `Experiment/legacy_stage3/`.
- Segregated local-only exploratory records and scripts from maintained files.
