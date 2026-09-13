# ProSys TODO

Updated: `2026-09-13`

User-directed freeze: no further experiments or split changes are scheduled.
The unfinished items below are deferred, not an active training queue. Completed
code repairs and aggregate evidence are retained; existing results are not replaced.

## Active

- New project-completion work is in progress under
  `Experiment/project_completion_20260913/PLAN.md`. Whole-side canonicalization
  is fixed in shared features and preprocessing; corrected condition-split
  audit passes. All six paired Diels-Alder sensitivity fits and replays passed;
  Sys@10 is 26.47 +/- 0.38% with the legacy helper and 27.43 +/- 1.72% after repair.
  This is not a new six-family result; support identities differ across arms.
- Prepared Stage-1 text/bin audit found 14610 empty family training pairs and
  80 validation pairs. Export now rejects empty pairs and records indices;
  versioned nonempty copies and independent full-tensor verification are done.
  Applying the chosen validation policy is still required before training.
  USPTO augmented inputs have none.
  All 39640 family validation pairs match their binary tensors exactly.
- Repeated same-seed R-GNN fits differed under default CUDA algorithms but
  matched tensor-for-tensor with deterministic algorithms. The maintained
  runner now enables deterministic graph fitting/inference with separate
  caches. A complete corrected Diels-Alder seed-0 refit matches 125600 candidate
  rows and predictions exactly; six-family metric promotion remains pending.
- USPTO raw train/val audit finds zero exact overlaps with condition tests,
  but 98 with condition validation (82 train, 16 val). Resolve clean-validation
  subset versus base retraining before expanding Stage-1 three-seed work.
  New Beckmann seed-0 pilot stopped deliberately; original models unchanged.
  The read-only preview retains 3750 validation reactions / 5504 condition rows.
  See `Experiment/project_completion_20260913/VALIDATION_DECISION.md`.

- The six-item evidence plan is complete; see
  `Experiment/mainline_evidence_completion_20260913/COMPLETION_REPORT.md`.
  Keep the historical promoted temperature result distinct from the newly
  identity-matched reconstruction; do not mix their values in one paired claim.
- Keep historical Diels-Alder models distinct from the versioned helper fix
  and its sensitivity rerun; do not attribute new-code inputs to old artifacts.

- Keep the six-family parallel KNN + ReaFNN post-fusion workflow as the only
  maintained Stage 2/3 mainline.
- Treat `Experiment/stage23_parallel_post_fusion_multiseed_20260903/`,
  `Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/`,
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
- Keep `stage2_KNN/` as a documented import-compatibility shim; all maintained
  Stage-2 implementation changes belong in `stage2_ReaFNN/`.

## Completed

- All 18 mainline reconstructions and row-level replays passed. Sys@k agrees
  with promoted results for every family/seed. New exact-support temperature
  MAE is `11.82 +/- 0.33 C`, versus `13.93 +/- 0.38 C` without R-GNN.
- Completed condition-memory seen/unseen and context-library diagnostics,
  exhaustive failure accounting, 18 fully traced first-hit-correct examples,
  and six cold full-product inference measurements.
- Thirty tests and the requirement-level completion gate passed, including
  candidate score alignment, stale route/split rejection and concurrent-cache
  writer rejection. Real cache reuse leaves model bytes/mtimes unchanged.
- Completed and replay-audited all 18 strict ReaFNN-only controls:
  Sys@10 `36.24 +/- 0.27%`; full-minus-control `+7.53 pp`.
- Added a read-only product-query CLI. A cached-route smoke matched 200
  candidate identities, 52 LTR feature values, ranking scores and temperatures.
  Fresh EditRetro decoding through temperature prediction also ran successfully.
- Added cache-mismatch and shell command-wiring tests. The shell workflow tests
  record commands in a temporary repository; they do not retrain Stage 1.
- Promoted the parallel KNN + ReaFNN post-fusion Stage 2 implementation.
- Completed the matched fixed-Stage-1 parallel seeds-0/1/2 study, including
  conditional temperature aggregation and compact source provenance.
- Kept XGB-LTR in a fixed 52-feature non-graph ranking space and R-GNN features
  in the separate temperature regressor only.
- Completed the paired current-parallel Stage-2 ReaFNN-removal control: KNN-only
  plus a separately re-trained 52-feature XGB-LTR reaches `53.39 +/- 0.00%`
  candidate recall and `39.86 +/- 2.08%` Sys@10, versus `54.26 +/- 0.15%` and
  `43.77 +/- 0.60%` for the full pool.
- Completed the protocol-matched current-parallel Stage-3 three-seed control:
  aggregate accounting agrees and the reported Sys@10 effect is `+7.74 pp`.
  Historical compact records do not establish candidate identity equality.
- Completed the protocol-matched current-parallel temperature three-seed control:
  ranked-system metrics and support counts agree, and the retained results show that
  the 128D R-GNN route representation reduces conditional temperature MAE by
  `2.43 C` and improves within-10 C accuracy by `7.53 pp`.
- Archived retired candidate-aware Stage 3 ranking probes under
  `Experiment/legacy_stage3/`.
- Segregated local-only exploratory records and scripts from maintained files.
