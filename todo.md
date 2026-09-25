# ProSys TODO

Updated: `2026-09-25`

## Accepted Release

The user has accepted the corrected 50K mainline and complete baseline/ablation
suite. No additional trick search or FULL retraining is queued. Publish source,
tests and aggregate evidence to GitHub; keep papers, raw data, per-query outputs
and model weights local. Current release index:
`Experiment/release_50k_20260925/README.md`.
The no-push statements below describe the earlier audit phase only; the user
has now explicitly requested GitHub publication.

## Current Base Decision

- User selected filtered USPTO-50K from scratch, with no FULL checkpoint restore.
  `scripts/run_stage1_50k_from_scratch.py --train` performs source verification,
  raw/transformed-identity filtering, augmentation, full binary/membership audit,
  then fresh base training in `Experiment/stage1_50k_from_scratch_20260924/`.
- The 50K base completed at epoch 50 / 16,500 updates; best is epoch 49,
  validation loss 3.573. Best/last provenance and new expert admission passed.
  The initially admitted expert queue has since been stopped on a new
  augmentation-fidelity finding. The audit/repair/admission and all-query guard
  preflight now pass. The restarted
  `stage1_50k_fidelity_v2_expert_multiseed_20260924` six-family three-seed study
  is now certified, including one explicitly admitted same-weight decode recovery.
  Historical FULL tables/checkpoints remain unpromoted diagnostic records.
- The earlier request to choose FULL retraining versus fixed FULL is superseded.
  No FULL retraining is requested. The completed FULL input audits remain evidence
  of the original issue, not scientific admission of the new 50K model.

## Historical Repair Context
The user reopened Stage 1 expert-seed training on 2026-09-15. Its 18-job queue
was terminated after confirmed post-augmentation train/test overlap.
Strict-option input membership audits have passed; model retraining is pending.
Existing results are retained for diagnosis, not certified as leakage-free.

## Final Verification Status

- All-18 expert certification and independent mean/sample-SD checks are complete.
  The final Diels-Alder decoder recovery is explicit in its separate admission;
  original failure logs are retained. Macro Route@10 is 55.42 +/- 0.39%, distinct
  from fixed-expert-seed-1 55.86%. Do not restart either historical controller.
- The isolated six-family cold-query cost measurement and independent audit
  are complete. One query per family is a deployment smoke, not throughput or
  population latency; report the load-inclusive wall time and sampled memory.
- The root main text and SI are published locally with verified original backups.
  Updated evidence includes 455 numeric cells, 56 paragraphs, 13 method cells,
  four quantitative figures and new expert/cost Tables S29/S30. Final PDFs have
  18/24 pages; all 12 SI contents entries match. Main Figures 1 and 2 remain fixed.
- Post-publication regression passed 252 project tests and eight decoder tests.
  Final source-binding and requirement checks are documented in
  `Experiment/final_release_50k_20260925/FINAL_CHECK.md`.
  No GitHub push is part of this final audit.
- Preserve all valid best/last checkpoints and fixed 3,860 test identities.
  Keep existing disk guards. No invalid-checkpoint deletion has been authorized.

The full downstream comparison, six-family query diagnostics, 18 fresh cases,
six-family deployment checks and auxiliary paper-field replay are complete.
Use `CURRENT_RESULTS.md` and their versioned evidence receipts, not the older
progress snapshots below, to decide whether a new experiment is needed.

## Earlier Progress Snapshots (Superseded)

The following entries retain the sequence of repairs and former pending work.
They are not instructions to rerun already completed experiments.

- Complete `scripts/audit_50k_query_diagnostics.py` on the final six-family
  grid. It rebuilds exact-system labels and route/pool/ranking-failure counts
  from fresh retained rows, checks saved query annotations, and exports
  condition-memory seen/unseen and context-availability subgroups. Do not reuse
  the historical diagnostic tables as scratch-50K results.
- Complete the fresh auxiliary paper-field replay for all six families:
  `scripts/audit_50k_paper_auxiliary.py` independently recounts direct
  Condition@k and replays the validation fusion grid without fitting. These
  fields are not included in the system-metric comparison CSV. The private
  manuscript replacement map also flags fixed-seed versus expert-seed scope,
  new case provenance, dependent figures, and removal of the obsolete separate
  headline/reconstructed temperature versions once final fresh evidence passes.
- The combined fresh-result collector has verified 25 model/seed records each
  for Beckmann, Chan-Lam, acylation, alkylation and Buchwald-Hartwig.
  After the remaining Diels-Alder family
  studies finish, run it without a partial
  `--families` selection using explicit 50K roots. Its output covers downstream
  comparisons only; Stage 1 three-seed uncertainty and manuscript completion
  still need their separate final checks.
- Storage was relieved by native two-week-expiry Git garbage pruning:
  8.36 GiB expired unreachable payload removed; about 14 GiB remains free
  immediately after Buchwald-Hartwig training (now about 13 GiB after the next
  checkpoint pair). Seventeen experts are complete; the final Diels-Alder
  seed-2 job is running. Keep space for its best/last and downstream outputs,
  plus working space. Preserve all checkpoints/data and the unchanged admission
  guards. No invalid-checkpoint deletion was performed. Receipt:
  `Experiment/storage_cleanup_20260925/receipt.json`.
- Same-seed full Stage 2/3 repeat is complete on new Beckmann routes, with exact
  candidate/model/temperature reproduction. Do not repeat this merely to select
  a better score; the pending work is the remaining families and final reports.
- New case exporter has passed a real three-case Beckmann run. Once all bundles
  are ready, export six families to a new directory, check links and intermediate
  parity, then replace root `example.md`; it currently labels its FULL cases as
  historical and links the new Beckmann subset.
- Product-only deployment now uses completed 50K bundles and guarded decoding;
  Beckmann fixed-query parity and fresh end-to-end smoke both pass. Extend this
  parity check to each remaining family after its retained bundle is available.
  Do not treat one passed query as a family-wide performance estimate.
  The cost entrypoint now requires explicit fresh artifact/output roots and an
  idle-training/GPU preflight. Real isolated cost collection remains deferred
  until the current queues finish; the live-queue rejection was tested.
- The live fixed-seed-1 route handoff waits for fully verified expert jobs and
  then generates guarded validation routes with the same selected checkpoint.
  New root: `Experiment/stage1_50k_downstream_routes_20260924/`. Its waiting state
  is not published route evidence. Downstream reruns must use paired test/val
  routes from this root, never historical defaults.
- Local main/SI drafts now visibly distinguish historical FULL scores from the
  ongoing scratch-50K study; tables and main Figure 1 are unchanged. Replace
  numerical results only after new experiments pass full replay and provenance
  checks; remove the temporary draft notice only at that point.
- September 24 augmentation fidelity: preserve failed receipts in
  `Experiment/stage1_augmentation_identity_20260924/`; distinguish stereochemical
  serialization drift from split overlap. Do not resume the interrupted expert
  checkpoints as clean models. Base train/validation full identity scan passes.
  New copies remove 35 train/five validation augmentations and pass independent
  copy verification. Stock test decoding is diagnostic; an identity-preserving
  augmentation guard is being applied equally to base and expert inference.
- Post-augmentation fault tracing is complete for all observed expert/condition
  held-out collisions; six independent binary witnesses are confirmed.
  The full base augmented audit and standalone strict expert-copy/chemical
  audits are complete. The strict base copy and independent retained-text/tensor
  checks are complete. The rebuilt base full chemical audit and all six combined
  expert/base checks pass with zero protected held-out overlaps. The September 24
  decision replaces the FULL option with a scratch-50K base. Bind its completed
  checkpoint, new combined audit and unchanged condition validation protocol
  before admitting expert training. See
  `Experiment/stage1_split_repair_20260915/README.md`.
- The original base full scan found zero exact original condition-test matches and zero
  base train/validation overlap, but three transformed expert-test reactions
  in base validation. Strict refiltering has removed 85 train and 19 validation
  reactions (850/190 augmented pairs) from new copies; their full audit passes,
  but no base checkpoint is thereby repaired.
  Do not train these historical FULL copies; continue the selected scratch-50K
  study and re-evaluate dependent models instead.
- Corrected baseline repeats must rebuild B3/B4 export packages with content
  manifests and use new output roots. The repeat entrypoint now rejects stale
  completed outputs/configurations and evaluates B1 on this study's routes.
  These safeguards are tested, but corrected scientific baseline runs still
  depend on the admitted Stage 1 models and regenerated route caches.
- Blocking update, 2026-09-15: actual augmented expert-training reactions
  overlap condition validation and test reactions in all six families.
  The expert-seed queue has been terminated; preserve its checkpoints as diagnostic
  artifacts, not valid independent-test results. See
  `Experiment/final_release_audit_20260915/augmented_split_audit.json`.
  Clean post-augmentation memberships and reevaluate affected experiments;
  excluding only the 98 USPTO/validation overlaps is insufficient.
- After clean-input admission, restart the six-family seeds 0/1/2 study in
  a new versioned study directory; do not resume the contaminated queue.
  Verify each best-checkpoint
  decode against the original evaluation manifest, and aggregate Route@1/3/5/10
  with sample SD. Update the SI only after complete results are verified.
- Final manuscript/code checks and their boundaries are recorded in
  `Experiment/final_release_audit_20260915/REPORT.md`: 61 source tests, 18 full
  plus 36 paired replays, 375 numeric document cells and DOCX render checks.
- New project-completion work is in progress under
  `Experiment/project_completion_20260913/PLAN.md`. Whole-side canonicalization
  is fixed in shared features and preprocessing; corrected condition-split
  audit passes. All six paired Diels-Alder sensitivity fits and replays passed;
  Sys@10 is 26.47 +/- 0.38% with the legacy helper and 27.43 +/- 1.72% after repair.
  This is not a new six-family result; support identities differ across arms.
- Prepared Stage-1 text/bin audit found 14610 empty family training pairs and
  80 validation pairs. Export now rejects empty pairs and records indices;
  versioned nonempty copies and independent full-tensor verification are done.
  The now-invalidated expert-seed study used these earlier copies with original
  validation membership and retained the known pretraining/validation overlap.
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
