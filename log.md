# ProSys Development Log

## 2026-09-13 - user-directed freeze and repository publication

- User requested publication of completed changes without further training,
  data changes or validation-policy selection. Existing model/results versions
  remain unchanged; unresolved audit limitations remain documented.
- Publication scope: maintained code, regression tests, aggregate experiment
  results and provenance. Exclude local checkpoints, candidate/query rows,
  manuscript drafts and case-analysis source material from Git.

## 2026-09-13 - nonempty expert copies independently verified

- Built six versioned copies without changing original inputs or retained
  token tensors: train 851350, validation 39560, prepared test 39740 pairs.
  Independent hash, full-tensor, lineage and creation-source checks pass.
- Added a training-entry guard against EOS-only data and an explicit data-bin
  override. All 60 regression tests pass. No new training was launched.
- Copies remove empty targets only. Validation-policy selection remains
  unresolved; formal expert three-seed work and new result promotion are paused.
  The fixed 3860 system test queries remain unchanged. No Git push occurred.

## 2026-09-13 - canonicalization sensitivity completed

- All six Diels-Alder legacy/corrected fits completed, retaining the fixed
  762-query denominator. Metric replays and input/source pairing checks pass.
  Mean Sys@10 is 26.47 +/- 0.38% versus 27.43 +/- 1.72%, a +0.96 pp change;
  seed 0 decreases, seeds 1 and 2 increase. No seed is discarded.
- Conditional MAE is 18.99 +/- 0.14 C versus 18.11 +/- 0.36 C, but selected
  support identities differ in seeds 0/1 despite each having 208 queries.
  This is a whole-pipeline repair result, not a same-support temperature-only
  comparison. All disposable scratch was removed after successful compaction.
- The full deterministic repeat, corrected inference parity and expanded
  57-test suite passed. No training worker from this phase remains. The overall
  goal is still incomplete: validation protocol selection, versioned clean
  expert inputs, expert three-seed training and new result promotion remain.

## 2026-09-13 - prepared-input and end-to-end repeatability checks

- Read-only validation preview retains 3750 of 3848 unique reactions and
  5504 of 5863 condition rows after excluding USPTO overlaps. No new selection
  protocol has been chosen or applied. All 39640 validation source/target
  text pairs match their retained binary tensors after dictionary encoding.
- Found and binary-verified empty augmented targets: family train 14610,
  validation 80, prepared test 150. Retained USPTO base train/validation have
  zero empty pairs. Added final paired nonempty filtering with an index
  manifest, and actual connected-component handling in Stage-1 preprocessing.
  A real CLI smoke accepts a mapped cross-dot molecule and removes two empty
  outputs from six augmented pairs. Existing data were not rewritten.
- A complete corrected Diels-Alder seed-0 Stage-2/3 refit matches all 125600
  retained candidate rows, ranking scores, temperatures and R-GNN tensors
  exactly. This is current-environment repeatability, not proof of the sole
  historical drift cause or a new six-family three-seed mainline.
- Added feature-source compatibility checks before read-only inference.
  The corrected cached-product smoke matches 100 chemical candidate identities,
  all 52 LTR inputs and both prediction columns exactly. Its initial diagnostic
  compared literal raw product strings; canonical chemical identity resolves
  that representation-only difference without changing model predictions.
- The expanded 57-test suite passed. Sensitivity aggregation restores the
  scorer's float32 temperature predictions before replaying MAE, avoiding a
  2.86e-9 C decimal-serialization artifact instead of weakening the assertion.
- Stage-1 training remains stopped pending the validation-protocol decision;
  the six-fit Diels-Alder repair sensitivity study continues separately.

## 2026-09-13 - correctness repairs and Stage-1 robustness work

- Added the project-completion plan without declaring it complete. Shared
  reaction-side canonicalization now uses whole-side parsing and connected
  components; preprocessing grouping uses the same helper. Frozen data and
  checkpoint files were not rewritten. Corrected full condition-split audit:
  zero mismatches, invalid sides or pairwise split reaction intersections.
- Added canonicalization, Stage-1 seed-wiring, unmapped-audit and deterministic
  runtime tests. Stage-1 runner isolates new checkpoints, reuses hashed data,
  preserves family-specific token batches, and checks training completion.
- Started Beckmann seed 0, then deliberately stopped the new study after its
  epoch-22/2464-update last checkpoint when a new validation-boundary issue was
  found. Old expert checkpoints and published route caches were untouched.
- Independently scanned 934575 retained filtered USPTO raw rows: zero exact
  condition-test reaction overlaps and zero base train/val intersections;
  98 condition-validation overlaps (82 base train, 16 base val). Whole-side
  parsing also identifies 41 raw invalid rows; their historical preprocessing
  disposition still needs separate tracing. Asked user to choose clean
  validation with fixed base versus filtered-base retraining.
- Four repeated Beckmann seed-0 R-GNN fits isolate current nondeterminism:
  default fits differ; deterministic fits have exactly identical tensors.
  Added scoped deterministic fitting/inference and config-bound caches to
  the maintained runner. No new headline temperature result is claimed yet.
- Started Diels-Alder legacy/corrected helper sensitivity: three seeds per arm,
  both with deterministic R-GNN. New outputs and disposable scratch are in
  a separate study directory. No Git commit/push performed.
- Replayed the retained EditRetro input guard on all 41 exact invalid raw rows:
  every row is rejected. Historical raw-to-augmentation membership remains
  unproven because no per-row historical manifest was retained. Completed a
  public-API deterministic GPU fit/inference repeat test with exact equality.

## 2026-09-13 - six-item evidence plan completed

- Completed 18/18 mainline reconstructions; independently replayed full,
  same-pool no-LTR and exact-support no-R-GNN metrics. Every family/seed Sys@k
  equals the promoted result. The mainline macro remains `43.77 +/- 0.60%`.
- Fresh temperature reconstruction gives `11.82 +/- 0.33 C` MAE and
  `61.58 +/- 0.77%` within-10 C, versus `13.93 +/- 0.38 C` and
  `55.56 +/- 1.23%` in its identity-matched no-R-GNN control. Historical
  promoted temperature values were not silently replaced.
- Added condition-memory-unseen diagnostics (3,569 identities,
  `45.32 +/- 0.62%` macro Sys@10), exhaustive route/pool/ranking failure tables,
  and 18 first-hit-correct cases with KNN neighbors, branch unions, input
  vectors, token heads, 52 tabular and 128 graph features.
- Fixed score-to-candidate alignment during report replay and a pandas Series
  attribute collision in case export; re-exported and independently checked
  every case's actual product and sparse molecular feature vector.
- Six fresh product-query smokes measured cold total latency of 9.91-19.93 s
  including imports, with sampled whole-device GPU peaks of 591-735 MiB.
  One query per family is a measured deployment smoke, not a throughput study.
- Thirty tests and the requirement-level checker passed. The known one-record
  Diels-Alder training representation limitation and missing first-pilot runtime
  source manifest remain explicit. Details: the evidence study's
  `COMPLETION_REPORT.md` and `completion_gate.json`.
- Data-disk scratch was compacted per job; reusable bundles and row-level
  traces are retained locally and excluded from routine Git payloads.
  No Git commit or push was requested or performed in this task.

## 2026-09-13 - read-only inference and cache verification

- Added `scripts/predict_product.py`: real product-only molecular inference,
  with explicit family-artifact selection, model/library hashes and no fitting
  or validation selection. Both cached-route and fresh EditRetro-to-temperature
  smokes passed. Retained parity covers 200 identities, 52 tabular values,
  ranking scores and temperatures for the tested query.
- The maintained shell suite now calls `scripts/run_verified_mainline.py`.
  Input/source/config/checkpoint/output hashes guard family caches; a process
  lock rejects simultaneous writers. Forced rebuild also retrains neural models.
- `inference_cache_audit.json` records a real bounded smoke rebuild, identical
  reuse with unchanged model bytes/mtimes, and rejection after a seed change.
  This 20/10-route ranker-training smoke is not an accuracy experiment.
- Twenty-five tests passed, including feature parity, no-fit inference,
  candidate/support identity checks, no-KNN isolation, cache mismatches and the
  raw-workflow command graph tested against recording stubs in a temporary repo.
- Existing strict family Stage 1/2 split audit passed. An additional complete-
  molecule scan found a legacy dot-splitting issue in one Diels-Alder training
  reaction (two sides), no invalid whole sides, no affected validation/test
  sides and zero split intersections. Details and scope limits are recorded in
  the evidence study's `CANONICALIZATION_NOTE.md`; frozen models are unchanged.
- Full 18-run reconstruction/report export remains in progress. Automated
  postprocessing will replay rows, write subgroup/failure/control reports,
  refresh first-hit-correct cases, and measure one cold product query per
  family after reconstruction workers exit. Do not infer completion from this
  log entry; inspect study status, replay audit and finalization artifacts.

## 2026-09-13 - throughput adjustment and completed ReaFNN-only audit

- All 18 ReaFNN-only family/seed records passed candidate-hash verification
  and metric replay. Macro Sys@10 is `36.24 +/- 0.27%`; full parallel ProSys
  remains `43.77 +/- 0.60%` (a `+7.53 pp` difference). See the study SUMMARY.
- The mainline evidence reconstruction pilot (Beckmann, seed 0) passed exact
  candidate, ranked-identity, temperature-support and reference-temperature
  checks. Its wall time was 146.49 seconds and process peak RSS 1870.18 MiB.
  This is Stage 2/3 reconstruction plus auditing, not single-query latency.
- Launched remaining evidence jobs with four independent processes and two
  OpenMP/MKL/OpenBLAS threads per child, within a 14-core/90-GiB allocation.
  Model hyperparameters, seeds, splits and evaluation definitions are unchanged.
  Per-child runtime/source manifests are retained; scratch is removed only
  after successful evidence compaction, with an 8-GiB free-space launch guard.
- The full reconstruction is still running. Its live progress is in
  `Experiment/mainline_evidence_completion_20260913/status.json`; no promoted
  mainline result has been replaced based on partial reconstructed results.

## 2026-09-13 - current documentation cleanup and strict no-KNN control

- Removed mixed-version numerical/method blocks from current results and
  Stage 2/3 detail pages; preserved their complete text under
  `Experiment/document_archive_20260913/`. Maintained model settings and
  promoted numerical results are unchanged.
- Corrected temperature-audit reporting: equal aggregate metrics and support
  counts do not establish candidate/support identity equality. All 18 retained
  family-seed pairs pass aggregate checks; identity status is now unknown,
  rather than incorrectly asserted true, when hashes are unavailable.
- Started the six-family seeds-0/1/2 ReaFNN-only control in
  `Experiment/stage2_reafnn_only_multiseed_20260913/`. It disables KNN lookup,
  proposals and KNN-derived feature values, retains the historical neural
  proposal library and retrains XGB-LTR with the same 20-context budget.
- Three isolated intervention tests pass. First family/seed completed with
  candidate-table zero-KNN checks; the full result is pending. Consult the
  study status and per-family CSV rather than treating partial results as a
  six-family macro comparison.
- New runs retain compressed per-candidate evidence, source/split/route hashes
  and candidate-identity hashes before pruning only their own scratch files.
- Corrected the raw reproduction entrypoint to generate Stage 1 validation
  routes needed by parallel fusion, clear that cache on processed-data reset,
  and reject resetting data while skipping preprocessing. Shell syntax passed;
  the destructive/full-training entrypoint was not executed for this repair.


## 2026-09-05 - current parallel Stage-2 ReaFNN three-seed ablation

- Completed `Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/`:
  six families, fixed Stage-1 test-route caches, and seeds 0/1/2 (18 compact
  records).
- The KNN-only control preserves product-Morgan KNN (`K=64`), the 64-context
  prefilter, top-20 cap, train-only memory, and reference-split candidate
  construction. It disables ReaFNN/post-fusion and re-trains a 52-feature
  tabular XGB-LTR on its own KNN-only candidate distribution.
- KNN-only achieves `53.39 +/- 0.00%` candidate recall and `39.86 +/- 2.08%`
  Sys@10, versus `54.26 +/- 0.15%` and `43.77 +/- 0.60%` for the full parallel
  pool. The paired full-minus-KNN-only effects are `+0.87 pp` recall and
  `+3.91 pp` Sys@10; all six family mean Sys@10 effects are positive.
- All 18 results pass the KNN-only protocol and fixed Stage-1/manifest pairing
  audit. The launcher was corrected to use the current
  `--no-reafnn_enable_independent_post_fusion` flag and successfully resumed
  from 17 compact records after an interrupted terminal session.

## 2026-09-04 - current parallel R-GNN temperature three-seed ablation

- Completed `Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/`:
  a six-family, fixed-Stage-1, seeds-0/1/2 matched temperature-only control.
- The full arm uses a 52-column tabular table plus a 128D R-GNN route embedding;
  the control retrains only temperature XGBoost from the identical 52 tabular
  fields and asserts that no `route_gnn_feat_*` feature is available.
- Every one of the 18 family-seed pairs passes exact Stage-1 recall, Stage-2
  pool/protocol, Sys@k/MRR/nDCG, and conditional-temperature-support checks.
- Conditional MAE is `11.49 +/- 0.26 C` with R-GNN versus `13.93 +/- 0.38 C`
  without it. Within `+/-5 / +/-10 / +/-20 C` improves by `+6.11 / +7.53 /
  +5.65 pp`; no system-ranking improvement is claimed because rank is fixed.

## 2026-09-04 - current parallel Stage-3 three-seed ablation

- Completed `Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/`:
  six families, fixed Stage-1 test-route caches, and seeds 0/1/2 (18 compact
  records).
- The deterministic no-XGB-LTR arm preserves the exact current parallel Stage-2
  protocol and candidate-availability metrics for every matched family/seed; it
  has no learned ranking parameters and skips temperature, which cannot alter
  Sys@k.
- Candidate recall is unchanged at `54.26 +/- 0.15%`; Sys@10 is
  `36.03 +/- 0.16%` versus `43.77 +/- 0.60%` for the full mainline, a
  `+7.74 pp` XGB-LTR within-pool reranking effect.
- Sys@1 changes by only `+0.65 pp` on average and is seed-variable; the
  report therefore makes no stable Top-1 claim.


## 2026-09-03 - current parallel mainline three-seed robustness

- Completed the maintained parallel KNN + ReaFNN post-fusion mainline at
  seeds 0, 1, and 2 over the six fixed-family test manifests.
- Persisted Stage-1 route caches, the 3,860-product denominator, KNN retrieval,
  family train-only condition memories, and validation-only fusion selection
  were fixed. ReaFNN, the temperature-only R-GNN, and XGBoost were rebuilt for
  every seed.
- Current macro result: candidate recall `54.26 +/- 0.15%`, Sys@1/3/5/10
  `25.13 +/- 1.20% / 35.12 +/- 1.27% / 39.11 +/- 1.04% /
  43.77 +/- 0.60%`, MRR `31.53 +/- 1.12%`, and nDCG@10 `33.16 +/- 1.02%`.
- Conditional temperature: MAE `11.49 +/- 0.26 C`; within `+/-5 / +/-10 /
  +/-20 C` is `41.41 +/- 2.16% / 63.09 +/- 1.76% / 83.74 +/- 0.69%`.
- Compact per-family records, source run summaries, manifest counts, and
  Stage-1 route-cache hashes are retained in
  `Experiment/stage23_parallel_post_fusion_multiseed_20260903/`.

## 2026-09-02 - mainline organization and archival

- Confirmed the maintained Stage 2/3 path is parallel product-Morgan KNN plus ReaFNN post-fusion, then tabular XGB-LTR with a separate R-GNN-assisted temperature regressor.
- Moved candidate-aware GNN ranking code and its three negative-result probes to `Experiment/legacy_stage3/`; they are not imported by the maintained pipeline.
- Added `scripts/README.md` and `Experiment/local_archive/README.md` to distinguish active entrypoints, tracked history, and local-only exploratory material.
- No end-to-end metric was recomputed during this organization pass.

## 2026-08-31 - matched current-mainline component ablations

- Completed Experiment/current_mainline_matched_ablation_multiseed_20260830/:
  six families, fixed Stage 1 route caches, seeds 0/1/2, and 36 compact
  family-seed records.
- The KNN-only plus re-trained XGB-LTR control gives coverage
  53.39 +/- 0.00% and Sys@10 39.86 +/- 2.08%; the promoted full system is
  higher by 1.06 pp coverage and 4.76 pp Sys@10.
- The deterministic no-XGB-LTR control has an exact matched Stage 2 pool
  (54.44 +/- 0.14% coverage) but Sys@10 36.45 +/- 0.08%; learned XGB-LTR
  supplies +8.17 pp Sys@10 from within-pool ranking.
- The independent audit passes the family/seed Stage 2-pool match, 3,860-record
  denominator, 3,833 candidate-slate, and 27 no-slate contracts.

## 2026-08-30 - post-hardening three-seed mainline promotion

- Promoted `Experiment/stage23_product_morgan_reafnn_multiseed_20260830/` as
  the current compact result source.
- The mainline fixes the six-family, 3,860-product Stage 1 route cache and
  independently rebuilds ReaFNN, R-GNN, and XGBoost at seeds 0/1/2.
- Current macro result: coverage `54.44 +/- 0.14%`, Sys@1/3/5/10
  `27.12 +/- 0.37% / 36.84 +/- 0.80% / 40.47 +/- 0.77% /
  44.62 +/- 0.42%`, MRR `33.28 +/- 0.48%`, nDCG@10 `34.70 +/- 0.53%`.
- Conditional temperature: MAE `11.73 +/- 0.54 C`; within +/-5 / +/-10 /
  +/-20 C is `40.59 +/- 0.56% / 61.80 +/- 2.36% / 83.04 +/- 1.10%`.
- The pre-hardening 2026-08-09 result remains a historical reference only.

## 2026-07-12 — six-family mainline consolidation

- The maintained project scope is now fixed to 6 reaction families:
  - `Beckmann`
  - `Buchwald-HartwigCross-Coupling`
  - `Chan_LamCoupling`
  - `DielsAlder`
  - `Friedel-CraftsAcylation`
  - `Friedel-CraftsAlkylation`
- Refreshed the maintained Non-Oracle outputs under:
  - `outputs/stage23_mainline/`
  - `outputs/checklist_stats/`
- Current 6-family macro results in `outputs/stage23_mainline/overview.md`:
  - `rr@10 = 63.2`
  - `cover = 55.0`
  - `sys@1 = 25.0`
  - `sys@5 = 37.4`
  - `sys@10 = 42.0`
  - `Temp MAE = 23.32`
  - `Temp±5C = 18.6`
  - `Temp±10C = 31.9`
  - `Temp±20C = 54.6`

## 2026-07-12 — default family lists synchronized

- Synchronized the maintained 6-family scope in:
  - `prosys_shared/mainline.py`
  - `baseline/common.py`
  - `baseline/render_stage23_nonoracle_reports.py`
  - `data_preprocess/preprocess.py`
  - `scripts/reproduce_mainline_from_raw.sh`
- Updated `README.md`, `checklist.md`, and `todo.md` so the written record matches the maintained code path.
- Removed stale checklist snapshots and outdated summary records that still referenced retired family results.

## 2026-07-12 — notes for future maintenance

- The maintained workflow remains:
  - `Stage 1 EditRetro`
  - `Stage 2 KNN screening`
  - `Stage 3 XGBoost reranking + temperature prediction`
- Historical or exploratory material should stay under `Experiment/` and should not be used as the current official result source.
- Chemistry labels such as `TurboGrignard` may still appear inside candidate tables for retained families; these are reagent names, not retired family result records.
