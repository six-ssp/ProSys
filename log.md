# ProSys Development Log

## 2026-09-25 - accepted 50K mainline and GitHub publication preparation

- User accepted the completed 50K mainline, baselines and ablations; no further
  hyperparameter search or FULL retraining is planned.
- README now displays only the accepted 37.87 +/- 0.27% Sys@10 result and
  explicitly equates FS@k with Sys@k. Historical FULL tables remain in their
  dated records, not as a competing headline.
- Corrected the baseline document's stale no-route count (27 to 66), the
  ablation document's obsolete expert-pending statement, and terminology for
  parallel candidate construction and the separate graph-assisted regressor.
- Git publication includes source/tests and family/seed aggregates. The
  experiment allowlist excludes row-level chemistry, weights, cases and private
  manuscripts. Earlier no-push statements apply to the completed audit phase;
  this publication is a new explicit user request. Prior maintenance snapshots
  are retained locally without rewriting scientific evidence or audit receipts.

## 2026-09-25 - cold cost audited and local manuscripts published

- Six fixed-query fresh cold runs completed after all training/diagnostic jobs
  exited. The independent audit verifies 137 input bindings, output ranks,
  float32 prediction equivalence and resource rows. Load-inclusive wall times
  are 13.62-24.01 s; these six observations do not establish throughput or
  population latency. One-ULP prediction changes are rejected by the check.
- Main text and SI are published locally with original-file SHA-verified
  backups. New expert S29 and cost S30 join the verified numerical, prose,
  method and figure refresh. Final PDFs contain 18/24 pages and 12 checked SI
  contents entries; changed/high-risk pages were visually inspected.
  Main Figure 1 remains explicitly out of scope; Main Figure 2 is preserved.
- Publication source bindings total 754, with original DOCX inputs redirected
  only to their exact backups. Earlier receipts retain historical pending
  fields; the publication and closing audit supersede them rather than
  rewriting failed logs or claiming untouched decoder completion.
- Post-publication regression: 252 project tests (14.151 s) and eight
  guard/parser tests (0.102 s), all passing. Current status text synchronized;
  closing requirements and limitations: `Experiment/final_release_50k_20260925/`.
  No GitHub push or unapproved checkpoint deletion was performed.

## 2026-09-25 - all expert seeds certified with explicit decoder recovery

- All 18 expert results passed the final source/checkpoint/query/metric checks;
  CSV means/sample SD were independently recomputed. Macro Route@1/3/5/10:
  36.57 +/- 0.42 / 48.53 +/- 0.09 / 51.81 +/- 0.09 / 55.42 +/- 0.39%.
- The recovered Diels-Alder seed-2 result retains 762 queries and 76,200 slots,
  with one over-capacity hypothesis invalidated, 5,767 routes and 43 empty
  queries. Its Route@10 is 33.858268%; no weight, seed or query substitution.
- Full seed-1 GPU replay has identical hypothesis strings and final ranked
  routes/scores/probabilities. Two EOS-score differences on invalid strings are
  explicitly recorded; universal token-score bitwise equality is not claimed.
  The original failed queue remains failed in its historical log. Recovery
  admission and the amended summary identify 17 untouched decodes plus one
  recovered decode. Eight CPU guard/parser tests pass.
- Started isolated six-family cold inference only after training, diagnostics,
  parity and expert verification exited. Cost measurement and document release
  remain pending. All 69 frozen production-source hashes are unchanged.

## 2026-09-25 - final expert trained; decoder failure preserved for diagnosis

- Synchronous execution and a pre-lookup trace localize the failure to a
  3,078-token hypothesis against 1,024 learned positions, at iteration 10.
  Frozen query 220 / augmentation 6 / beam 7 is retained. An isolated fail-closed
  guard invalidates unsupported hypotheses rather than truncating chemistry;
  seven CPU tests pass and full 762-query GPU execution is underway. No recovery
  output is promoted yet. Source and recovery-provenance admission are separate
  from merely completing inference; all 69 frozen sources remain unchanged.
- Diels-Alder expert seed 2 completed normally at epoch 101 / 22,119 updates,
  best loss 4.266, with the unchanged 15-check patience. Its subsequent guarded
  test decoding failed with CUDA indexSelectLargeIndex / device-side assertion.
  The expert controller and dependent tail queue exited. No model is retrained,
  seed changed or query removed. A same-checkpoint, same-parameter diagnostic
  with CUDA_LAUNCH_BLOCKING=1 uses a new output directory and separate log.
- Private header corrections passed layout/visual checks (main 18 pages, SI 23)
  without changing text or values. Added a final-candidate auditor for independent
  expert mean/sample-SD arithmetic, cost transcription, inverse XML restoration,
  unchanged media/formulas, rendered bounds and twelve SI contents labels.
  Full integration/acceptance awaits the outstanding experimental evidence.
- The latest ablation unittest discovery passed all 252 tests in 14.949 s.
  Log: `/root/autodl-tmp/prosys_ablation_regression_final_docs_20260925.log`.
  This is not a claim that the newly failing real GPU decoding has passed.

## 2026-09-25 - auxiliary fields verified and private manuscripts staged

- Complete auxiliary replay produced 48 direct-condition and 18 validation-fusion
  rows. Independent checks cover all 139 input/eight output bindings, seed grids,
  integer hit counts and the larger-KNN-weight tie-break. Evidence:
  `Experiment/paper_auxiliary_50k_20260924/all_families/independent_auxiliary_check.json`.
  Seed-0 Product-NB/GNN Condition@10 is 42.21/49.50%; seed-0 ReaFNN/KNN/fused
  validation coverage is 40.04/44.77/45.90%. Diels-Alder seed 2 selects w=1.
- Private main/SI previews apply 455 numeric cells, 56 paragraphs and 13 method
  cells. Inverse XML audit restores the original exactly; unrelated ZIP parts,
  equations and Main Figures 1/2 are unchanged. Source-bound S1/S2 panels passed
  independent arithmetic/export checks. Both DOCX previews rendered to PDF
  (18/23 pages); no searched historical headline literals remain. Two header
  wrapping issues remain for final layout correction. Root DOCX files unchanged.
- Started a no-timeout tail queue bound to the exact expert controller process.
  After normal completion it verifies all 18 experts, independently summarizes
  seed variability, then measures isolated cold inference with explicit fresh
  artifact roots. It never changes training or restarts a quiet process.
  Queue scope does not include final manuscript publication or a GitHub push.

## 2026-09-25 - full downstream comparison independently verified

- The full collector completed all 150 family/model/seed rows. Independently
  checked 69 sources/17 outputs, equal-family macro and sample-SD arithmetic,
  all earlier family reports and unchanged mainline/Stage 3 rows. Evidence:
  `Experiment/50k_verified_comparisons_20260924/full/independent_summary_check.json`.
- ProSys Sys@10 37.87 +/- 0.27%; strongest baseline B3 26.16 +/- 0.61%,
  an unrounded 11.711804 pp gain. Without KNN/ReaFNN/LTR gives
  31.31 +/- 0.50% / 34.16 +/- 1.80% / 29.90 +/- 0.10%.
  B3 temperature support is 1,251/1,219/1,230 versus mainline
  1,529/1,537/1,538; B3 has higher within-10 C accuracy on its own support.
  Do not claim paired or uniform baseline-temperature superiority.
- Started full auxiliary replay only after the collector exited normally.
  Expert seed 2 still trains. Standalone Main Figure 3 / SI Figure S3 exports
  passed numeric/export and visual checks; manuscripts remain unchanged until
  coordinated prose/table/figure replacement. Figure S1/S2 and final layout
  remain pending. No model or test-selection rule was changed.

## 2026-09-25 - final downstream family completed; full comparison replay

- Controller 35155 exited normally after Diels-Alder baselines, Stage 2
  removals, retained-evidence audits and guarded scratch pruning. Its configured
  five-family completion plus the independently completed Beckmann covers all
  six families. Started the full explicit-root comparison collector at
  `Experiment/50k_verified_comparisons_20260924/full/`; full-table verification
  is not claimed until that collector and output checks finish.
- Prepared 232 unapplied, source-bound numeric manuscript cell replacements.
  All 138 inputs, original DOCX bytes and physical merged-cell coordinates
  passed checks. Private plan: `manuscript/50k_numeric_replacement_20260925/`.
  Baseline/Stage 2/auxiliary fields, coordinated prose/figures and final render
  remain separate completion requirements.
- Diels-Alder's retained fusion weights are 0.9/0.8/1.0, not nonzero neural
  fusion in every run. Seed 2 still passes nonzero ReaFNN token scores into
  XGB-LTR; a zero branch fusion weight is not the no-ReaFNN intervention.
  Final auxiliary replay and manuscript qualification will retain this boundary.

## 2026-09-25 - complete fresh mainline and exact Stage 3 replay

- Full six-family query diagnostics also passed: 11,580 query-seed rows,
  exactly 3,860 distinct family/query identities. Rebuilt every candidate's
  exact-system label from test gold, checked query and predicted-route
  membership and replayed annotations. Seventy-two inputs and four outputs
  passed hash checks. Exhaustive failure counts and seen-product/context
  subgroups: `Experiment/query_diagnostics_50k_20260924/all_families/`.
  Repeated seeds are not independent test samples. The private terminology
  ledger was corrected from obsolete serial Stage 2 / gated 64-d GNN to
  parallel Stage 2 / ungated 128-d temperature features, with 66 empty queries.

- All 18 downstream fits passed original-query candidate/metric replay.
  Independently recomputed equal-family macro arithmetic and sample SD;
  all saved family rows agree. New Sys@10 is 37.87 +/- 0.27%, versus
  same-pool no-LTR 29.90 +/- 0.10% (+7.962408 pp before rounding).
  Macro Sys@1 is lower with LTR by 0.416707 pp; report this limitation.
- Conditional temperature MAE full/no-R-GNN is 11.32 +/- 0.33 C /
  13.63 +/- 0.29 C, with identical supports 1,529/1,537/1,538 per seed.
  Receipt: `Experiment/stage23_50k_evidence_20260924/independent_full_summary.json`.
  Final Diels-Alder baselines, Stage 2 removals and expert seed 2 remain live.
- Rechecked all 18 model metadata sets against certified manifests, including
  the final family. Dimensions and ReaFNN 30/8 / R-GNN 20/5 training settings
  agree; private method receipt binds 93 sources. All 69 controller-frozen
  source hashes remain unchanged. No test-driven model change was made.

## 2026-09-25 - publish complete fresh-route case traces

- Root `example.md` now derives from the complete six-family, 18-case export
  at `Experiment/examples_50k_20260924/all_families_seed1/`. Fixed expert
  seed 1 and downstream seed 1; descriptive first-exact-hit cases span
  ranks 1/3/5/10 across the study. Verified manifest/output hashes, all 18
  unique case links and lossless report publication apart from path rewriting.
- Preserved the previous root document byte-for-byte as
  `Experiment/examples_50k_20260924/historical_FULL_example.md`; publication
  records before/after hashes and generator identity in `root_publication.json`.
  No model, candidate or test-selection protocol was modified. This milestone
  does not certify unfinished expert, baseline or control experiments.

## 2026-09-24 - switch to USPTO-50K from scratch

- The final Diels-Alder seed-1 validation/test route pair passed independent
  11-input admission: queries 761/762, routes 6,004/6,104, no-route queries
  24/22, all retained. Route publication finished normally, with six receipts
  in `summary.json`. The full six-family readmission and source/output hash
  checks passed again, recorded in the route study's `independent_admission.json`
  and `route_counts.csv`. Test totals: 3,860 queries, 31,745 routes, 66 empty
  queries; validation totals: 3,855 queries, 31,846 routes, 63 empty queries.
  These supersede historical counts only for the fresh route version.
  Downstream controller 35155 automatically started Diels-Alder seeds 0/1
  (PIDs 197585/197586), while expert seed 2 continues (PID 193848).
  Full downstream-suite count stays five of six. The private replacement
  map now points to complete route evidence, not unfinished system metrics.
- Diels-Alder expert seed 1 passed independent verification on all 762
  original queries, retaining 22 no-route queries. Route@1/3/5/10 =
  17.4541/27.0341/30.1837/33.7270%. Normal epoch-86 / 18,834-update
  patience-15 early stopping; logged training time 8,077.5 s. Best SHA256:
  `7cbbb3ef1c5aa9524c9f410d7c6b3e72b40a512c1af621ddf6bbd279001d803a`;
  route-cache SHA256:
  `8c055db9ce3fed505932f1dd9193fd13e6c201e12a58148b32405c43ce387615`.
  Seventeen of 18 experts are verified. Seed 2 (PID 193848) and seed-1
  validation decoding (PID 193912) run concurrently under the original queue.
  All six fixed-seed-1 experts were then independently reverified for the
  complete guarded base/expert comparison: macro Route@1/3/5/10 base
  3.59/6.43/8.04/10.08%, expert 36.55/48.43/51.72/55.86%. Exact tables,
  query support 3,860, and source/output hashes pass. Evidence:
  `Experiment/stage1_50k_paired_comparison_20260924/all_families_seed1_verification.json`.
  This fixed-seed comparison is not expert-three-seed uncertainty or completed
  downstream performance. No model or test-driven selection was changed.
- Diels-Alder expert seed 0 completed normal patience-15 early stopping at
  epoch 95 / 20,805 updates (9,098.8 s logged training). Independent
  `verify_job` replay passed all 762 original queries, keeping 42 no-route
  queries and 5,493 predicted routes. Route@1/3/5/10 =
  20.3412/26.9029/30.9711/33.2021%. Best SHA256:
  `e121fe62a9b64259279b0df0d3c91004504edcdcd0242491a520682b209cf208`;
  route-cache SHA256:
  `2dca06a99d8b9336951c16fcd018fae0b2e1d94841d4d44a8ea3135d2b48b938`.
  Receipt: expert study `DielsAlder_seed0_independent_verification.json`.
  Sixteen of 18 experts are verified. The original queue started seed 1
  (trainer PID 179187); fixed-seed-1 downstream route publication still waits.
  No full-family/full-study completion is claimed. Best/last are retained.
- Minimal manuscript method audit found SI ReaFNN training-length drift:
  all 15 completed new fits use max_epochs=30 / patience=8, not 20/5.
  Corrected S11 and distinguished R-GNN's own 20/5 settings, auxiliary
  token supervision, Stage 1/2 ranking prior, ranker early stopping and
  fixed-300-tree temperature fitting. No experiments or performance values
  changed. Exact recorded XML edits, all other DOCX members, 165 static
  cells against prior source-bound expectations, both PDF renders (18/23
  pages) and 12 SI contents-page headings passed checks. Changed pages were
  visually inspected. Private evidence: `manuscript/50k_methods_revision_20260924/`.
  Main Figure 1 is untouched; final six-family performance revision remains
  pending. The private replacement map records new hashes and unchanged indices.
- Buchwald-Hartwig auxiliary paper fields passed read-only replay on all
  three seeds: 1,097 validation queries, 8,604 predicted routes, KNN weights
  0.7/0.6/0.6, fused coverage 50.59/50.77/50.23% versus KNN-only 49.04%.
  Direct Condition@k was recounted separately before route pairing, keeping
  all 1,099 test queries. All 24 input and three output hashes passed, as did
  the unified report's 14 source and seven output hashes after the rerun.
  Five auxiliary families are complete; full six-family evidence remains
  pending. Diels-Alder seed 0 is still live; no existing queue was restarted.
- Buchwald-Hartwig's complete downstream suite passed controller audits and
  the unified 25-row collector; all 14 source and seven output hashes were
  independently rechecked. Sys@10 B1/B2/B3/B4 = 19.47%, 12.80 +/- 0.53%,
  17.74 +/- 1.07%, 13.32 +/- 0.58%; without KNN/ReaFNN/LTR =
  32.42 +/- 1.79%, 43.28 +/- 0.26%, 33.73 +/- 0.34%, versus full
  45.86 +/- 0.55%. B3's conditional temperature MAE is lower
  (8.84 +/- 0.65 C versus 12.02 +/- 0.48 C), on different supports
  304/272/281 versus 484/486/492; no paired superiority claim is made.
  Five of six complete suites are now verified. The controller waits for
  Diels-Alder; auxiliary validation/Condition@k replay is running separately.
  Frozen source hashes remain unchanged (69). No valid checkpoints were
  deleted, no models were tuned, and no manuscript headline was replaced.
- Buchwald-Hartwig expert seed 2 completed normally at epoch 73 / 11,023
  updates, patience 15, logged training time 5,219.1 s. All three experts
  passed fresh independent verification on the same 1,099 original queries.
  Route@1/3/5/10 means = 40.64/54.23/57.66/60.33%, sample SD
  1.37/1.50/1.62/1.55 pp. Seed-2 best SHA256:
  `12073e22088ccf33be3486356b1f7606e723d1f71966d47df930ca9ef1bd81bc`;
  route-cache SHA256:
  `de2bd6053f53be0b6f3549b2e6c866dad3d145fd6d6469dc20a8e9fa93969c99`.
  All 45 seed-2 no-route queries remain. Tables/receipts are under the expert
  study's `family_reports/Buchwald-HartwigCross-Coupling/`. The higher-scoring
  expert does not replace the predeclared seed 1 downstream. Fifteen of 18
  experts are complete; the original queue started Diels-Alder seed 0
  (trainer PID 161038), with 234,418 admitted training augmentation pairs.
  Separately, the three no-KNN controls passed independent candidate/metric
  replay: Sys@10 32.42 +/- 1.79%. Baselines and KNN-only controls still run;
  the complete downstream-suite count remains four of six.
- All three Buchwald-Hartwig downstream seeds and exact Stage 3 controls
  passed independent replay. Regenerated per-seed and family-statistic CSVs
  and verification receipts match the controller's output. Mean Sys@10:
  full 45.86 +/- 0.55%, no-LTR 33.73 +/- 0.34%; +12.132241 pp unrounded.
  Matched temperature supports 484/486/492; MAE full/no-R-GNN is
  12.02 +/- 0.48 / 12.42 +/- 0.71 C. Mean within-5 C declines slightly
  (35.16% vs 35.35%); no uniform temperature benefit is claimed. The family
  README records the full scope. Independent query diagnostics passed all
  3,297 query-seed rows and source/output hash checks. Seed-1 partition:
  436 route misses, 93 pool misses, 70 ranking misses, 500 hits. Baselines
  and Stage 2 removals have started, so only four full suites are complete.
- Buchwald-Hartwig downstream seeds 0/1 passed independent retained-candidate
  replay and exact Stage 3 controls on all 1,099 queries. Sys@10 full/no-LTR:
  45.5869/33.4850% and 45.4959/33.5760%. Temperature MAE full/no-R-GNN:
  12.5249/13.1876 C and 11.9662/12.2966 C, identical per-seed supports 484/486.
  Seed 1's within-10 C accuracy is lower with R-GNN (55.35% vs 57.20%);
  do not claim uniformly improved temperature metrics. Seed 2 is pending.
  Fixed-index-0 deployment replay matched all 200 candidates, order and final
  predictions; fresh guarded decoding also completed with ten routes and
  200 candidates. Source/checkpoint/evidence hashes passed. Three seed-1 cases
  (indices 1/231/306; first exact ranks 1/5/10) were exported and rehashed.
  These are descriptive/deployment checks, not additional accuracy seeds or
  isolated cost measurements. See the Buchwald-Hartwig directories under
  `Experiment/product_inference_50k_20260924/` and
  `Experiment/examples_50k_20260924/`. Full suites remain four of six families.
- Buchwald-Hartwig fixed-seed-1 route publication passed independent
  `admit_routes` verification of 11 bound inputs. Validation/test have
  1,097/1,099 original queries, 8,604/8,476 routes and 20/25 no-route queries,
  all retained. The existing controller automatically started downstream
  seeds 0/1 (PIDs 149880/149879), concurrent with expert seed 2. Five route
  pairs are admitted, but only four complete downstream suites are verified.
  No training sources or resource guards were changed.
- Buchwald-Hartwig expert seed 1 passed independent `verify_job`: all 1,099
  original queries, including 25 without routes. Route@1/3/5/10 =
  39.9454/53.4122/57.3248/60.3276%. Normal epoch-69 early stopping,
  10,419 updates, patience 15, logged training time 4,721.4 s. Best SHA256:
  `93179694a5047e347e7c84cd898151c90dc179be1959be3b050b00d61c5ce0e0`;
  route-cache SHA256:
  `69826c40e56ea1558692ff685b6d384622b729016f567e09b211fac0e3d961be`.
  Guarded paired base comparison gives Route@10 32.7571% versus 60.3276%,
  +27.5705 pp; CSV/Markdown under `Experiment/stage1_50k_paired_comparison_20260924/`.
  Fourteen of 18 experts are complete. Seed 2 (trainer PID 145756) and seed-1
  validation decoding (PID 145814) started automatically and run concurrently.
  Full downstream suites remain four of six; no family-three-seed or global
  completion claim is made.
- SI static-data audit completed without editing data or document bytes:
  105 S4/S5/S7/S8 numeric cells were recounted from current persisted splits
  and retained raw route files; 12 S1 cells were recounted from source-ID
  lists and export files. All match. Another 48 S2/S3 cells match the historical
  preprocessing diagnostic CSV; this is not a new cleaning run or exact split
  lineage proof. Private receipts and scope notes are under
  `manuscript/50k_status_revision_20260924/`; final performance/manuscript
  completion remains pending. Original raw-route counts remain distinguished
  from repaired effective augmented training counts.
- Fresh read-only query diagnostics passed for Beckmann, Chan-Lam, acylation
  and alkylation across downstream seeds 0/1/2: 12 runs, 5,997 query-seed rows.
  Every retained candidate's exact-system label, query identity and predicted
  route membership was checked against persisted test gold and fresh routes.
  Rebuilt annotations equal saved `queries.csv`; failure categories partition
  every original query, including empty slates. Inputs/outputs were rehashed.
  Seed-1 route/pool/ranking misses and hits: Beckmann 73/65/39/58;
  Chan-Lam 117/19/47/207; acylation 181/17/68/209; alkylation 537/22/46/294.
  Evidence: `Experiment/query_diagnostics_50k_20260924/first_four/`.
  Subgroups describe condition-training memory only, not all Stage 1 sources;
  the partial report is not a six-family macro. No fitting or test tuning.
  Nine new regression tests passed; full maintained discovery passed 252 tests
  in 15.515 s, no skips. See `Experiment/regression_20260924/README.md`.
- Native two-week-expiry Git pruning removed 515 expired unreachable loose
  objects (8.36 GiB payload), after checking refs, reflogs and index exclusion.
  HEAD/refs/index/reflog and staged/unstaged changes were unchanged; Git
  connectivity and the controller's 69 frozen source bindings passed.
  Free data space rose from about 8.00 to 16.37 GiB. No worktree/model files
  were removed, including the invalid checkpoints awaiting a separate decision.
  Evidence: `Experiment/storage_cleanup_20260925/`. Buchwald-Hartwig seed 1
  and all three controllers remained live; resource admission guards unchanged.
- Buchwald-Hartwig expert seed 0 passed independent `verify_job` on all
  1,099 original queries, including 36 without routes. Route@1/3/5/10 =
  39.7634/53.3212/56.2329/58.7807%. Normal epoch-60 early stopping,
  9,060 updates, patience 15, logged training time 4,134.1 s. Best SHA256:
  `6b9b69dbd037bdd692a9ade4818edcacc316f247eadd3090764a08e1caf989d8`;
  route-cache SHA256:
  `3c420c7e9e7ff2379ba1f41e0de6e5d34b7c9b30320fa5b2c5a4fd84509cc7fd`.
  Thirteen of 18 experts are complete. Seed 1 started automatically
  (trainer PID 134855); no configuration changes. Full downstream remains
  four of six families. Disk free space was about 8.9 GiB before the new
  seed's checkpoint pair; continue monitoring the unchanged space guards.
- Alkylation expert seed 2 completed normally at epoch 92 / 13,892 updates,
  patience 15, logged training time 6,966.3 s. All three expert jobs passed
  fresh independent verification on the same 899 original test queries.
  Route@1/3/5/10 means = 22.21/33.30/37.37/41.64%, sample SD
  1.43/1.14/1.84/1.52 pp. Unrounded results and hashes are in the expert
  study's `family_reports/Friedel-CraftsAlkylation/`. Seed 2's higher score
  does not change the predeclared expert-seed-1 downstream protocol.
  Twelve of 18 experts are now complete, with four of six full downstream
  suites. The original queue automatically started Buchwald-Hartwig seed 0
  (trainer PID 123671); no restart or test-driven configuration change.
- Alkylation auxiliary paper fields passed direct-condition replay and all
  three read-only validation-grid replays. Original validation denominator:
  896 queries, 8,052 routes; selected KNN weights 0.8/0.8/0.5. Fused
  validation coverage = 36.27/36.38/36.50%, KNN-only = 35.94% each seed.
  Inputs/outputs and unified comparison hashes all passed after the run.
  This remains validation-only calibration, not test tuning. Auxiliary
  evidence is now complete for four families, with two still pending.
- Alkylation's complete downstream suite passed all controller audits and
  the independent unified collector (25 model/seed rows). Report scope,
  source hashes and output hashes were rechecked. Mean Sys@10 for
  B1/B2/B3/B4 is 8.34%, 17.20 +/- 0.42%, 18.21 +/- 0.17%,
  11.38 +/- 1.14%; without KNN 31.59 +/- 0.19%, without ReaFNN
  29.85 +/- 1.00%. B3's conditional temperature MAE is lower than ProSys,
  8.14 +/- 0.48 versus 10.63 +/- 0.91 C, but on different supports
  (295/296/299 versus 321/322/318); this limitation is explicitly retained.
  Four of six complete downstream family suites are now available. The
  controller waits for Buchwald-Hartwig seed-1 routes; no old routes are used.
  Verified scratch cleanup restored approximately 9.9 GiB free data space.
  Auxiliary condition/fusion replay is running separately; no new fitting.
- Alkylation downstream seeds 0/1/2 and exact Stage 3 controls completed.
  An independent three-seed `verify_run`/`aggregate` replay matched every
  saved family CSV row and verification receipt. All 899 original queries
  remain in the denominator. Sys@10 = 33.04 +/- 0.68% versus same-pool
  no-LTR 18.84 +/- 0.28%, +14.200964 pp before rounding. Sys@1/3/5,
  MRR and nDCG@10 also improve in this family; other families' mixed effects
  remain reported. Temperature MAE = 10.63 +/- 0.91 C versus no-R-GNN
  13.39 +/- 0.57 C on identical per-seed supports 321/322/318. The
  controller has moved on to baselines and Stage 2 removals. Its generated
  CSVs were preserved; a human-readable family README was added under
  `Experiment/stage23_50k_evidence_20260924/family_reports/Friedel-CraftsAlkylation/`.
  This completes four families' mainline/Stage 3 evidence, but only three
  families' entire downstream suites. Expert seed 2 is still training.
- Alkylation downstream seeds 0/1 passed independent `verify_run`: original
  899-query metrics, same-pool no-LTR controls and same-support no-R-GNN
  temperature controls all replayed. Sys@10 full/no-LTR is
  32.5918/18.7987% (seed 0) and 32.7030/19.1324% (seed 1).
  Temperature MAE full/no-R-GNN is 10.1525/13.3743 C on 321 matched queries
  and 11.6829/13.9662 C on 322 matched queries. Seed 2 remains incomplete;
  no three-seed or six-family aggregate is claimed. Verified compact outputs
  allowed scratch removal and data free space recovered to about 11 GiB.
  Fixed-index-0 product replay passed all 200 candidate identities/order and
  predictions. Fresh guarded decoding also passed (10 routes, 200 candidates,
  no input normalization/replacement); hashes were independently checked.
  Three traced seed-1 cases (indices 0/29/129; first exact ranks 1/5/10) were
  exported and rehashed. See the alkylation directories in
  `Experiment/product_inference_50k_20260924/` and
  `Experiment/examples_50k_20260924/`. These are descriptive/deployment
  checks, not additional accuracy replicates or isolated cost measurements.
- Alkylation fixed-seed-1 validation/test route publication completed, followed
  by independent `admit_routes` verification of all 11 bound inputs. All
  896 validation and 899 test queries are retained, including two validation
  queries without predictions. The controller automatically launched fresh
  downstream seeds 0/1 while expert seed 2 continued training. The paired
  collector verified identical original test queries and decoding protocols:
  Route@10 = 0.4449% base versus 40.2670% expert, +39.8220 pp.
  Unrounded single-family CSV/Markdown:
  `Experiment/stage1_50k_paired_comparison_20260924/Friedel-CraftsAlkylation_seed1.*`.
  Its aggregate rows cover only this family, not the complete six-family study.
- Alkylation expert seed 1 passed independent `verify_job` after normal
  early stopping at epoch 71 / 10,721 updates (patience 15; 5,189.1 s).
  All 899 original queries were retained. Route@1/3/5/10 =
  21.0234/32.0356/35.5951/40.2670%. Best checkpoint SHA-256:
  `5949414393cc881a90b3fe537dd4850d4beecb79815c89dfe0da7cd5cd96ec9b`;
  route-cache SHA-256:
  `c7a950dbad51caf5e8398efe02babe862eee749f2c9b38d839013aaf11993f91`.
  Eleven of 18 experts are complete. The unchanged queue automatically
  started seed 2 (trainer PID 105909). The predeclared seed-1 downstream
  choice is retained despite the slightly higher seed-0 test Route@10.
  Training-log timestamps use the host logger's clock, not necessarily UTC.
- Alkylation expert seed 0 passed independent `verify_job` after normal
  early stopping at epoch 94 / 14,194 updates (patience 15; 6,847.3 s).
  All 899 original queries were retained. Route@1/3/5/10 =
  21.8020/33.5929/37.2636/41.3793%. Best checkpoint SHA-256:
  `e73982270a6ea18a585dfb5bb9a9f8c19dec86f853af44cb9fb183d6799f3dae`;
  route-cache SHA-256:
  `3411cfaee8b203296a4db43d5122469758d62d1ba90ca1877ddc050a49332a64`.
  Ten of 18 experts are complete. The unchanged queue automatically started
  alkylation seed 1 (trainer PID 95272). Downstream remains three of six
  completed families, waiting for this family's predeclared seed-1 routes.
  Data/root free space is approximately 13/2.7 GiB. No checkpoints deleted.
- Acylation expert seed 2 completed normally at epoch 84 / 6,804 updates,
  patience 15, logged training time 3,325.3 s. All three expert jobs passed
  fresh `verify_job` checks and same-475-query aggregation. Route@1/3/5/10
  means are 39.23/51.86/55.51/60.56%, sample SD 0.88/1.08/0.64/1.49 pp.
  Unrounded CSVs and bound hashes are in the expert study's
  `family_reports/Friedel-CraftsAcylation/`. Fixed expert seed 1 continues
  to supply downstream results; this is not post-test seed selection.
  Nine of 18 experts are complete. The original queue automatically entered
  alkylation seed 0 (trainer PID 81579), confirmed through epoch 2.
- Acylation's full downstream suite passed all controller audits, followed by
  the explicit-root unified collector (25 rows, single-family scope). Sys@10
  for B1/B2/B3/B4 is 33.47%, 24.84 +/- 0.96%, 33.19 +/- 2.21%,
  25.12 +/- 0.74%; without KNN 36.77 +/- 0.32%, without ReaFNN
  37.54 +/- 6.11%. No-ReaFNN seed 0 exceeds full Top-10; retain this
  replicate-level limitation. Report inputs/outputs were independently rehashed.
  Auxiliary direct Condition@k and full validation-grid replay also passed:
  476 validation queries, 3,729 routes, KNN weights 0.9/0.6/0.6. Fused
  coverage is 50.63/50.84/50.84%, versus 50.42% KNN-only. Auxiliary hashes
  passed after completion; no retraining or test-driven selection occurred.
  Three of six downstream families are complete; the controller waits for
  alkylation routes. Acylation expert seed 2 remains live.
- Acylation downstream seeds 0/1/2 and exact Stage 3 controls completed.
  Independent retained-row replay and comparison with every saved family
  row passed, using all 475 original queries. Sys@10 is 44.14 +/- 2.11%
  versus same-pool no-LTR 36.63 +/- 0.36% (gain 7.5088 pp before rounding).
  Mean Top-1/3, MRR and nDCG@10 are lower with LTR. Matched-support
  temperature MAE is 10.08 +/- 0.07 C versus no-R-GNN 12.27 +/- 0.44 C;
  supports are 238/239/239, with identical per-seed identity/reference hashes.
  The controller has advanced to baselines and Stage 2 removal studies;
  expert seed 2 remains live. The family report explicitly excludes these
  unfinished components and does not claim full six-family completion.
- Acylation fixed-seed-1 validation/test routes passed independent
  `admit_routes`: 476/475 original queries, the same best checkpoint and
  unchanged condition splits. Validation has 14 no-route queries, retained in
  the denominator. The controller started fresh downstream replicates.
  New detailed cases (indices 10/31/133, first exact ranks 1/3/10) passed
  source/output hash checks. Fixed index-0 CPU deployment replay has all 160
  candidate identities/full ordering equal to retained results, with exact
  final scores/temperatures. A fresh guarded end-to-end prediction passed
  (10 routes, 200 candidates); source/checkpoint/guard hashes and 1-query,
  10-variant support were independently checked. Fresh single-query decoding
  need not reproduce batched cached routes; concurrent timings are not a cost
  benchmark. See `Experiment/product_inference_50k_20260924/Friedel-CraftsAcylation/`.
- Friedel-Crafts acylation seed 1 completed at epoch 85 / 6,885 updates
  (3,222.8 s training), normal patience-15 early stopping. `verify_job` passed
  all bindings/guard/original-query checks and replayed 475 test queries:
  Route@1/3/5/10 = 38.9474/53.0526/56.2105/61.8947%.
  Best SHA256 `d05bf56e631eedd8005ace14a9b124874d429a1b68df4ca9f5db9f81bf786cd7`;
  route SHA256 `41033d52c0f62d63f50aec243d8d1d8bbc276d2ef99c4fd730aef853db8f70d1`.
  The explicit-root paired collector checked identical base/expert queries
  and guarded decoding settings: base Route@10 14.9474%, expert 61.8947%,
  delta 46.9474 pp. CSV/Markdown are in
  `Experiment/stage1_50k_paired_comparison_20260924/Friedel-CraftsAcylation_seed1.*`.
  Eight of 18 experts complete; seed 2 trainer PID 70123 and seed-1 validation
  decoder PID 70157 were confirmed live. No three-seed acylation summary yet.
- Friedel-Crafts acylation seed 0 completed at epoch 86 / 6,966 updates
  (3,334.9 s training), with normal patience-15 early stopping. Independent
  `summarize_stage1_multiseed.verify_job` passed input/admission, best/last,
  guarded-cache and unchanged-query checks and replayed all 475 queries:
  Route@1/3/5/10 = 40.2105/50.9474/55.3684/60.8421%.
  Best checkpoint SHA256 is
  `993eb8d205459cbc8f00b2ddb1de0f9444a6307cc356cfeb7f0ebb0a3998623d`;
  route cache SHA256 is
  `5951bf5c44657dfcc91ed17820ec19545bd78867147759f4692a5475b34a0b47`.
  The existing queue automatically started seed 1 (trainer PID 62631).
  Seven of 18 expert jobs are complete; full downstream families remain 2/6.
  No training/configuration changes or manuscript aggregate replacements made.
- Chan-Lam expert seed 2 completed at epoch 67 / 4,355 updates with normal
  patience-15 termination and guarded test decoding. All three experts passed
  independent verification: mean Route@1/3/5/10 is 55.30/65.21/67.01/69.32%,
  sample SD 1.50/0.82/0.78/1.18 pp. The fixed seed-1 paired comparison gives
  base Route@10 12.3077% versus expert 70.0000%. Six of 18 expert jobs are now
  complete; the original queue advanced to Friedel-Crafts acylation seed 0.
- Chan-Lam's entire downstream suite passed the controller audits and the
  unified collector produced 25 source/output-bound records. Mean Sys@10 is
  53.50 +/- 0.74% for ProSys; B1/B2/B3/B4 are 35.13%, 34.87 +/- 4.13%,
  49.15 +/- 1.16%, 32.22 +/- 2.33%. No-KNN and no-ReaFNN are 47.61 +/- 0.78%
  and 49.91 +/- 1.04%. The ProSys-minus-B3 Top-10 gap is 4.36 pp before
  rounding; B3 Top-1 is higher. KNN-only scratch was pruned only after compact
  evidence certification (about 8.9 MiB retained).
- Chan-Lam auxiliary Condition@k and all three validation-fusion replays also
  passed; weights are 0.7 KNN / 0.3 ReaFNN for all seeds. Actual report hashes
  and live experimental source bindings were checked. README no longer calls
  the old FULL table the current reportable result or recommends old-cache
  reproduction. Two of six family suites are complete; no full-study or final
  manuscript completion is claimed.
- Read-only paper inventory found that direct Condition@k and validation fusion
  require their own fresh evidence export, beyond the unified system table.
  Added an unbound standalone audit; the real Beckmann run independently
  recounts B1/B2 conditions and replays all three mainline validation grids
  without fitting. Counts, curves and chosen weights match; input/output
  hashes pass. Default six-family export correctly rejects missing evidence.
  Full regression now passes 243 tests in 15.916 seconds, no skips. The private
  manuscript replacement map preserves Figure 1 and flags stale case/figure,
  expert-seed and historical-versus-reconstructed temperature statements.
- Chan-Lam downstream seed 2 also passed independent candidate replay; the
  three-seed family summary is complete on 390 fixed queries. Mean +/- sample
  SD Sys@10 is 53.50 +/- 0.74% versus no-LTR 50.94 +/- 0.30%. Mean Top-1/3/5,
  MRR and nDCG@10 are lower with LTR; the family report explicitly retains this
  qualification. Matched-support temperature MAE is 5.88 +/- 0.22 C with R-GNN
  versus 7.12 +/- 0.15 C without it, n=238/239/238. Fresh baseline exports
  completed and the existing controller started baselines and Stage 2 controls.
  No six-family summary or manuscript result promotion is implied.
- Chan-Lam downstream seeds 0/1 completed and passed independent full-candidate
  replay: Sys@10 54.36% / 53.08%, versus same-pool no-LTR 50.77% / 50.77%.
  LTR reduces Top-1/3/5 for these two seeds; retain this mixed result rather
  than claiming uniform ranking improvement. Matched-support temperature MAE
  is 6.1266 / 5.7851 C with R-GNN versus 6.9502 / 7.2315 C without it
  (238 / 239 eligible queries). Seed 2 is running; no three-seed mean yet.
- Chan-Lam first-query deployment parity passed on all 200 candidates and
  their complete order; final scores and temperatures are exact. Maximum
  feature differences are 2.3842e-7 (ranking) and 6.0000e-6 (temperature).
  Fresh guarded product-only Stage 1/2/3 also completed. These are scoped
  smoke checks, not isolated cost measurements. Three hashed illustrative
  cases were exported (indices 4/16/34, first exact ranks 1/3/10). Receipts
  and details are in `Experiment/product_inference_50k_20260924/Chan_LamCoupling/`
  and `Experiment/examples_50k_20260924/Chan_LamCoupling_seed1/`.
- Chan-Lam fixed-expert-seed-1 validation/test route publication completed:
  390 unchanged queries per split, identical bound expert checkpoint, no
  identity-fallback replacements in either decode. Independent route admission
  and the live downstream controller's frozen-source checks passed again.
  The existing controller started downstream seeds 0/1 concurrently while
  Stage 1 continues Chan-Lam seed 2; no duplicate queue or training-source edit
  was introduced. This verifies handoff only, not completed downstream results.
- Chan-Lam expert seed 1 independently verified after normal patience-15
  termination at epoch 68 / 4,420 updates. On the unchanged 390 test queries,
  Route@1/3/5/10 is 56.41 / 64.62 / 67.18 / 70.00 percent. Its best checkpoint
  SHA256 is `6b63bdf999d2305fdd4136272d3239a723f89e6ed53dec123b46cf0b0920499c`.
  Five of 18 expert fits and guarded test evaluations are now complete; the
  existing queue advanced to Chan-Lam seed 2. The fixed-seed-1 publisher is
  decoding validation routes before downstream admission. These are partial
  results, not a six-family or three-seed Chan-Lam summary. The base training
  receipt again confirms random initialization and 50 completed epochs on
  filtered USPTO-50K, without FULL neural-weight inheritance.
- Added an explicit-root combined 50K comparison collector without changing
  frozen experimental sources. Real Beckmann audit produced 25 model/seed rows:
  all system metrics and temperature hit rates exactly match the earlier family
  report; maximum temperature-MAE difference is 2.52e-7 C from retained-value
  replay, within the declared serialization tolerance. Both source and output
  hashes were independently rechecked. Full-scope preflight correctly withheld
  a six-family report and created no output directory while five families lack
  complete evidence. Deterministic B1 is never duplicated to create seed SD.
- Nine new summary tests passed, followed by full CPU discovery: 232 tests,
  16.936 seconds, no skips. Log is
  `/root/autodl-tmp/prosys_ablation_regression_v2_20260924.log`, SHA256
  `e0f65ef48513f19f4e201304a3604cb2c6c5ef06f85653a8cbfaa5912b8b110d`.
- Full discovery of maintained `ablation/test_*.py` on CPU passed 223 tests in
  17.097 seconds with no skips. Log is
  `/root/autodl-tmp/prosys_ablation_regression_20260924.log`, SHA256
  `7909d43039dee6b5c66e8c18b75d0b5e2e3d549ef02b21f675220f2df3c44d72`.
  This does not claim the unused vendored third-party test suites were run.
- Storage forecast: 10 of 36 expert best/last files currently retained; remaining
  checkpoint requirement about 11.40 GiB, against 16.90 GiB available. That leaves
  only 5.50 GiB before new decode/evidence/scratch costs, below the downstream
  admission threshold. Previously identified invalid-weight cleanup (4.38 GiB)
  still awaits the user response; current valid models and raw data remain intact.
  No training source or disk-admission threshold was changed.
- Repeated the complete fresh-route Beckmann seed-0 Stage 2/3 fit with unchanged
  inputs, hyperparameters and sources. Independent verification passed both runs;
  all 40,520 candidate rows, 2,013 route embeddings, rankings and temperatures are
  exact. ReaFNN/R-GNN payloads and both XGBoost model files are identical. This
  confirms one same-environment repeat, not universal determinism. The repeat
  took 196.31 seconds in concurrent training; its scratch was pruned after
  evidence certification. Retained receipt/models occupy about 27 MiB.
- Replaced the old hard-coded case-export entrypoint with explicit roots and
  retained-row verification. Fresh Beckmann seed-1 examples have first exact
  ranks 1/3/10 (queries 2/53/125). Full intermediate JSONs are compressed; the
  root case document is now clearly historical until all families are available.
- Stage 1 single-expert collector now requires explicit guarded route roots;
  macro summary sample count is total rather than mean family size. Ten repeat/
  paired-collector tests passed. Chan-Lam seed 0 passed independent verification:
  390 queries, Route@10 70.00%; training stopped by patience at epoch 69/4,485
  updates. The original identified queue started seed 1 without restart.
- Deployment audit found `predict_product.py` still resolved historical expert
  aliases and called stock unguarded decoding. Only the deployment entrypoint
  and independent helpers/tests were changed; no live training/controller-bound
  source was edited. It now checks the completed 50K bundle, bound route caches,
  feature sources and expert hash, and replays its guarded decode command.
  Fifteen targeted tests passed. Beckmann fixed first-test-query replay matched
  all 200 candidates and complete order: 52 ranking features differ by at most
  1.20e-7, 180 temperature features by 2.50e-6; final scores/temperatures match
  exactly. Fresh product-only Stage 1/2/3 also passed. This is deployment parity,
  not a six-family metric or isolated latency benchmark. Evidence is under
  `Experiment/product_inference_50k_20260924/`.
- The combined deployment/parity and existing downstream-evidence suite passed
  67 tests in 10.750 seconds. Three additional cost-admission tests passed in
  0.003 seconds. The cost entrypoint no longer hard-codes historical inputs or
  reuses old predictions; it requires explicit new roots. The actual live
  training queue correctly blocked isolated-cost measurement. Chan-Lam seed 0
  reached epoch 62 / 4,030 updates at 16:28 UTC; expert, route publisher and
  downstream controllers are live, with frozen experiment sources unchanged.
- Beckmann is the first fully replayed family: three expert seeds (Route@10
  67.09 +/- 1.61%), three downstream seeds with exact Stage 3 controls,
  four baselines and both Stage 2 removal controls. Combined source-bound
  table: `Experiment/50k_family_reports_20260924/Beckmann/RESULTS.md`.
  Baseline independent replay reconstructs direct-product joint systems,
  relabels external candidates, rechecks validation fusion and test metrics
  across all ten fits. No temperature output is reported as NA, not zero.
- KNN-only audit initially rejected four nonzero legacy-prefixed fields.
  Tracing showed ordinary train-library metadata rather than neural output;
  independent count/support/yield recomputation passed all rows, including
  leave-one-canonical-reaction-out training exclusion. Neural fields are zero.
  All three candidate replays passed, models/predictions were certified, and
  raw scratch was deleted. The fitted experiment/schema was not changed.
- Started remaining-five-family downstream controller PID 35155, two children
  maximum and two threads each, bound to route controller PID 23100/start ticks.
  It freezes source hashes, waits only on an identified live publisher, and
  performs no automatic retries or old-result fallback. Current wait is
  Chan-Lam. New audit/controller regression: 48 tests / 0.384 s. Full six-family
  results and manuscript numerical replacement still remain pending.
- All three fresh Beckmann downstream seeds passed retained-row replay.
  Mean +/- sample SD: Sys@10 full 26.38 +/- 2.95%, no-LTR 24.11 +/- 0.25%;
  temperature MAE full 12.58 +/- 1.65 C, no-R-GNN 13.75 +/- 1.71 C.
  LTR does not improve mean Sys@1/Sys@3 for this family. No test-driven tuning
  or uniform-benefit claim is made. Family reports and source-bound comparison
  references are retained; the full six-family study remains incomplete.
- Started fresh Beckmann four-baseline study (deterministic B1 once; B2/B3/B4
  seeds 0/1/2) on newly exported admitted routes. Started both three-seed Stage 2
  removal controls. KNN-only uses a new scratch root and retains original
  outputs pending independent row replay/compaction; it has an initial source/
  input fingerprint. No-KNN uses the maintained bound-evidence wrapper. No
  completed baseline or Stage 2 result is claimed at this launch snapshot.
- Fresh Beckmann downstream seed 0 finished and passed independent retained-row
  replay (`stage23_50k_evidence_20260924/pilot_replay.json`). On 235 unchanged
  queries, full Sys@10 = 29.7872%, exact-pool no-LTR = 24.2553%. Full temperature
  MAE = 10.68885 C versus no-R-GNN = 11.77915 C on the same 65 conditional
  examples; system order is identical for the temperature control. This is
  one family/seed, not the six-family/three-seed conclusion. Fitting/control
  wall time 195.3 s during concurrent expert training. Retained evidence/model
  bundle ~27 MiB; scratch tables removed after certification. Downstream seeds
  1 and 2 started in separate processes with the same fixed expert seed 1.
- Beckmann expert seed 1 completed normally at epoch 58 / 6,322 updates,
  with 15 no-improvement validation checks and 1,876.6 s training wall time.
  Independent Route@1/3/5/10 on 235 queries: 45.5319/60.4255/63.8298/68.9362%.
  The queue advanced to expert seed 2. Fixed seed 1 now has an admitted
  235-query validation / 235-query test pair; the route controller waits for
  Chan-Lam next. Fresh Beckmann downstream seed 0 launched as PID 30338,
  two CPU threads and CUDA, with exact no-LTR/no-R-GNN controls included.
  No completed downstream metric is claimed at this launch snapshot.
- Added an explicit-root fresh downstream replayer, separate from the
  historical summary CLI. It requires all 18 certified runs, independently
  reevaluates retained candidate/control predictions, verifies fixed query
  identities and same temperature support, and reports equal-family means
  with sample SD across downstream seeds. Missing temperature support remains
  NA with counts. Summary/admission/control regression: 30 tests / 0.343 s.
  No complete new downstream table is claimed before the real fits finish.
- Final guarded base decoding completed all six families and all 3,860 original
  queries. Independent checks recomputed all Route@k, validated guard receipts,
  original query identities, output digests and bound sources/checkpoint.
  Macro Route@1/3/5/10: 3.5881/6.4316/8.0408/10.0762%. Beckmann and Diels-Alder
  are genuinely zero in this base-only evaluation; do not omit those families
  or tune training against this test result. Full expert/downstream results
  remain pending. Receipt: guarded-v2 base `independent_verification.json`.
- Fresh downstream evidence entrypoint now requires explicit admitted 50K
  route roots and fits through the content-bound mainline wrapper. It keeps
  compact candidate rows, trained models, no-LTR and no-R-GNN controls, checks
  inputs again after fitting, certifies retained file hashes, then removes
  scratch tables. Historical evidence CLI defaults are not used for this
  replacement. New admission/cache tests plus existing paired-control tests:
  24 passed in 0.243 s. This is synthetic regression evidence, not a completed
  downstream fit; the first seed-1 route pair is still pending.
- Beckmann expert seed 0 completed at epoch 54 / 5,886 updates (15 validation
  checks without improvement); training wall time 1,885.5 s. Independent
  job verification reports Route@1/3/5/10 = 41.2766/60.4255/63.4043/66.3830%
  on the unchanged 235 queries. Seed 1 is running. This is one expert result,
  not a six-family or three-seed summary; no manuscript score is replaced.
- Added a fixed-seed-1 route handoff after complete expert-job verification.
  It identifies the existing queue by PID/start time, never restarts training,
  verifies product-identity guard receipts and publishes same-checkpoint
  validation/test routes into `stage1_50k_downstream_routes_20260924`.
  The live controller is waiting for Beckmann seed 1; no routes are yet
  claimed published. Stage 1 summarization now checks retained guard evidence
  rather than trusting only metric/cache hashes. Stage 1 regression passes
  94 tests in 3.082 seconds; this is not a full-project or model-result audit.
- Minimal local manuscript update: main text and SI explicitly distinguish
  historical FULL performance from the running scratch-50K replacement.
  Only draft notices/method qualifiers and three SI contents page labels were
  revised; original numerical tables, drawings (including main Figure 1),
  equations and all non-document XML package entries remain byte-identical.
  Both DOCX files render; all 12 SI contents headings match the PDF pages.
  No new model performance was inserted, and private manuscript files remain
  outside Git. Numerical manuscript/SI replacement is still pending.
- Final guard freeze/relaunch: all 7,715 original validation/test queries and
  their 77,150 sampled augmentations pass CPU fidelity preflight. The 151-query
  Buchwald GPU test includes and successfully normalizes the CXSMILES case.
  Final Stage 1 regression: 83 tests / 3.137 s. New expert study
  `stage1_50k_fidelity_v2_expert_multiseed_20260924` and paired guarded-base study
  `stage1_50k_guarded_v2_base_test_20260924` are running. Beckmann seed 0 loads
  the completed scratch-50K best at update zero and 79,248 repaired train pairs.
  No new final performance or manuscript score is claimed. Initial observed
  GPU utilization 99%, memory 7,530 MiB; data disk about 22 GiB free.
- Complete augmentation audit: scratch-50K train/val have zero mismatches.
  Expert fidelity copies remove Beckmann 32 train/five val, Diels-Alder two
  train, Friedel-Crafts acylation one train. Total retained train/val pairs:
  846,045/39,375. Stored test tensors and all original condition queries stay
  unchanged. Independent six-dataset byte verification, fresh combined overlap
  audit and completed-base admission pass. One stored Beckmann test augmentation
  mismatch remains disclosed; actual inference regenerates guarded variants.
- Product-only inference guard: exactly one random draw per augmentation slot,
  identity-changing variants fall back to the original molecule; no score-based
  selection. A macrocycle CPU probe catches 81/200 altered variants. The initial
  GPU smoke passed; full Buchwald input handling exposed CXSMILES metadata.
  Parse complete CXSMILES and serialize equivalent ordinary SMILES before SPE
  when needed; never truncate metadata blindly. The short expert attempt was
  stopped before changing bound code. Final all-query preflight/relaunch pending.
- Subsequent augmentation-fidelity audit supersedes the earlier running-queue
  status: stopped identified expert handoff/runner/train PIDs
  8038/16475/16511/16529 after confirming altered Beckmann training identities.
  Existing best/last and logs are retained for diagnosis, not final promotion.
  Original condition splits and test queries are unchanged. The base full
  source-to-augmentation audit is still running; do not infer base failure
  from the expert finding. The original split-overlap audit and this chemical
  fidelity audit answer different questions.
- CPU probe of Beckmann test query 214: 81/200 random SMILES differ in
  stereo-sensitive canonical identity before SPE, while all 200 tokenization
  roundtrips are exact. Initial expert membership findings: Beckmann train/val
  32/5 unmatched augmentations, Friedel-CraftsAcylation train 1; full audit
  pending. No empirical test score is used to choose the repair policy.
- Initial Beckmann base decode completed on all 235 queries with zero exact
  hits, independently confirmed across 23,500 raw hypotheses. ID counts,
  ordering and hypothesis/probability pairing agree. A stricter source fidelity
  check failed: 418/2,350 augmented inputs contain UNK and two non-UNK variants
  of query 214 fail stereo-sensitive identity. The failed audit is preserved;
  DECODING_DIAGNOSTICS.md records the distinction and unresolved boundary.
  No query removal, test-driven tuning or paper-result promotion was performed.
- Started full six-family original-test base decoding alongside expert training
  in `Experiment/stage1_50k_base_test_20260924/`. Initial total GPU memory was
  about 7.7 GiB, versus about 5.8 GiB for the expert alone. Decoding settings
  and full query lists are unchanged; source/checkpoint/query bindings and
  per-family completion receipts guard this new base-only arm. Expert logs
  continue to advance during decoding. No paired final scores exist yet.
- The base/expert collector now rejects missing requested families and different
  ordered queries instead of silently producing a partial comparison. Full
  regression passed 138 tests in 17.402 seconds; model-bound training files
  were not edited for these scheduling/reporting changes.
- Completion update: scratch-50K base exited normally after 50 epochs /
  16,500 updates in 8,112.3 seconds. Actual checkpoint metadata confirms best
  at epoch 49 / 16,170 updates (validation loss 3.573), last at epoch 50
  (validation loss 3.701). Stop reason is the epoch cap, not early stopping or
  the update ceiling. Both hashes and completion/input/log bindings passed.
  `training_summary.json` records the observed checkpoint metadata.
- The automatic handoff independently reverified expert inputs and published
  new admission manifests bound to the completed base. Six-family/three-seed
  runner preflight passed; Beckmann seed 0 has completed epoch 1 on the repaired
  79,280-pair train split, restoring the new 50K best with optimizer updates
  reset to zero. Expert/downstream completion and paper numerical refresh are
  still pending; base training is not final evaluation.
- User explicitly chose filtered USPTO-50K random initialization, not upstream
  50K weights with possible FULL pretraining. No FULL retraining is scheduled.
- Downloaded RetroSim's Schneider corpus and verified Git blob
  `6baad99c6d412ce06a84c6144bf70a16bb1bad89`, pinned commit
  `0a272f0b5de833c448f41491e81e4dc00b4d85b0`. Its class-stratified nonshuffled
  80/10/10 split contains 40,008/5,001/5,007 source records (50,016 total).
- Added a versioned prepare/audit/train entrypoint. It filters original and
  transformed reaction identities before augmentation, then audits every actual
  text/bin pair and held-out membership before launching fresh training. Fixed
  SPE/ChEMBL tokenizer resources are reused, not pretrained model weights.
- Default base name is `USPTO_50K_FILTERED`. Restore checkpoints are forbidden
  in that mode; expert scripts no longer silently search for a FULL fallback.
  The legacy destructive raw-reset wrapper refuses this new profile and points
  to the audited entrypoint. Historical FULL evidence is preserved.
- Ten focused split/shell wiring tests passed. The new preparation job is started;
  no new experimental scores or completed model are claimed at launch.
- Full regression subsequently passed 112 tests in 15.588 seconds. Raw filtering
  retained train 39,688 and validation 4,988 after removing 320/13 source records;
  the 5,007 upstream test records remain unused for fitting. Augmentation and
  binarization started; the augmented-data audit is a separate prerequisite.
- Augmentation/binarization subsequently completed: 396,830 train pairs from
  39,683 distinct reactions and 49,880 validation pairs from 4,988 reactions.
  Five additional training records were rejected by the existing small-molecule
  rules (small_p=4, small_r=1). The full audit passed: zero unparseable pairs,
  exact text/tensor agreement, zero internal train/validation overlap, and zero
  intersections in all 48 Reaxys held-out comparisons.
- Fresh training started only after that audit. The actual log records no
  existing checkpoint and 47,056,896 trainable parameters. At 11:20 UTC,
  epoch 1 completed with 330 updates and validation loss 15.311; GPU execution
  was observed. This is ongoing training, not a completed model or new Route@k
  result. Best/last retention is enabled and old aliases remain untouched.
- The six-family combined chemical audit was rerun against the new 50K base
  membership and passed, rather than reusing the historical FULL receipt.
  Added scratch-50K expert admission with independent copy verification,
  completed-base/source/log/checkpoint bindings, unchanged condition validation
  inputs, and new hard-linked manifests to avoid duplicating tensor storage.
  An actual attempt during base training correctly refused intermediate best.
  Full regression passed 123 tests in 14.759 seconds; no expert training or
  repaired downstream performance is claimed by this input/admission work.
- The real input-only expert preflight then passed all six families, including
  exhaustive retained-text/tensor-byte checks: train 846,080, validation 39,380,
  test 39,740 augmented pairs. Base completion is still required for admission.
- Queued a process-identity-bound handoff (PID/start ticks) behind the live
  50K base coordinator. It serially admits, trains/decodes 18 expert jobs and
  verifies final tables; failure or missing completion never triggers an
  automatic base restart. Six unique historical family configs were checked.
  Best/last and per-job free-space admission remain enforced. Full regression
  passed 129 tests in 14.819 seconds. The live handoff is waiting, not evidence
  of completed expert training; downstream and manuscript numerics stay pending.
- Audited downstream entrypoints for the new Stage 1 source. The no-KNN runner
  had hardcoded historical routes/references; it now accepts explicit routes,
  requires bound matching reference provenance, checks shared validation/test
  expert weights, replays macro statistics and rejects stale compact evidence.
  Removed a hardcoded historical Route@10 from the shared reference loader.
  A real historical-table replay agrees with original Sys@10 mean/SD; it is
  not a new-model score. Final full tests passed 135 cases in 17.088 seconds.
  All 322 files bound by the live base-training input manifest were rehashed
  after these downstream edits: no changes. Base and handoff processes remain
  live; model-bound training code/data were not edited mid-run.
- Predeclared the unchanged statistical scope in DOWNSTREAM_HANDOFF.md:
  expert seeds reported separately; downstream seeds share the new seed-1
  experts, matching all six original fine-tuning logs and avoiding test-seed
  selection. Fresh validation routes and fresh base comparisons are required;
  the historical fixed-root evidence launcher must not be reused unchanged.

## 2026-09-15 - strict input repair audits complete; model decision pending

- Rebuilt base full scan passed in 1,288.10 seconds: 7,436,440 train and
  1,858,870 validation augmented pairs; 743,644/185,887 distinct reactions.
  Counts match the exclusion plan. All text/bin pairs agree, all sides parse,
  base internal overlap is zero and all 48 split/held-out intersections are zero.
- The subsequent combined chemical audit passed for all six expert families.
  Expert validation versus rebuilt-base train/validation overlap is zero for
  every family, in addition to the existing expert internal/condition-held-out
  checks. Before/after hashes bind 100 files. No original condition test query
  or prepared expert test pair was removed by this repair.
- The base scan exited successfully with multiprocessing semaphore-cleanup
  warnings after writing its passing receipt; its worker processes exited.
  The combined audit also completed successfully. No GPU training was started.
- Updated README, CURRENT_RESULTS, checklist, TODO and audit summaries to
  distinguish completed strict-input repair from pending trained-model repair.
  The strict fresh-base versus limited fixed-base decision remains unanswered;
  old results and manuscript numbers remain historical and qualified.

## 2026-09-15 - combined-audit binding and lossless baseline evidence retention

- Strengthened the expert chemical auditor to bind original condition files,
  expert text, base membership keys/receipt and the base audit's source hashes
  before and after scanning. Base-key file lengths must equal the audited row
  counts, and invalid reaction sentinels are rejected. A failed base audit cannot
  become a passing combined audit. This changes audit evidence, not model data.
- Queued the six-family combined audit behind the already-running base audit
  (PID 24544), without restarting the scan or admitting GPU training. A full
  passing base receipt is required before the combined command runs.
- Future B3/B4 cleanup now preserves full validation/test predictions and
  candidate tables as lossless gzip, plus explicit query manifests. Decompressed
  hashes must equal source bytes before work-directory removal. This supports
  later numerical replay while reducing disk use; it does not reconstruct
  previously pruned historical predictions.
- All 106 source regression tests passed in 13.099 seconds. Added tests cover
  truncated base-key membership, invalid sentinels, exact gzip preservation,
  corrupted copies and missing prediction evidence. No corrected training or
  model performance has been produced by these checks.

## 2026-09-15 - strict base copy verified; baseline reuse made fail-closed

- Strict base-input preparation completed with 7,436,440 train and 1,858,870
  validation augmented pairs, excluding 850/190 pairs. Independent verification
  passed exhaustive row partition, dictionary equality, and all retained text
  and tensor bytes. A separate full chemical/text-binary audit is running under
  `Experiment/stage1_split_repair_20260915/base_rebuilt_audit`; no new base
  checkpoint or scientific training result exists yet.
- Found a forward-reproduction risk in the baseline-repeat entrypoint: it could
  write new configuration before skipping old completed seeds, and imported B1
  from a hard-coded historical directory. This is not evidence of a numerical
  error in the original same-route comparison. New repeats bind input/code/model
  hashes, validate B3/B4 export contents, lock their study directory, verify
  completed output hashes on resume, and compute B1 once on current-study routes.
- Old data, checkpoints and result tables are preserved. New baseline exports
  and result roots are required after Stage 1 repair; no GPU jobs were started.
- Corrected the results-page claim to distinguish original split disjointness
  from augmented-input overlap. Manuscript numbers remain provisional and no
  additional manuscript rewrite was made during this step.
- Full source regression: 100 tests passed in 12.770 seconds. The 10 new tests
  include real small synthetic B3/B4 export generation, wrong-route cache bytes,
  changed checkpoint bytes, stale configuration and tampered output rejection.
  Synthetic exporter fixtures are not trained chemical models or new results.

## 2026-09-15 - strict expert inputs prepared and independently verified

- Strict-option expert copies remove 5,270 augmented training pairs and 180
  validation pairs from the earlier nonempty inputs. Retained totals are
  train 846,080, validation 39,380; all 39,740 prepared test pairs are unchanged.
  These are not raw reaction counts or evaluation query counts.
- Independent full-copy verification passed for all six families, including
  exhaustive source-row partition, unchanged dictionaries/text, and every
  surviving tensor byte. A separate full chemical scan finds zero internal
  expert train/val/test overlap and zero expert-training versus condition
  validation/test overlap. Base/model admission is not implied by these passes.
- The chemical audit's relative-path receipt bug was fixed and the failed
  audit rerun from the same unchanged copies. No successful audit was reused
  to conceal an incomplete check.
- The invalid process group 6300 was terminated using pending SIGTERM plus
  SIGCONT. All five recorded processes disappeared and the GPU compute-process
  list became empty. All six previous best/last checkpoint files remain.
- Base training now accepts an explicit versioned DATA_BIN, seed and run name,
  and UPDATE_ALIAS=0 prevents promotion. Expert seed entrypoints accept an
  explicit base checkpoint and strict post-augmentation protocol. Admission
  binds the actual checkpoint, expert output hashes and condition split hashes.
  Shell tests use a fake trainer, not a scientific training run.
- A strict base-input copy is in progress. New GPU training and model promotion
  remain withheld while the larger base-retraining decision is pending.

## 2026-09-15 - traced split fault and repair safeguards

- Full base audit completed in 1,288.85 seconds: 7,437,290 train and 1,859,060
  validation augmented pairs all agree with actual tensors. No unparseable
  text pairs, base train/validation collisions, or exact original condition-test
  matches were found. Base validation contains three transformed expert-test
  reactions (Buchwald-Hartwig, Diels-Alder, alkylation), not original full test
  reactions. Strict base refiltering requires excluding 85/19 unique train/val
  reactions (850/190 augmented pairs). The exact plan is saved; no new base
  training was launched while the user considers the compute tradeoff.
- All six independent positive overlap examples match actual source/target
  tensors and contain no unknown tokens. Every observed expert-training versus
  condition-held-out collision has a matching transformed raw training source;
  no traced original full reaction is itself in the held-out set. Per-family
  trace coverage is 36/36, 13/13, 5/5, 49/49, 23/23, and 61/61.
- Added a full USPTO augmented text/binary and membership scan, including
  base train/validation overlap, instead of certifying from raw CSV alone.
  Completion is recorded only in its final `audit.json`, not progress output.
- Added versioned paired-exclusion preparation with exact augmented row lineage,
  immutable test inputs, preserved surviving tensors, and independent membership
  re-audit support. This code does not admit training or propagate condition
  validation automatically; those steps remain pending.
- Invalidated-study summaries are refused. Expert training also refuses a
  merely nonempty prepared copy without scientific admission evidence. Tested
  both rejections against the actual old study/copy, without launching training.
- Main text and SI now explicitly qualify retained performance after the new
  training/test overlap discovery. Their 375 numeric cells still reproduce
  retained sources; this is not a clean-performance certificate. Rendered main
  text/SI are 18/23 pages after the final base-audit note; SI contents page
  numbers were updated. Main Figure 1 is unchanged. Draft changes remain local.
- The final bounded regression suite passed all 78 tests in 11.927 seconds.
- Vocabulary coverage audit found unknown-token encodings in base and expert
  tensors; this is a representational limitation, distinct from split leakage.
  The six independent leakage witnesses themselves contain no unknown tokens.
  Counts are in `Experiment/stage1_split_repair_20260915/vocabulary_audit.json`.

## 2026-09-15 - blocking post-augmentation split finding

- The actual nonempty augmented input audit found expert-training/condition-test
  canonical reaction overlaps in all families: Beckmann 20, Buchwald-Hartwig 6,
  Chan-Lam 2, Diels-Alder 19, Friedel-Crafts acylation 10, alkylation 27.
  These are exact reaction pairs, not product-only similarities.
- Raw-CSV split checks and successful metric replay were too narrow to establish
  leakage-free training. This supersedes any earlier overall integrity claim;
  saved numbers remain historical observations only pending clean reevaluation.
- Paused dedicated process group 6300 with SIGSTOP after verifying command,
  process group and session ownership. All descendants show stopped states;
  no checkpoints deleted. Requested validation-policy clarification remains
  pending, and removing the previously reported 98 overlaps alone is not enough.
- Evidence: `scripts/audit_stage1_augmented_splits.py` and
  `Experiment/final_release_audit_20260915/augmented_split_audit.json`.

## 2026-09-15 - final replay, manuscript correction and Stage 1 seeds

- Started the user-requested six-family Stage 1 expert fine-tuning queue for
  seeds 0/1/2 under `Experiment/stage1_finetune_multiseed_20260915/`.
  Fixed base and original validation/test memberships; verified nonempty
  augmented copies; old models untouched; best/last only. This supersedes the
  no-new-experiments freeze only for the requested expert-seed study.
- The first Beckmann seed-0 run has real GPU optimizer updates and checkpoints;
  no completed three-seed Stage 1 result is claimed yet. Existing 98 pretraining/
  condition-validation overlaps remain disclosed and are not fixed by seeds.
- Replayed 18 retained full runs and 36 exact paired controls, including model,
  split and route-cache hashes, candidate/support identities and temperatures.
  The source regression suite passed 61 tests on CPU without contending for GPU.
- Locally corrected the main manuscript and SI without changing the headline
  system result. 375 numeric cells verified; SI directory pages and rendering
  checked. The first main illustration remains excluded per user instruction.
- Synchronized current result/ablation descriptions: the identity-matched
  temperature comparison is 11.82 versus 13.93 C, not the original headline
  11.49 C versus the new no-graph control. Historical data remain intact.
- Evidence, limitations and still-running work:
  `Experiment/final_release_audit_20260915/REPORT.md`. Private drafts and row-level
  artifacts stay ignored; this step did not push GitHub changes.
- First new expert job completed and independently verified: Beckmann seed 0,
  52 epochs / 5,720 updates, validation-patience stop, 235 test queries including
  four no-route failures. Route@1/3/5/10 = 48.51/63.40/66.81/71.49%. Seed 1
  started automatically. Added a final-summary verifier that refuses incomplete
  grids and checks checkpoint/query identity and metric replay before reporting
  equal-family means and sample SD; five aggregation tests passed.

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
