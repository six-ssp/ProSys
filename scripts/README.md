# Script Map

## Maintained Entrypoints

- `run_stage1_50k_from_scratch.py`: current filtered USPTO-50K base preparation,
  full augmented-input audit, and random-initialized training without FULL weights.
- `admit_stage1_50k_experts.py`: verify repaired expert tensors against that
  base; full admission requires normally completed training and bound best/last.
- `continue_stage1_50k_experts.py`: wait on an identified live base coordinator,
  then serially admit, train and verify six-family expert seeds; no automatic
  restart or old-checkpoint fallback on a failed/interrupted base run.
- `decode_stage1_50k_base.py`: generate complete original-family test routes
  from the admitted completed 50K base, with frozen decoding and query checks;
  this base-only arm does not substitute for the new expert comparison.
- `run_stage1_multiseed.py`: six-family fixed-base expert repeats on admitted
  inputs, with isolated checkpoints and unchanged original condition queries.
- `reproduce_mainline_from_raw.sh`: historical raw-input pipeline; refuses the
  new 50K profile before destructive reset and redirects to its audited entrypoint.
- `run_stage23_non_oracle_suite.sh`: current parallel KNN + ReaFNN Stage 2/3
  suite.
- `run_verified_mainline.py`: maintained cache-safe wrapper used by the suite;
  content hashes and family locks prevent stale or concurrent cache reuse.
- `predict_product.py`: read-only product-to-system inference with retained
  completed 50K family bundles, no training or validation selection, optional
  bound frozen routes. It rejects historical bundles, altered feature sources,
  checkpoints and unbound caches. The checkpoint defaults to the expert used
  to fit that bundle, not the historical alias. `--checkpoint` allows only a
  byte-identical relocation.
- `product_route_inference.py`: reuses the bundle's retained guarded decoder
  command for a new product and verifies one-query identity-protection evidence.
  Fresh single-product augmentations need not equal a batched test-cache draw.
- `audit_product_prediction.py`: compares a fixed-route prediction with a
  retained test query, including all 52 ranking / 180 temperature features,
  complete candidate order, final scores and temperature predictions. This
  checks deployment parity, not a new generalization estimate.
- `benchmark_product_inference.py`: requires explicit `--artifact-root` (one
  fresh downstream seed) and a new `--output-root`. No historical result reuse;
  live training queues or other GPU compute processes block measurement.
  Run after the studies finish, not concurrently as a latency benchmark.
- `run_stage23_mainline_non_oracle.py`: family-level Stage 2/3 driver.
- `run_current_mainline_temperature_ablation.py`: matched three-seed R-GNN
  temperature-representation control for the current parallel mainline; it
  verifies that the 52-feature tabular control leaves Stage 1/2/3 system
  metrics and conditional support unchanged.
- `setup_prosys_env.sh`: environment setup.

## Evaluation and Reporting

- `../Experiment/release_50k_20260925/verify_public_release.py`: standard-library
  verification of public release hashes, family/seed grids, denominators and
  macro/sample-SD arithmetic; runs from a clone without models or raw data.
- `../Experiment/final_release_50k_20260925/audit.py` (local-only, not uploaded):
  data/checkpoint/query/summary/private-document reconciliation. Requires the
  retained licensed inputs and manuscript files; not the public clone checker.
- `../Experiment/stage1_decode_diagnostic_20260925/run_length_guarded.py`:
  versioned same-checkpoint recovery for a generated hypothesis that exceeds
  the decoder's actual positional capacity. Invalidates that hypothesis while
  retaining every query and beam slot. This is the explicit final-expert
  recovery, not a silently changed default decoder; see its README/admission.
- `audit_50k_query_diagnostics.py`: explicit-root, read-only failure and subgroup
  replay for the fresh study. Reconstructs every candidate's exact-system label,
  checks query/route membership, regenerates query annotations and verifies
  retained records. Keeps no-route queries, reports empty subgroups as NA and
  defaults to all six families; missing requested runs prevent report creation.
  Partial reports are labeled and cannot supply a full-study macro result.
- `audit_50k_paper_auxiliary.py`: explicit-root, no-fit replay for direct
  Condition@k (before route pairing, all query identities) and validation
  Stage 2 fusion curves/selected weights. Requires completed baseline and
  mainline evidence, rechecks their provenance, and exports per-family/seed
  CSVs for SI S12/S20. Full scope is six families by default; partial output
  is labeled and cannot support a full manuscript table.
- `collect_50k_comparisons.py`: explicit-root, requested-scope summary of the
  fresh mainline, four baselines and four removal controls. Replays mainline,
  baselines and no-KNN evidence; verifies and replays retained KNN-only rows
  after scratch pruning. Requires the exact family/model/seed grid, keeps B1
  deterministic (one fit per family, no manufactured SD), and reports separate
  conditional-temperature support. Default six-family export refuses missing
  evidence before creating an output directory. Partial exports are labeled.
- `audit_50k_repeat.py`: independently verifies two same-seed downstream studies
  against the same route admission, then compares all retained candidate values,
  graph embeddings, both neural payloads and both XGBoost models. Differences
  are reported rather than used to choose the more favorable run.
- `export_evidence_examples.py`: requires explicit new study/route/output roots,
  verifies each requested family, and exports three deterministic first-hit-aware
  cases with branch/token/neighborhood and graph/ranker traces. Partial-family
  export is labeled as partial and never overwrites root `example.md`.
- `collect_stage1_base_vs_tuned.py`: requires both explicit route roots and
  guarded, identical-query base/expert caches. It reports total query counts in
  both macro and weighted rows; single-expert comparisons are not three-seed SD.
- `continue_50k_downstream_studies.py`: identified-live-controller handoff for
  the remaining admitted family route pairs. Freezes sources, runs at most
  two downstream children, verifies three-seed mainline results, then runs
  and audits all baselines and both Stage 2 removal controls. No automatic
  retry, partial-cache recovery or historical route fallback. Its configured-
  family completion is not the manuscript/global final-audit certificate.
- `audit_50k_baseline_evidence.py`: reconstructs direct product systems from
  retained condition predictions/new routes, relabels external candidates,
  replays validation-only fusion selection and all retained test metrics.
- `audit_50k_knn_only_evidence.py`: verifies zero neural scores/proposals and
  independently recomputes train-library statistics (including reaction-level
  exclusion in training rows); replays all seeds before optional scratch
  pruning. Four legacy `reafnn_` metadata columns are library statistics,
  not learned ReaFNN outputs.
- `run_50k_downstream_evidence.py`: fresh 50K-expert route evidence, one
  family/downstream seed per invocation. Requires explicit route-study,
  study-root and scratch-root arguments, admitted fixed expert seed 1,
  matching guarded validation/test routes and original split identities.
  Fits through the verified mainline wrapper, retains exact-pool no-LTR and
  same-support no-R-GNN temperature controls, and hashes compact evidence
  before removing scratch tables. Only its evidence manifest authorizes reuse;
  a standalone completion file is insufficient. New fits are recorded in
  `Experiment/stage23_50k_evidence_20260924/`; launch is not completion.
- `summarize_50k_downstream_evidence.py`: explicit-root summary for those fresh
  fits; requires all 18 content-certified runs, replays retained candidates and
  paired controls, and exports per-family/per-seed plus equal-family macro
  mean/sample-SD tables. Missing temperature support remains NA. It does not
  read historical result roots or report Stage 2/baselines as complete.
- `run_mainline_evidence.py`: bounded-concurrency historical reconstruction;
  its default CLI paths are historical.
  The reusable child supports explicit routes, but new 50K studies must use
  `run_50k_downstream_evidence.py`, not that historical CLI default.
- `finalize_mainline_evidence.py`: obsolete historical orchestration. Do not
  launch it for 50K studies; its no-argument case/cost calls predate the required
  explicit-root interfaces and are no longer a valid finalization path.
- `summarize_mainline_evidence.py`: historical retained-row metric/subgroup
  report; reusable alignment helpers only are used by the new 50K auditors.
- `audit_inference_and_cache.py`, `audit_canonical_fragments.py`: deployment
  parity/cache smokes and whole-molecule canonical-component audit.
- `run_reafnn_only_ablation.py`, `audit_reafnn_only_evidence.py`: strict
  KNN-removal intervention and its per-candidate evidence audit.
  New no-KNN runs accept explicit test/validation route roots and require
  matching `--mainline_reference` / `--mainline_compact_root` arguments;
  reference provenance and retained-result inputs are checked before reuse.
- `collect_stage1_base_vs_tuned.py`, `collect_checklist_stats.py`, and
  `analyze_fixed_manifest_checks.py`: reproducibility and reporting audits.
- `run_current_mainline_matched_ablations.py`: paired current-parallel Stage-2
  and Stage-3 controls. The three-seed `knn_only` and `no_xgb_ltr` arms are
  complete in separate artifacts; the launcher reuses compact records safely
  when a run is resumed after interruption.
- `build_paper_statistics_bundle.py`: archival July-artifact builder only. It
  is not a source for the maintained parallel mainline and requires a rewrite
  before generating a current paper-statistics bundle.

## Historical Launchers

`Experiment/legacy_stage2/launchers/` contains the neural-V2 launchers and
hit-rate summarizer. They reproduce the archived neural-V2 line and are not
inputs to the maintained workflow. Retired joint/negative scripts live in
`Experiment/local_archive/retired_scripts/` locally.
