# Script Map

## Maintained Entrypoints

- `reproduce_mainline_from_raw.sh`: end-to-end reproduction from raw input.
- `run_stage23_non_oracle_suite.sh`: current parallel KNN + ReaFNN Stage 2/3
  suite.
- `run_verified_mainline.py`: maintained cache-safe wrapper used by the suite;
  content hashes and family locks prevent stale or concurrent cache reuse.
- `predict_product.py`: read-only product-to-system inference with retained
  family models, no training or validation selection, optional frozen routes.
  It rejects missing/mismatched feature-source manifests before inference;
  historical weights need matching code, not silent reuse with repaired inputs.
- `run_stage23_mainline_non_oracle.py`: family-level Stage 2/3 driver.
- `run_current_mainline_temperature_ablation.py`: matched three-seed R-GNN
  temperature-representation control for the current parallel mainline; it
  verifies that the 52-feature tabular control leaves Stage 1/2/3 system
  metrics and conditional support unchanged.
- `setup_prosys_env.sh`: environment setup.

## Evaluation and Reporting

- `run_mainline_evidence.py`, `finalize_mainline_evidence.py`: bounded-concurrency
  reconstruction and complete-study export, with per-candidate paired controls.
- `summarize_mainline_evidence.py`, `export_evidence_examples.py`: retained-row
  metric replay, subgroup/failure reports and first-hit-correct cases.
- `benchmark_product_inference.py`: one cold full-product query per family
  after reconstruction workers exit; not a throughput benchmark.
- `audit_inference_and_cache.py`, `audit_canonical_fragments.py`: deployment
  parity/cache smokes and whole-molecule canonical-component audit.
- `run_reafnn_only_ablation.py`, `audit_reafnn_only_evidence.py`: strict
  KNN-removal intervention and its per-candidate evidence audit.
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
