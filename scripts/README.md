# Script Map

## Maintained Entrypoints

- `reproduce_mainline_from_raw.sh`: end-to-end reproduction from raw input.
- `run_stage23_non_oracle_suite.sh`: current parallel KNN + ReaFNN Stage 2/3
  suite.
- `run_stage23_mainline_non_oracle.py`: family-level Stage 2/3 driver.
- `run_current_mainline_temperature_ablation.py`: matched three-seed R-GNN
  temperature-representation control for the current parallel mainline; it
  verifies that the 52-feature tabular control leaves Stage 1/2/3 system
  metrics and conditional support unchanged.
- `setup_prosys_env.sh`: environment setup.

## Evaluation and Reporting

- `collect_stage1_base_vs_tuned.py`, `collect_checklist_stats.py`, and
  `analyze_fixed_manifest_checks.py`: reproducibility and reporting audits.
- `run_current_mainline_matched_ablations.py`: historical serial control only;
  it is not a paired ablation of the current parallel candidate pool.

## Historical Launchers

`Experiment/legacy_stage2/launchers/` contains the neural-V2 launchers and
hit-rate summarizer. They reproduce the archived neural-V2 line and are not
inputs to the maintained workflow. Retired joint/negative scripts live in
`Experiment/local_archive/retired_scripts/` locally.
