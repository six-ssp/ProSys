# ProSys TODO

更新日期：`2026-08-30`

## Active

- Keep the maintained scope fixed to the current 6-family mainline and avoid reintroducing retired families into default configs, scripts, or reports.
- Treat `Experiment/stage23_product_morgan_reafnn_multiseed_20260830/` and
  `CURRENT_RESULTS.md` as the authoritative fixed-Stage-1 three-seed result
  record. Do not mix it with the pre-hardening 2026-08-09 snapshot.
- Treat Experiment/current_mainline_matched_ablation_multiseed_20260830/ as
  the authoritative source for current ReaFNN and XGB-LTR attribution figures;
  do not substitute historical direct-R-GNN ablation values.
- When rerunning from raw data, keep `data_preprocess/`, `scripts/reproduce_mainline_from_raw.sh`, and `outputs/checklist_stats/` synchronized.
- Continue pruning large intermediate artifacts once the corresponding summary tables have been confirmed stable.

## Maintenance

- Keep `README.md`, `CURRENT_RESULTS.md`, `checklist.md`, `MAINTENANCE.md`, and
  `log.md` synchronized with the promoted compact mainline record.
- Put new exploratory notebooks, one-off scripts, and obsolete result trees into `Experiment/` instead of the repo root.
- Do not extend the archived neural-V2 path as the maintained experiment line.

## Done

- Promoted `KNN + XGBoost` to the maintained mainline.
- Split the maintained post-Stage-1 code into `stage2_KNN/` and `stage3_XGBoost/`.
- Consolidated the official reporting scope to the current 6 retained families.
- Refreshed the maintained Non-Oracle result roots under `outputs/stage23_mainline/` and `outputs/checklist_stats/`.
- Unified temperature reporting to `Temp MAE` plus `Temp±5C / Temp±10C / Temp±20C`.
- Promoted the post-hardening product-Morgan KNN + ReaFNN three-seed result as
  the official Stage 2/3 mainline: `Sys@10 = 44.62 +/- 0.42%`.
- Completed the current matched three-seed ablations: removing ReaFNN lowers
  Sys@10 by 4.76 pp, and removing XGB-LTR from an identical Stage 2 pool lowers
  Sys@10 by 8.17 pp.
