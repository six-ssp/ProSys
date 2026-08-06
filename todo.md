# ProSys TODO

更新日期：`2026-07-12`

## Active

- Keep the maintained scope fixed to the current 6-family mainline and avoid reintroducing retired families into default configs, scripts, or reports.
- When rerunning from raw data, keep `data_preprocess/`, `scripts/reproduce_mainline_from_raw.sh`, and `outputs/checklist_stats/` synchronized.
- Continue pruning large intermediate artifacts once the corresponding summary tables have been confirmed stable.

## Maintenance

- Keep `README.md`, `checklist.md`, `MAINTENANCE.md`, and `log.md` synchronized with `outputs/stage23_mainline/` and `outputs/checklist_stats/`.
- Put new exploratory notebooks, one-off scripts, and obsolete result trees into `Experiment/` instead of the repo root.
- Do not extend the archived neural-V2 path as the maintained experiment line.

## Done

- Promoted `KNN + XGBoost` to the maintained mainline.
- Split the maintained post-Stage-1 code into `stage2_KNN/` and `stage3_XGBoost/`.
- Consolidated the official reporting scope to the current 6 retained families.
- Refreshed the maintained Non-Oracle result roots under `outputs/stage23_mainline/` and `outputs/checklist_stats/`.
- Unified temperature reporting to `Temp MAE` plus `Temp±5C / Temp±10C / Temp±20C`.
