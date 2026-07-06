# ProSys TODO

更新日期：`2026-07-05`

## Active

- Improve Stage 1 route recall on weak coupling families:
  - `Kumada`
  - `Negishi`
  - `Grignard`
- Audit why `Original FNN` on `Beckmann` has zero full positives in the Non-Oracle pool while `KNN` and `Cluster` still retrieve positives.
- Decide whether the final paper table should report:
  - all 10 families
  - or the filtered subset where `KNN+XGB sys@10 > 20%`
- Recheck temperature quality on the current mainline:
  - `Temp@10C`
  - `Temp@20C`
  - now both are top-10 end-to-end hit rates
- Add one short methods note or figure explaining:
  - why `KNN` is screening
  - why `XGBoost` is reranking
  - why `SVM` is not chosen despite stronger raw `sys@k` on the filtered subset

## Maintenance

- Keep `README.md`, `ProSys_goal.md`, `MAINTENANCE.md`, `log.md` synchronized with the current mainline.
- Put new exploratory notebook / one-off script / old result tree into `Experiment/` instead of the repo root.
- Do not extend the old neural-V2 path as the main experiment line.

## Done

- Promoted `KNN + XGBoost` to the maintained mainline.
- Split code into:
  - `stage2_KNN/`
  - `stage3_XGBoost/`
- Completed the 10-family Non-Oracle stage2/stage3 suite at `outputs/stage23_non_oracle_all10/`.
- Added historical baseline and Stage 2 / Stage 3 ablation reports.
- Switched temperature reporting from error-only summaries to:
  - `Temp@10C`
  - `Temp@20C`
- Unified the current temperature metric definition:
  - top-10 contains a full system hit whose temperature error is within threshold
- Cleaned the repo root and moved non-mainline but valuable material into `Experiment/`.
