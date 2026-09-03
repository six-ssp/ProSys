# ProSys Development Log

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
