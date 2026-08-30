# Current Mainline Compact Result Record (2026-08-30)

## Scope

This promoted artifact is a conditional Stage-2/3 robustness experiment. The
persisted fixed split, the six test manifests, and the Stage-1 route caches are
held fixed across seeds. It does not claim variability from retraining EditRetro
Stage 1.

Randomized learned components are ReaFNN, Reaction-GNN, and XGBoost (subsample and column-subsample). KNN retrieval, route caches, data split, vocabularies derived from the fixed training split, candidate evaluation, and validation-only score-fusion selection are fixed by protocol.

Values below are unweighted macro averages over the six families. Rates are percentages; standard deviations use the sample definition (`ddof=1`). Temperature MAE is the unweighted mean of family-level conditional MAEs, and its support can vary by seed.

## Mean +/- Std

| System | Seeds | Coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 | Temp MAE (C) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProSys (current mainline) | 0,1,2 | 54.44 +/- 0.14 | 27.12 +/- 0.37 | 36.84 +/- 0.80 | 40.47 +/- 0.77 | 44.62 +/- 0.42 | 33.28 +/- 0.48 | 34.70 +/- 0.53 | 11.73 +/- 0.54 |

Temperature within +/-5 / +/-10 / +/-20 C is
`40.59 +/- 0.56% / 61.80 +/- 2.36% / 83.04 +/- 1.10%`.

## Per-Seed Macro Results

| System | Seed | Families | Test manifest | Candidate slates | Sys@1 | Sys@10 | MRR | nDCG@10 | Temp support | Temp MAE (C) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProSys (Stage 2/3 fixed-Stage-1 robustness) | 0 | 6 | 3860 | 3833 | 27.53 | 45.08 | 33.83 | 35.30 | 1801 | 11.16 |
| ProSys (Stage 2/3 fixed-Stage-1 robustness) | 1 | 6 | 3860 | 3833 | 26.84 | 44.51 | 33.04 | 34.51 | 1795 | 11.78 |
| ProSys (Stage 2/3 fixed-Stage-1 robustness) | 2 | 6 | 3860 | 3833 | 26.98 | 44.27 | 32.96 | 34.28 | 1799 | 12.24 |

## Retention policy

The `compact/` directory keeps per-family metric records and model metadata. When `--prune-run` is used, large candidate tables, scored tables, and checkpoint files under that explicitly supplied `*_work` directory are removed only after this collector has successfully written the compact record.
