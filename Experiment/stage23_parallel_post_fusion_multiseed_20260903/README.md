# Fixed-Stage-1 Multi-Seed Robustness

## Scope

This is a conditional Stage-2/3 robustness experiment. The persisted fixed split, the six test manifests, and the Stage-1 route caches are held fixed across seeds. It does not claim variability from retraining EditRetro Stage 1.

Randomized learned components are ReaFNN, Reaction-GNN, and XGBoost (subsample and column-subsample). KNN retrieval, route caches, data split, vocabularies derived from the fixed training split, candidate evaluation, and validation-only score-fusion selection are fixed by protocol.

Values below are unweighted macro averages over the six families. Rates are percentages; standard deviations use the sample definition (`ddof=1`). Temperature MAE is the unweighted mean of family-level conditional MAEs, and its support can vary by seed.

## Current Headline

This is the reportable robustness record for the maintained parallel mainline.
The fixed test manifest contains 3,860 product identities. Every seed has
3,833 candidate slates; the remaining 27 no-slate identities remain in all
full-system metric denominators and therefore contribute zero.

| Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |

The fixed Stage-1 macro Route@1/3/5/10 is `43.46/56.69/59.97/63.20%` for all
three runs. It is fixed by design and is not included in the reported Stage-2/3
standard deviations.

Temperature is conditional on the highest-ranked exact full-system match with
a valid temperature. Its macro MAE is `11.49 +/- 0.26 C`; hit rates within
`+/-5`, `+/-10`, and `+/-20 C` are `41.41 +/- 2.16%`, `63.09 +/- 1.76%`, and
`83.74 +/- 0.69%`, respectively. Supports are 1,785, 1,795, and 1,797 for
seeds 0, 1, and 2 because ranked exact system matches can differ by seed.

## Per-Family Mean +/- Std

| Family | Coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 | Temp. MAE (C) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 42.55 +/- 0.43 | 13.90 +/- 4.93 | 20.00 +/- 3.32 | 23.83 +/- 1.85 | 28.09 +/- 2.98 | 18.58 +/- 4.06 | 19.64 +/- 3.75 | 11.03 +/- 1.39 |
| Buchwald-Hartwig | 59.21 +/- 0.28 | 33.36 +/- 0.21 | 45.13 +/- 0.40 | 48.50 +/- 0.48 | 52.44 +/- 0.19 | 40.30 +/- 0.02 | 42.14 +/- 0.14 | 11.92 +/- 0.41 |
| Chan-Lam | 72.91 +/- 0.15 | 37.35 +/- 5.79 | 50.68 +/- 5.14 | 56.07 +/- 3.64 | 61.62 +/- 1.92 | 45.68 +/- 4.95 | 48.01 +/- 4.32 | 5.83 +/- 0.39 |
| Diels-Alder | 34.73 +/- 0.08 | 15.84 +/- 1.45 | 21.43 +/- 0.87 | 23.36 +/- 0.68 | 26.47 +/- 0.38 | 19.51 +/- 1.06 | 20.23 +/- 0.85 | 18.55 +/- 1.10 |
| Friedel-Crafts acylation | 64.84 +/- 0.00 | 24.14 +/- 2.46 | 35.93 +/- 3.05 | 40.84 +/- 2.83 | 48.91 +/- 2.47 | 32.15 +/- 2.56 | 34.06 +/- 2.78 | 10.86 +/- 0.02 |
| Friedel-Crafts alkylation | 51.32 +/- 0.23 | 26.21 +/- 2.58 | 37.52 +/- 1.87 | 42.08 +/- 1.51 | 45.09 +/- 1.29 | 32.96 +/- 2.15 | 34.87 +/- 1.94 | 10.78 +/- 0.52 |
| MACRO-AVG | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 | 11.49 +/- 0.26 |

## Reproduction and Retention

Each run uses the maintained parallel configuration: product-Morgan KNN with
`K=64`, independent ReaFNN top-64 historical contexts, validation-only family
post-fusion over `w in {0.0, 0.1, ..., 1.0}`, and a route-local top-20 context
cap. Stage 3 retrains the fixed 52-feature XGB-LTR, while the separate
128-dimensional R-GNN plus XGBoost branch reports temperature only.

`per_family_seed_metrics.csv`, `macro_by_seed.csv`, `macro_mean_std.csv`, and
`per_family_mean_std.csv` are the machine-readable tables. `compact/` retains
the 18 full `result.json` records, including protocol and model metadata.
`source_runs/` retains each run's `all_results.json`, overview, flattened table,
and cleanup manifests. Large candidate tables and model checkpoints were
intentionally pruned after each completed family to respect disk limits.

## Mean +/- Std

| System | Seeds | Sys@1 | Sys@10 | MRR | nDCG@10 | Temp MAE (C) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ProSys (Stage 2/3 fixed-Stage-1 robustness) | 0,1,2 | 25.13 +/- 1.20 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 | 11.49 +/- 0.26 |

## Per-Seed Macro Results

| System | Seed | Families | Test manifest | Candidate slates | Sys@1 | Sys@10 | MRR | nDCG@10 | Temp support | Temp MAE (C) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProSys (Stage 2/3 fixed-Stage-1 robustness) | 0 | 6 | 3860 | 3833 | 23.89 | 43.44 | 30.35 | 32.12 | 1785 | 11.53 |
| ProSys (Stage 2/3 fixed-Stage-1 robustness) | 1 | 6 | 3860 | 3833 | 25.23 | 43.40 | 31.65 | 33.18 | 1795 | 11.22 |
| ProSys (Stage 2/3 fixed-Stage-1 robustness) | 2 | 6 | 3860 | 3833 | 26.28 | 44.47 | 32.59 | 34.16 | 1797 | 11.74 |

## Retention policy

The `compact/` directory keeps per-family metric records. This run used
per-family cleanup during execution, so full protocol metadata remains embedded
in each `result.json`, while regenerable candidate tables and checkpoint files
were intentionally not retained.
