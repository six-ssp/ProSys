# Multi-Seed Baseline Results

**Status:** current formal baseline record, completed `2026-08-10`.

This report supersedes the single-seed numerical comparison in
[`current_baseline_results_20260727.md`](current_baseline_results_20260727.md),
which remains an archived point snapshot for provenance. It uses the current
six-family fixed Stage 1 route caches and is directly comparable with the
three-seed ProSys results in [`CURRENT_RESULTS.md`](../CURRENT_RESULTS.md).

## Protocol

- B2 Product-GNN, B3 EditRetro + Sequential FNN, and B4 EditRetro +
  Reaction-GCNN were independently retrained from scratch with seeds `0`,
  `1`, and `2`.
- The family splits, train-only condition libraries, label normalization,
  Stage 1 test/validation route caches, top-20 context cap, and evaluator were
  fixed across all runs.
- Early stopping and route/condition fusion selection use validation data only.
  The test manifest is consumed once for final evaluation.
- The full denominator is `3,860` held-out products. In every method and seed,
  `3,833` products have at least one Stage 1 candidate slate and the remaining
  `27` products stay as zero-valued end-to-end failures.
- B1 Product-Bernoulli Naive Bayes is a deterministic closed-form estimator
  under its fixed data and hyperparameters. It is reported once rather than
  pseudo-replicated.
- Percentages below are unweighted six-family macro averages. `+/-` is the
  sample standard deviation across the three independent seeds (`ddof=1`).

The exact fixed-cache hashes, hyperparameters, and compact per-seed metadata
are stored in `outputs/baselines/multiseed_20260810/experiment_config.json`
and `seed_<n>/external_compact/`.

## Macro-Average System Results

| Method | Replication | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 Product-Bernoulli Naive Bayes | deterministic | 31.68 | 9.10 | 14.87 | 17.45 | 21.48 | 13.15 | 14.56 |
| B2 Product-GNN | 3 seeds | 38.33 +/- 0.28 | 6.11 +/- 0.49 | 12.84 +/- 0.56 | 16.75 +/- 0.43 | 23.03 +/- 0.73 | 11.40 +/- 0.54 | 13.32 +/- 0.58 |
| B3 EditRetro + Sequential FNN | 3 seeds | 45.85 +/- 0.26 | 16.75 +/- 0.27 | 24.32 +/- 0.28 | 27.68 +/- 0.40 | 31.71 +/- 0.10 | 22.01 +/- 0.22 | 23.64 +/- 0.18 |
| B4 EditRetro + Reaction-GCNN | 3 seeds | 38.15 +/- 0.24 | 7.43 +/- 0.21 | 13.16 +/- 0.10 | 16.39 +/- 0.34 | 21.10 +/- 0.28 | 11.88 +/- 0.11 | 13.27 +/- 0.10 |
| ProSys current mainline | 3 seeds | 48.52 +/- 0.59 | 28.96 +/- 0.91 | 36.61 +/- 1.05 | 39.53 +/- 1.24 | 42.90 +/- 1.00 | 33.78 +/- 0.96 | 35.13 +/- 0.97 |

The strongest baseline is B3 at `31.71 +/- 0.10%` Sys@10. The descriptive
gap to the current ProSys mean is `11.19` percentage points. This is a
comparison of independently repeated methods on the same fixed manifest; it
does not by itself constitute a hypothesis test.

## Family-Resolved Sys@10

| Family | B1 deterministic | B2 Product-GNN | B3 Sequential FNN | B4 Reaction-GCNN | ProSys current mainline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 8.51 | 10.07 +/- 1.23 | 22.27 +/- 1.49 | 15.32 +/- 0.74 | 27.94 +/- 1.72 |
| Buchwald-Hartwig | 21.75 | 14.41 +/- 0.38 | 19.93 +/- 0.36 | 12.92 +/- 1.00 | 46.07 +/- 0.56 |
| Chan-Lam | 39.49 | 41.62 +/- 1.57 | 56.58 +/- 1.21 | 35.90 +/- 1.33 | 62.39 +/- 1.55 |
| Diels-Alder | 12.73 | 15.66 +/- 0.38 | 21.39 +/- 0.73 | 13.21 +/- 0.67 | 26.99 +/- 0.87 |
| Friedel-Crafts Acylation | 35.16 | 31.16 +/- 3.04 | 43.37 +/- 2.32 | 30.46 +/- 0.64 | 47.86 +/- 5.16 |
| Friedel-Crafts Alkylation | 11.23 | 25.25 +/- 0.59 | 26.73 +/- 0.72 | 18.80 +/- 1.65 | 46.16 +/- 0.22 |
| Macro average | 21.48 | 23.03 +/- 0.73 | 31.71 +/- 0.10 | 21.10 +/- 0.28 | 42.90 +/- 1.00 |

## Conditional Temperature Results

Temperature is evaluated only after recovery of an exact ranked system with a
valid temperature label. Its support therefore differs by model and seed and
must not be interpreted as an unconditional regression benchmark.

| Method | Valid exact-system matches per seed | MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | --- | ---: | ---: | ---: | ---: |
| B3 EditRetro + Sequential FNN | 1,464 / 1,424 / 1,455 | 12.20 +/- 0.29 | 47.81 +/- 2.32 | 66.04 +/- 1.32 | 81.11 +/- 0.47 |
| ProSys current mainline | 1,603 / 1,592 / 1,576 | 10.75 +/- 0.14 | 41.66 +/- 1.39 | 64.14 +/- 1.65 | 84.74 +/- 0.38 |

B1, B2, and B4 do not include temperature heads. The temperature comparison
is conditional and is reported separately from system-ranking accuracy.

## Audit and Artifacts

- `per_family_seed_metrics.csv` contains `54` records, exactly
  `3 methods x 3 seeds x 6 families`.
- `macro_by_seed.csv` records each seed-specific macro average; all three
  methods have the same manifest size and candidate-slate count in each seed.
- `macro_mean_std.csv` is the source of the macro table above.
- `RESULTS.md` is a compact machine-generated rendering of the baseline-only
  aggregate.
- Large B3/B4 candidate tables were deleted only after copying the family
  run metadata, validation fusion selections, and model metadata into the
  compact audit folders. This preserves reproducibility evidence within the
  available disk budget.

The direct Product-GNN condition model uses the target product only and sees
Stage 1 routes only after it has ranked contexts. B3 and B4 receive the same
predicted routes used by the mainline. None of these models receives gold
reactants, gold conditions, or a family label as a learned molecular feature at
test time.
