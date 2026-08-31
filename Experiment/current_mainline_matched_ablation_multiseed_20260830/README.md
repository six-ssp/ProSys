# Matched Current-Mainline Ablations

## Scope

This record is matched to the promoted product-Morgan mainline: fixed Stage 1 route caches, product-Morgan KNN (radius 2, 4,096 bits, K=64), a 64-context wide pool, and a 20-context cap.

Each ablation was evaluated over 6 family/families and seeds 0, 1, 2. Each family/seed run was compacted immediately after validation, so raw candidate tables and binary checkpoints are intentionally absent.

## Arms

- **KNN-only + XGB-LTR:** removes ReaFNN but retrains a tabular 52-feature XGB-LTR on its changed candidate distribution.
- **Full Stage 2 + deterministic no-XGB-LTR:** preserves the full ReaFNN Stage 2 generation configuration and sorts using only the fixed Stage 1/2 prior: route rank, route probability, Stage 2 initial score, KNN evidence, and stable reagent/solvent tie-breaks. It has no fitted ranking parameters.
- Both ablation arms skip temperature because the temperature branch does not add candidates or contribute to system ranking; temperature omission cannot change Sys@k.

## Macro Results

| Arm | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full current mainline | 54.44 +/- 0.14 | 27.12 +/- 0.37 | 36.84 +/- 0.80 | 40.47 +/- 0.77 | 44.62 +/- 0.42 | 33.28 +/- 0.48 | 34.70 +/- 0.53 |
| KNN-only + XGB-LTR | 53.39 +/- 0.00 | 22.72 +/- 1.86 | 31.44 +/- 2.45 | 35.19 +/- 2.23 | 39.86 +/- 2.08 | 28.63 +/- 1.98 | 29.85 +/- 2.05 |
| Full Stage 2 + deterministic no-XGB-LTR | 54.44 +/- 0.14 | 24.87 +/- 0.35 | 32.23 +/- 0.14 | 34.27 +/- 0.11 | 36.45 +/- 0.08 | 29.56 +/- 0.20 | 30.35 +/- 0.12 |

## Per-Family Sys@10

| Family | KNN-only + XGB-LTR | Full Stage 2 + deterministic no-XGB-LTR |
| --- | ---: | ---: |
| Beckmann | 24.96 +/- 3.57 | 27.23 +/- 1.13 |
| Buchwald-Hartwig | 49.59 +/- 0.40 | 38.25 +/- 0.14 |
| Chan-Lam | 55.30 +/- 1.82 | 56.92 +/- 0.26 |
| Diels-Alder | 25.98 +/- 1.07 | 20.60 +/- 0.23 |
| Friedel-Crafts Acyl. | 42.25 +/- 7.50 | 47.16 +/- 0.42 |
| Friedel-Crafts Alkyl. | 41.08 +/- 0.93 | 28.55 +/- 0.32 |

## Independent Audit Report

The completed independent check is retained in audit.md. It verifies the
compact-record count, fixed manifest denominator, Stage 2 limits, arm-specific
model boundaries, and exact full-mainline Stage 2 pool agreement for every
no-XGB-LTR family-seed result.

## Audit Contracts

- Every compact result is checked for the expected family, seed, fixed Stage 1 manifest, product-Morgan retrieval, K=64, 64-context wide pool, 20-context cap, and oracle-only train/validation tables.
- The KNN-only arm must have a tabular non-graph XGB-LTR and ReaFNN disabled.
- The no-XGB-LTR arm must have the deterministic Stage 1/2 prior, no fitted ranking parameters, and an exactly matching official Stage 2 protocol plus candidate-coverage record for the same family and seed.
- The official full-mainline row is read from the retained three-seed compact artifact; it is not recomputed or numerically mixed with old direct-R-GNN snapshots.
