# Matched Current-Mainline Ablations

## Scope

This record is matched to the maintained parallel post-fusion mainline: fixed Stage 1 route caches, product-Morgan KNN (radius 2, 4,096 bits, K=64), independently generated KNN and ReaFNN 64-context pools, validation-only post-fusion, and a 20-context cap.

Each ablation was evaluated over 6 family/families and seeds 0, 1, 2. Each family/seed run was compacted immediately after validation, so raw candidate tables and binary checkpoints are intentionally absent.

## Arms

- **KNN-only + XGB-LTR:** removes ReaFNN but retrains a tabular 52-feature XGB-LTR on its changed candidate distribution.
- The selected ablation arms skip temperature because the temperature branch does not add candidates or contribute to system ranking; temperature omission cannot change Sys@k.

## Macro Results

| Arm | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full current mainline | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |
| KNN-only + XGB-LTR | 53.39 +/- 0.00 | 22.72 +/- 1.86 | 31.44 +/- 2.45 | 35.19 +/- 2.23 | 39.86 +/- 2.08 | 28.63 +/- 1.98 | 29.85 +/- 2.05 |

## Per-Family Sys@10

| Family | Full current mainline | KNN-only + XGB-LTR |
| --- | ---: | ---: |
| Beckmann | 28.09 +/- 2.98 | 24.96 +/- 3.57 |
| Buchwald-Hartwig | 52.44 +/- 0.19 | 49.59 +/- 0.40 |
| Chan-Lam | 61.62 +/- 1.92 | 55.30 +/- 1.82 |
| Diels-Alder | 26.47 +/- 0.38 | 25.98 +/- 1.07 |
| Friedel-Crafts Acyl. | 48.91 +/- 2.47 | 42.25 +/- 7.50 |
| Friedel-Crafts Alkyl. | 45.09 +/- 1.29 | 41.08 +/- 0.93 |

## Audit Contracts

- Every compact result is checked for the expected family, seed, fixed Stage 1 manifest, product-Morgan retrieval, K=64, 64-context pools, 20-context cap, and reference-split training/validation tables.
- The KNN-only arm must have a tabular non-graph XGB-LTR and ReaFNN disabled.
- The official full-mainline macro row and per-family Sys@10 entries are read from the retained three-seed compact artifact; they are not recomputed or mixed with historical serial snapshots.
