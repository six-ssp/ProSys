# Matched Current-Mainline Ablations

## Scope

This record is matched to the maintained parallel post-fusion mainline: fixed Stage 1 route caches, product-Morgan KNN (radius 2, 4,096 bits, K=64), independently generated KNN and ReaFNN 64-context pools, validation-only post-fusion, and a 20-context cap.

Each ablation was evaluated over 6 family/families and seeds 0, 1, 2. Each family/seed run was compacted immediately after validation, so raw candidate tables and binary checkpoints are intentionally absent.

## Arms

- **Full Stage 2 + deterministic no-XGB-LTR:** preserves the official parallel KNN + ReaFNN Stage 2 pool, including validation-only fusion selection, but sorts using only the fixed Stage 1/2 prior: route rank, route probability, Stage 2 initial score, KNN evidence, and stable reagent/solvent tie-breaks. It has no fitted ranking parameters.
- The selected ablation arms skip temperature because the temperature branch does not add candidates or contribute to system ranking; temperature omission cannot change Sys@k.

## Macro Results

| Arm | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full current mainline | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |
| Full Stage 2 + deterministic no-XGB-LTR | 54.26 +/- 0.15 | 24.48 +/- 0.09 | 32.23 +/- 0.34 | 34.05 +/- 0.02 | 36.03 +/- 0.16 | 29.24 +/- 0.07 | 29.99 +/- 0.06 |

## Per-Family Sys@10

| Family | Full current mainline | Full Stage 2 + deterministic no-XGB-LTR |
| --- | ---: | ---: |
| Beckmann | 28.09 +/- 2.98 | 26.24 +/- 0.25 |
| Buchwald-Hartwig | 52.44 +/- 0.19 | 37.94 +/- 0.18 |
| Chan-Lam | 61.62 +/- 1.92 | 56.84 +/- 0.53 |
| Diels-Alder | 26.47 +/- 0.38 | 20.78 +/- 0.33 |
| Friedel-Crafts Acyl. | 48.91 +/- 2.47 | 45.89 +/- 0.21 |
| Friedel-Crafts Alkyl. | 45.09 +/- 1.29 | 28.48 +/- 0.29 |

## Audit Contracts

- Every compact result is checked for the expected family, seed, fixed Stage 1 manifest, product-Morgan retrieval, K=64, 64-context pools, 20-context cap, and reference-split training/validation tables.
- The no-XGB-LTR arm must have the deterministic Stage 1/2 prior, no fitted ranking parameters, and an exactly matching official Stage 2 protocol plus candidate-coverage record for the same family and seed.
- The official full-mainline macro row and per-family Sys@10 entries are read from the retained three-seed compact artifact; they are not recomputed or mixed with historical serial snapshots.
