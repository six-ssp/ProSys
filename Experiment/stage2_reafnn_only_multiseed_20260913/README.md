# Matched Current-Mainline Ablations

## Scope

This record is matched to the maintained parallel post-fusion mainline: fixed Stage 1 route caches, product-Morgan KNN (radius 2, 4,096 bits, K=64), independently generated KNN and ReaFNN 64-context pools, validation-only post-fusion, and a 20-context cap.

Each ablation was evaluated over 6 family/families and seeds 0, 1, 2. Each family/seed run was compacted immediately after validation, so raw candidate tables and binary checkpoints are intentionally absent.

## Arms

- **ReaFNN-only + XGB-LTR:** removes KNN proposals and all KNN-derived feature values, fixes w=0, and retrains the 52-column ranker on neural historical proposals. The KNN configuration slots are inactive, not retrieval performed with a zero fusion weight.
- The selected ablation arms skip temperature because the temperature branch does not add candidates or contribute to system ranking; temperature omission cannot change Sys@k.

## Macro Results

| Arm | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full current mainline | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |
| ReaFNN-only + XGB-LTR (no KNN evidence) | 46.48 +/- 0.30 | 18.14 +/- 1.12 | 27.76 +/- 0.91 | 31.80 +/- 0.71 | 36.24 +/- 0.27 | 24.35 +/- 0.90 | 26.32 +/- 0.67 |

## Per-Family Sys@10

| Family | Full current mainline | ReaFNN-only + XGB-LTR (no KNN evidence) |
| --- | ---: | ---: |
| Beckmann | 28.09 +/- 2.98 | 22.55 +/- 3.78 |
| Buchwald-Hartwig | 52.44 +/- 0.19 | 36.28 +/- 2.24 |
| Chan-Lam | 61.62 +/- 1.92 | 53.76 +/- 0.39 |
| Diels-Alder | 26.47 +/- 0.38 | 22.40 +/- 0.77 |
| Friedel-Crafts Acyl. | 48.91 +/- 2.47 | 40.35 +/- 1.40 |
| Friedel-Crafts Alkyl. | 45.09 +/- 1.29 | 42.12 +/- 1.67 |

## Audit Contracts

- Compact results use fixed Stage 1 manifests, 64-context proposal prefilters, a 20-context cap and reference-split training/validation tables. Active branches follow the intervention specified above; no KNN retrieval is performed in the ReaFNN-only arm.
- The official full-mainline macro row and per-family Sys@10 entries are read from the retained three-seed compact artifact; they are not recomputed or mixed with historical serial snapshots.
