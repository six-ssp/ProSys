# Mainline Evidence Reconstruction

All 18 family/seed jobs completed. Retained candidate files were replayed independently; no-LTR uses identical candidate identities, and no-R-GNN uses identical ranked identities, temperature support and reference temperatures. Graph augmentation preserves the shared training/validation feature values. These controls pair to the reconstruction, not deleted historical rows.

## Reconstructed Controls

Equal-family macro mean +/- sample SD over fixed-split downstream seeds 0/1/2. Rates in percent.

| Arm | Sys@1 | Sys@3 | Sys@5 | Sys@10 | Temp MAE (C) | Within 10 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 11.82 +/- 0.33 | 61.58 +/- 0.77 |
| no_ltr | 24.48 +/- 0.09 | 32.23 +/- 0.34 | 34.05 +/- 0.02 | 36.03 +/- 0.16 | NA | NA |
| no_rgnn_temperature | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 13.93 +/- 0.38 | 55.56 +/- 1.23 |

Maximum absolute per-family/seed Sys@k difference from promoted results: 0.00000000 pp. Maximum absolute temperature-MAE difference: 2.458091 C. See reconstructed_minus_promoted.csv for every difference. Promoted figures are not silently overwritten.

## Seen/Unseen Product Diagnostics

Seen means canonical product occurs in this family's condition-training split. The existing split groups canonical reactions, not products. This is a descriptive subgroup analysis, not an independent external benchmark. Denominators are query identities, not unique molecules; the same fixed identities recur across seeds. Unseen here means absent from the condition-training memory only; Stage 1 pretraining or expert-training data may still contain these products.

| Family | Seen n | Seen Sys@10 | Unseen n | Unseen Sys@10 | Train-context available n |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 53 | 21.38 +/- 6.63 | 182 | 30.04 +/- 2.08 | 216 |
| Buchwald-HartwigCross-Coupling | 97 | 36.43 +/- 0.60 | 1002 | 53.99 +/- 0.20 | 1057 |
| Chan_LamCoupling | 28 | 23.81 +/- 4.12 | 362 | 64.55 +/- 1.75 | 385 |
| DielsAlder | 22 | 39.39 +/- 2.62 | 740 | 26.08 +/- 0.36 | 735 |
| Friedel-CraftsAcylation | 40 | 20.00 +/- 2.50 | 435 | 51.57 +/- 2.74 | 458 |
| Friedel-CraftsAlkylation | 51 | 34.64 +/- 3.00 | 848 | 45.72 +/- 1.34 | 878 |

Training-context availability asks whether at least one gold reagent/solvent pair belongs to the train-only historical library; it is an evaluation annotation, not a model input. Empty subgroups report NA.

## Exhaustive Failure Accounting

Seed-0 counts below; all seeds are in failures_per_family_seed.csv. Route miss: no predicted reference-matching route; pool miss: route retained but no exact system; ranking miss: first exact system below rank 10; hit: first exact system within rank 10. No-slate queries remain in the denominator.

| Family | n | Route miss | Pool miss | Ranking miss | Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 235 | 71 | 65 | 25 | 74 |
| Buchwald-HartwigCross-Coupling | 1099 | 330 | 121 | 70 | 578 |
| Chan_LamCoupling | 390 | 88 | 18 | 48 | 236 |
| DielsAlder | 762 | 478 | 19 | 65 | 200 |
| Friedel-CraftsAcylation | 475 | 141 | 26 | 89 | 219 |
| Friedel-CraftsAlkylation | 899 | 410 | 30 | 66 | 393 |

## Resource Scope

cost_per_family_seed.csv records concurrent reconstruction wall time, nested stage times and per-process peak RSS/PyTorch allocated GPU memory. Stage 1 decoding is excluded; stage intervals include table work or model loading as their names indicate. Concurrent wall times are not isolated single-query latency or additive total GPU-hours. The product-query CLI records cold-load inference separately. No new hyperparameter selection used these test results.

Source provenance note: the initial Beckmann seed-0 pilot predates runtime/source-manifest instrumentation; its model/data hashes and exact-control audit are retained, but its missing runtime source manifest is not fabricated retrospectively. The other jobs retain that manifest. Configuration and feature-schema comparisons against promoted compact records pass for every family/seed.

Known implementation boundary: see CANONICALIZATION_NOTE.md for one training-only cross-dot ring-closure record. The independent whole-molecule split scan found no cross-split reaction overlap.
