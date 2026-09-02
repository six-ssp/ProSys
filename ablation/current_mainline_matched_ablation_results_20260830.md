# Current Mainline Matched Ablation Results

## Historical Serial Scope

This artifact is paired only to the former serial wide-pool candidate distribution.
It is retained for auditability, but it is not an ablation of the maintained
parallel KNN + ReaFNN post-fusion mainline.

Status: completed and independently audited
Result artifact: Experiment/current_mainline_matched_ablation_multiseed_20260830/
Execution entry point: scripts/run_current_mainline_matched_ablations.py

## 1. Purpose

These experiments isolate the contributions of the maintained Stage 2 and Stage
3 modules under the exact promoted product-Morgan mainline. They are not
historical point snapshots. Each test sample is evaluated through the same
persisted family-specific Stage 1 route cache used by the historical serial reference.

The two questions are deliberately separated:

1. Does ReaFNN improve the route-conditioned Stage 2 candidate pool and final
   system recommendation beyond product-Morgan KNN retrieval alone?
2. Once the full Stage 2 pool is fixed, does learned XGB-LTR improve ranking
   beyond a deterministic Stage 1/Stage 2 prior?

## 2. Shared Protocol

All three rows below use the same six reaction families, fixed Stage 1 route
caches, test manifest of 3,860 identities, and product-Morgan KNN configuration:

- Morgan radius 2 and 4,096 bits.
- 64 retrieved neighbors and a 64-context historical wide pool.
- At most 20 historical contexts per proposed route.
- No condition augmentation and no novel reagent-solvent combinations.
- Train-only family memory, vocabularies, and condition libraries.
- Three independent Stage 2/3 seeds: 0, 1, and 2.

The 27 identities without a Stage 1 candidate slate remain in every system
metric denominator. Training and validation candidate tables use their
reference split routes exactly as in the promoted mainline; this known
train/test route-distribution difference is unchanged in every arm and is
documented in Experiment/stage23_legality_audit_20260830.md.

All values are equal-family macro averages in percent. +/- is the sample
standard deviation across the three seeds.

## 3. Controlled Arms

### A2: KNN-only + XGB-LTR

This arm disables ReaFNN while retaining product-Morgan KNN and the same
candidate budget. Because removing ReaFNN changes the candidate distribution,
a fresh 52-feature tabular XGB-LTR is trained for each family and seed using
that arm's own train and validation candidate tables. This avoids attributing
ranker miscalibration to ReaFNN.

The comparison is KNN-only pool plus re-trained XGB-LTR versus KNN plus ReaFNN
pool plus re-trained XGB-LTR.

### A3: Full Stage 2 + deterministic no-XGB-LTR

This arm preserves the complete full-mainline Stage 2 protocol, including
ReaFNN, the 12 KNN anchors, bounded correction, and final candidate-generation policy.
It trains no learned ranker. Candidate rows are sorted deterministically by the
fixed Stage 1/2 prior:

1. retro_rank ascending
2. retro_probability descending
3. stage2_initial_score descending
4. stage2_knn_rank ascending
5. knn_similarity_sum descending
6. knn_similarity_max descending
7. knn_neighbor_count descending
8. knn_weighted_mean_yield descending
9. reagent_norm ascending
10. solvent_norm ascending

The audit checks that, for each family and seed, Stage 2 candidate-pool
coverage fields match the official full-mainline result exactly. The observed
difference is therefore a ranking comparison under matched Stage 2 generation
and recorded candidate availability. Raw candidate tables were pruned after
compaction, so the retained audit does not claim a per-row candidate-key hash
equality.

Temperature is skipped in both ablation arms because it neither adds a
candidate nor changes the ranking score. Its omission cannot change candidate
coverage or Sys@k.

## 4. Macro Results

| Method | Candidate coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full historical serial reference | 54.44 +/- 0.14 | 27.12 +/- 0.37 | 36.84 +/- 0.80 | 40.47 +/- 0.77 | 44.62 +/- 0.42 | 33.28 +/- 0.48 | 34.70 +/- 0.53 |
| KNN-only + XGB-LTR | 53.39 +/- 0.00 | 22.72 +/- 1.86 | 31.44 +/- 2.45 | 35.19 +/- 2.23 | 39.86 +/- 2.08 | 28.63 +/- 1.98 | 29.85 +/- 2.05 |
| Full Stage 2 + deterministic no-XGB-LTR | 54.44 +/- 0.14 | 24.87 +/- 0.35 | 32.23 +/- 0.14 | 34.27 +/- 0.11 | 36.45 +/- 0.08 | 29.56 +/- 0.20 | 30.35 +/- 0.12 |

| Removed component | Delta coverage | Delta Sys@1 | Delta Sys@3 | Delta Sys@5 | Delta Sys@10 | Delta MRR | Delta nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ReaFNN, while retraining XGB-LTR | -1.06 pp | -4.40 pp | -5.40 pp | -5.28 pp | -4.76 pp | -4.65 pp | -4.84 pp |
| XGB-LTR, with matched full Stage 2 generation and coverage | 0.00 pp | -2.25 pp | -4.61 pp | -6.20 pp | -8.17 pp | -3.71 pp | -4.34 pp |

## 5. Family-Resolved Sys@10

| Family | KNN-only + XGB-LTR | Full Stage 2 + deterministic no-XGB-LTR | Full historical serial reference |
| --- | ---: | ---: | ---: |
| Beckmann | 24.96 +/- 3.57 | 27.23 +/- 1.13 | 29.36 +/- 1.13 |
| Buchwald-Hartwig | 49.59 +/- 0.40 | 38.25 +/- 0.14 | 52.65 +/- 0.14 |
| Chan-Lam | 55.30 +/- 1.82 | 56.92 +/- 0.26 | 62.14 +/- 2.15 |
| Diels-Alder | 25.98 +/- 1.07 | 20.60 +/- 0.23 | 26.29 +/- 1.18 |
| Friedel-Crafts Acyl. | 42.25 +/- 7.50 | 47.16 +/- 0.42 | 51.79 +/- 1.64 |
| Friedel-Crafts Alkyl. | 41.08 +/- 0.93 | 28.55 +/- 0.32 | 45.49 +/- 0.69 |

The mean Sys@10 direction favors the full system in all six families for both
controls. With only three seeds, these results are reported as effect sizes and
replicate variation, not as a formal significance test.

## 6. Validity Checks

The independent audit passed all of the following contracts:

- 36 compact family-seed records are present: 6 families x 3 seeds x 2 arms.
- Every arm/seed has the same 3,860-identity manifest, 3,833 candidate slates,
  and 27 retained no-slate failures.
- All runs use product-Morgan KNN with the 64-neighbor, 64-context, and
  20-context limits specified above.
- The KNN-only arm has ReaFNN disabled and uses only the 52-feature tabular
  XGB-LTR; graph features and temperature fitting are absent.
- The no-XGB-LTR arm has no fitted ranking parameters and its Stage 2 protocol
  plus candidate-pool coverage match the official full-mainline record for the
  same family and seed.

The compact artifact retains per-family/per-seed metrics, result metadata, the
run manifest, completion status, and this summary's source CSVs. Large raw
candidate tables and checkpoints were deleted only after compaction to respect
the storage limit.

## 7. Paper-Safe Interpretation

The matched controls support three bounded claims:

1. Family-specific Stage 1 fine-tuning increases Route@10 from 14.97% to
   63.20% (+48.22 pp; existing fixed-cache Stage 1 comparison).
2. ReaFNN makes a positive current-mainline contribution: compared with KNN
   alone and a re-trained matching ranker, it increases macro candidate
   coverage by 1.06 pp and Sys@10 by 4.76 pp.
3. XGB-LTR supplies substantial ranking value after candidate construction:
   the deterministic control has matched Stage 2 generation and coverage but loses 8.17 pp
   Sys@10, 6.20 pp Sys@5, and 4.34 pp nDCG@10.

The R-GNN is not a ranking component in the maintained mainline. It is used
only by the temperature XGBoost regressor, so these system-ranking ablations do
not make a causal R-GNN ranking claim.
