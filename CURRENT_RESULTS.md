# ProSys Current Results

Accepted on 2026-09-25: filtered-USPTO-50K base trained from random neural
initialization, repaired experts and the parallel KNN/ReaFNN mainline.
No FULL weights, FULL retraining or new hyperparameter search is part of this
release. Historical FULL-based scores must not be used as its results.

Sys@k (code) and FS@k (paper abbreviation) are the same full-system Top-k
accuracy: joint route/reagent-set/solvent-set matching, excluding temperature.
All 3,860 original queries remain, including 66 without routes. Means weight
families equally before computing sample SD across downstream seeds 0/1/2.
The downstream expert is fixed at seed 1, not selected by test performance.

## Mainline, Baselines and Ablations

| Model | Sys@1 (%) | Sys@3 (%) | Sys@5 (%) | Sys@10 (%) | Conditional MAE (C) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ProSys | 19.81 +/- 0.30 | 29.74 +/- 0.51 | 33.33 +/- 0.17 | 37.87 +/- 0.27 | 11.32 +/- 0.33 |
| B1 Product-NB | 7.88 | 13.05 | 15.35 | 19.44 | NA |
| B2 Product-GNN | 5.06 +/- 0.51 | 10.11 +/- 0.74 | 13.35 +/- 1.10 | 18.85 +/- 0.89 | NA |
| B3 Sequential FNN | 13.80 +/- 0.59 | 20.11 +/- 0.50 | 22.76 +/- 0.53 | 26.16 +/- 0.61 | 12.28 +/- 0.19 |
| B4 Reaction-GCNN | 5.31 +/- 0.17 | 10.09 +/- 0.37 | 12.96 +/- 0.76 | 17.84 +/- 0.42 | NA |
| Without KNN | 15.01 +/- 0.67 | 23.54 +/- 0.21 | 27.12 +/- 0.07 | 31.31 +/- 0.50 | NA |
| Without ReaFNN | 18.22 +/- 1.69 | 26.42 +/- 1.98 | 29.52 +/- 2.03 | 34.16 +/- 1.80 | NA |
| Without LTR | 20.22 +/- 0.14 | 26.52 +/- 0.23 | 28.20 +/- 0.18 | 29.90 +/- 0.10 | NA |
| Without temperature R-GNN | 19.81 +/- 0.30 | 29.74 +/- 0.51 | 33.33 +/- 0.17 | 37.87 +/- 0.27 | 13.63 +/- 0.29 |

B1 is deterministic, not triplicated. Full-minus-control Sys@10 differences
before rounding: B3 11.711804 pp; without KNN 6.553670 pp; without ReaFNN
3.708961 pp; without LTR 7.962408 pp. Stage 2 removals rebuild the pool and
refit XGB-LTR; only no-LTR isolates ranking in the identical pool.
Contributions are not additive and do not establish statistical significance.

Mainline candidate recall is 47.57 +/- 0.08%, MRR is 26.06 +/- 0.15% and
nDCG@10 is 27.74 +/- 0.14%. LTR decreases macro Sys@1 by 0.416707 pp.
Do not describe reranking as uniformly beneficial at every cutoff or family.

Source: [150 family/model/seed records and summaries](Experiment/50k_verified_comparisons_20260924/full/RESULTS.md).

## Family Results

Route@10 uses fixed expert seed 1. Other values summarize three downstream seeds.

| Family | Route@10 (%) | Candidate recall (%) | Sys@10 (%) | Conditional MAE (C) |
| --- | ---: | ---: | ---: | ---: |
| Beckmann | 68.94 | 41.56 | 26.38 +/- 2.95 | 12.58 |
| Buchwald-Hartwig | 60.33 | 51.99 | 45.86 +/- 0.55 | 12.02 |
| Chan-Lam | 70.00 | 65.13 | 53.50 +/- 0.74 | 5.88 |
| Diels-Alder | 33.73 | 30.93 | 24.28 +/- 1.12 | 16.70 |
| Friedel-Crafts acylation | 61.89 | 58.18 | 44.14 +/- 2.11 | 10.08 |
| Friedel-Crafts alkylation | 40.27 | 37.67 | 33.04 +/- 0.68 | 10.63 |

## Stage 1 and Temperature

Fixed-seed-1 base/expert macro Route@10 is 10.08/55.86%. Separate expert
seeds 0/1/2 sharing one scratch-50K base give Route@1/3/5/10:
36.57 +/- 0.42 / 48.53 +/- 0.09 / 51.81 +/- 0.09 / 55.42 +/- 0.39%.
These are not independent base-pretraining or full-pipeline repeats.

All 18 experts are certified: 17 original decodes and one explicitly admitted
Diels-Alder seed-2 recovery using the same weights. One over-capacity hypothesis
became an empty invalid generation slot; all 762 queries and 76,200 slots remain.
Normal-path GPU replay preserves every hypothesis string and final ranked route;
two EOS-score differences on already invalid strings are documented separately.
Original failure logs remain historical; no seed or query was substituted.

Temperature uses the highest-ranked exact system with finite reference/predicted
temperature anywhere in the slate, not just Top-10. Mainline and no-R-GNN have
identical supports 1,529/1,537/1,538. Within-10-C accuracy is 62.27 +/- 0.13%
versus 55.62 +/- 1.27%; MAE improves by 2.310916 C. B3 uses its own supports
1,251/1,219/1,230 and has higher within-10-C accuracy (66.14 +/- 1.10%).
Do not claim a paired or uniformly superior baseline temperature result.

## Evidence and Boundaries

The release includes aggregate seed/family tables for experts, baseline and
ablation comparisons, 48 direct-condition auxiliary rows, 18 validation-fusion
rows, grouped failure analysis and six fixed-query cold-start measurements.
Detailed cases, per-query predictions, weights and manuscript drafts remain
local. No raw Reaxys records are redistributed.

Seed-0 ReaFNN/KNN/fused validation coverage is 40.04/44.77/45.90%.
Diels-Alder seed 2 selects KNN fusion weight 1.0 while retaining ReaFNN features
for ranking; this is not the ReaFNN-removal ablation.
Cold load-inclusive wall times are 13.62-24.01 s on six predeclared queries,
not throughput or a population-latency estimate. The benchmark was used during
development and is not product-disjoint or prospective validation.

Current public index: [accepted 50K release](Experiment/release_50k_20260925/README.md).
Historical numerical records remain in dated directories. Their old headline
and pending fields describe their original phase, not the current status.
