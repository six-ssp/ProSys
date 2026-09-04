# Current Parallel Stage-3 Matched Ablation

## Status

Completed on `2026-09-04`. The compact, machine-readable result record is
[`Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/`](../Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/README.md).
It contains six families x three seeds = 18 matched family-seed records.

## Question and Controlled Comparison

This experiment isolates whether the maintained 52-feature tabular XGB-LTR
contributes after the current parallel Stage 2 pool has been constructed. It
compares the official current mainline with a deterministic no-XGB-LTR control.
The Stage-1 test-route cache is fixed; no Stage-1 retraining variation is being
measured.

The control preserves, for the same family and seed:

- product-Morgan KNN (`radius=2`, 4,096 bits, `K=64`);
- ReaFNN independent top-64 historical contexts;
- predicted-validation-route-only KNN/ReaFNN fusion-weight selection;
- the route-local cap of 20 historical contexts;
- reference-split train/validation candidate tables and persisted Stage-1 test routes;
- the 3,860-product full-system denominator, including 3,833 candidate slates
  and 27 retained no-slate failures.

The only ranking intervention is replacement of XGB-LTR with the deterministic
Stage-1/2 prior. It uses already-emitted route rank/probability, initial Stage-2
score, KNN evidence, and stable reagent/solvent tie breaks. It has no fitted
ranking parameters. The temperature branch is intentionally skipped because it
adds neither candidates nor a system-ranking score; omitting it cannot change
Sys@k.

## Macro Results

All values are unweighted six-family macro averages over seeds 0, 1, and 2.
Rates are percentages and `+/-` is the sample standard deviation (`ddof=1`).

| System | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full parallel mainline | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |
| Full Stage 2 + deterministic no-XGB-LTR | 54.26 +/- 0.15 | 24.48 +/- 0.09 | 32.23 +/- 0.34 | 34.05 +/- 0.02 | 36.03 +/- 0.16 | 29.24 +/- 0.07 | 29.99 +/- 0.06 |
| Full minus no-XGB-LTR (pp) | 0.00 | +0.65 | +2.89 | +5.06 | +7.74 | +2.29 | +3.16 |

Candidate recall is exactly equal per family and seed, not merely equal after
rounding. The `+7.74 pp` Sys@10 difference therefore identifies the learned
reranker as the source of the improvement within a matched candidate pool.

## Family-Resolved Sys@10

| Family | Full parallel mainline | Deterministic no-XGB-LTR | Delta (pp) |
| --- | ---: | ---: | ---: |
| Beckmann | 28.09 +/- 2.98 | 26.24 +/- 0.25 | +1.84 |
| Buchwald-Hartwig | 52.44 +/- 0.19 | 37.94 +/- 0.18 | +14.50 |
| Chan-Lam | 61.62 +/- 1.92 | 56.84 +/- 0.53 | +4.79 |
| Diels-Alder | 26.47 +/- 0.38 | 20.78 +/- 0.33 | +5.69 |
| Friedel-Crafts acylation | 48.91 +/- 2.47 | 45.89 +/- 0.21 | +3.02 |
| Friedel-Crafts alkylation | 45.09 +/- 1.29 | 28.48 +/- 0.29 | +16.61 |

All six family means favor XGB-LTR at Sys@10. The Beckmann estimate has larger
three-seed variability, so this table is reported as effect sizes and replicate
variation rather than as a formal significance test.

## Validity and Paper-Safe Interpretation

The independent post-run audit passed all 18 records. It checked exact equality
of the complete `stage2_protocol` object and the candidate-availability fields
(`pool_coverage`, route/context coverage, candidate slates, and missing slates)
against the official current-mainline compact result for the same family and
seed. It also checked the deterministic ranking contract and the fixed full
manifest denominator. See the retained [`audit.md`](../Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/audit.md).

The evidence supports a bounded statement: once the parallel KNN + ReaFNN pool
has made the same candidates available, XGB-LTR substantially improves their
ordering, especially at deeper recommendation cutoffs (`Sys@3/5/10`). The
Top-1 difference is smaller (`+0.65 pp`) and changes direction across seeds
(`-0.69`, `+0.81`, `+1.83 pp`), so it should not be described as a stable
Top-1 gain. This experiment does not attribute a ranking effect to the R-GNN,
which remains temperature-only in the maintained mainline.
