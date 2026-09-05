# Current Parallel Stage-2 Matched ReaFNN Ablation

## Status

Completed on `2026-09-05`. The compact, machine-readable record is
[`Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/`](../Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/README.md).
It contains six families x three seeds = 18 completed family-seed records.

## Question and Controlled Comparison

Does ReaFNN improve the maintained parallel Stage-2 candidate pool after the
downstream ranker is re-trained for the candidate distribution produced by each
arm?

The full arm is the official parallel KNN + ReaFNN post-fusion mainline. KNN
and ReaFNN independently nominate up to 64 train-only historical contexts per
Stage-1 route, a validation-only family-specific fusion weight combines their
rank priors, and the route-local top 20 contexts are delivered to XGB-LTR.

The KNN-only control preserves, for every family and seed:

- persisted Stage-1 test-route caches and the fixed 3,860-product manifest;
- product-Morgan KNN (radius 2, 4,096 bits, `K=64`) and a 64-context prefilter;
- a route-local cap of 20 historical contexts;
- family-train-only condition memory and reference-split training/validation
  candidate tables;
- a separately trained 52-feature, tabular non-graph XGB-LTR.

It disables ReaFNN and post-fusion, so only the KNN route-conditioned pool is
passed to the re-trained ranker. Temperature is skipped in this Sys@k-only
control because it neither creates candidates nor contributes to their ranking.
The comparison therefore changes candidate availability and composition, unlike
the matched Stage-3 no-XGB-LTR ablation, which holds the pool identical.

## Macro Results

All values are unweighted six-family macro averages across seeds 0, 1, and 2.
Rates are percentages and `+/-` is the sample standard deviation (`ddof=1`).
The final difference row is computed from the unrounded paired seed means.

| System | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full parallel KNN + ReaFNN + XGB-LTR | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |
| KNN-only + XGB-LTR | 53.39 +/- 0.00 | 22.72 +/- 1.86 | 31.44 +/- 2.45 | 35.19 +/- 2.23 | 39.86 +/- 2.08 | 28.63 +/- 1.98 | 29.85 +/- 2.05 |
| Full minus KNN-only (pp) | +0.87 | +2.42 | +3.68 | +3.92 | +3.91 | +2.90 | +3.30 |

The KNN-only candidate pool is deterministic under the fixed Stage-1 routes,
so its candidate-recall standard deviation is zero. The learned XGB-LTR is
retrained at every seed and therefore retains seed-dependent ranking variance.

## Paired Seed and Family Patterns

| Seed | Full minus KNN-only candidate recall (pp) | Full minus KNN-only Sys@10 (pp) |
| --- | ---: | ---: |
| 0 | +0.71 | +1.42 |
| 1 | +0.91 | +3.72 |
| 2 | +1.00 | +6.59 |

| Family | Full parallel Sys@10 | KNN-only Sys@10 | Delta (pp) |
| --- | ---: | ---: | ---: |
| Beckmann | 28.09 +/- 2.98 | 24.96 +/- 3.57 | +3.12 |
| Buchwald-Hartwig | 52.44 +/- 0.19 | 49.59 +/- 0.40 | +2.85 |
| Chan-Lam | 61.62 +/- 1.92 | 55.30 +/- 1.82 | +6.32 |
| Diels-Alder | 26.47 +/- 0.38 | 25.98 +/- 1.07 | +0.48 |
| Friedel-Crafts acylation | 48.91 +/- 2.47 | 42.25 +/- 7.50 | +6.67 |
| Friedel-Crafts alkylation | 45.09 +/- 1.29 | 41.08 +/- 0.93 | +4.00 |

All six family means and all three seed-level macro Sys@10 effects favor the
full KNN + ReaFNN pool. The small Diels-Alder mean effect and the larger
Beckmann/acylation variability should be reported as effect sizes with
replicate variation, not as standalone formal significance claims.

## Audit and Paper-Safe Interpretation

The independent post-run audit passed all 18 compact records. It verifies:

- the expected KNN-only baseline identifier, disabled ReaFNN policy, and
  disabled temperature branch;
- product-Morgan retrieval, `K=64`, 64-context prefilter, top-20 cap, and
  reference-split training candidate construction;
- a separately trained 52-feature XGB-LTR with no target-named input field;
- exact equality with the full counterpart of Stage-1 route recall, manifest
  count, candidate-slate count, missing-slate count, and shared KNN settings.

Candidate recall intentionally differs because removing ReaFNN changes the
Stage-2 pool. Thus the supported conclusion is that ReaFNN provides a useful
Stage-2 candidate-availability/composition contribution under the maintained
parallel workflow: it adds `+0.87 pp` candidate recall and `+3.91 pp` Sys@10
after the ranker is fairly re-trained on each arm. It is not valid to describe
this result as a pure reranking effect; the matched no-XGB-LTR experiment is
the appropriate evidence for that separate Stage-3 claim.

## Reproduction

From the repository root, run:

```bash
conda run -n ProSys python scripts/run_current_mainline_matched_ablations.py \
  --repo_root . \
  --arms knn_only \
  --output_root Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904 \
  --scratch_root /tmp/prosys_stage2_parallel_post_fusion_ablation_20260904 \
  --python_bin /root/miniconda3/envs/ProSys/bin/python \
  --reafnn_device cuda:0 \
  --cpu_threads 8
```

The launcher processes one family at a time, compacts each successful result,
and removes large candidate tables and checkpoints. A resumed run validates and
reuses completed compact records before computing only missing family-seed
items.
