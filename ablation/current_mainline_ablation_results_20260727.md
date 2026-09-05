# Historical Stage 1-3 Ablation Snapshot

Date: `2026-07-27`

> **Historical evidence only.** This controlled snapshot evaluates the earlier direct-R-GNN-ranking configuration. It is not the maintained headline mainline; use [`CURRENT_RESULTS.md`](../CURRENT_RESULTS.md) and the current parallel artifacts for `candidate recall = 54.26 +/- 0.15%`, `full-system Top-10 accuracy = 43.77 +/- 0.60%`, and conditional `MAE = 11.49 +/- 0.26 C`.

## Status and Scope

This is the completed, corrected ablation run for the historical six-family ProSys direct-R-GNN-ranking configuration:

```text
target product
  -> family-tuned EditRetro route proposals
  -> widened KNN retrieval + ReaFNN condition-pool selection
  -> R-GNN route features + XGB-LTR reranking and temperature regression
```

All results use the fixed end-to-end test manifest from the Stage 1 route cache. The denominator is every cached test identity (`n = 3,860` across the six families); a sample without a candidate slate remains a zero hit. Stage 2 retains at most 20 reagent-solvent contexts per route.

Two safeguards were applied before this run:

- Training KNN retrieval is leave-one-reaction-out. The query's own canonical reaction and its context aggregates are excluded from its training evidence.
- The regenerated test candidate-key multisets are identical to the pre-correction tables. All non-GNN test fields are exact; GNN CSV serialization differs by at most `1.1e-5`.

The independent audit passed for all six families and all six model variants: fixed manifest, candidate partition, 20-context cap, XGB-LTR target exclusion, GNN feature dimensions, and equality between the reported historical full reference and the rebuilt configuration.

- Candidate stability audit: [`loo_test_candidate_stability_20260727.md`](../outputs/stage23_mainline_reafnn_gnn_fused_20260723/loo_test_candidate_stability_20260727.md)
- Independent audit: [`audit.md`](../outputs/ablation_reafnn_gnn_20260726/audit.md)
- Detailed generated tables: [`Stage 1`](../outputs/ablation_reafnn_gnn_20260726/stage1_route_ablation.md), [`Stage 2`](../outputs/ablation_reafnn_gnn_20260726/stage2_pool_ablation.md), [`Stage 3`](../outputs/ablation_reafnn_gnn_20260726/stage3_reranking_ablation.md), and [`interaction control`](../outputs/ablation_reafnn_gnn_20260726/stage23_interaction_ablation.md).

## A1: Family-Specific Stage 1 Fine-Tuning

| Family | Base R@1 | Tuned R@1 | Base R@3 | Tuned R@3 | Base R@5 | Tuned R@5 | Base R@10 | Tuned R@10 | Delta R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Beckmann | 0.00% | 47.66% | 0.00% | 66.81% | 0.85% | 68.09% | 1.70% | 69.79% | +68.09 pp |
| Buchwald-Hartwig | 16.11% | 45.68% | 26.21% | 61.06% | 30.03% | 65.97% | 35.30% | 69.97% | +34.67 pp |
| Chan-Lam | 1.54% | 62.56% | 5.38% | 72.82% | 6.67% | 75.90% | 8.72% | 77.44% | +68.72 pp |
| Diels-Alder | 0.52% | 23.49% | 0.92% | 31.36% | 1.18% | 33.86% | 1.71% | 37.27% | +35.56 pp |
| Friedel-Crafts Acyl. | 14.11% | 50.32% | 23.79% | 63.79% | 28.21% | 66.95% | 33.05% | 70.32% | +37.26 pp |
| Friedel-Crafts Alkyl. | 2.89% | 31.03% | 5.45% | 44.27% | 7.45% | 49.05% | 9.34% | 54.39% | +45.05 pp |
| Macro average | 5.86% | 43.46% | 10.29% | 56.69% | 12.40% | 59.97% | 14.97% | 63.20% | +48.22 pp |

Family fine-tuning improves Route@10 for every family. The macro-average gain is `+48.22 pp`, so the Stage 1 adaptation is a necessary part of the end-to-end system rather than a cosmetic specialization.

## A2: Stage 2 Candidate Screening

All three arms use the same fine-tuned Stage 1 routes, the same 20-context cap, a frozen family R-GNN encoder, and a separately trained XGB-LTR ranker and temperature regressor. The table shows candidate recall and end-to-end full-system Top-10 accuracy.

| Family | Top-20 frequency candidate recall / full-system Top-10 accuracy | KNN + ReaFNN candidate recall / full-system Top-10 accuracy | KNN only candidate recall / full-system Top-10 accuracy |
|---|---:|---:|---:|
| Beckmann | 17.02% / 10.21% | 39.57% / 28.51% | 42.55% / 28.51% |
| Buchwald-Hartwig | 24.75% / 14.01% | 48.68% / 46.86% | 59.42% / 53.96% |
| Chan-Lam | 39.49% / 25.90% | 66.67% / 61.79% | 71.03% / 66.15% |
| Diels-Alder | 17.59% / 11.42% | 29.79% / 27.30% | 34.38% / 28.35% |
| Friedel-Crafts Acyl. | 44.84% / 36.21% | 60.63% / 45.89% | 65.89% / 42.74% |
| Friedel-Crafts Alkyl. | 10.12% / 8.57% | 49.72% / 45.72% | 51.06% / 40.27% |
| Macro average | 25.63% / 17.72% | 49.18% / 42.68% | 54.06% / 43.33% |

The route-conditioned KNN pool is decisively better than a global frequency prior. Relative to Top-20 frequency, the full pool gains `+23.54 pp` candidate recall and `+24.96 pp` full-system Top-10 accuracy; KNN-only gains `+28.42 pp` candidate recall and `+25.61 pp` full-system Top-10 accuracy. This supports the central Stage 2 claim: local precedent retrieval, not a global condition prior, is what makes the feasible candidate pool useful.

The ReaFNN expansion is not a stable macro-average full-system Top-k accuracy improvement over raw KNN in this run: full pool full-system Top-10 accuracy is `42.68%` versus `43.33%` for KNN-only. It helps Friedel-Crafts acylation and alkylation, is tied for Beckmann, and is lower on the other three families. It should therefore be described as a controlled candidate-composition mechanism, not as a universally improving ranking component. Conditional temperature quality is better in the full arm (`11.28 C` MAE and `61.60%` within `+/-10 C`) than KNN-only (`13.62 C`, `54.70%`), but the supports differ (`1,603` vs `1,795`) because temperature is evaluated only on exact retrieved systems.

## A3: Stage 3 Learned Reranking

All rows below use the same saved KNN + ReaFNN candidate tables. The no-Stage-3 arm applies only deterministic Stage 1/Stage 2 priors and has no temperature model.

| Family | Full full-system Top-10 accuracy | Without R-GNN full-system Top-10 accuracy | Without XGB-LTR full-system Top-10 accuracy |
|---|---:|---:|---:|
| Beckmann | 28.51% | 29.79% | 23.40% |
| Buchwald-Hartwig | 46.86% | 46.77% | 28.30% |
| Chan-Lam | 61.79% | 60.77% | 50.00% |
| Diels-Alder | 27.30% | 27.56% | 15.75% |
| Friedel-Crafts Acyl. | 45.89% | 53.05% | 43.16% |
| Friedel-Crafts Alkyl. | 45.72% | 44.61% | 27.25% |
| Macro average | 42.68% | 43.76% | 31.31% |

Removing XGB-LTR reduces macro full-system Top-1 accuracy from `27.64%` to `12.30%` and full-system Top-10 accuracy from `42.68%` to `31.31%` (`-11.37 pp`). Every family loses full-system Top-10 accuracy without learned reranking. This is the cleanest Stage 3 conclusion: KNN/ReaFNN make the correct system available, while XGB-LTR is needed to move it near the top of the recommendation list.

The R-GNN feature block has a mixed ranking effect: removing it increases macro full-system Top-10 accuracy by `1.08 pp` (`43.76%` versus `42.68%`). It is better on three families and lower on three families. It does, however, improve the conditional temperature metrics on the same `n = 1,603` exact-system support: MAE `11.28 C` versus `12.51 C`, and `+/-10 C` hit rate `61.60%` versus `58.30%`. The correct claim is therefore that graph features are an auxiliary representation with a temperature-regression benefit in this run, not a demonstrated universal full-system Top-k accuracy gain.

## A4: ReaFNN-Graph Interaction Control

| ReaFNN pool | R-GNN features | Macro full-system Top-1 accuracy | Macro full-system Top-10 accuracy | MRR | MAE (deg C) | Within +/-10 deg C |
|---|---|---:|---:|---:|---:|---:|
| Yes | Yes | 27.64% | 42.68% | 32.66% | 11.28 | 61.60% |
| No | Yes | 27.68% | 43.33% | 33.07% | 13.62 | 54.70% |
| Yes | No | 29.14% | 43.76% | 34.29% | 12.51 | 58.30% |
| No | No | 27.56% | 42.95% | 32.99% | 16.55 | 48.57% |

The factorial control confirms that neither ReaFNN nor the R-GNN block has a simple, additive macro full-system Top-10 accuracy contribution. ReaFNN changes from `-0.65 pp` with graph features to `+0.81 pp` without them; graph features change from `-1.08 pp` with ReaFNN to `+0.38 pp` without it. The modules interact, so a single-arm comparison would overstate their individual effect. The full configuration provides the best macro temperature MAE and `+/-10 C` hit rate, while the no-graph ReaFNN arm gives the highest macro full-system Top-10 accuracy in this specific run.

## Temperature Result

For the historical full configuration, temperature is evaluated only when the highest-ranked prediction is an exact route-and-condition match with a valid gold temperature. Across the six families this gives `n = 1,603`, MAE `11.28 C`, and hit rates of `37.96%`, `61.60%`, and `82.98%` within `+/-5 C`, `+/-10 C`, and `+/-20 C`, respectively. These are conditional regression results and must not be interpreted as temperature accuracy over all 3,860 test products.

## Paper-Safe Conclusions

1. Family-specific Stage 1 fine-tuning is strongly supported: Route@10 improves for all six families, by `+48.22 pp` macro-average.
2. Local KNN retrieval is strongly supported: it substantially outperforms a size-matched global frequency pool in coverage and all reported full-system Top-k accuracy metrics.
3. XGB-LTR reranking is strongly supported: removing it reduces full-system Top-10 accuracy for every family and lowers macro full-system Top-10 accuracy by `11.37 pp`.
4. ReaFNN and R-GNN should not be claimed as universal full-system Top-k accuracy improvements from this run. Their controlled effects are mixed, while the full configuration has the strongest conditional temperature metrics.

This conservative interpretation is important: it separates the components that are empirically necessary for the headline ranking claim from the components that remain useful architectural extensions but need further tuning or repeated-seed validation before a stronger performance claim.

## Reproduction

The corrected run uses:

```bash
cd /root/autodl-tmp/ProSys
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 /root/miniconda3/envs/ProSys/bin/python ablation/run_current_mainline_ablation.py --repo_root . --families all --run_set all --output_root outputs/ablation_reafnn_gnn_20260726 --mainline_root outputs/stage23_mainline_reafnn_gnn_fused_20260723 --route_root outputs/stage1_routes --base_route_root outputs/stage1_base_vs_tuned/base_route_caches --gnn_device cuda --knn_workers 8 --force
```
