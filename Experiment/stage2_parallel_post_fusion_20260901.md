# Parallel KNN + ReaFNN Post-Fusion Evaluation

Date: `2026-09-01`

Status: promoted as the maintained Stage 2 procedure by project decision. The
seed-0 comparison with the historical serial system remains below as an honest
record. Joint Stage-2/Stage-3 training and wrong-route negative-sample
supervision were retired and do not affect this record.

## Question

This experiment tests a genuinely parallel Stage 2 construction. KNN and
ReaFNN independently propose historical reagent-solvent contexts, and their
scores are combined as:

```text
S(context) = w * S_KNN(context) + (1 - w) * S_ReaFNN(context)
```

KNN retrieves a 64-context product-Morgan precedent pool from the family train
split. ReaFNN independently scores the full train-only historical condition
library from the route representation and retains its top 64 contexts. The
union is rank-normalized separately for each branch, then the fused top 20
contexts are retained per Stage 1 route.

For each family, `w` is selected only from the predicted Stage 1 validation
route cache using the grid `0.0, 0.1, ..., 1.0`. The selection metric is exact
Stage 2 candidate coverage on the validation manifest. Ties favor the larger
KNN weight. Test conditions are not used to train either branch, build the
context library, select `w`, or tune Stage 3.

Stage 3 is unchanged from the maintained pipeline: the 52-feature XGB-LTR
reranker is retrained on the matching candidate distribution, and the separate
R-GNN plus XGBoost temperature branch is enabled for the full six-family run.

## Validation Stage 2 Ablation

Values are exact candidate coverage in percent. `ReaFNN-only` is `w=0`,
`KNN-only` is `w=1`, and `fused` is the validation-selected mixture. Every
family selects a nonzero ReaFNN contribution and the fused pool is at least as
good as the KNN-only endpoint on its validation manifest.

| Family | w(KNN) | ReaFNN-only | KNN-only | Fused |
| --- | ---: | ---: | ---: | ---: |
| Beckmann | 0.5 | 38.72 | 40.43 | 42.98 |
| Buchwald-Hartwig | 0.7 | 45.85 | 56.24 | 58.07 |
| Chan-Lam | 0.6 | 66.92 | 70.00 | 71.79 |
| Diels-Alder | 0.9 | 22.86 | 31.67 | 31.80 |
| Friedel-Crafts acylation | 0.9 | 50.84 | 59.45 | 59.66 |
| Friedel-Crafts alkylation | 0.8 | 49.55 | 52.57 | 53.12 |
| MACRO-AVG | - | 45.79 | 51.73 | 52.91 |

This is the clean Stage 2 result: ReaFNN adds information beyond KNN retrieval
when both branches are allowed to supply independently selected historical
contexts.

## End-to-End Seed-0 Result

The formal run used the same Stage 1 test route caches and fixed test manifests
as the historical serial mainline. Candidate recall and Sys@k are percentages.
The final column is the absolute Sys@10 change relative to
`outputs/stage23_knn_reafnn_checked_prior_seed0_20260828`.

| Family | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | Delta Sys@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 42.13 | 19.57 | 23.83 | 25.96 | 31.49 | +3.83 |
| Buchwald-Hartwig | 58.96 | 33.48 | 44.86 | 49.04 | 52.59 | +0.18 |
| Chan-Lam | 72.82 | 31.28 | 45.38 | 52.82 | 60.51 | -0.51 |
| Diels-Alder | 34.78 | 14.30 | 20.47 | 22.97 | 26.25 | -0.92 |
| Friedel-Crafts acylation | 64.84 | 21.47 | 32.42 | 37.68 | 46.11 | -2.74 |
| Friedel-Crafts alkylation | 51.06 | 23.25 | 35.37 | 40.38 | 43.72 | -1.00 |
| MACRO-AVG | 54.10 | 23.89 | 33.72 | 38.14 | 43.44 | -0.19 |

The pooled candidate-recall macro average is unchanged at 54.10 percent, but
the reordered candidate distributions interact differently with the current
Stage 3 reranker. Therefore this run supports the Stage 2 complementarity claim
without supporting a headline replacement claim.

## Rejected Follow-Up Trials

1. Protecting the top 12 KNN members before filling the remaining fused slots
   was tested on Friedel-Crafts acylation. Candidate recall was unchanged at
   64.84 percent, but Sys@10 fell from 46.11 to 43.58 percent. The code path was
   removed.
2. Per-branch min-max score normalization was tested on the same family. It
   selected `w=1.0`, so it reduced to KNN-only behavior and did not demonstrate
   a neural contribution. The code path was removed.

## Reproduction

```bash
cd /root/autodl-tmp/ProSys
/root/miniconda3/envs/ProSys/bin/python scripts/run_stage23_mainline_non_oracle.py \
  --repo_root . \
  --families Buchwald-HartwigCross-Coupling,Chan_LamCoupling,DielsAlder,Friedel-CraftsAcylation,Friedel-CraftsAlkylation,Beckmann \
  --output_root /tmp/prosys_stage2_parallel_fusion_vectorized_seed0_20260901 \
  --route_root outputs/stage1_routes \
  --knn_retrieval_mode product_morgan \
  --knn_top_k 64 --prefilter_contexts 64 --max_contexts 20 \
  --reafnn_enable_independent_post_fusion \
  --reafnn_independent_contexts 64 \
  --reafnn_post_fusion_weights 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --reafnn_post_fusion_validation_route_root outputs/stage1_routes_validation \
  --reafnn_device cuda --gnn_device cuda --seed 0 \
  --cleanup_family_intermediates
```

The temporary run root retains `all_results.json`, `results_flat.csv`, and a
compact `result.json` plus cleanup manifest for each family. Regenerable model
and table files are pruned after each successful family to control disk use.
