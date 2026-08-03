# Current Mainline Ablation Protocol

更新日期：`2026-07-27`

## Scope

本文件定义当前 ProSys 主线的消融口径。当前系统为：

```text
target product
  -> Stage 1 family-tuned EditRetro routes
  -> Stage 2 widened KNN retrieval + ReaFNN condition-pool selection
  -> Stage 3 Reaction-GNN features + XGBoost reranking and temperature regression
```

所有结果均为端到端的 Non-Oracle 测试：Stage 2 的测试输入只来自对应家族的 Stage 1 `route_cache.json`，不以真实反应路线替代预测路线。

## Unified Protocol

- Families: Beckmann, Buchwald-Hartwig, Chan-Lam, Diels-Alder, Friedel-Crafts acylation, and Friedel-Crafts alkylation.
- Test denominator: every identity in the Stage 1 test route cache. A product without any Stage 1 route remains in the denominator and contributes a zero hit.
- Stage 1 budget: at most 10 predicted routes per product.
- Stage 2 budget: at most 20 reagent-solvent contexts per route.
- Condition evidence, vocabularies, context frequencies, and yield summaries are built from the family training split only.
- All XGBoost fitting, early stopping, and score-fusion weight selection use train/validation tables only. The test table is scored once after selection.
- Candidate rows are stably sorted before ranking, so score ties have a deterministic resolution.

### Training-Time KNN Safeguard

For a training reaction, KNN retrieval excludes the same canonical reaction from the train memory. Its own condition records are also subtracted from the global context count, support, and mean-yield statistics used as KNN features. This prevents a training candidate from receiving a trivial self-neighbor similarity of 1.0.

Validation and test queries are already outside the train memory under the reaction-group split, so their retrieval path is unchanged. The test path remains entirely Stage-1-route driven.

## A1: Stage 1 Family Adaptation

**Question:** Does family-specific Stage 1 fine-tuning improve route proposal quality?

**Comparison:** the full-data base model versus the corresponding family-tuned model, evaluated on the identical cached test identities.

**Metrics:** Route@1, Route@3, Route@5, and Route@10.

**Controlled variables:** the product list, test manifest, decoding budget, and route evaluation code are identical. The runner asserts cache-identity equality before computing a result.

## A2: Stage 2 Candidate Screening

**Question:** Does route-conditioned retrieval provide more useful feasible-system candidates than a simple frequency prior, and does the ReaFNN expansion add value beyond KNN?

All arms use the same Stage 1 routes, maximum 20 contexts per route, frozen family Reaction-GNN encoder, and a separately retrained XGBoost ranker/temperature regressor.

| Arm | Candidate-pool construction | Purpose |
| --- | --- | --- |
| Full pool | KNN top-64 retrieval, prefilter 64 contexts, then ReaFNN re-scores/augments and retains at most 20 contexts | Current Stage 2 design |
| KNN only | KNN top-64 retrieval followed by the same 20-context cap, without ReaFNN | Isolate the ReaFNN pool contribution |
| Top-20 frequency | The 20 most frequent training-split reagent-solvent contexts, independent of the query route | Replace local retrieval with a non-local prior |

For every arm, its own train, validation, and Non-Oracle test tables are built and its own XGBoost models are trained. This is necessary because a changed candidate pool changes both the available positives and the candidate-feature distribution.

**Primary metrics:** pool coverage, Sys@1, Sys@3, Sys@5, Sys@10, MRR, and nDCG@10.

## A3: Stage 3 Learned Reranking

**Question:** Once Stage 2 has constructed a feasible pool, does learned reranking materially improve the order of complete reaction systems?

All A3 rows use the saved full KNN+ReaFNN Stage 2 candidate tables and the same Stage 1 test routes.

| Arm | Stage 3 input / ranking | Purpose |
| --- | --- | --- |
| Full | tabular features plus 64 Reaction-GNN route features, XGBRanker | Current Stage 3 design |
| Without Reaction-GNN | the same table with all `route_gnn_feat_*` features absent, XGBRanker | Isolate graph-derived route representation |
| Without XGBoost | deterministic Stage 1/Stage 2 prior only | Isolate learned listwise reranking |

The deterministic ranking orders candidates by route rank, route probability, ReaFNN evidence, KNN support, and stable textual tie breakers. It does not fit another model. Because it has no temperature head, temperature metrics are intentionally reported as N/A for this arm.

## A4: ReaFNN-GNN Interaction Control

The single-factor A2 and A3 comparisons can hide interactions. We therefore also evaluate the fourth factorial corner:

| ReaFNN pooling | Reaction-GNN features | Method |
| --- | --- | --- |
| yes | yes | Full mainline |
| no | yes | KNN only + Reaction-GNN + XGBoost |
| yes | no | KNN + ReaFNN + XGBoost |
| no | no | KNN only + XGBoost |

This control uses the same full-manifest denominator, Stage 1 routes, candidate cap, and train/validation-only fitting as the other arms. It is required before assigning an apparent gain or loss to ReaFNN or Reaction-GNN alone.

## Labels and Features

Candidate labels are created only after candidate-pool generation:

- `route_match`: predicted route equals the gold route.
- `context_match`: candidate reagent-solvent context equals the gold context.
- `label`: exact route-and-context match.
- `rank_relevance`: graded supervision for the listwise XGBoost objective.
- `temperature_gold`: retained only for exact matches with a valid temperature label.

XGBoost always excludes `label`, `route_match`, `context_match`, `rank_relevance`, `sample_weight`, `temperature_gold`, and `yield_gold` from its input matrix. Each saved model metadata file is audited for this contract.

## Temperature Evaluation

Temperature is evaluated separately from system ranking. For each product, it is counted only when the highest-ranked exact route-and-condition match has a valid temperature label. We report MAE and the fraction within plus or minus 5, 10, and 20 degrees C. Temperature support `n` is reported alongside every aggregate because it is conditional on exact system retrieval.

## Reproduction

The maintained experiment runner is:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
/root/miniconda3/envs/ProSys/bin/python ablation/run_current_mainline_ablation.py \
  --repo_root . \
  --families all \
  --run_set all \
  --output_root outputs/ablation_reafnn_gnn_20260726 \
  --mainline_root outputs/stage23_mainline_reafnn_gnn_fused_20260723 \
  --gnn_device cuda \
  --knn_workers 8 \
  --force
```

The corrected Stage 2/3 mainline must be rebuilt first with `scripts/run_stage23_mainline_non_oracle.py --force_rebuild`; the current run records this prerequisite in the final result audit.

## Result Files

The final tables are written under `outputs/ablation_reafnn_gnn_20260726/`:

- `stage1_route_ablation.md`
- `stage2_pool_ablation.md`
- `stage3_reranking_ablation.md`
- `stage23_interaction_ablation.md`
- `interaction_results.json`
- `audit.md`

The completed run and its conservative, paper-safe interpretation are recorded in
[`current_mainline_ablation_results_20260727.md`](current_mainline_ablation_results_20260727.md).

Interpretation follows the data rather than the intended architecture: a component is described as beneficial only if the controlled family-level and macro-average comparisons support it. A mixed or negative effect is reported as such rather than being used to claim an architectural improvement.
