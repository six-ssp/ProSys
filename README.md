# ProSys

**ProSys** is a family-conditioned, target-product-driven framework for
complete reaction-system recommendation. Given a target product and a specified
reaction family, it proposes retrosynthetic routes, selects feasible
reagent-solvent contexts, ranks complete systems, and estimates temperature.

## Maintained Mainline

```text
target product + specified reaction family
  -> Stage 1: family-tuned EditRetro route generation
  -> Stage 2: parallel product-Morgan KNN and ReaFNN context proposals
  -> validation-only KNN/ReaFNN post-fusion and a 20-context route-local pool
  -> Stage 3: tabular XGB-LTR reranking + R-GNN-assisted XGBoost temperature regression
  -> ranked route / reagent-set / solvent-set systems with temperature
```

Stage 2 is deliberately parallel. KNN retrieves up to 64 historical contexts
from the family training split using only the target product's 4,096-bit,
radius-2 Morgan fingerprint. ReaFNN independently scores the full
train-only historical context library from the 8,218-dimensional route vector
and retains its own top 64 contexts. Their union is fused by
`w * score_KNN + (1 - w) * score_ReaFNN`; `w` is selected separately for each
family only on persisted predicted validation routes, and the top 20 historical
contexts per Stage-1 route are passed to Stage 3.

The maintained mainline does **not** use joint Stage-2/Stage-3 optimization,
wrong-route negative-sample supervision, generated contexts, or novel
reagent-solvent combinations. XGB-LTR ranks a fixed 52-column non-graph table.
A separate XGBoost temperature regressor receives those tabular fields plus a
128-dimensional R-GNN route representation; temperature never alters system
rank or candidate membership.

## Verified Parallel Result

The current verified end-to-end record is the fixed-Stage-1, six-family,
seed-0 parallel evaluation in
[Experiment/stage2_parallel_post_fusion_20260901.md](Experiment/stage2_parallel_post_fusion_20260901.md).
All rates below are equal-family macro averages over the fixed test manifest.

| Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 |
| ---: | ---: | ---: | ---: | ---: |
| 54.10 | 23.89 | 33.72 | 38.14 | 43.44 |

- Fixed Stage-1 Route@10: `63.20%`.
- The parallel run includes the temperature branch, but a parallel multi-seed
  temperature summary has not yet been aggregated; no new temperature headline
  is claimed here.
- The earlier serial three-seed result (`44.62 +/- 0.42%` Sys@10) and its
  matched ablations remain historical references only. They are not numerical
  evidence for the current parallel candidate distribution.

## Reproduction

```bash
conda activate ProSys
bash scripts/setup_prosys_env.sh
python data_preprocess/audit_data_splits.py --strict

OUTPUT_ROOT=outputs/stage23_parallel_mainline \
ROUTE_ROOT=outputs/stage1_routes \
REAFNN_DEVICE=cuda:0 \
GNN_DEVICE=cuda:0 \
bash scripts/run_stage23_non_oracle_suite.sh .
```

The maintained launcher always trains Stage-3 tables from the family reference
train/validation splits and evaluates only on persisted Stage-1 test routes.
The validation route cache is used only to choose the Stage-2 fusion weight.

## Repository Map

| Path | Purpose |
| --- | --- |
| `stage1_retrosynthesis/` | EditRetro training, fine-tuning, and route caches |
| `stage2_ReaFNN/` | Parallel KNN/ReaFNN condition-pool construction |
| `stage3_XGBoost/` | XGB-LTR reranking and R-GNN-assisted temperature regression |
| `baseline/` | Reproducible comparison methods |
| `ablation/` | Controlled component analyses and historical controls |
| `data_preprocess/` | Cleaning, normalization, splits, and audits |
| `Experiment/` | Promoted result records and archived exploratory material |

## Documentation

- [Current result status](CURRENT_RESULTS.md)
- [Parallel Stage-2 record](Experiment/stage2_parallel_post_fusion_20260901.md)
- [Stage 2 details](stage2_ReaFNN/stage2_KNN_detail.md)
- [Stage 3 details](stage3_XGBoost/stage3_XGBoost_detail.md)
- [Baseline and ablation scope](baseline&ablation.md)
- [Metric nomenclature](NOMENCLATURE.md)
