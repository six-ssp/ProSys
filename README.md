# ProSys

**ProSys** is a family-conditioned, target-product-driven framework for
complete reaction-system recommendation. Given a target product and a specified
reaction family, it proposes retrosynthetic routes, selects feasible
reagent-solvent contexts, ranks complete systems, and estimates temperature.

**Verification in progress:** whole-side canonicalization and deterministic
R-GNN runtime repairs are being checked against frozen results. A new USPTO
audit also identifies a validation-overlap boundary; see
[current findings](Experiment/project_completion_20260913/FINDINGS.md).
Published result tables have not been silently replaced by unfinished runs.

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

## Verified Parallel Three-Seed Result

The current reportable end-to-end record is the fixed-Stage-1, six-family,
parallel evaluation repeated at seeds `0, 1, 2` in
[Experiment/stage23_parallel_post_fusion_multiseed_20260903](Experiment/stage23_parallel_post_fusion_multiseed_20260903/README.md).
All values are equal-family macro averages; rates are percentages and `+/-`
denotes sample standard deviation (`ddof=1`).

| Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |

- The fixed manifest has 3,860 product identities: 3,833 have a candidate
  slate and 27 no-slate identities remain in every Sys@k denominator.
- Fixed Stage-1 Route@1/3/5/10 is `43.46/56.69/59.97/63.20%`; this study
  measures only stochasticity in the learned Stage-2/3 modules.
- Conditional temperature MAE is `11.49 +/- 0.26 C`; conditional hit rates
  within `+/-5/+/-10/+/-20 C` are `41.41 +/- 2.16% / 63.09 +/- 1.76% /
  83.74 +/- 0.69%`. A matched three-seed control that removes only the 128D
  R-GNN route representation gives `13.93 +/- 0.38 C` MAE and `55.56 +/-
  1.23%` within `+/-10 C`. Retained records match aggregate system-ranking
  metrics and temperature-support counts; see the audit boundary in
  [CURRENT_RESULTS.md](CURRENT_RESULTS.md).
- [Experiment/stage2_parallel_post_fusion_20260901.md](Experiment/stage2_parallel_post_fusion_20260901.md)
  remains the detailed seed-0 development record, not the three-seed headline.
- The paired current Stage-3 ablation holds Stage-2 generation settings fixed and
  gives `36.03 +/- 0.16%` Sys@10 without XGB-LTR, a `7.74 pp` lower macro
  value than the current mainline.
- The paired current Stage-2 ablation removes only ReaFNN, rebuilds its
  KNN-only candidate pool, and retrains its 52D XGB-LTR on that pool. It gives
  `53.39 +/- 0.00%` candidate recall and `39.86 +/- 2.08%` Sys@10, versus
  `54.26 +/- 0.15%` and `43.77 +/- 0.60%` for the full parallel pipeline.
  Thus, ReaFNN contributes `+0.87 pp` candidate recall and `+3.91 pp` Sys@10
  under this end-to-end candidate-composition control.

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
The suite now checks content-bound cache manifests; use a new output root for
legacy artifacts or explicitly rebuild rather than silently reusing them.
Read-only product inference and the completed six-item evidence plan are
documented in [the completion report](Experiment/mainline_evidence_completion_20260913/COMPLETION_REPORT.md).

## Repository Map

| Path | Purpose |
| --- | --- |
| `stage1_retrosynthesis/` | EditRetro training, fine-tuning, and route caches |
| `stage2_ReaFNN/` | Parallel KNN/ReaFNN condition-pool construction |
| `stage2_KNN/` | Backward-compatible import shim; not a second implementation |
| `stage3_XGBoost/` | XGB-LTR reranking and R-GNN-assisted temperature regression |
| `baseline/` | Reproducible comparison methods |
| `ablation/` | Controlled component analyses and historical controls |
| `data_preprocess/` | Cleaning, normalization, splits, and audits |
| `Experiment/` | Promoted result records and archived exploratory material |

## Documentation

- [Current result status](CURRENT_RESULTS.md)
- [Parallel multi-seed record](Experiment/stage23_parallel_post_fusion_multiseed_20260903/README.md)
- [Parallel seed-0 development record](Experiment/stage2_parallel_post_fusion_20260901.md)
- [Matched Stage 2 ReaFNN three-seed ablation](Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/README.md)
- [Matched Stage 3 three-seed ablation](Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/README.md)
- [Matched R-GNN temperature three-seed ablation](Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/README.md)
- [Stage 2 details](stage2_ReaFNN/stage2_ReaFNN_detail.md)
- [Stage 3 details](stage3_XGBoost/stage3_XGBoost_detail.md)
- [Baseline and ablation scope](baseline&ablation.md)
- [Metric nomenclature](NOMENCLATURE.md)
- [Script map](scripts/README.md)

## Storage Policy

Version-controlled files contain source code, configuration, documentation, and
compact audited result records. `outputs/` is intentionally excluded from Git:
it contains regenerable checkpoints, route caches, candidate tables, and
per-sample predictions. The fixed `outputs/stage1_routes/` and
`outputs/stage1_routes_validation/` caches are useful accelerators for a
Stage-2/3-only rerun, but they are not the paper record and can be recreated
from the retained Stage-1 checkpoints. Reportable results must be promoted to a
named `Experiment/` record before a run root is removed.
