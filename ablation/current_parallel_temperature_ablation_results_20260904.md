# Current Parallel Temperature-Representation Ablation

> **Reportable paired control, 2026-09-04.** The compact artifact is
> [`Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/`](../Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/README.md).
> This document gives the paper-ready interpretation of the verified
> three-seed result.

## Question

Does the Reaction-GNN route representation improve conditional temperature
prediction after the same complete reaction system has already been selected?

This is deliberately not a system-ranking experiment. Temperature is emitted
only after ranking, and it never modifies candidate membership, the Stage-3
XGB-LTR score, or the final route/reagent-set/solvent-set order.

## Matched Protocol

The current full arm is read from the official parallel mainline compact record
at `Experiment/stage23_parallel_post_fusion_multiseed_20260903/`. For each of
six families and seeds `0`, `1`, and `2`, the no-R-GNN arm recomputes the
pipeline using the same:

- persisted Stage-1 test-route cache and fixed test manifest;
- product-Morgan KNN (`radius=2`, `4096` bits, `K=64`);
- independently proposed ReaFNN top-64 historical contexts, validation-only
  post-fusion, and route-local top-20 context cap;
- 52-feature tabular XGB-LTR, with the same training/validation candidate-table
  construction and seed.

Only the temperature XGBoost input changes. The full model receives 52 tabular
candidate descriptors plus the 128-dimensional R-GNN route embedding (180
inputs total). The control re-trains temperature XGBoost with the exact same 52
tabular descriptors and has zero `route_gnn_feat_*` columns. The temperature
head is trained only on strict exact-system candidate rows with a valid
`temperature_gold` label.

At test time, temperature is evaluated only for the highest-ranked exact
full-system match with a valid recorded temperature. This makes it conditional
on exact route-and-complete-context recovery. It is not an all-product
regression metric.

## Macro Results

All values are equal-family macro averages across three seeds; `+/-` is sample
standard deviation (`ddof=1`). The support is the number of eligible exact
matches after combining the six family supports within each seed.

| Temperature representation | Conditional support | MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | ---: | ---: | ---: | ---: | ---: |
| 52 tabular + 128D R-GNN | 1,792.33 +/- 6.43 | 11.49 +/- 0.26 | 41.41 +/- 2.16 | 63.09 +/- 1.76 | 83.74 +/- 0.69 |
| 52 tabular only | 1,792.33 +/- 6.43 | 13.93 +/- 0.38 | 35.30 +/- 0.81 | 55.56 +/- 1.23 | 78.09 +/- 1.22 |
| Full minus tabular-only | 0.00 | -2.43 C | +6.11 pp | +7.53 pp | +5.65 pp |

A negative MAE difference favors the R-GNN arm. Because the support is exactly
matched in each family-seed pair, the MAE and tolerance differences cannot be
explained by recovering an easier or different set of systems.

## Family-Resolved Results

| Family | R-GNN MAE (C) | No-R-GNN MAE (C) | MAE gain (C) | R-GNN +/-10 C | No-R-GNN +/-10 C | +/-10 C gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 11.03 +/- 1.39 | 14.14 +/- 1.34 | -3.11 | 65.11 +/- 7.95 | 50.89 +/- 7.76 | +14.22 pp |
| Buchwald-Hartwig | 11.92 +/- 0.41 | 12.32 +/- 0.67 | -0.39 | 56.89 +/- 0.74 | 56.27 +/- 1.48 | +0.62 pp |
| Chan-Lam | 5.83 +/- 0.39 | 6.98 +/- 0.24 | -1.14 | 84.41 +/- 0.94 | 77.31 +/- 0.91 | +7.11 pp |
| Diels-Alder | 18.55 +/- 1.10 | 24.16 +/- 1.51 | -5.61 | 43.27 +/- 3.94 | 34.29 +/- 1.00 | +8.97 pp |
| Friedel-Crafts acylation | 10.86 +/- 0.02 | 12.55 +/- 0.30 | -1.69 | 66.05 +/- 1.67 | 60.74 +/- 1.61 | +5.31 pp |
| Friedel-Crafts alkylation | 10.78 +/- 0.52 | 13.44 +/- 0.10 | -2.66 | 62.81 +/- 2.06 | 53.84 +/- 1.92 | +8.97 pp |

The graph representation reduces MAE in all six family means and improves the
within-10 C rate in all six family means. The effect is largest for Diels-Alder
in MAE (`-5.61 C`) and for Beckmann in within-10 C accuracy (`+14.22 pp`).

## Audit and Interpretation Boundary

The experiment writes a compact audit for all 18 family-seed pairs. It requires
exact equality of:

- Stage-1 route recall;
- Stage-2 protocol and candidate-pool availability;
- candidate recall, Sys@1/3/5/10, MRR, and nDCG@10;
- conditional temperature support.

It also requires the no-R-GNN regressor to expose exactly 52 features and no
`route_gnn_feat_*` feature. All 18 pairs pass. Accordingly, it is valid to
state that the retained R-GNN route representation improves **conditional
temperature prediction** under the maintained mainline. It is not valid to
state that it improves Sys@k, because the system-ranking pipeline is fixed by
design in this control.

## Reproduction

From the repository root, run:

```bash
conda run -n ProSys python scripts/run_current_mainline_temperature_ablation.py \
  --repo_root . \
  --reafnn_device cuda:0 \
  --python_bin /root/miniconda3/envs/ProSys/bin/python
```

The launcher reuses completed compact records when available. During a fresh
run it uses a temporary per-family scratch root and removes raw candidate
tables, scored tables, and binary checkpoints after compact result and metadata
retention. This preserves the audit trail while respecting the disk limit.
