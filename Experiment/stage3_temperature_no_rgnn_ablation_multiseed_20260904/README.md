# Matched R-GNN Temperature Ablation

## Scope

This is a three-seed, six-family temperature-only ablation matched to the maintained parallel mainline. The R-GNN arm is read from the official current mainline compact records. The new no-R-GNN arm reruns the same fixed Stage 1 routes, product-Morgan KNN (radius 2, 4,096 bits, K=64), independent ReaFNN top-64 post-fusion pool, route-local top-20 cap, and 52-feature XGB-LTR.

The only intentional difference is the temperature XGBoost input: the full arm uses the 52 tabular candidate features plus a 128D Reaction-GNN route embedding, whereas the control uses exactly the 52 tabular features and contains no route_gnn_feat_* column. Temperature never contributes candidates or ranking scores.

Temperature is evaluated only on the highest-ranked exact full-system match with a valid temperature label. The support and all Stage 1/2/3 ranking metrics are checked to be identical within each family/seed pair, so any reported temperature difference is not attributable to a changed route, candidate pool, or ranked system.

## Macro Results

| Arm | Conditional support (mean +/- sd) | MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | ---: | ---: | ---: | ---: | ---: |
| R-GNN + XGBoost temperature | 1792.33 +/- 6.43 | 11.49 +/- 0.26 | 41.41 +/- 2.16 | 63.09 +/- 1.76 | 83.74 +/- 0.69 |
| Tabular XGBoost, no R-GNN | 1792.33 +/- 6.43 | 13.93 +/- 0.38 | 35.30 +/- 0.81 | 55.56 +/- 1.23 | 78.09 +/- 1.22 |
| R-GNN gain (full - no R-GNN) | 0.00 | -2.43 | +6.11 pp | +7.53 pp | +5.65 pp |

A negative MAE difference in the final row favors the R-GNN arm; a positive hit-rate difference favors the R-GNN arm.

## Per-Family Results

| Family | R-GNN MAE (C) | No-R-GNN MAE (C) | R-GNN MAE gain (C) | R-GNN +/-10 C | No-R-GNN +/-10 C | R-GNN +/-10 C gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 11.03 +/- 1.39 | 14.14 +/- 1.34 | -3.11 | 65.11 +/- 7.95 | 50.89 +/- 7.76 | +14.22 pp |
| Buchwald-Hartwig | 11.92 +/- 0.41 | 12.32 +/- 0.67 | -0.39 | 56.89 +/- 0.74 | 56.27 +/- 1.48 | +0.62 pp |
| Chan-Lam | 5.83 +/- 0.39 | 6.98 +/- 0.24 | -1.14 | 84.41 +/- 0.94 | 77.31 +/- 0.91 | +7.11 pp |
| Diels-Alder | 18.55 +/- 1.10 | 24.16 +/- 1.51 | -5.61 | 43.27 +/- 3.94 | 34.29 +/- 1.00 | +8.97 pp |
| Friedel-Crafts Acyl. | 10.86 +/- 0.02 | 12.55 +/- 0.30 | -1.69 | 66.05 +/- 1.67 | 60.74 +/- 1.61 | +5.31 pp |
| Friedel-Crafts Alkyl. | 10.78 +/- 0.52 | 13.44 +/- 0.10 | -2.66 | 62.81 +/- 2.06 | 53.84 +/- 1.92 | +8.97 pp |

## Audit Contract

- 18 matched family/seed pairs passed exact Stage 1 route, Stage 2 protocol/pool, XGB-LTR ranking-metric, and conditional-temperature-support checks.
- Every full temperature regressor used exactly 180 features, including 128 route_gnn_feat_* dimensions.
- Every no-R-GNN temperature regressor used exactly 52 tabular features and zero route_gnn_feat_* dimensions.
- Raw candidate tables, scored tables, and binary checkpoints were removed after compact retention to respect disk limits.
