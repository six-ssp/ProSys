# Fresh 50K-Route Downstream Evidence

All 18 family/downstream-seed runs passed retained-row replay. All downstream seeds use the fixed family expert seed 1. These are not three independent base-pretraining runs.

Rates are percentages; mean +/- sample SD across downstream seeds 0/1/2. Macro averages weight families equally, not queries. Temperature is evaluated separately on the first ranked exact system with a finite reference/prediction, without a top-10 restriction. Missing temperature support is NA, never zero; support counts and available-family counts are in CSV.

| Arm | Sys@1 | Sys@3 | Sys@5 | Sys@10 | Temp MAE (C) | Within 10 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 19.81 +/- 0.30 | 29.74 +/- 0.51 | 33.33 +/- 0.17 | 37.87 +/- 0.27 | 11.32 +/- 0.33 | 62.27 +/- 0.13 |
| no_ltr | 20.22 +/- 0.14 | 26.52 +/- 0.23 | 28.20 +/- 0.18 | 29.90 +/- 0.10 | NA | NA |
| no_rgnn_temperature | 19.81 +/- 0.30 | 29.74 +/- 0.51 | 33.33 +/- 0.17 | 37.87 +/- 0.27 | 13.63 +/- 0.29 | 55.62 +/- 1.27 |

No-LTR reuses exactly the full candidate pool. No-R-GNN changes only the temperature regressor and preserves system order and conditional temperature support. These controls do not replace Stage 2 removal experiments or the four baseline models.
