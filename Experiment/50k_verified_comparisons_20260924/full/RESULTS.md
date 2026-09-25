# Verified 50K Model Comparisons

Scope: Beckmann, Buchwald-HartwigCross-Coupling, Chan_LamCoupling, DielsAlder, Friedel-CraftsAcylation, Friedel-CraftsAlkylation.
Full six-family study: True. Partial exports do not replace the full study.

Rates are percentages; neural-model uncertainty is mean +/- sample SD across downstream seeds 0/1/2. B1 is deterministic and fitted once per family: its SD is NA, not zero and not three duplicated runs. Macro rates weight families equally before seed averaging. All use the fixed expert seed 1.

Temperature uses the highest-ranked exact system with finite reference/prediction over the entire slate, not just Top-10. Missing support is NA. Available-family and per-seed support counts are in the CSVs. The R-GNN removal is paired on identical support; B3 temperature errors have their own support and are not a paired comparison.

| Model | Sys@1 | Sys@3 | Sys@5 | Sys@10 | Temp MAE (C) | Within 10 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ProSys | 19.81 +/- 0.30 | 29.74 +/- 0.51 | 33.33 +/- 0.17 | 37.87 +/- 0.27 | 11.32 +/- 0.33 | 62.27 +/- 0.13 |
| B1_ProductNB | 7.88 | 13.05 | 15.35 | 19.44 | NA | NA |
| B2_ProductGNN | 5.06 +/- 0.51 | 10.11 +/- 0.74 | 13.35 +/- 1.10 | 18.85 +/- 0.89 | NA | NA |
| B3_SequentialFNN | 13.80 +/- 0.59 | 20.11 +/- 0.50 | 22.76 +/- 0.53 | 26.16 +/- 0.61 | 12.28 +/- 0.19 | 66.14 +/- 1.10 |
| B4_ReactionGCNN | 5.31 +/- 0.17 | 10.09 +/- 0.37 | 12.96 +/- 0.76 | 17.84 +/- 0.42 | NA | NA |
| Without_KNN | 15.01 +/- 0.67 | 23.54 +/- 0.21 | 27.12 +/- 0.07 | 31.31 +/- 0.50 | NA | NA |
| Without_ReaFNN | 18.22 +/- 1.69 | 26.42 +/- 1.98 | 29.52 +/- 2.03 | 34.16 +/- 1.80 | NA | NA |
| Without_LTR | 20.22 +/- 0.14 | 26.52 +/- 0.23 | 28.20 +/- 0.18 | 29.90 +/- 0.10 | NA | NA |
| Without_RGNN_temperature | 19.81 +/- 0.30 | 29.74 +/- 0.51 | 33.33 +/- 0.17 | 37.87 +/- 0.27 | 13.63 +/- 0.29 | 55.62 +/- 1.27 |
