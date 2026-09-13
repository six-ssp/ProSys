# Strict Stage 2 Three-Way Control

All six families and seeds 0/1/2 are included. Values are equal-family macro means and sample SD across training seeds, in percent. Stage 1 and data splits are fixed. No mainline promotion is performed.

| Family | KNN-only + XGB-LTR | ReaFNN-only + XGB-LTR | Full parallel ProSys | Full minus ReaFNN-only (pp) |
| --- | ---: | ---: | ---: | ---: |
| Beckmann | 24.96 +/- 3.57 | 22.55 +/- 3.78 | 28.09 +/- 2.98 | +5.53 |
| Buchwald-HartwigCross-Coupling | 49.59 +/- 0.40 | 36.28 +/- 2.24 | 52.44 +/- 0.19 | +16.17 |
| Chan_LamCoupling | 55.30 +/- 1.82 | 53.76 +/- 0.39 | 61.62 +/- 1.92 | +7.86 |
| DielsAlder | 25.98 +/- 1.07 | 22.40 +/- 0.77 | 26.47 +/- 0.38 | +4.07 |
| Friedel-CraftsAcylation | 42.25 +/- 7.50 | 40.35 +/- 1.40 | 48.91 +/- 2.47 | +8.56 |
| Friedel-CraftsAlkylation | 41.08 +/- 0.93 | 42.12 +/- 1.67 | 45.09 +/- 1.29 | +2.97 |
| MACRO-AVG | 39.86 +/- 2.08 | 36.24 +/- 0.27 | 43.77 +/- 0.60 | +7.53 |

The ReaFNN-only intervention removes KNN proposals and KNN evidence, then retrains its ranker. Historical context priors and neural features remain. Thus this is a downstream branch-removal comparison, not a pure same-candidate reranking test.

Retained ReaFNN-only candidate hashes and replayed evaluation agree for all 18 records. Historical full/KNN-only controls provide compact results, not matching per-candidate files; this summary does not claim a per-query paired bootstrap or identity-matched pool across changed Stage 2 arms.

The final outcome is retained regardless of which arm performs best. Detailed Top-1/3/5/10, candidate recall, MRR and nDCG tables are in the study CSVs.
