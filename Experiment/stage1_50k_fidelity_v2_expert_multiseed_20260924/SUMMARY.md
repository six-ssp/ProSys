# Stage 1 Expert Fine-Tuning: Verified Seed Results

One shared filtered-USPTO-50K base trained from random initialization; expert seeds 0/1/2. Means and sample SD use all 3,860 unchanged test queries per seed, with equal-family macro averaging. These are not independent base-pretraining or full-pipeline repetitions.

**Decoder recovery:** Diels-Alder expert seed 2 uses the same trained best checkpoint, same queries and same decode hyperparameters, with an explicit positional-capacity safety guard. One unsupported hypothesis is retained as an empty invalid generation slot; no query is removed, no sequence is truncated into different chemistry and no replacement hypothesis is drawn. A complete seed-1 no-overflow GPU replay reproduces all 76,200 hypothesis strings and all final ranked route lists/scores exactly. Two EOS-score differences on already invalid strings are recorded separately; universal token-score bitwise equality is not claimed. The other 17 original decoding outputs are unchanged. See `../stage1_decode_diagnostic_20260925/expert_recovery_admission.json`. The original failed queue log remains preserved; this is explicit recovered certification, not silent completion.

| Family | Route@1 | Route@3 | Route@5 | Route@10 |
| --- | ---: | ---: | ---: | ---: |
| Beckmann | 43.55 +/- 2.14 | 59.57 +/- 1.47 | 62.84 +/- 1.37 | 67.09 +/- 1.61 |
| Chan_LamCoupling | 55.30 +/- 1.50 | 65.21 +/- 0.82 | 67.01 +/- 0.78 | 69.32 +/- 1.18 |
| Friedel-CraftsAcylation | 39.23 +/- 0.88 | 51.86 +/- 1.08 | 55.51 +/- 0.64 | 60.56 +/- 1.49 |
| Friedel-CraftsAlkylation | 22.21 +/- 1.43 | 33.30 +/- 1.14 | 37.37 +/- 1.84 | 41.64 +/- 1.52 |
| Buchwald-HartwigCross-Coupling | 40.64 +/- 1.37 | 54.23 +/- 1.50 | 57.66 +/- 1.62 | 60.33 +/- 1.55 |
| DielsAlder | 18.50 +/- 1.60 | 26.99 +/- 0.08 | 30.49 +/- 0.42 | 33.60 +/- 0.35 |
| MACRO-AVG | 36.57 +/- 0.42 | 48.53 +/- 0.09 | 51.81 +/- 0.09 | 55.42 +/- 0.39 |
