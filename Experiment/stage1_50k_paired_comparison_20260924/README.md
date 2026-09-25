# Guarded Scratch-50K Base Versus Fixed Experts

All six family comparisons are complete on the same 3,860 original test
queries. Each expert is the predeclared seed 1 used by the downstream study,
not the best test-scoring seed. Both arms use the product-identity augmentation
guard, matching query identities and decoding settings. No query is dropped
when the model returns no route.

| Equal-family macro (%) | Route@1 | Route@3 | Route@5 | Route@10 |
| --- | ---: | ---: | ---: | ---: |
| Shared scratch-50K base | 3.59 | 6.43 | 8.04 | 10.08 |
| Family experts, fixed seed 1 | 36.55 | 48.43 | 51.72 | 55.86 |
| Difference (pp) | 32.96 | 42.00 | 43.68 | 45.78 |

| Family | Queries | Base Route@10 (%) | Expert Route@10 (%) |
| --- | ---: | ---: | ---: |
| Beckmann | 235 | 0.00 | 68.94 |
| Buchwald-Hartwig | 1,099 | 32.76 | 60.33 |
| Chan-Lam | 390 | 12.31 | 70.00 |
| Diels-Alder | 762 | 0.00 | 33.73 |
| Friedel-Crafts acylation | 475 | 14.95 | 61.89 |
| Friedel-Crafts alkylation | 899 | 0.44 | 40.27 |

`all_families_seed1.csv` retains unrounded values and separately labeled
query-weighted results. `all_families_seed1.md` is the standard one-decimal
collector export. Differences are calculated before rounding, not from the
displayed cells. Macro rates weight the six families equally.

`all_families_seed1_verification.json` records independent verification of
all six expert jobs, guarded paired-cache replay, exact table reconstruction
and source/output hashes. The file hashes were independently rechecked after
the report was generated. Individual family files remain available alongside
the complete report.

This comparison does not itself quantify expert-seed uncertainty or full-system
performance. Both separate studies are now complete, including Diels-Alder.
The expert three-seed Route@10 is 55.42 +/- 0.39%; do not mix it with this
fixed-seed-1 mean or reuse historical FULL scores. Detailed local verification
receipts require retained route/model artifacts and are not all public payloads.
