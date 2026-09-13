# ProSys Current Results

Updated: 2026-09-13. This page contains only the maintained parallel mainline.
Pre-cleanup mixed-version text is retained in
[the document archive](Experiment/document_archive_20260913/CURRENT_RESULTS_before_cleanup.md).

**Code/result version boundary:** A subsequent canonicalization repair and
deterministic R-GNN runtime are under verification in
[the project-completion study](Experiment/project_completion_20260913/FINDINGS.md).
They do not retroactively change the promoted values below. Stage-1 three-seed
training is paused after a new audit found condition-validation overlaps in
the USPTO base corpus; no full-reaction condition-test overlap was found.
Do not describe current downstream repeats as full-pipeline training repeats.

## Maintained implementation

```text
target product -> selected family-specific EditRetro expert -> top 10 routes
  -> parallel product-Morgan KNN and route-conditioned ReaFNN proposals
  -> validation-selected rank-prior fusion -> top 20 contexts per route
  -> 52-column XGB-LTR plus Stage 1/2 prior -> ranked reaction systems
  -> separate temperature XGBoost: 52 tabular + 128 R-GNN features
```

All contexts come from the corresponding family training library. The current
configuration has no serial KNN-core correction, novel-context generation,
joint training, explicit wrong-route negative training, graph-to-ranker input
or temperature gate. Family artifacts are selected externally; the target
product is the sole external molecular query.

## Reporting protocol

- Six families; fixed test manifest of 3,860 query identities. The 27 identities
  without candidate slates remain zero-valued full-system failures.
- An exact system jointly matches canonical reactants, normalized reagent set
  and normalized solvent set. Temperature is not part of full-system Top-k.
- Temperature selects the highest-ranked finite-temperature exact match from
  the entire candidate slate, without a Top-10 cutoff. It is conditional and
  is not end-to-end temperature accuracy on all test queries.
- Each seed first computes metrics per family, then averages equally across
  families. Reported variability is sample SD across seeds 0, 1 and 2.
- Stage 1 caches and data splits are fixed across these downstream repeats.
  They do not measure EditRetro retraining or split variability.
- Canonical reactions are split-disjoint; products need not be split-disjoint.
  Fitting uses training data and configured selection uses validation data.
  These repeats are not a fresh held-out benchmark after method development.

## Mainline

Source: [promoted three-seed results](Experiment/stage23_parallel_post_fusion_multiseed_20260903/).
Rates below are percentages; differences use percentage points (pp).

| Candidate recall | Full-system Top-1 | Top-3 | Top-5 | Top-10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |

Fixed Stage 1 Route@1/3/5/10: `43.46/56.69/59.97/63.20%`.

| Family | Full-system Top-10 |
| --- | ---: |
| Beckmann | 28.09 +/- 2.98 |
| Buchwald-Hartwig | 52.44 +/- 0.19 |
| Chan-Lam | 61.62 +/- 1.92 |
| Diels-Alder | 26.47 +/- 0.38 |
| Friedel-Crafts acylation | 48.91 +/- 2.47 |
| Friedel-Crafts alkylation | 45.09 +/- 1.29 |

Conditional temperature: MAE `11.49 +/- 0.26 C`; within +/-5/10/20 C:
`41.41 +/- 2.16 / 63.09 +/- 1.76 / 83.74 +/- 0.69%`.
Pooled eligible support is `1,785 / 1,795 / 1,797` for seeds 0/1/2.

## Baselines

Source: [baseline protocol and results](baseline/baseline.md), with compact
records in [baseline/results](baseline/results/multiseed_20260810/).

| Method | Full-system Top-10 |
| --- | ---: |
| Product-Bernoulli Naive Bayes (deterministic) | 21.48 |
| Product-GNN | 23.03 +/- 0.73 |
| EditRetro + Sequential FNN | 31.71 +/- 0.10 |
| EditRetro + Reaction-GCNN | 21.10 +/- 0.28 |
| ProSys | 43.77 +/- 0.60 |

The current difference from the strongest evaluated baseline is `+12.06 pp`.
Sequential FNN has temperature MAE `12.20 +/- 0.29 C` and within-10-C rate
`66.04 +/- 1.32%`. Its conditional support differs from ProSys; do not claim
uniform temperature superiority or a paired common-support baseline test.

## Component controls

| Control | Candidate recall | Full-system Top-10 | Full-minus-control Top-10 |
| --- | ---: | ---: | ---: |
| KNN-only + retrained XGB-LTR | 53.39 +/- 0.00 | 39.86 +/- 2.08 | +3.91 pp |
| ReaFNN-only + retrained XGB-LTR | 46.48 +/- 0.30 | 36.24 +/- 0.27 | +7.53 pp |
| Full Stage 2 + deterministic Stage 1/2 ranking | 54.26 +/- 0.15 | 36.03 +/- 0.16 | +7.74 pp |

The Stage 2 control changes candidate composition and retrains its ranker. The
Stage 3 control holds candidate-generation settings fixed and changes ranking.
These effects are not additive. Stage 3 Top-1 gains change direction across
seeds, so a stable Top-1 benefit is not established.

| Temperature model | MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | ---: | ---: | ---: | ---: |
| 52 tabular + 128 R-GNN | 11.49 +/- 0.26 | 41.41 +/- 2.16 | 63.09 +/- 1.76 | 83.74 +/- 0.69 |
| 52 tabular, no R-GNN | 13.93 +/- 0.38 | 35.30 +/- 0.81 | 55.56 +/- 1.23 | 78.09 +/- 1.22 |

Temperature MAE decreases by `2.43 C` when calculated before rounding.

Sources:

- [Stage 2 control](Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/).
- [Strict ReaFNN-only control and three-way comparison](Experiment/stage2_reafnn_only_multiseed_20260913/SUMMARY.md):
  all 18 retained candidate files pass hash verification and metric replay;
  this arm removes KNN proposals and KNN-derived evidence, then retrains LTR.
- [Stage 3 control](Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/).
- [Temperature control](Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/).

Audit boundary: retained controls match protocol fields and aggregate candidate
statistics; the temperature pairs also match ranking metrics and support
counts. The compact artifacts do not independently establish per-candidate
or per-query support identity by hashes. Numerical results are retained, but
aggregate agreement must not be described as a completed identity-level audit.

## Completed evidence reconstruction (2026-09-13)

The [six-item completion report](Experiment/mainline_evidence_completion_20260913/COMPLETION_REPORT.md)
adds 18 identity-matched reconstructions, subgroup diagnostics, exhaustive
failure counts, 18 traced cases and six cold product-query smokes. All
family/seed Sys@1/3/5/10 results reproduce the promoted values exactly.
No-LTR remains `36.03 +/- 0.16%` Sys@10 on exactly the reconstructed candidates.

Temperature is a distinct fresh reconstruction, not a silent replacement:
R-GNN-assisted MAE is `11.82 +/- 0.33 C` and within-10 C is
`61.58 +/- 0.77%`; its exact-support no-R-GNN pair is `13.93 +/- 0.38 C`
and `55.56 +/- 1.23%`. The newly paired gains are `2.11 C` and `6.02 pp`.
The original `11.49 C` headline remains historical promoted evidence above.

Condition-memory-unseen products account for 3,569/3,860 query identities and
have `45.32 +/- 0.62%` macro Sys@10. This does not establish that they are
unseen to Stage 1 pretraining or fine-tuning, nor an independent external test.
The known one-record training-side canonicalization issue and missing pilot
runtime-source manifest are explicitly described in the completion report.

## Writing boundaries

The contribution is the evaluated product-to-system workflow and its
complementary evidence/ordering design, not a new KNN, XGBoost or general GNN
algorithm. Exact-match recovery is not experimental feasibility validation.
Current results do not establish automatic family routing, pipeline-wide unseen-product
generalization, novel-context generation or prospective synthesis success.
The older document snapshots and immutable experiment histories remain in
`Experiment/`; their headline scores must not be mixed into this mainline.
