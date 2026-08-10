# ProSys: A Product-to-System Framework for Target Product-Driven Reaction-System Recommendation

## Current Verified Results (2026-08-09)

**Current maintained mainline.** Every family uses a fixed R-GNN
(128-dimensional, four-layer) plus XGBoostRegressor temperature branch. There
is no family-level no-GNN fallback and no temperature-model selection gate. The
current official figures below are from a completed six-family, three-seed
reproduction; the older gated point snapshot is historical only.

## Current Scope

This is a family-stratified, target-product-driven, end-to-end specialist
evaluation. The molecular query is the target product only. A family identifier
selects separately trained specialist artifacts, condition memory, and label
vocabularies outside feature encoders; it is not concatenated to a product
fingerprint, route representation, ReaFNN input, or XGB feature row.

    target product
      -> family-specific specialist artifacts selected by the evaluation partition
      -> family-tuned EditRetro route proposals
      -> KNN wide recall plus ReaFNN condition-pool selection
      -> tabular XGB-LTR full-system reranking
      -> fixed R-GNN structural representation plus XGBoost temperature prediction
      -> ranked reaction systems

## Evaluation Protocol

- Fixed end-to-end test manifest: 3,860 product identities across six families.
- Fixed Stage 1 route caches: 3,833 products have a candidate slate in every
  seed; the remaining 27 products remain in the denominator and receive zero
  for full-system metrics.
- The fixed Stage 1 macro Route@1/3/5/10 is
  43.46/56.69/59.97/63.20%; this robustness experiment measures Stage 2/3
  stochastic variation, not Stage 1 retraining variation.
- A full-system match is an exact joint match of canonical route, normalized
  reagent set, and normalized solvent set.
- Stage 2 emits at most 20 reagent-solvent contexts per proposed route.
- ReaFNN, the R-GNN, and XGBoost are retrained independently at seeds 0, 1,
  and 2. KNN retrieval, split, route caches, vocabulary construction, and
  evaluation are held fixed.
- Temperature is a conditional statistic. For a product with an available
  valid-temperature exact system match, the highest-ranked such match supplies
  one absolute error. It is not an all-candidate regression average and it is
  not included in the full-system ranking score.

## Current Mainline Performance

All values are unweighted six-family macro averages over three seeds. Rates are
percentages and +/- denotes sample standard deviation (ddof=1).

| Candidate coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48.52 +/- 0.59 | 28.96 +/- 0.91 | 36.61 +/- 1.05 | 39.53 +/- 1.24 | 42.90 +/- 1.00 | 33.78 +/- 0.96 | 35.13 +/- 0.97 |

| Family | Candidate coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 35.89 +/- 3.19 | 18.30 +/- 0.74 | 22.27 +/- 1.72 | 23.83 +/- 1.95 | 27.94 +/- 1.72 | 21.41 +/- 1.07 | 22.03 +/- 0.98 |
| Buchwald-Hartwig | 48.35 +/- 0.37 | 36.12 +/- 0.98 | 42.77 +/- 1.03 | 44.59 +/- 0.66 | 46.07 +/- 0.56 | 39.77 +/- 0.79 | 40.90 +/- 0.68 |
| Chan-Lam | 68.12 +/- 1.41 | 46.92 +/- 1.36 | 56.50 +/- 1.04 | 59.66 +/- 1.21 | 62.39 +/- 1.55 | 52.52 +/- 1.23 | 53.94 +/- 1.37 |
| Diels-Alder | 28.70 +/- 1.00 | 18.02 +/- 0.89 | 23.71 +/- 0.08 | 25.55 +/- 0.89 | 26.99 +/- 0.87 | 21.34 +/- 0.46 | 22.40 +/- 0.12 |
| Friedel-Crafts Acyl. | 60.63 +/- 0.42 | 25.89 +/- 5.37 | 34.88 +/- 6.23 | 40.42 +/- 5.73 | 47.86 +/- 5.16 | 32.89 +/- 5.36 | 34.71 +/- 5.68 |
| Friedel-Crafts Alkyl. | 49.46 +/- 0.23 | 28.51 +/- 1.38 | 39.53 +/- 0.84 | 43.16 +/- 0.95 | 46.16 +/- 0.22 | 34.78 +/- 0.87 | 36.78 +/- 0.79 |

## Current Temperature Performance

The temperature support is 1,603, 1,592, and 1,576 matched products at seeds
0, 1, and 2, respectively. The support changes with the learned ranking, so
temperature results should be reported together with system accuracy rather
than as a manifest-wide score.

| Family | Conditional MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | ---: | ---: | ---: | ---: |
| Beckmann | 10.52 +/- 1.73 | 38.27 +/- 6.57 | 62.93 +/- 10.86 | 86.13 +/- 4.14 |
| Buchwald-Hartwig | 10.72 +/- 0.24 | 36.09 +/- 1.58 | 61.13 +/- 1.62 | 84.59 +/- 1.53 |
| Chan-Lam | 5.85 +/- 0.53 | 66.05 +/- 2.37 | 83.42 +/- 2.86 | 94.73 +/- 1.52 |
| Diels-Alder | 17.44 +/- 1.31 | 26.19 +/- 2.68 | 48.13 +/- 2.61 | 69.62 +/- 3.74 |
| Friedel-Crafts Acyl. | 9.94 +/- 0.33 | 42.82 +/- 4.18 | 65.75 +/- 2.44 | 86.43 +/- 0.94 |
| Friedel-Crafts Alkyl. | 10.01 +/- 0.12 | 40.57 +/- 2.49 | 63.47 +/- 0.38 | 86.93 +/- 1.62 |
| Macro average | 10.75 +/- 0.14 | 41.66 +/- 1.39 | 64.14 +/- 1.65 | 84.74 +/- 0.38 |

## Reproducibility Record

The compact formal artifacts are retained at
outputs/unified_rgnn_multiseed_20260809/:

- README.md: macro and per-seed summary.
- macro_mean_std.csv, macro_by_seed.csv, and per_family_mean_std.csv:
  reportable aggregates.
- per_family_seed_metrics.csv: family-level seed records.
- compact/: audited per-family metrics and model metadata.

The raw candidate tables and checkpoints were intentionally pruned only after
the collector retained these compact records, to respect the storage budget.
All 18 family-seed temperature metadata records state always_enabled = true
and selection = none; every ranker has 52 non-graph features and every
temperature regressor has 180 features, including 128 R-GNN columns.

## Historical Context

The prior 2026-08-03 gated point snapshot used a different temperature
selection policy and an earlier R-GNN configuration. It remains useful for
traceability in its archived output directory, but is not the current
headline result and must not be compared as a gate-only ablation: the current
implementation simultaneously changed R-GNN capacity to 128 dimensions and
four message-passing layers. A matched architecture with and without a gate
would be required for a causal gate ablation.

## Evidence Strength From Ablation

| Controlled comparison | Main result | Interpretation |
| --- | --- | --- |
| Base Stage 1 vs family-tuned Stage 1 | Route@10: `14.97% -> 63.20%` | Fine-tuning improves every family; the macro gain is `+48.22 pp`. |
| Global top-20 frequency pool vs full Stage 2 | candidate recall: `25.63% -> 49.18%`; historical full-system Top-10 accuracy: `17.72% -> 42.68%` | Route-conditioned local retrieval is essential. |

| Fixed R-GNN temperature branch | Current three-seed conditional MAE: 10.75 +/- 0.14 C; Within +/-10 C: 64.14 +/- 1.65% | Every family uses the same R-GNN plus XGBoost temperature architecture with no fallback or model-selection gate; it does not affect ranking metrics. |

| Full Stage 2 vs KNN-only Stage 2 | Historical full-system Top-10 accuracy: `42.68%` vs `43.33%` | ReaFNN changes candidate composition but is not a universal macro full-system Top-k accuracy gain in that control. |
| Candidate-aware route-context GNN residual | Beckmann interaction pilot: selected alpha `0`; related six-family residual probe: `0/6` accepted | The branch is validation-gated and remains exploratory; it does not alter headline `full-system Top-k accuracy`. |

The current temperature head is fixed before test evaluation: every family uses
the same R-GNN plus XGBoost architecture. The unrelated candidate-aware residual
remains a historical validation-only exploratory branch and is not part of the
headline ranking or temperature results.

## Candidate-aware GNN Update

`stage3_XGBoost/condition_aware_gnn.py` adds a route-context interaction
network on top of frozen 64-dimensional R-GNN features and learned reagent
and solvent token embeddings. Its score fusion is selected on validation
full-system Top-10 accuracy only. The completed Beckmann interaction-model pilot selected
`alpha = 0`, and the related six-family auxiliary-token residual probe selected
no nonzero residual. This implementation is preserved for further development,
but it is not part of the official model or of the result tables above.

## Baseline Scope

Four reproducible baselines are complete on the same fixed manifest. Two are
direct Product-to-Condition models, which predict contexts from the target
product before pairing them with frozen Stage 1 routes; two are downstream
route-conditioned baselines.

`Product-Bernoulli Naive Bayes` (B1) is deterministic under fixed data and
hyperparameters. B2 Product-GNN, B3 EditRetro + Sequential FNN, and B4
EditRetro + Reaction-GCNN were independently retrained at seeds `0`, `1`, and
`2` with fixed Stage 1 caches and validation-only fusion selection.

| Method | Replication | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Product-Bernoulli Naive Bayes | deterministic | 31.68 | 9.10 | 14.87 | 17.45 | 21.48 | 13.15 | 14.56 |
| Product-GNN | 3 seeds | 38.33 +/- 0.28 | 6.11 +/- 0.49 | 12.84 +/- 0.56 | 16.75 +/- 0.43 | 23.03 +/- 0.73 | 11.40 +/- 0.54 | 13.32 +/- 0.58 |
| EditRetro + Sequential FNN | 3 seeds | 45.85 +/- 0.26 | 16.75 +/- 0.27 | 24.32 +/- 0.28 | 27.68 +/- 0.40 | 31.71 +/- 0.10 | 22.01 +/- 0.22 | 23.64 +/- 0.18 |
| EditRetro + Reaction-GCNN | 3 seeds | 38.15 +/- 0.24 | 7.43 +/- 0.21 | 13.16 +/- 0.10 | 16.39 +/- 0.34 | 21.10 +/- 0.28 | 11.88 +/- 0.11 | 13.27 +/- 0.10 |
| ProSys current mainline | 3 seeds | 48.52 +/- 0.59 | 28.96 +/- 0.91 | 36.61 +/- 1.05 | 39.53 +/- 1.24 | 42.90 +/- 1.00 | 33.78 +/- 0.96 | 35.13 +/- 0.97 |

B2-B4 are now matched to the mainline in replication depth and use the same
fixed full test manifest. B1 remains a deterministic reference, not a
pseudo-replicated stochastic model. The reaction-level split is not
product-disjoint, which remains a required disclosure for every product-only
model.

The detailed comparison is documented in
[`baseline/multiseed_baseline_results_20260810.md`](baseline/multiseed_baseline_results_20260810.md).
The direct Product-to-System Transformer remains unreported because no
verified compatible checkpoint was available. It is not a zero-valued baseline.

## Audit Status

- Strict canonical-reaction split audit: PASS; every reported train/validation/
  test overlap count is zero.
- Mainline candidate stability after the leave-one-reaction-out correction:
  PASS; held-out candidate keys are unchanged.
- Candidate-aware GNN residual probes: PASS as validation-only exploratory
  controls; no nonzero residual was accepted, so headline rankings are unchanged.
- Mainline ablation audit: PASS; fixed manifest, candidate cap, target-feature
  exclusion, and full-reference equality all pass.
- Multi-seed baseline audit: PASS; `54` family-level records cover B2-B4 across
  three seeds and six families, with the same `3,860`-product manifest,
  `3,833` candidate slates, and `27` retained no-route failures in every run.
- Fixed R-GNN temperature branch audit: PASS; all 18 ranker feature lists
  exclude route_gnn_feat_*, all 18 temperature models contain the 128 graph
  features, and metadata records always_enabled = true with selection = none.

See [`project_audit_20260727.md`](project_audit_20260727.md) for scope,
remaining limitations, and links to the raw audit artifacts.
