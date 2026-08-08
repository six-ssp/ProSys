# ProSys: A Product-to-System Framework for Target-Product-Driven Reaction-System Recommendation

## Current Verified Results

Date: `2026-08-03`

## Canonical Scope

This is the only current result summary to cite for the maintained ProSys
pipeline. It covers six Reaxys reaction families in a family-stratified,
end-to-end specialist evaluation setting. The target product is the only
molecular query and no model feature encodes a reaction-family or
reaction-type label:

```text
target product
  -> family-specific specialist artifacts selected by the evaluation partition
  -> family-tuned EditRetro route proposals
  -> KNN wide recall + ReaFNN condition-pool selection
  -> tabular XGB-LTR full-system reranking + validation-gated R-GNN temperature prediction
  -> ranked reaction systems
```

The family identifier selects the separately trained expert, condition memory,
and label vocabulary outside the feature encoder. It is not concatenated to a
product fingerprint, route representation, ReaFNN input, or XGB-LTR feature
row. Thus these results are not product-plus-reaction-type feature fusion;
they are reported for family-specific specialists.

Canonical result artifacts:

- `outputs/stage23_mainline_gnn_temperature_gated_20260803/`
- `outputs/stage23_mainline_gnn_temperature_gated_20260803/gnn_temperature_gate_audit.tsv`
- `outputs/current_mainline_multiseed_20260807/` (fixed-Stage-1 three-seed robustness)
- `outputs/ablation_reafnn_gnn_20260726/`
- `outputs/baselines/non_oracle_external_b23_20260726/`

The earlier `outputs/stage23_mainline_reafnn_gnn_fused_20260723/` snapshot is
retained for ablation traceability. It exposed a mixed GNN ranking effect and
is no longer the source of current headline metrics.

## Evaluation Protocol

- Fixed end-to-end test manifest: `3,860` product identities across six
  families.
- `27` identities without a Stage 1 route candidate remain in the denominator
  and receive zero for end-to-end system metrics.
- A complete system is an exact joint match of canonical route, normalized
  reagent set, and normalized solvent set.
- Stage 2 emits at most `20` reagent-solvent contexts per proposed route.
- Temperature is evaluated only for a valid-temperature sample whose
  highest-ranked exact system match is available. It is a conditional
  regression statistic, not a score over all test products.

## Mainline Performance

| Family | Route@10 | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 69.79 | 39.57 | 18.72 | 23.83 | 25.96 | 29.79 | 22.64 | 23.13 |
| Buchwald-Hartwig | 69.97 | 48.68 | 37.22 | 44.68 | 45.77 | 46.77 | 41.03 | 42.17 |
| Chan-Lam | 77.44 | 66.67 | 45.90 | 55.38 | 59.23 | 60.77 | 51.45 | 52.68 |
| Diels-Alder | 37.27 | 29.79 | 17.32 | 23.62 | 26.25 | 27.56 | 20.94 | 22.27 |
| Friedel-Crafts Acyl. | 70.32 | 60.63 | 32.00 | 41.68 | 46.95 | 53.05 | 38.89 | 40.94 |
| Friedel-Crafts Alkyl. | 54.39 | 49.72 | 29.48 | 39.04 | 42.60 | 45.49 | 35.21 | 36.78 |
| Macro average | 63.20 | 49.18 | 30.11 | 38.04 | 41.13 | 43.91 | 35.03 | 36.33 |

| Temperature statistic | Value |
| --- | ---: |
| N_temp | 1,603 |
| MAE (deg C) | 11.11 |
| Within +/-5 deg C | 39.22% |
| Within +/-10 deg C | 62.62% |
| Within +/-20 deg C | 82.93% |

## Current Fixed-Stage-1 Three-Seed Robustness

The current Stage 2/3 mainline was freshly rebuilt with seeds `0`, `1`, and
`2` on 2026-08-07. The persisted split, six-family 3,860-product test
manifest, and Stage 1 route caches were frozen; ReaFNN, the Reaction-GNN
temperature representation, XGBoost learning-to-rank, and both temperature
regressors were independently retrained for each seed. This measures Stage
2/3 stochastic variation only, not variance from retraining the family-tuned
EditRetro models.

Every run contained `3,833` products with a Stage 1 candidate slate and `27`
products without one; all `3,860` products remained in the full-system
denominator. The frozen Stage 1 macro Route@1/3/5/10 values were
`43.46/56.69/59.97/63.20%` in all rounds. All 18 rankers used the same
52-column non-graph feature schema; Reaction-GNN features were used only in
the validation-gated temperature branch.

| Metric | Mean +/- sample SD |
| --- | ---: |
| Candidate coverage | 48.52 +/- 0.59% |
| Full-system Top-1 accuracy | 28.96 +/- 0.91% |
| Full-system Top-3 accuracy | 36.61 +/- 1.05% |
| Full-system Top-5 accuracy | 39.53 +/- 1.24% |
| Full-system Top-10 accuracy | 42.90 +/- 1.00% |
| MRR | 33.78 +/- 0.96% |
| nDCG@10 | 35.13 +/- 0.97% |
| Conditional temperature MAE | 11.20 +/- 0.39 C |
| Temperature within +/-5 C | 40.70 +/- 2.99% |
| Temperature within +/-10 C | 62.43 +/- 2.02% |
| Temperature within +/-20 C | 83.91 +/- 0.85% |

Temperature is calculated only when a valid candidate exactly matches both the
route and normalized condition context and has a valid temperature label. For
each such product, the highest-ranked valid full match supplies the absolute
temperature error; it is therefore a conditional regression statistic rather
than a full-manifest metric. Temperature supports were 1,603, 1,592, and
1,576 for seeds 0, 1, and 2, respectively.

The canonical single-run table above remains the archived point snapshot
(`43.91%` Full-system Top-10). The three-seed estimate should be used when
reporting current Stage 2/3 robustness; it must not be described as full
end-to-end retraining variance.


## Evidence Strength From Ablation

| Controlled comparison | Main result | Interpretation |
| --- | --- | --- |
| Base Stage 1 vs family-tuned Stage 1 | Route@10: `14.97% -> 63.20%` | Fine-tuning improves every family; the macro gain is `+48.22 pp`. |
| Global top-20 frequency pool vs full Stage 2 | candidate recall: `25.63% -> 49.18%`; historical full-system Top-10 accuracy: `17.72% -> 42.68%` | Route-conditioned local retrieval is essential. |
| R-GNN rank features | Historical ablation: `42.68%` with graph rank features vs `43.76%` without | Graph features have a mixed ranking effect and are not used by the current ranker. |
| Validation-gated R-GNN temperature branch | Fixed-ranking MAE (deg C): `12.85 -> 11.11`; Within +/-10 deg C: `56.77% -> 62.62%` | R-GNN is retained only where validation MAE improves by at least `0.25 C`; ranking metrics cannot decrease by design. |
| Full Stage 2 vs KNN-only Stage 2 | Historical full-system Top-10 accuracy: `42.68%` vs `43.33%` | ReaFNN changes candidate composition but is not a universal macro full-system Top-k accuracy gain in that control. |
| Candidate-aware route-context GNN residual | Beckmann interaction pilot: selected alpha `0`; related six-family residual probe: `0/6` accepted | The branch is validation-gated and remains exploratory; it does not alter headline `full-system Top-k accuracy`. |

The new temperature gate is selected per family on validation MAE only. It
enabled the GNN regressor for four of six families; the test set is used only
for the final, fixed-policy report.

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
route-conditioned baselines:

- `Product-Bernoulli Naive Bayes` (product Bernoulli naive Bayes): macro full-system Top-10 accuracy `21.48%`.
- `Product-GNN` (product graph neural network): macro full-system Top-10 accuracy `23.05%`.
- `EditRetro + Sequential FNN`: macro full-system Top-10 accuracy `31.67%`.
- `EditRetro + Reaction-GCNN`: macro full-system Top-10 accuracy `21.03%`.

| Method | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Product-Bernoulli Naive Bayes | 31.68 | 9.10 | 14.87 | 17.45 | 21.48 | 13.15 | 14.56 |
| Product-GNN | 38.03 | 6.55 | 12.90 | 16.77 | 23.05 | 11.63 | 13.52 |
| EditRetro + Sequential FNN | 45.99 | 17.06 | 24.64 | 27.98 | 31.67 | 22.27 | 23.84 |
| EditRetro + Reaction-GCNN | 38.01 | 7.93 | 13.16 | 16.36 | 21.03 | 12.18 | 13.50 |
| ProSys current mainline | 49.18 | 30.11 | 38.04 | 41.13 | 43.91 | 35.03 | 36.33 |

The mainline exceeds the formal low-capacity Product-Bernoulli Naive Bayes and Product-GNN direct
controls at all reported macro cutoffs. The reaction-level
split is not product-disjoint, which remains a required disclosure for every
product-only model.

The detailed comparison is documented in
[`baseline/current_baseline_results_20260727.md`](baseline/current_baseline_results_20260727.md).
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
- Canonical direct-baseline artifact and validation-only fusion audit: PASS for
  all twelve Product-Bernoulli Naive Bayes/Product-GNN family-model runs.
- Validation-gated GNN temperature branch audit: PASS; all ranker feature lists
  exclude `route_gnn_feat_*`, and the four selected temperature branches were chosen without test labels.

See [`project_audit_20260727.md`](project_audit_20260727.md) for scope,
remaining limitations, and links to the raw audit artifacts.
