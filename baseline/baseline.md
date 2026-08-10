# ProSys Baseline Experiments

Updated: `2026-08-10`

The current citable comparison is maintained in
[`multiseed_baseline_results_20260810.md`](multiseed_baseline_results_20260810.md).
[`current_baseline_results_20260727.md`](current_baseline_results_20260727.md)
is retained as an archived single-seed point snapshot only.
The direct Product-to-Condition implementation is specified in
[`product_condition_baselines_detail.md`](product_condition_baselines_detail.md).

## Completed Methods

| ID | Method | Condition-model input | Output |
| --- | --- | --- | --- |
| Baseline 1 | Product-Bernoulli Naive Bayes (product Bernoulli naive Bayes) | target product only | ranked historical reagent-solvent contexts |
| Baseline 2 | Product-GNN (product graph neural network) | target product only | ranked historical reagent-solvent contexts |
| Baseline 3 | EditRetro + Sequential FNN | predicted route plus product | reagent set, solvent set, temperature |
| Baseline 4 | EditRetro + Reaction-GCNN | predicted route plus product | reagent set and solvent set |

The two canonical Product-to-Condition baselines are direct models: they do
not receive a Stage 1 route while predicting conditions. After condition
inference, each of their top-20 contexts is paired with frozen family-specific
Stage 1 route proposals and ranked using a validation-selected route/condition
score fusion. This makes their final `full-system Top-k accuracy` directly comparable with the
mainline without leaking reference reactants into the condition model.

For every method, the product is the only molecular condition-model input.
Family-specific training and evaluation select the relevant expert artifacts
outside the model; no reaction-family or reaction-type label is concatenated
to a fingerprint, graph embedding, or ranking feature vector.

The current ProSys mainline is:

```text
target product
  -> family-specific EditRetro route proposals
  -> KNN wide recall + ReaFNN condition-pool selection
  -> tabular XGB-LTR full-system reranking
  -> fixed R-GNN temperature prediction
```

## Shared Protocol

- Family splits, label normalization, canonical matching, and the fixed test
  manifest are shared across comparisons.
- The denominator contains `3,860` product identities. The `27` identities
  without a Stage 1 route remain zero-valued end-to-end failures.
- `full-system Top-k accuracy` requires one ranked candidate to jointly match the canonical
  route, complete reagent set, and complete solvent set.
- All candidate budgets and route/condition fusion choices are selected on
  validation data only; test labels are used only for final evaluation.
- Product-Bernoulli Naive Bayes and Product-GNN do not predict temperature, so
  temperature is not reported for them.

## Current Multi-Seed Comparison

B2 Product-GNN, B3 EditRetro + Sequential FNN, and B4 EditRetro +
Reaction-GCNN were independently retrained at seeds `0`, `1`, and `2` using
the current fixed Stage 1 route caches. B1 Product-Bernoulli Naive Bayes is a
deterministic closed-form model, so it is reported once rather than
pseudo-replicated. Rates are equal-family macro averages and `+/-` is sample
standard deviation across the three seeds.

| ID | Method | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 1 | Product-Bernoulli Naive Bayes | 31.68 | 9.10 | 14.87 | 17.45 | 21.48 | 13.15 | 14.56 |
| Baseline 2 | Product-GNN | 38.33 +/- 0.28 | 6.11 +/- 0.49 | 12.84 +/- 0.56 | 16.75 +/- 0.43 | 23.03 +/- 0.73 | 11.40 +/- 0.54 | 13.32 +/- 0.58 |
| Baseline 3 | EditRetro + Sequential FNN | 45.85 +/- 0.26 | 16.75 +/- 0.27 | 24.32 +/- 0.28 | 27.68 +/- 0.40 | 31.71 +/- 0.10 | 22.01 +/- 0.22 | 23.64 +/- 0.18 |
| Baseline 4 | EditRetro + Reaction-GCNN | 38.15 +/- 0.24 | 7.43 +/- 0.21 | 13.16 +/- 0.10 | 16.39 +/- 0.34 | 21.10 +/- 0.28 | 11.88 +/- 0.11 | 13.27 +/- 0.10 |
| Mainline | ProSys current three-seed result | 48.52 +/- 0.59 | 28.96 +/- 0.91 | 36.61 +/- 1.05 | 39.53 +/- 1.24 | 42.90 +/- 1.00 | 33.78 +/- 0.96 | 35.13 +/- 0.97 |

The full protocol, family-resolved System@10 results, conditional temperature
metrics, and compact artifact inventory are in
[`multiseed_baseline_results_20260810.md`](multiseed_baseline_results_20260810.md).

## Archived Point Comparison

The baseline table below is a fixed historical point comparison. Its ProSys row is retained only because all rows share that archived evaluation run; current three-seed mainline results are defined by [`CURRENT_RESULTS.md`](../CURRENT_RESULTS.md) and must not be numerically mixed with this table.

| ID | Method | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 1 | Product-Bernoulli Naive Bayes | 31.68 | 9.10 | 14.87 | 17.45 | 21.48 | 13.15 | 14.56 |
| Baseline 2 | Product-GNN | 38.03 | 6.55 | 12.90 | 16.77 | 23.05 | 11.63 | 13.52 |
| Baseline 3 | EditRetro + Sequential FNN | 45.99 | 17.06 | 24.64 | 27.98 | 31.67 | 22.27 | 23.84 |
| Baseline 4 | EditRetro + Reaction-GCNN | 38.01 | 7.93 | 13.16 | 16.36 | 21.03 | 12.18 | 13.50 |
| Mainline | Archived ProSys point snapshot | 49.18 | 30.11 | 38.04 | 41.13 | 43.91 | 35.03 | 36.33 |

Product-Bernoulli Naive Bayes is the low-capacity conventional-ML control: it uses no neighbor
lookup, route input, learned graph encoder, or route-aware feature. Product-GNN
is the direct product-graph neural comparison. Baselines 3 and 4 are
route-conditioned downstream controls evaluated on the same frozen Stage 1
route cache.

## Artifacts

- Current multi-seed baseline result record:
  [`multiseed_baseline_results_20260810.md`](multiseed_baseline_results_20260810.md)
- Current compact multi-seed artifacts:
  `outputs/baselines/multiseed_20260810/`
- Direct Product-to-Condition results:
  `outputs/baselines/direct_product_condition_nb_20260727/RESULTS.md`
- Product-Bernoulli Naive Bayes per-family predictions, selected fusion weights, model
  artifacts, and compressed Top-10 audit candidates:
  `outputs/baselines/direct_product_condition_nb_20260727/`
- Product-GNN artifacts:
  `outputs/baselines/direct_product_condition_20260727/`
- External downstream baselines:
  `outputs/baselines/non_oracle_external_b23_20260726/`
- Direct-baseline data-flow and split audit:
  [`direct_product_condition_audit_20260727.md`](direct_product_condition_audit_20260727.md)

## Unreported Direct Transformer

The Product-to-System Transformer remains excluded from numerical tables. No
compatible, validated checkpoint has been obtained, and an unavailable model
must never be represented as a zero-valued baseline.
