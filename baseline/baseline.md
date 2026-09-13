# ProSys Baseline Experiments

Updated: `2026-09-13`

The current citable comparison is maintained in
[`multiseed_baseline_results_20260810.md`](multiseed_baseline_results_20260810.md).
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

For every method, the product is the only external molecular query. B1 and B2
use only the product in their condition models; B3 and B4 also receive the
predicted precursor route from Stage 1.
Family-specific training and evaluation select the relevant expert artifacts
outside the model; no reaction-family or reaction-type label is concatenated
to a fingerprint, graph embedding, or ranking feature vector.

The current ProSys mainline is:

```text
target product
  -> family-specific EditRetro route proposals
  -> parallel product-Morgan KNN + ReaFNN post-fusion condition-pool selection
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
| Mainline | ProSys current parallel three-seed result | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |

The full protocol, family-resolved System@10 results, conditional temperature
metrics, and compact artifact inventory are in
[`multiseed_baseline_results_20260810.md`](multiseed_baseline_results_20260810.md).

Product-Bernoulli Naive Bayes is the low-capacity conventional-ML control: it uses no neighbor
lookup, route input, learned graph encoder, or route-aware feature. Product-GNN
is the direct product-graph neural comparison. Baselines 3 and 4 are
route-conditioned downstream controls evaluated on the same frozen Stage 1
route cache.

## Artifacts

- Current multi-seed baseline result record:
  [`multiseed_baseline_results_20260810.md`](multiseed_baseline_results_20260810.md)
- Current compact multi-seed artifacts:
  [`results/multiseed_20260810/`](results/multiseed_20260810/)
- Direct-baseline data-flow and split audit:
  [`direct_product_condition_audit_20260727.md`](direct_product_condition_audit_20260727.md)

Large historical output directories were cleaned for storage. Use retained
compact records and result documents, not removed output paths, for evidence.
Older point snapshots are historical records and do not define the current
mainline comparison.
