# Current Baseline Comparison

Date: `2026-07-27`

## Scope

This is the current, citable baseline comparison for the six-family
target-product-to-system task. It contains four completed baselines:

1. `Product-Bernoulli Naive Bayes Condition Prediction`
2. `Product-GNN Condition Prediction`
3. `EditRetro + Sequential FNN`
4. `EditRetro + Reaction-GCNN`

Product-Naive-Bayes and Product-GNN predict a condition from the target product
alone. They receive no reactant route during condition inference. Their
predicted contexts are then paired with the frozen mainline Stage 1 top-10 route
cache to form an end-to-end system slate. Sequential FNN and Reaction-GCNN
instead receive a predicted route plus product. Thus all four methods are
evaluated on the same final route hypotheses, while the direct pair isolates the
value of product-only condition inference.

## Common Protocol

- Family-specific train/validation/test splits, normalized labels, and
  canonical matching follow the mainline protocol.
- Every method is evaluated on the complete fixed manifest (`n = 3,860`).
  The `27` identities without a Stage 1 candidate remain in the denominator as
  zero hits.
- `System@k` requires a single ranked candidate to jointly match the canonical
  route, complete reagent set, and complete solvent set.
- Candidate/model parameters and route/condition score fusion are selected on
  validation data only.
- The direct models score historical complete training contexts only. They do
  not invent a new reagent-solvent combination and do not contain a temperature
  head.
- The direct runner validates Stage 1 cache identities against the formal split
  and does not consume cache-side gold reactants before evaluation.

Primary baseline artifacts:

- `outputs/baselines/direct_product_condition_nb_20260727/RESULTS.md`
- `outputs/baselines/direct_product_condition_nb_20260727/summary.csv`
- `outputs/baselines/direct_product_condition_20260727/product_gnn/`
- `outputs/baselines/non_oracle_external_b23_20260726/RESULTS.md`
- `outputs/baselines/non_oracle_external_b23_20260726/summary.csv`
- `baseline/evaluation_audit_20260726.md`
- `baseline/direct_product_condition_audit_20260727.md`

Implementation details for the direct models are in
[`product_condition_baselines_detail.md`](product_condition_baselines_detail.md).

## Direct Product-to-Condition Models

`Product-Bernoulli-Naive-Bayes` uses a radius-2, 4,096-bit binary Morgan
fingerprint of the product and fits independent multi-label Bernoulli Naive
Bayes heads for reagent and solvent tokens from training records only. It has
Laplace smoothing `alpha=1.0`, no neighbor lookup, no hidden layer, and no
route-derived feature. Token log-probabilities score the same top-20 historical
training-context library as Product-GNN. `Product-GNN` is trained separately
per family from a product molecular graph with a 128-hidden three-step
message-passing encoder and two multi-label heads. Their route/condition fusion
weight is selected on validation from `{0, 0.25, 0.5, 1.0, 1.5, 2.0}` by
`Sys@10`, then `Sys@1`.

## Macro-Average Comparison

All ranking percentages are equal-family macro averages.

| Method | Cover | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Product-Bernoulli Naive Bayes Condition Prediction | 31.68 | 9.10 | 14.87 | 17.45 | 21.48 | 13.15 | 14.56 |
| Product-GNN Condition Prediction | 38.03 | 6.55 | 12.90 | 16.77 | 23.05 | 11.63 | 13.52 |
| EditRetro + Sequential FNN | 45.99 | 17.06 | 24.64 | 27.98 | 31.67 | 22.27 | 23.84 |
| EditRetro + Reaction-GCNN | 38.01 | 7.93 | 13.16 | 16.36 | 21.03 | 12.18 | 13.50 |
| ProSys current mainline | 49.18 | 27.64 | 35.55 | 38.47 | 42.68 | 32.66 | 34.12 |

The mainline is higher than Product-Bernoulli-Naive-Bayes by `+18.54 pp` at
Sys@1 and `+21.20 pp` at Sys@10. It is also higher than Product-GNN by `+19.63
pp` at Sys@10. These direct baselines deliberately receive neither a predicted
reactant route nor a reaction-aware learned representation.

## Family-Resolved Sys@10

| Family | Product-NB | Product-GNN | Sequential FNN | Reaction-GCNN | Mainline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 8.51 | 10.64 | 23.83 | 14.47 | 28.51 |
| Buchwald-Hartwig | 21.75 | 13.56 | 19.93 | 14.10 | 46.86 |
| Chan-Lam | 39.49 | 41.03 | 56.15 | 35.38 | 61.79 |
| Diels-Alder | 12.73 | 16.93 | 20.73 | 13.12 | 27.30 |
| Friedel-Crafts Acyl. | 35.16 | 31.58 | 43.37 | 32.63 | 45.89 |
| Friedel-Crafts Alkyl. | 11.23 | 24.58 | 26.03 | 16.46 | 45.72 |
| Macro average | 21.48 | 23.05 | 31.67 | 21.03 | 42.68 |

## Product-Only Split Caveat

The data split is grouped by canonical reaction, which gives zero canonical
reaction overlap across train/validation/test in the project audit. It is not
product-disjoint: a product can occur with a different route in both train and
test. Train/test product overlap is `51/231`, `91/1,085`, `27/387`, `20/757`,
`38/473`, and `50/888` unique test products for Beckmann, Buchwald-Hartwig,
Chan-Lam, Diels-Alder, Friedel-Crafts Acylation, and Friedel-Crafts Alkylation
respectively. This is valid under the shared reaction-level split and must be
disclosed for every product-only method.

Unlike a retrieval model, Product-Naive-Bayes cannot look up an individual
neighbor product, its context count, or its complete condition record. It only
uses aggregate train-split estimates of fingerprint-bit likelihoods for each
reagent/solvent token, then maps independent token scores into the fixed
training-context library. This lower-capacity design is the formal conventional
ML control in the main table.

## Temperature Comparison

| Method | Conditional support | MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | ---: | ---: | ---: | ---: | ---: |
| EditRetro + Sequential FNN | 1,464 | 11.87 | 49.52 | 67.25 | 81.58 |
| ProSys current mainline | 1,603 | 11.28 | 37.96 | 61.60 | 82.98 |
| EditRetro + Reaction-GCNN | N/A | N/A | N/A | N/A | N/A |

The supports are model-dependent because temperature is assessed only after an
exact system match. This table therefore supports a conditional regression
comparison, not a claim that one method is uniformly more accurate over all
test products.

## Interpretation

The staged mainline substantially exceeds the two canonical direct
product-to-condition baselines and both route-conditioned external baselines.
The result supports the benefit of route-aware candidate construction and
system-level reranking over a product-only, low-capacity conventional ML
control. All comparisons remain single-run results; repeated seeds and a
product-disjoint stress test remain appropriate follow-up analyses.
