# Direct Product-to-Condition Baselines

## Purpose

This document specifies the two canonical direct baselines implemented by
[`run_direct_product_condition_baselines.py`](run_direct_product_condition_baselines.py):

1. `product_naive_bayes`: Product-Bernoulli Naive Bayes (product Bernoulli naive Bayes).
2. `product_gnn`: Product-GNN (product graph neural network).

The canonical pair answers a deliberately different question from the ProSys Stage 2/3
pipeline: can a model infer a useful reaction-condition context from the target
product alone, without receiving a proposed reactant route? The resulting
contexts are paired with frozen Stage 1 route proposals only after the direct
condition model has made its predictions.

## Fixed Data Protocol

- Scope: six maintained Reaxys families and the existing family-specific
  train/validation/test splits.
- Training rows: only the selected family's condition-training split.
- Validation rows: used only for early stopping of Product-GNN and selection
  of route/condition fusion weight.
- Test rows: used only to construct final labels and metrics after prediction.
- Stage 1 source: frozen `outputs/stage1_routes/<family>/route_cache.json`
  for test and `outputs/stage1_routes_validation/<family>/route_cache.json`
  for validation.
- Cache fields consumed before evaluation: `sample_index`, `reaction_id`,
  `product`, and predicted route fields (`reactants`, `retro_rank`,
  `retro_score`, `retro_probability`). The runner intentionally does not read
  cache-side gold reactants or gold conditions.

The runner verifies that every route-cache identity exactly matches the formal
split manifest before any model is trained or evaluated. A missing Stage 1
route remains a zero-valued end-to-end result instead of being dropped.

## Shared Condition Representation

A complete condition is the normalized pair:

```text
context = (reagent_norm, solvent_norm)
```

The candidate context library is built from training rows only. It contains
complete historical reagent-solvent pairs, tokenized by semicolon-separated
normalized labels. Both canonical direct baselines rank this same library;
therefore they do not invent a new complete reagent-solvent combination. This
is an intentional closed-context baseline, not a generative condition model.

## Product-Bernoulli Naive Bayes: Product Bernoulli Naive Bayes

### Input and supervision

For each raw training condition record, the model encodes the canonical target
product as an RDKit Morgan bit fingerprint with radius `2`, `4096` bits, and
chirality enabled. This gives a fixed binary vector `x` in `{0, 1}^4096`.
The record's normalized reagent tokens and solvent tokens are retained as two
independent multi-label targets. If a product appears in several training
records, it remains several examples; labels are never unioned across those
records.

The model has one independent Bernoulli Naive Bayes binary classifier per
reagent token and per solvent token. It has no neighbor database, hidden layer,
graph encoder, reactant input, route score, or Stage 1 feature.

### Fitting rule

For one token `t`, with binary presence target `y_t`, the model estimates the
class prior and each fingerprint-bit conditional probability from the training
split only. With Laplace smoothing `alpha=1.0`:

```text
P(y_t = 1) = (n_t_positive + alpha) / (N + 2 * alpha)
P(x_i = 1 | y_t = c) = (n_i,c + alpha) / (n_c + 2 * alpha)
```

The test-time token logit is the Bernoulli Naive Bayes log-posterior odds:

```text
logit_t(x) = log P(y_t=1, x) - log P(y_t=0, x)
```

The implementation evaluates all token heads by matrix operations over the
binary fingerprint matrix. Smoothing keeps rare tokens and absent fingerprint
bits finite. There is no iterative optimizer or validation-based parameter
fitting; the validation split is used only for the shared route/condition score
fusion described below.

### Mapping token logits to complete contexts

For a historical training context `c`, let `T(c)` be its reagent and solvent
tokens. Its score is:

```text
score(c) = mean over t in T(c) [ log_sigmoid(logit_t(x)) ]
         + 0.05 * normalized_log_frequency(c)
```

The top `20` complete contexts are emitted. This mapping makes Naive Bayes and
Product-GNN comparable: both predict marginal reagent/solvent token evidence,
then rank the same train-only complete-context library.

## Product-GNN: Product Graph Neural Network

### Input and network

Product-GNN receives only one product molecular graph. Nodes use the shared
atom features already used by the project graph encoder, including element,
degree, formal charge, aromatic/ring flags, hydrogen count, hybridization, and
mass. Bonds are represented as bidirectional graph edges.

The family-specific network is:

```text
product graph
  -> GraphEncoder(hidden=128, message-passing steps=3, dropout=0.10)
  -> Linear(128, 128) + ReLU + Dropout
  -> reagent multi-label linear head
  -> solvent multi-label linear head
```

There is no reactant graph, route embedding, route score, or Stage 1 feature
in this network.

### Supervision and optimization

One supervised example is retained per training condition record. If the same
product occurs in multiple training reactions, it appears multiple times with
the condition labels of each record; labels are not unioned across records.
This trains marginal reagent and solvent token probabilities without treating
all historically observed conditions as simultaneous positives.

- Loss: sum of reagent and solvent `BCEWithLogitsLoss` terms.
- Class weighting: per-token negative/positive ratio, clipped to `[1, 20]`.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-5`.
- Batch size: `64`; maximum epochs: `50`; patience: `7` validation epochs.
- Formal robustness run: independently retrained at seeds `0`, `1`, and `2`; each seed uses its validation-best state for prediction.
- Performance implementation: product graphs are parsed once per unique
  train/validation product and cached in memory. This changes no model input
  or numerical objective.

### Mapping heads to complete contexts

For every train-library context, the GNN scores its reagent and solvent tokens
with mean log-sigmoid probability, then adds the same `0.05` normalized
log-frequency prior used for deterministic tie support. The top `20` scored
complete contexts are emitted. The heads can score novel token pairings
internally, but final candidates are restricted to historical full contexts so
that the candidate budget and evaluation semantics match Product-Naive-Bayes.

## From Direct Conditions to Systems

For each validation or test product:

1. Predict up to 20 direct condition contexts before reading any predicted
   route.
2. Deduplicate the frozen Stage 1 route list by canonical reactant side, then
   pair every remaining route with every predicted context.
3. Rank the joint candidates with a per-product score:

```text
system_score = w_route * zscore(retro_score) + 1.0 * zscore(condition_score)
             + deterministic rank tie-break
```

4. Select `w_route` from `{0, 0.25, 0.5, 1.0, 1.5, 2.0}` on the validation
   set by maximum `full-system Top-10 accuracy`, then `full-system Top-1 accuracy`, then smaller route weight.
5. Apply the selected weight unchanged to the test set.

The uncompressed candidate tables are materialized only in an automatically
deleted temporary directory because the shared label function accepts CSV
input. Each result directory retains condition predictions, the fusion record,
model metadata, and compressed Top-10 candidates for audit.

## Metrics

- `Condition@k`: direct complete-context hit among the model's top-k contexts.
- `candidate recall`: a joint route-context candidate exists that exactly matches the
  route, reagent set, and solvent set.
- `full-system Top-k accuracy`: such an exact joint candidate appears within the final top-k ranked
  systems.
- `MRR` and `nDCG@10`: computed on the same fixed full test manifest.

All system metrics retain products with no Stage 1 route in the denominator.
The direct models do not contain a temperature head, so they have no
temperature metric.

## Formal Results

The table below is the current formal system-level comparison. Percentages are
equal-family macro averages; `+/-` is sample standard deviation across three
independent Product-GNN retrainings.

| Method | Replication | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Product-Bernoulli Naive Bayes | deterministic | 31.68 | 9.10 | 14.87 | 17.45 | 21.48 |
| Product-GNN | 3 seeds | 38.33 +/- 0.28 | 6.11 +/- 0.49 | 12.84 +/- 0.56 | 16.75 +/- 0.43 | 23.03 +/- 0.73 |

The archived Product-GNN `Condition@1=15.25` and `Condition@10=48.76` values
are seed-0 auxiliary diagnostics, not a multi-seed aggregate. The formal
multi-seed comparison is maintained in
[`multiseed_baseline_results_20260810.md`](multiseed_baseline_results_20260810.md).
Family-level compact artifacts are in `outputs/baselines/multiseed_20260810/`.


## Interpretation and Limits

- Product-Bernoulli Naive Bayes is the formal conventional-ML control. It is
  intentionally low capacity: aggregate fingerprint-bit likelihoods replace
  both nearest-neighbor precedent retrieval and a learned molecular encoder.
- Product-GNN is a lightweight, independently trained neural baseline rather
  than a claim of a fully optimized graph model.
- The formal Product-Bernoulli Naive Bayes and Product-GNN Sys@10 values are `21.48%` and
  `23.03 +/- 0.73%`, compared with `43.77 +/- 0.60%` for the current parallel staged mainline.
- The official split is grouped by canonical reaction, not made product
  disjoint. Train/test product overlap is therefore possible for distinct
  routes; this is not reaction leakage and must be disclosed.
- All direct condition models are restricted to historical complete contexts and cannot test
  the value of de novo condition combination generation.
- Product-GNN is reported over three independent seeds; B1 remains a single
  deterministic reference because its closed-form fit has no random initialization.
