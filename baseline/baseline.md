# ProSys Baseline Experiments

Updated: `2026-09-25`

**Scientific status:** the table below uses the fresh filtered-USPTO-50K base,
repaired expert inputs and predeclared expert seed 1. All six families passed
the new retained-evidence replay. Historical FULL comparisons are not these
results; see [the repair record](../Experiment/stage1_split_repair_20260915/README.md).
Exact-reaction split checks do not establish unseen-product or prospective validity.

The historical comparison is maintained in
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
- The denominator contains `3,860` original query identities. The `66` identities
  without a Stage 1 route remain zero-valued end-to-end failures.
- `full-system Top-k accuracy` requires one ranked candidate to jointly match the canonical
  route, complete reagent set, and complete solvent set.
- All candidate budgets and route/condition fusion choices are selected on
  validation data only; test labels are used only for final evaluation.
- Product-Bernoulli Naive Bayes and Product-GNN do not predict temperature, so
  temperature is not reported for them.

## Corrected-Rerun Safeguards

Use new input and result directories after replacing Stage 1 routes. The
maintained `baseline.run_multiseed_baselines` entrypoint now requires B3/B4
exports from `baseline.external_adapters.build_datasets` with content manifests.
Both commands must receive the same explicit `--route-root` and
`--validation-route-root`; export manifests bind splits, route bytes, checkpoint
bytes and source code. Historical packages without this evidence are not
silently admitted.

B1 is evaluated once inside the current study using the same routes, rather
than loaded from a fixed historical directory. B2/B3/B4 retain seed repeats.
`--resume` accepts only identical study inputs/settings and content-verified
completed runs. Partial or legacy unverified runs require a new output root;
their files are not deleted automatically. A writer lock prevents concurrent
modification of one study. These checks prevent stale-result reuse; they do not
replace Stage 1 split/model eligibility checks or constitute new test results.

Before pruning B3/B4 working directories, the runner now retains manifests and
losslessly compressed full validation/test predictions, validation candidate
tables, and raw/labeled test candidate tables. It verifies decompressed bytes
against the source before deletion. This preserves later metric-replay evidence
without retaining every model or uncompressed intermediate file; the historical
JSON-only archives are not retroactively upgraded.

## Current Multi-Seed Comparison

B2 Product-GNN, B3 EditRetro + Sequential FNN, and B4 EditRetro +
Reaction-GCNN were independently retrained at seeds `0`, `1`, and `2` using
the current fixed Stage 1 route caches. B1 Product-Bernoulli Naive Bayes is a
deterministic closed-form model, so it is reported once rather than
pseudo-replicated. Rates are equal-family macro averages and `+/-` is sample
standard deviation across the three seeds.

| ID | Method | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 1 | Product-Bernoulli Naive Bayes | 27.82 | 7.88 | 13.05 | 15.35 | 19.44 | 11.60 | 12.92 |
| Baseline 2 | Product-GNN | 33.00 +/- 0.98 | 5.06 +/- 0.51 | 10.11 +/- 0.74 | 13.35 +/- 1.10 | 18.85 +/- 0.89 | 9.28 +/- 0.64 | 10.79 +/- 0.70 |
| Baseline 3 | EditRetro + Sequential FNN | 40.09 +/- 0.19 | 13.80 +/- 0.59 | 20.11 +/- 0.50 | 22.76 +/- 0.53 | 26.16 +/- 0.61 | 18.21 +/- 0.50 | 19.41 +/- 0.48 |
| Baseline 4 | EditRetro + Reaction-GCNN | 33.11 +/- 0.46 | 5.31 +/- 0.17 | 10.09 +/- 0.37 | 12.96 +/- 0.76 | 17.84 +/- 0.42 | 9.22 +/- 0.27 | 10.60 +/- 0.39 |
| Mainline | ProSys current parallel three-seed result | 47.57 +/- 0.08 | 19.81 +/- 0.30 | 29.74 +/- 0.51 | 33.33 +/- 0.17 | 37.87 +/- 0.27 | 26.06 +/- 0.15 | 27.74 +/- 0.14 |

The full protocol, family-resolved full-system Top-10 results, conditional temperature
metrics, and compact artifact inventory are in
[the verified 50K comparison](../Experiment/50k_verified_comparisons_20260924/full/RESULTS.md).
The unrounded difference from strongest baseline B3 is 11.711804 percentage
points. B3 temperature MAE is 12.28 +/- 0.19 C versus ProSys 11.32 +/- 0.33 C,
but B3 has higher within-10 accuracy (66.14% versus 62.27%) on different
eligible queries. This is not a paired baseline-temperature comparison.

Product-Bernoulli Naive Bayes is the low-capacity conventional-ML control: it uses no neighbor
lookup, route input, learned graph encoder, or route-aware feature. Product-GNN
is the direct product-graph neural comparison. Baselines 3 and 4 are
route-conditioned downstream controls evaluated on the same frozen Stage 1
route cache.

## Artifacts

- Current multi-seed baseline result record:
  [`verified 50K comparison`](../Experiment/50k_verified_comparisons_20260924/full/RESULTS.md)
- Current compact multi-seed artifacts:
  [`baseline_50k_multiseed_20260924`](../Experiment/baseline_50k_multiseed_20260924/)
- Current direct-condition metric replay:
  [`paper_auxiliary_50k_20260924`](../Experiment/paper_auxiliary_50k_20260924/all_families/)
- Historical direct-baseline data-flow and split audit:
  [`direct_product_condition_audit_20260727.md`](direct_product_condition_audit_20260727.md)

Large historical output directories were cleaned for storage. Use retained
compact records and result documents, not removed output paths, for evidence.
Older point snapshots are historical records and do not define the current
mainline comparison.
