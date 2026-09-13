# Stage 2: Parallel KNN + ReaFNN Condition Proposals

Updated: 2026-09-13. This document describes only the maintained parallel
implementation. Detailed historical serial/combination experiments are
[archived separately](../Experiment/document_archive_20260913/stage2_detail_before_cleanup.md).

## Purpose and interface

Stage 2 associates each Stage 1 precursor hypothesis with a compact list of
historically observed reagent-solvent contexts. KNN supplies product-similarity
precedent and ReaFNN supplies route-conditioned context evidence. They propose
independently; ReaFNN does not merely edit a KNN shortlist.

Input is a proposed route (`reactants`, `product`) with Stage 1 rank, score and
probability. `sample_index` and `reaction_id` identify and group records, not
learned molecular features. A context is the pair of normalized reagent and
solvent sets; catalysts are included among reagents.

Output rows contain a proposed route, a historical context, KNN/ReaFNN evidence
and an initial fusion score. At most 20 contexts are retained per route and
at most 200 systems for 10 routes. Actual counts may be lower after deduplication
or when fewer routes/contexts are available.

## KNN branch

Implementation: `KNNContextPoolBuilder` in `knn_condition_selector.py`.

The maintained retrieval representation is a radius-2, 4,096-bit Morgan
fingerprint of the product only. Proposed reactants and reaction differences
do not enter the similarity calculation. The memory contains the corresponding
family's training reactions and their observed contexts, not validation/test
condition records. The 64 nearest training routes are retrieved and their
contexts are aggregated into at most 64 proposals. Context evidence includes
similarity sum and maximum, supporting-neighbor count and neighbor-weighted
historical yield. Historical yield is training-memory evidence, not query yield.

Fingerprints are L2-normalized and similarity is their dot product (cosine
similarity), not Tanimoto/Jaccard similarity. The sparse posting implementation
computes the same dot product while avoiding a full dense scan for each query.

Training candidate construction excludes the query's canonical reaction from
its KNN memory lookup. This leave-one-reaction-out safeguard does not make the
whole staged training process out-of-fold.

## ReaFNN branch

Implementation: `ReaFNNSelector` in `reafnn_selector.py`.

| Input block | Width |
| --- | ---: |
| Product Morgan fingerprint | 4,096 |
| Product minus reactant Morgan fingerprint | 4,096 |
| Reactant/product/difference descriptors | 24 |
| Reactant-component count and reactant-string length | 2 |
| Total | 8,218 |

Features are standardized using training-fitted statistics. Two 512-unit
hidden layers use ReLU and dropout 0.10. Separate linear reagent and solvent
heads output multilabel token logits; sigmoid probabilities are used when
scoring contexts. Each target set is multi-hot over the family training
vocabulary. Training uses positive-weighted binary cross-entropy for both
heads. Retained settings include learning rate 0.001, weight decay 0.00001,
batch size 64, up to 30 epochs and validation patience 8.

The network scores the entire train-only context library independently of KNN.
`_compose_context_score` combines reagent/solvent token evidence, token-frequency
prior (weight 0.05), historical-context bonus (0.40), context-support contribution
(weight 0.15) and additional-token size penalty (0.02). The historical bonus
is common to historical candidates; it is not evidence that one such candidate
is more feasible than another. The top 64 contexts form the ReaFNN proposal list.

Separate token heads do not imply Cartesian generation in this mainline. Their
probabilities score complete contexts already present in the training library.
`from_reafnn_generated` is a legacy proposal-origin field; it does not establish
novelty. `from_reafnn_novel` is zero in the maintained configuration.

## Parallel post-fusion

For each route, take the deduplicated union of the two branch lists. Convert
within-list ranks to priors on a common `[0, 1]` scale; a context absent from a
branch has zero support from that branch. Then use:

```text
s_initial(context | route) = w * A_KNN + (1 - w) * B_ReaFNN
```

These are ranking priors, not calibrated reaction-success probabilities.
Retain the top 20 contexts under the fused score. Each family chooses `w` on
predicted Stage 1 validation routes over `0.0, 0.1, ..., 1.0`, maximizing exact
candidate coverage and resolving ties toward larger KNN weight. Test labels do
not enter the implemented weight-selection procedure.

## Training and evaluation boundary

ReaFNN fitting uses training routes and conditions. Stage 3 training and
validation candidate tables use reference routes from their respective splits;
Stage 2 fusion selection specifically uses predicted validation routes. Test
slates always use persisted Stage 1 predictions. The current procedure uses
separate model training, not joint optimization or explicit wrong-route
negative training. Reference-route training creates a distribution shift
relative to predicted-route inference and should be reported accurately.

## Results and interpretation

The [current three-seed mainline](../CURRENT_RESULTS.md) has candidate recall
`54.26 +/- 0.15%` and full-system Top-10 accuracy `43.77 +/- 0.60%`.
Removing ReaFNN and fusion, then retraining the ranker on KNN-only candidates,
gives `53.39 +/- 0.00%` and `39.86 +/- 2.08%`, respectively. The `+3.91 pp`
system-level effect supports complementary candidate construction. It is not
a pure ordering intervention within an identical pool, and it does not prove
that each branch is universally indispensable.

The new strict ReaFNN-only control has now completed all six families and seeds
0/1/2. It disables KNN similarity lookup, proposals and KNN-derived ranker
evidence, fixes the KNN fusion weight to zero, and retrains XGB-LTR. It reaches
`46.48 +/- 0.30%` candidate recall and `36.24 +/- 0.27%` full-system Top-10.
The full-minus-ReaFNN-only difference is `+7.53 pp`. All 18 retained candidate
files passed hash checks and metric replay. See the
[three-way control table](../Experiment/stage2_reafnn_only_multiseed_20260913/SUMMARY.md).
These branch-removal effects include retraining on changed candidate
distributions; they are not additive effects under a common fixed pool.

The separate seed-0 validation sweep includes ReaFNN-only, KNN-only and fusion.
It remains a model-selection analysis, not the source of the new test control.
See [the current Stage 2 control](../ablation/current_parallel_stage2_ablation_results_20260905.md).

## Entry points

- `scripts/run_stage23_non_oracle_suite.sh`: maintained family suite.
- `scripts/run_verified_mainline.py`: content-bound, fail-closed cache wrapper
  used by the suite; rejects stale tables, model changes and concurrent writers.
- `scripts/predict_product.py`: read-only deployment using saved family bundles;
  no model fitting or validation-weight selection occurs during inference.
- `scripts/run_stage23_mainline_non_oracle.py`: candidate construction and
  downstream training orchestration.
- `knn_condition_selector.py`: family memory, retrieval, proposal union/fusion.
- `reafnn_selector.py`: neural training and historical-library scoring.
- `prosys_shared/`: canonicalization, feature definitions and evaluation.

Legacy experimental branches remain in source for retained experiment
compatibility; their existence does not activate them in the maintained suite.
