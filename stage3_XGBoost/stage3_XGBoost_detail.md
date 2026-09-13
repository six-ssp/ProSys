# Stage 3: System Ranking and Independent Temperature Regression

Updated: 2026-09-13. This is the current parallel-Stage-2 interface.
[Earlier mixed-version details](../Experiment/document_archive_20260913/stage3_detail_before_cleanup.md)
are archived and do not define the current feature inputs or results.

### Numerical reproducibility update (pending result promotion)

New maintained fits explicitly use `ReactionGNNConfig(deterministic=True)`.
The fit and embedding-inference scopes enable deterministic PyTorch algorithms
and restore the caller's previous runtime settings on exit. CUDA uses a fixed
cuBLAS workspace configuration. No architecture, input dimension, loss or
temperature gate was added. Cache keys now include the complete graph config.
Historical payloads without this field retain `deterministic=False`; loading
them does not silently relabel old training. A four-fit same-seed probe found
different tensors in two default CUDA runs and identical tensors in two
deterministic runs. This is a current-environment result, not a cross-device
guarantee or a complete attribution of historical MAE drift. Final numerical
promotion awaits the new paired checks; see
`Experiment/project_completion_20260913/FINDINGS.md`.

A subsequent full corrected Diels-Alder seed-0 repeat also matches all 125600
retained candidate rows and their ranking/temperature predictions exactly,
including R-GNN tensors and data/route hashes. A cached-product deployment
smoke matches 100 candidate identities and all 52 ranking inputs with zero
prediction difference. These are current-environment, single-family checks;
they do not turn the unfinished expert study into a full-pipeline three-seed
result. Inference rejects model bundles whose feature-source hashes no longer
match the maintained code.

## Two separate XGBoost models

| Branch | Model | Learned input | Target | Changes system order? |
| --- | --- | --- | --- | --- |
| System recommendation | XGBRanker / XGB-LTR | 52 tabular columns | Graded route-context relevance | Yes, with a Stage 1/2 prior |
| Temperature | XGBRegressor | 52 tabular + 128 R-GNN columns | Recorded temperature | No |

Implementation is `xgb_reranker.py`; graph representation learning is in
`reaction_gnn_features.py`. Stage 3 receives Stage 2 route-context rows and creates no
additional candidates. The graph representation never enters the maintained
ranker. There is no family-specific temperature gate or joint training.

## Tabular schema

`TABULAR_FEATURE_COLUMNS` is a fixed allowlist in `xgb_reranker.py`, assembled
from `prosys_shared/constants.py` and Stage 2 evidence fields.

| Block | Width | Meaning |
| --- | ---: | --- |
| Route evidence and size | 5 | EditRetro rank/score/probability, component count, reactant-string length |
| Context composition | 2 | Reagent and solvent counts |
| Product descriptors | 8 | Fixed product descriptor vector |
| Compatibility support fields | 15 | Retained historical schema slots |
| KNN/ReaFNN/context evidence | 22 | Retrieval support, token/context scores, origin and support statistics, legacy cluster slots |
| Total | 52 | Fixed schema, not 52 independent chemical descriptors |

Some retained slots are constant or zero under this protocol. Describe the
active information as route confidence, molecular/context descriptors,
retrieval support and neural context evidence. Do not draw the ranker as taking
raw molecular fingerprints or a sparse one-hot context vector directly.

`sample_index`, `reaction_id`, SMILES strings, reference match labels,
reference yield and reference temperature are excluded from the learned
feature matrix. Initial Stage 2 fusion-score fields are also excluded from
the fixed learned schema; they contribute via the explicit ranking prior.

## Ranking supervision and fitting

All candidate rows belonging to one product-query identity form an LTR group.
The label definition is exact route plus exact context = 3; route only = 2;
context only = 1; neither = 0. Yield does not directly define rank relevance.
Training/validation candidate tables use reference split routes, so the label
definition must not be mistaken for evidence that current training includes
explicit Stage 1 wrong-route negatives.

The retained XGB-LTR settings include `rank:ndcg`, validation `ndcg@10`, up to
300 trees, learning rate 0.05, depth 6, row/column subsampling 0.8, L2 penalty
1.0 and histogram tree construction, with validation early stopping. Neural
models are fitted separately before producing features for XGBoost; this is
not a jointly optimized differentiable Stage 2/3 network or out-of-fold stack.

## Final system score

Raw XGB-LTR scores are standardized within a query's complete candidate slate,
across routes as well as contexts. Final scoring is:

```text
score_final = z_query(score_XGB) + beta * prior_Stage1_Stage2
```

The deterministic prior follows Stage 1 route ordering and Stage 2 condition
ordering, with stable tie-breaking. `beta` is selected from validation data
using full-system Top-10 accuracy, then Top-1 as a tie-breaker. The prior is an
ordering signal, not a calibrated success probability. The no-XGB-LTR control
uses the deterministic Stage 1/2 order alone.

## R-GNN representation

The proposed reactants and product are converted into molecular graphs. The
shared encoder uses atom features and graph connectivity, not a separate
bond-feature message vector. Four message-passing steps are followed by graph
pooling. Reactant, product and difference embeddings are projected into a
128-dimensional route representation. Auxiliary reagent and solvent multilabel
heads supervise graph training using the corresponding family training data.

The representation is route-specific; all contexts on that route share the
same graph embedding. Candidate-specific variation remains in the tabular
features. The auxiliary graph heads do not directly recommend temperature.

## Temperature fitting and evaluation

`TEMPERATURE_FEATURE_COLUMNS` concatenates the fixed 52 columns and
`route_gnn_feat_0` through `route_gnn_feat_127`. A separate XGBoost regressor
fits exact-match rows with finite reference temperatures. Temperature
predictions neither change candidate membership nor enter the ranking score.

Evaluation in `prosys_shared/mainline.py:evaluate_scored_frame` sorts the full
slate and selects the first exact-system row with finite predicted and gold
temperature. It does not restrict selection to Top-10. Queries without an
eligible row are excluded from temperature metrics, not assigned an error of
zero. Report conditional MAE and within +/-5, +/-10 and +/-20 C, with support.
This gold-conditioned evaluation is distinct from deployment, where all
candidates can receive predictions without knowledge of the correct system.

## Current controls and evidence

Full-system Top-10 is `43.77 +/- 0.60%`, versus `36.03 +/- 0.16%` under the
Stage 1/2 deterministic order, with unchanged aggregate candidate recall of
`54.26 +/- 0.15%`. The `+7.74 pp` effect supports system prioritization.
The Top-1 mean gain is small and changes sign across seeds.

Conditional temperature MAE is `11.49 +/- 0.26 C` with R-GNN versus
`13.93 +/- 0.38 C` without it. Within-10-C rates are `63.09 +/- 1.76%` versus
`55.56 +/- 1.23%`. Pooled supports are 1,785/1,795/1,797 for seeds 0/1/2
in both arms. Differences are computed before rounding.

Retained controls agree on configuration and aggregate candidate/support
statistics; temperature pairs also agree on aggregate ranking metrics. The
compact audit does not independently verify candidate or support identities
with per-row hashes. Do not upgrade these checks to an identity-level claim.
Full numbers and source links are in [CURRENT_RESULTS.md](../CURRENT_RESULTS.md).

The [2026-09-13 reconstruction](../Experiment/mainline_evidence_completion_20260913/COMPLETION_REPORT.md)
now supplies new identity-level evidence for all 18 family/seed pairs.
Every Sys@k value reproduces the promoted result, including the same-pool
no-LTR comparison. Reconstructed R-GNN temperature MAE is
`11.82 +/- 0.33 C`, versus `13.93 +/- 0.38 C` on the exact same support without
R-GNN; within-10-C values are `61.58 +/- 0.77%` and `55.56 +/- 1.23%`.
These new paired gains (`2.11 C`, `6.02 pp`) must not be confused with the
historical aggregate-only comparison above. The temperature drift is reported,
not hidden or attributed to an experimentally isolated cause.

## Artifacts

- `xgb_ranker.json` and `xgb_ranker_meta.json`: learned system ranking model
  and feature/configuration metadata.
- `xgb_temperature.json` and `xgb_temperature_meta.json`: separate regressor.
- Current compact mainline and control records are under the dated parallel
  experiment directories linked from `CURRENT_RESULTS.md`.
- Temporary candidate tables and trained models may be cleaned for storage;
  retained metrics do not imply those large per-row files are still present.
