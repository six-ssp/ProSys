# ProSys Paper Terminology Contract

Updated: `2026-09-25`
Source aligned: accepted filtered-USPTO-50K mainline and current paper terminology.

Official paper title: *ProSys: A Product-to-System Framework for
Target-Product-Driven Reaction-System Recommendation*.

This file defines the reader-facing terminology for ProSys. Existing machine
keys in CSV, JSON, and checkpoint metadata are intentionally retained so that
historical outputs remain reproducible.

## Input Boundary

The target product is the only molecular query. Reaction-family labels select
separately trained expert artifacts, train-only memories, and label
vocabularies outside the feature encoders; they are not concatenated to any
molecular, route, ReaFNN, or XGB-LTR feature vector.

## Canonical Module Names

| Project path or internal name | Reader-facing name | Role |
| --- | --- | --- |
| `stage1_retrosynthesis/`, EditRetro | Stage 1: family-tuned EditRetro route generation | Generate ranked precursor routes from the target product. |
| `stage2_ReaFNN/`, KNN + ReaFNN | Stage 2: condition candidate construction | Parallel product-Morgan retrieval and route-conditioned neural proposals, with validation-selected fusion of historical contexts. |
| `stage3_XGBoost/`, XGBRanker | XGB-LTR system reranking | Rank the Stage 2 candidates using 52 non-graph fields; it never adds candidates. |
| Separate XGBoost regressor | Temperature regression | Regress temperature from 52 tabular fields plus 128 R-GNN features; not XGB-LTR. |
| `Reaction-GNN` | R-GNN (reaction graph neural network) | A temperature-only representation with no inclusion gate; never changes system ranking. |

## Canonical Metrics

| Internal key or legacy display name | Public name | Definition |
| --- | --- | --- |
| `route_recall_top{k}`, `rr@k` | `Route@k` | The canonical reference route occurs among the Stage 1 top-k proposals. |
| `pool_route_coverage` | Route recall | A candidate slate contains the canonical route. |
| `pool_context_coverage` | Condition recall | A candidate slate contains the normalized reference reagent-solvent context. |
| `pool_coverage`, `cover`, `Cover` | Candidate recall | A candidate jointly matches the canonical route and complete normalized reagent-solvent context. |
| `system_top{k}_all`, `sys@k`, `System@k`, `Sys@k` | Full-system Top-k accuracy | An exact full-system candidate is ranked within the final top-k list. |
| `system_mrr` | `MRR` | Reciprocal rank of the first exact full-system match, averaged over the fixed manifest. |
| `system_ndcg10` | `nDCG@10` | Binary exact-system nDCG at rank 10. Graded 3/2/1/0 relevance is used only to train XGB-LTR. |
| `temperature.mae`, `temp_mae` | `MAE (deg C)` | Conditional temperature error on the highest-ranked exact-system match with a valid reference temperature. |
| `temperature.rmse`, `temp_rmse` | `RMSE (deg C)` | Conditional root mean squared temperature error on the same support. |
| `within_5c`, `within_10c`, `within_20c` | `Within +/-5/10/20 deg C` | Conditional temperature tolerance rates on the same support. |

`N_temp` is the number of temperature-evaluable exact-system recoveries. It is
not the full test-manifest size and must always be reported with conditional
temperature metrics.

## Canonical Baseline Names

| Internal method key or legacy label | Paper-facing name |
| --- | --- |
| `product_naive_bayes`, `P-BNB` | Product-Bernoulli Naive Bayes |
| `product_gnn`, `P-GNN` | Product-GNN |
| `sequential_fnn`, `Seq-MLP` | EditRetro + Sequential FNN |
| `reaction_gcnn`, `R-GCN` | EditRetro + Reaction-GCNN |

## Reader-facing Rule

Use `candidate recall` and `full-system Top-k accuracy` in new prose. The compact
paper abbreviation `FS@k` and repository display `Sys@k` denote exactly the same
metric, not different matching rules. An exact hit requires route, reagent set
and solvent set simultaneously; temperature is evaluated separately. Retain
machine-readable field names for compatibility. Expand baseline names after B1-B4.

## Compatibility Rule

Do not rename stored data columns or model artifact keys solely for reporting
purposes. New reports should use the public names above and may parenthetically
state the internal key where reproducibility requires it.
