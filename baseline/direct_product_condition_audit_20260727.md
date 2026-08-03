# Direct Product-to-Condition Baseline Audit

Date: `2026-07-27`

## Scope

This audit covers the six completed canonical `product_naive_bayes` runs in
`outputs/baselines/direct_product_condition_nb_20260727/` and the six completed
canonical `product_gnn` runs in
`outputs/baselines/direct_product_condition_20260727/product_gnn/`.

## Data-Flow Checks

| Check | Status | Evidence |
| --- | --- | --- |
| Family train-only condition library | Pass | Product-Naive-Bayes and Product-GNN construct their context/token stores from `train` rows only. |
| Product-Naive-Bayes training isolation | Pass | Six metadata records show `4,189`-`15,469` train records, fixed radius-2/4,096-bit fingerprints, and `alpha=1.0`; no validation or test labels are fitted. |
| Product-GNN validation isolation | Pass | Validation rows are used for early stopping only; no validation rows are added to the vocabulary or context library. |
| Test-label isolation during prediction | Pass | Test labels enter only direct-context and final candidate evaluation after predictions are written. |
| Direct model has no route input | Pass | Product-Naive-Bayes uses product Morgan fingerprints; Product-GNN uses a product graph only. |
| Stage 1 cache identity | Pass | Every run calls `_assert_cache_matches_split` for validation and test. Any mismatch raises an error; all 12 canonical runs completed. |
| Gold route exclusion before evaluation | Pass | Cache parsing consumes only identity/product/predicted-route fields; cache-side gold reactants are not read by the direct model or candidate builder. |
| Validation-only score fusion | Pass | Route weight is selected on validation by Sys@10, then Sys@1, then lower weight, and is frozen for test. |
| Exact system metric | Pass | A hit requires canonical route plus complete normalized reagent and solvent sets in one ranked candidate. |
| Missing Stage 1 route handling | Pass | Manifest identities without a route remain in the denominator with zero end-to-end contribution. |

## Candidate and Artifact Checks

- Direct condition slate: at most 20 complete training contexts per product.
- Joint slate: each direct context is paired with deduplicated frozen Stage 1
  routes; no reference route is inserted.
- Candidate tables used by the shared label function are temporary and deleted
  automatically. No `candidate_work_*` directory remains in the completed
  Naive Bayes root, which occupies 45 MB after completion.
- Retained evidence per family-method run: condition-prediction JSONL, selected
  fusion record, run metadata, compressed validation/test Top-10 candidates,
  and a compressed Naive Bayes parameter artifact or Product-GNN validation-best
  model weights and training history.
- Direct baselines have no temperature head. They must not be assigned a
  temperature score or compared in the temperature table.

## Naive Bayes Metadata Audit

The read-only metadata audit passed for all six formal Naive Bayes runs. Every
run records `condition_model_input=target product only`,
`condition_model_uses_stage1_route=false`, and the fixed configuration
`{n_bits: 4096, radius: 2, alpha: 1.0}`.

| Family | Train records | Reagent tokens | Solvent tokens | Validation-selected route weight | Sys@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 4,189 | 315 | 34 | 1.50 | 8.51 |
| Buchwald-Hartwig | 15,469 | 551 | 50 | 1.00 | 21.75 |
| Chan-Lam | 4,184 | 144 | 22 | 1.00 | 39.49 |
| Diels-Alder | 6,648 | 257 | 54 | 1.00 | 12.73 |
| Friedel-Crafts Acyl. | 5,024 | 173 | 41 | 0.25 | 35.16 |
| Friedel-Crafts Alkyl. | 11,293 | 693 | 55 | 1.00 | 11.23 |

## Split Interpretation

The project-level canonical-reaction split audit reports zero train/validation,
train/test, and validation/test canonical reaction overlap. The split is not
product-disjoint, which is important for a target-product-only baseline:

| Family | Train unique products | Test unique products | Shared products |
| --- | ---: | ---: | ---: |
| Beckmann | 1,573 | 231 | 51 |
| Buchwald-Hartwig | 8,276 | 1,085 | 91 |
| Chan-Lam | 2,963 | 387 | 27 |
| Diels-Alder | 5,972 | 757 | 20 |
| Friedel-Crafts Acyl. | 3,616 | 473 | 38 |
| Friedel-Crafts Alkyl. | 6,891 | 888 | 50 |

Shared products can correspond to different canonical routes. This is not a
violation of the official reaction-level split. It can favor a product-only
method in principle, so it is disclosed and motivates a future product-disjoint
stress test.

## Result Sanity Check

| Method | Macro Cover | Macro Sys@1 | Macro Sys@10 |
| --- | ---: | ---: | ---: |
| Product-Bernoulli Naive Bayes | 31.68 | 9.10 | 21.48 |
| Product-GNN | 38.03 | 6.55 | 23.05 |
| Current mainline | 49.18 | 27.64 | 42.68 |

The canonical Naive Bayes and Product-GNN controls are both well below the
mainline at every reported macro cutoff.

## Remaining Limits

- The direct condition models rank historical full contexts only and cannot demonstrate
  de novo reagent-solvent combination generation.
- Results use one random seed. The Product-GNN and other stochastic baselines
  need repeated-seed runs before significance claims.
- This audit checks implementation/data flow. It does not establish chemical
  plausibility of unrecorded but valid alternative conditions under strict
  exact-match evaluation.
