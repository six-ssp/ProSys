# Stage 2/3 Legality and Correctness Audit

**Date:** 2026-08-30
**Scope:** retained six-family Stage 1 route caches, the maintained Stage 2
KNN + ReaFNN implementation, Stage 3 XGB-LTR and temperature branch, and the
retained Stage 2/3 result artifacts.

## Bottom-Line Status

- **Direct test-label leakage:** not detected.
- **Canonical-reaction split isolation:** pass for all six families.
- **Metric implementation:** pass after independent recomputation; one
  historical Chan-Lam MRR differs by `5.94e-7` after a new deterministic
  tie-break rule, while all Top-k and temperature values agree exactly.
- **Strict end-to-end training protocol:** conditional, not yet fully sealed.
  The current runner builds training and validation candidate tables from the
  reference route, while test candidates are built from Stage 1 predicted
  routes. ReaFNN and R-GNN scores/embeddings for the XGBoost training rows are
  also in-sample rather than out-of-fold.

Thus, the retained results are valid for the stated **reaction-disjoint,
fixed-Stage-1** evaluation, but should not be described as product-disjoint
generalization or as an OOF-stacked fully end-to-end training protocol.

## Checks That Passed

### 1. Split isolation

`data_preprocess/audit_data_splits.py --strict` reported zero canonical
reaction intersections for every family:

| Family | Train | Validation | Test | Train/Val | Train/Test | Val/Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 1,876 | 234 | 235 | 0 | 0 | 0 |
| Buchwald-Hartwig | 8,777 | 1,097 | 1,098 | 0 | 0 | 0 |
| Chan-Lam | 3,112 | 389 | 390 | 0 | 0 | 0 |
| Diels-Alder | 6,078 | 760 | 760 | 0 | 0 | 0 |
| Friedel-Crafts Acyl. | 3,797 | 475 | 475 | 0 | 0 | 0 |
| Friedel-Crafts Alkyl. | 7,145 | 893 | 894 | 0 | 0 | 0 |

The Stage 1 validation/test manifests also match the Stage 2 validation/test
splits by identity, and the Stage 1 train/validation reaction set has zero
canonical overlap with the Stage 2 test set.

### 2. Stage 1 cache and inference boundary

All six `outputs/stage1_routes/<family>/route_cache.json` files passed an
identity audit: sequential `sample_index`, matching reaction ID/product test
manifest, at most ten routes, contiguous ranks, unique canonical predicted
routes, and normalized per-slate probabilities.

The cache format contains `gold_reactants` for evaluation convenience, but
`load_route_records_from_cache()` supplies Stage 2 only with
`sample_index`, `reaction_id`, `product`, and the predicted route fields. A
static search found no `gold_reactants` access in the Stage 2 selector or Stage
3 scorer. The only maintained access is the Stage 1 route-recall evaluator.

### 3. Candidate-pool provenance

The retained 2026-08-29 Product-Morgan/ReaFNN snapshot records the following
for every family: zero duplicate route-condition rows, zero non-historical
rows, zero ReaFNN-generated rows, zero novel combinations, and no more than
20 contexts per predicted route. ReaFNN therefore reorders/refines the KNN
wide pool in this production configuration; it does not inject a test-derived
or newly invented condition into the evaluated pool.

The KNN memory is built from the family training split only. A behavior-level
regression test on a duplicated canonical Friedel-Crafts alkylation route
confirmed that all three same-canonical route entries and all six associated
condition records are excluded from leave-one-reaction-out train-table
retrieval.

### 4. Feature and target separation

The XGB-LTR allowlist has 52 non-graph schema columns and the temperature
regressor allowlist has 180 columns (the same 52 plus 128 R-GNN embedding
coordinates). Their intersections with every target/evaluation field
(`label`, `route_match`, `context_match`, `rank_relevance`, `sample_weight`,
`temperature_gold`, and `yield_gold`) are empty. The ranker also excludes all
`route_gnn_feat_*` columns and all bounded `stage2_*` correction fields.

Temperature is evaluated separately: for each product with at least one
full-match candidate and a valid temperature, the first ranked full match
provides one absolute error. It is not added to the system-ranking score.

### 5. Result recomputation

The retained scored tables under `outputs/stage23_mainline/` were independently
re-evaluated against each Stage 1 test manifest. Five families reproduced every
stored metric exactly. For Chan-Lam, Top-k and temperature values reproduced
exactly; MRR changed from `0.38018404063618494` to
`0.3801846348931191` only because the current evaluator now applies an
explicit deterministic tie order. This difference is `5.94e-7` and does not
change any reported percentage at any sensible precision.

## Corrections Applied During This Audit

1. **Canonical leave-one-reaction-out KNN.** The training-table KNN exclusion
   was strengthened from a raw `(reaction_id, reactants, product)` key to the
   canonical `(reactants, product)` identity. This prevents another Reaxys ID
   representing the same reaction from supplying its condition statistics to
   the training query.
2. **Deterministic metric ties.** `evaluate_scored_frame()` now uses a stable,
   explicit tie order after the score: route rank, route probability, Stage 2
   initial score, KNN similarity, reagent label, and solvent label.
3. **Explicit Stage 1 RDKit guard.** The Stage 1 preprocessor now rejects any
   raw reaction with an unparseable reactant or product fragment before
   atom-mapped preprocessing. The one suspicious Diels-Alder raw record
   (`reaction_id=11028191`) had an empty `mapped_reaction_smiles` field and was
   already discarded by the pre-existing blank-mapped-record path, so it did
   not enter existing Stage 1 training artifacts.

All modified Python modules compile and `git diff --check` passes.

The archived three-seed result predates the canonical leave-one-reaction-out
hardening. The change affects training-table construction only, not test-route
or test-candidate provenance; it is therefore not direct test leakage.
Nevertheless, the archived score must be treated as a pre-hardening historical
result rather than the final post-audit headline until it is rerun.

## Required Disclosures and Follow-Up Work

### A. Reaction-disjoint is not product-disjoint

The canonical reaction overlap is zero, but identical products can occur in
different reactions. The exact test-product overlap with the training split is
as follows:

| Family | Test reaction records | Test records with product seen in train |
| --- | ---: | ---: |
| Beckmann | 235 | 53 |
| Buchwald-Hartwig | 1,099 | 97 |
| Chan-Lam | 390 | 28 |
| Diels-Alder | 762 | 22 |
| Friedel-Crafts Acyl. | 475 | 40 |
| Friedel-Crafts Alkyl. | 899 | 51 |
| Total | 3,860 | 291 (7.54%) |

This is not an exact reaction or condition-label leak because the canonical
reactions are split-disjoint. It can, however, make a product-fingerprint KNN
retriever easier for the seen-product subset. The manuscript must call this a
reaction-disjoint split and should report a seen/unseen-product stratification
or a product-grouped sensitivity split if making a strong product-generalization
claim.

### B. Candidate-table distribution shift

The current default `train_table_mode=oracle` constructs train/validation
candidate routes from reference reactants, whereas test candidates originate
from Stage 1 predictions. Labels are never read from the test set during
training or validation selection, so this is not direct test leakage. It is a
teacher-forcing distribution shift and must not be hidden behind an unqualified
"fully end-to-end trained" claim.

The runner already supports a stricter rerun using
`--train_table_mode non_oracle` with Stage 1 train and validation route caches.
That is the appropriate final protocol if the goal is an end-to-end training
claim rather than a fixed-Stage-1 evaluation claim.

### C. In-sample stacked learned features

ReaFNN is trained on the full training split and then scores the same training
rows used to fit XGB-LTR. The R-GNN is likewise trained on the training split
before its embeddings enter the temperature regressor's training rows. No test
label is used, but this is in-sample stacking and can make the downstream
training fit optimistic. A fully rigorous rerun should generate out-of-fold
ReaFNN scores and R-GNN embeddings for XGBoost training, then refit each
upstream model on all train data only for validation/test inference.

### D. The 52-column label is a schema count, not 52 active signals

Across all retained scored tables, 19 of the 52 ranker columns are constants:
the 15 legacy product-memory support fields and four legacy cluster fields.
The current historical-only KNN/ReaFNN path therefore has 33 varying tabular
signals in practice. This does not leak information or change a fitted tree
through a useful split, but manuscript text and figures must describe it as a
fixed 52-slot compatibility schema with 33 varying fields in the maintained
configuration, not as 52 independently extracted descriptors.

### E. Final ranking score must be named accurately

The deployed score is not always pure raw XGB-LTR output. Per family, validation
selects a bounded Stage 2 correction strength and may select a fusion weight
for:

`zscore(XGB-LTR raw score within a slate) + beta * Stage-2 heuristic prior`.

This is valid because `beta` is selected on validation data and test labels are
not consulted. It must nevertheless be disclosed as validation-calibrated
XGB-LTR plus a Stage 2 prior, rather than abbreviated to only "XGB-LTR".

### F. Artifact provenance

The completed 2026-08-09 three-seed archive records route-cache hashes and
hyperparameters, but does not preserve a source-code hash or the KNN retrieval
mode. The newer 2026-08-29 Product-Morgan/ReaFNN result is a seed-0 development
snapshot whose large raw tables/models were intentionally removed; it is not a
replacement for the archived three-seed headline until it is replicated with a
fully recorded manifest. Future runs should write hashes for split files,
route caches, source modules, candidate tables, and environment versions before
pruning raw artifacts.

## Publication Decision

It is scientifically defensible to report the retained results as
reaction-disjoint, fixed-Stage-1 route-cache results, with exact full-system
matching and no detected direct test-label leakage. Before presenting the
current Product-Morgan/ReaFNN configuration as the final mainline result, run
the stricter non-oracle/OOF protocol for three seeds, preserve a complete
provenance manifest, and update the method text using the disclosures above.
