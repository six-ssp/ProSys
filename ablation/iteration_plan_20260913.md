# Evidence-first iteration plan

Frozen before the new test runs on 2026-09-13. The promoted parallel mainline
remains unchanged; new controls are recorded separately regardless of outcome.

## First experiment: remove KNN

The completed three-seed KNN-only control tests removal of ReaFNN. Its
complement, a strict ReaFNN-only test control, is missing. This experiment
removes both KNN proposals and KNN evidence from the downstream ranker, retains
the same neural input/architecture/training settings and historical context
library, fixes KNN fusion weight to zero, retains 20 contexts per route, and
retrains the 52-column XGB-LTR. KNN-related columns remain zero for schema
compatibility. No temperature model is trained because it cannot affect Sys@k.

Six families, seeds 0/1/2, fixed data splits and Stage 1 route caches. Report
all families and seeds, even if the control outperforms the mainline. Compare
candidate recall, full-system Top-1/3/5/10, MRR and nDCG@10. The intervention
changes both proposal composition and evidence available to a retrained ranker;
it is not a pure fixed-pool score ablation.

Implementation uses an explicitly isolated subclass in the experiment child
process, leaving maintained model defaults unchanged. Tests assert no KNN
similarity lookup and zero KNN-derived candidate fields. Each completed family
retains source/input SHA256, sorted candidate-identity hash, a compressed
per-candidate audit CSV and compact metrics before its own scratch is deleted.
Scratch and temporary files must live on the data disk; stop below 5 GiB free.

## Next evidence checks

- New exact-pair controls should retain candidate membership, ranked identities
  and temperature-support identities, rather than inferring them from aggregate
  equality. Historical compact results cannot retroactively establish these.
- Reconstruct a full-mainline family with retained per-query evidence before
  attempting paired bootstrap or shared-support temperature comparisons.
- Evaluate seen/unseen-product subgroups with the same frozen model; do not
  describe the original reaction-group split as product-disjoint.
- Architecture changes, if attempted later, use training/validation selection
  and a predeclared trial budget. They are exploratory until independently
  evaluated and must not replace the paper mainline on test-score preference.

This plan does not promise higher scores. Its purpose is to distinguish a
useful component from a favorable but incompletely controlled comparison.
