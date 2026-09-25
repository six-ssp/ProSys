# Stage 1 Augmentation Fidelity Repair

## Current Status

Independent copy verification, combined held-out audit, completed-base admission
and full 7,715-query / 77,150-variant CPU guard preflight pass. The 151-query
Buchwald GPU smoke including the CXSMILES case passes. Final Stage 1 software
regression: 83 tests in 3.137 seconds. These checks do not establish completed
model performance.

Completed expert study: `../stage1_50k_fidelity_v2_expert_multiseed_20260924/`.
Completed paired base evaluation: `../stage1_50k_guarded_v2_base_test_20260924/`.
Both use the same final guard source; the two earlier interrupted expert roots
and the stock/initial guarded base decodes must not supply final paper results.
The final Diels-Alder expert seed has a separate capacity-safe recovery admission;
see `../release_50k_20260925/README.md`. Earlier test counts below are phase
snapshots; the final regression is 252 project tests plus eight decoder tests.

## Fixed Policy

The completed scratch-50K base passes the full source-to-augmentation identity
audit: 396,830 training and 49,880 validation pairs, zero unmatched identities.
It is retained without resuming or inheriting any FULL checkpoint.

Expert inputs are copied without changing original raw splits or evaluation
queries. Every train/validation augmented pair absent from the deterministic
same-split mapped raw transformation membership is excluded. This rule uses no
model score or test hit rate. All retained token tensors and vocabulary bytes
stay identical. Stored test tensors are unchanged, including one disclosed
Beckmann augmentation mismatch; actual evaluation re-generates augmentations
from the original product queries with a product-only identity guard.

| Family | Train Before | Train Removed | Train After | Validation Before | Removed | After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 79,280 | 32 | 79,248 | 2,310 | 5 | 2,305 |
| Buchwald-Hartwig | 166,600 | 0 | 166,600 | 11,010 | 0 | 11,010 |
| Chan-Lam | 63,760 | 0 | 63,760 | 3,900 | 0 | 3,900 |
| Diels-Alder | 234,420 | 2 | 234,418 | 8,210 | 0 | 8,210 |
| Friedel-Crafts Acylation | 102,400 | 1 | 102,399 | 4,860 | 0 | 4,860 |
| Friedel-Crafts Alkylation | 199,620 | 0 | 199,620 | 9,090 | 0 | 9,090 |
| Total | 846,080 | 35 | 846,045 | 39,380 | 5 | 39,375 |

These are augmented pair counts, not unique raw reaction counts. The exhaustive
text/tensor copy verifier passes all six datasets. Combined held-out boundary
audit and new completed-base admission must pass before starting fresh expert
seeds 0/1/2. No checkpoint from the interrupted expert queue is resumed.

## Inference Guard

`scripts/build_stage1_guarded_routes.py` reuses the frozen route builder and
stock EditRetro model/beam/aggregation settings. Its explicit interactive
wrapper replaces only product randomization with an identity-preserving guard.
Each slot consumes exactly one RDKit random draw; if stereo-sensitive canonical
identity changes, the original product SMILES fills that slot. No rejection
resampling, beam reduction, query filtering or gold-conditioned choice occurs.
All tokenized strings must reconstruct their pre-tokenization strings exactly.
The guard is applied equally to base and expert models and writes per-query
replacement counts and source hashes. Fixed-vocabulary UNK remains a separate,
disclosed limitation; the guard is not an UNK-repair algorithm.

Eleven focused unit tests passed, plus a GPU smoke decode of three original
Beckmann queries / 300 hypotheses. The first smoke attempt failed on a Python
package-name collision before model inference; the wrapper import path was
corrected and the second version completed. Full-query execution then exposed
CXSMILES annotations unsupported by direct SPE tokenization. When the original
string does not tokenize losslessly, the guard now parses the complete input
with RDKit and serializes an equivalent ordinary SMILES before tokenization;
it does not simply truncate annotations. An explicit radical-CXSMILES unit
test ensures this distinction. All validation/test queries undergo CPU
preflight before the final protocol is frozen. Smoke outputs are not paper data.

The first fidelity-repaired expert attempt was interrupted solely to avoid
changing its bound inference source during training. Its training input audit
still passes, but the short-run checkpoint is not resumed or promoted. A new
study root binds the finalized inference source.
