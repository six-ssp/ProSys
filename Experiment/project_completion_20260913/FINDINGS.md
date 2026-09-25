# Findings and pending protocol decision

> Historical findings and interim decisions. The user subsequently chose a
> filtered-USPTO-50K base trained from random initialization. Repaired inputs,
> all expert/downstream experiments and local manuscript publication are now
> complete; see `../final_release_50k_20260925/FINAL_CHECK.md`. Pending decisions
> below are retained for provenance, not current requests or live-job status.

## Whole-side canonicalization

The shared helper now parses whole reaction sides before splitting connected
components and rejects invalid whole sides. The preprocessing grouping helper
uses this same implementation. Existing raw data, splits and historical
checkpoints have not been rewritten. The corrected six-family condition audit
reports zero invalid whole sides, zero helper/reference mismatches and zero
train/validation/test canonical-reaction intersections in each family.
The affected Diels-Alder training record is retained intact. The six-fit
old/new-helper sensitivity study (three seeds per arm) is complete. Both arms
use deterministic R-GNN numerical settings to avoid conflating the helper
intervention with uncontrolled GPU aggregation differences.

| Diels-Alder repair study | Legacy helper | Corrected helper |
| --- | ---: | ---: |
| Sys@10, mean +/- sample SD (%) | 26.47 +/- 0.38 | 27.43 +/- 1.72 |
| Conditional temperature MAE (C) | 18.99 +/- 0.14 | 18.11 +/- 0.36 |

The mean Sys@10 change is +0.96 pp, but one seed decreases. Candidate identities
change in every seed, and selected temperature-support identities change in
seeds 0 and 1, despite equal support counts of 208. Thus the MAE comparison is
a whole-pipeline repair result, not a pure same-support temperature ablation.
No test-based selection of the best seed or helper is made. This one-family
study does not replace the six-family mainline.

## Temperature repeatability

Four full Beckmann seed-0 encoder fits were performed before changing the
maintained R-GNN implementation. Training and validation files, architecture,
hyperparameters, thread count, GPU and training seed were identical.

| Runtime | Repeat 0 best validation loss | Repeat 1 best validation loss | Learned tensors identical |
| --- | ---: | ---: | --- |
| Default CUDA algorithms | 0.9052387476 | 0.8944042325 | No |
| Deterministic algorithms | 0.8738905787 | 0.8738905787 | Yes |

This isolates a current same-seed numerical reproducibility issue. It does
not retrospectively prove that all historical temperature MAE drift came from
the same cause. New maintained R-GNN fits and embedding inference enable
deterministic algorithms, retain the architecture, and use distinct caches.
Old payloads preserve their original numerical setting. The full corrected
Diels-Alder seed-0 Stage-2/3 pipeline was also independently refit: all 125600
retained candidate rows, their scores and temperatures, and learned R-GNN
tensors agree exactly. Both runs use identical data, routes, seed and original
validation. This is full-regressor repeatability evidence for this family/seed
in this environment, not a three-seed six-family result promotion.

## USPTO overlap audit

The independent scan covers all 747660 retained base-training rows and 186915
base-validation rows. Atom maps are removed on RDKit molecules and complete
connected-component reaction keys are compared, retaining stereochemistry.

- Full-reaction intersections with the six condition test sets: zero.
- Base train/validation canonical-reaction intersection: zero.
- Full-reaction intersections with condition validation: 82 base-training
  records and 16 base-validation records, 98 records total.
- Same-product intersections are nonzero and separately reported; they are
  not the same as exact-reaction leakage or product-disjoint generalization.
- The retained mapped raw corpus contains 32 training and 9 validation rows
  that fail whole-side RDKit parsing. They are listed, not silently dropped
  from the audit. Replaying the retained EditRetro input guard rejects all
  41 exact raw rows. No historical per-row raw-to-augmentation manifest was
  retained, so this does not retrospectively prove their membership or absence
  in the July tensors. Exact-reaction intersection counts apply to parseable
  reactions; do not claim all raw rows were valid.

The original preprocessing explicitly removed only condition-test overlaps.
Thus the test-isolation claim is supported by this new scan, but an assertion
that the entire condition validation set was unseen to base pretraining is
not supported. Current hashes bind retained files; historical training logs
identify the data-bin path but do not supply missing historical raw-to-bin
hashes. Do not represent current hashing as retrospective proof.

The user has been asked to choose between retaining the fixed base and using
a clean validation subset, or filtering USPTO and retraining the base and all
dependent stages. Do not silently switch datasets or promote old-validation
training repeats as a clean-validation experiment.

The newly launched Stage-1 group was stopped with SIGINT after its Beckmann
seed-0 epoch-22 last checkpoint (2464 updates). Both best and last remain;
the partial run is not a completed replicate. No original model was stopped
or modified. The independently running Diels-Alder sensitivity study has now
finished; no training workers remain from this phase.

## Prepared Stage-1 input audit

The clean-validation preview finds 3750 unique reactions remaining after
excluding 98; condition validation rows would decrease from 5863 to 5504.
All 39640 retained family validation text pairs were re-encoded with the stored
dictionaries and exactly match the binary inputs. No new split is promoted.

This audit uncovered actual empty-target binary sequences: 14610 family
training pairs, 80 validation pairs and 150 prepared test pairs. Root-aligned
augmentation can select no reactant component, and the old export accepted
the resulting empty target. All flagged empty sides were verified as EOS-only
binary tensors. The USPTO base augmented inputs have zero empty pairs.

The preprocessing export now filters empty pairs, records original augmented
indices in a per-split manifest, and uses whole-side connected components for
cross-dot ring closures. A real three-record preprocessing CLI smoke exercises
valid mapping, disjoint mapping that produces empty targets, and a valid
cross-dot mapped reactant. Six augmented pairs become four valid pairs, with
the two empty pairs explicitly recorded. Frozen inputs are unchanged. Separate
nonempty copies now exist; validation policy still requires a decision before
new expert training. See `VALIDATION_DECISION.md`.

## Deployment compatibility

Read-only inference now rejects missing/mismatched feature-source manifests
and legacy monkeypatched sensitivity models before decoding or model use.
This prevents silently using historical weights with repaired input code.
Old artifacts require compatible code; no bypass fabricates their provenance.
A corrected Diels-Alder cached-route product smoke gives 100 identical
chemical candidate identities, exact equality of all 52 ranking inputs, and
zero difference in ranking scores and predicted temperatures versus batch
evaluation. The CLI canonicalizes its product string, so parity is checked
by canonical chemical identity, not literal raw SMILES equality.

## Verification status

The expanded 60-test suite passed; see `tests_nonempty_inputs.log`. It includes the
public-API deterministic GPU fit/inference repeat, actual preprocessing CLI,
text/bin mismatch rejection, and inference feature-source guards. Bounded
interface tests do not establish multi-family chemistry performance. The
Diels-Alder six-fit sensitivity study is complete; its fully replayed table
and exact pairing boundaries are in `canonical_sensitivity/SUMMARY.md`.
Temperature replay restores the scorer's float32 prediction dtype before
aggregation, rather than mistaking decimal serialization for new model drift.

## Nonempty expert input copies

`stage1_nonempty_inputs/independent_verification.json` passes for all six
families. It independently checks original/output hashes, exact retained
tensor equality, ordered augmented-row lineage, and the archived builder source.
Retained augmented pairs are train 851350, validation 39560 and prepared test
39740; these are not counts of unique reactions or system evaluation queries.
The fixed 3860 system-query denominator is unchanged. These copies only remove
empty pairs, not USPTO validation overlaps, and remain explicitly unapproved
for formal three-seed training. The training shell supports an explicit
`DATA_BIN` override with `SKIP_PREPARE=1` and rejects EOS-only input sequences.
No new expert training was launched. Available data-disk space is about 30 GiB.
