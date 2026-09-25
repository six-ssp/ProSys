# Stage 1 Post-Augmentation Split Repair

Status update (2026-09-24): the user chose filtered USPTO-50K trained from
scratch, not FULL retraining or restoration. See
`../stage1_50k_from_scratch_20260924/README.md` for live work. The strict FULL
copies and audits below remain historical evidence; their checkpoints are not
being trained or promoted. Repaired expert tensors require a new combined
audit and admission bound to the completed 50K base. No corrected performance
result has been promoted.
The previous expert-seed queue was suspended and then terminated after scientific invalidation.
Its checkpoints must not warm-start repaired expert runs.

Descriptions of a pending FULL-versus-fixed-base decision below record the
September 15 state and are superseded by the September 24 decision.

## Prepared Expert Copies

All six strict-option expert copies have been generated. Their independent
copy verification checks every retained text line and tensor byte, not a sample.
The independent chemical audit reports zero train/validation, train/test and
validation/test overlap within the new expert data; zero expert-training overlap
with original condition validation/test reactions; and zero expert-validation
overlap with original condition-test reactions. All inspected sides parse.
The subsequent combined audit also checks these expert inputs against the rebuilt
base inputs. Neither audit certifies a trained model.

| Family | Train pairs before | Train pairs after | Validation pairs before | Validation pairs after | Test pairs removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 80,660 | 79,280 | 2,410 | 2,310 | 0 |
| Buchwald-Hartwig | 167,620 | 166,600 | 11,010 | 11,010 | 0 |
| Chan-Lam | 63,910 | 63,760 | 3,920 | 3,900 | 0 |
| Diels-Alder | 235,760 | 234,420 | 8,260 | 8,210 | 0 |
| Friedel-Crafts acylation | 102,740 | 102,400 | 4,870 | 4,860 | 0 |
| Friedel-Crafts alkylation | 200,660 | 199,620 | 9,090 | 9,090 | 0 |
| Total | 851,350 | 846,080 | 39,560 | 39,380 | 0 |

These are augmented pairs relative to the earlier nonempty copies, not unique
raw reactions or final evaluation queries. Original condition test queries remain
unchanged. Evidence is in `expert_inputs/summary.json`,
`expert_inputs/independent_copy_verification.json` and
`expert_inputs/chemical_split_audit.json`. The copies target the proposed strict
refiltered base, not the old base checkpoint; the fresh-base training decision
and trained-model admission remain pending.

The strict base-data copy is complete under `base_inputs/`: train
7,436,440 and validation 1,858,870 augmented pairs remain after excluding
850/190 pairs. Independent copy verification passed every retained text line
and tensor byte. The separate full chemical/text-binary scan passed in
1,288.10 seconds; `base_rebuilt_audit/audit.json` is its completion receipt.
Retained unique reactions are 743,644 train and 185,887 validation, exactly as
planned. Both splits have zero unparseable pairs and zero overlaps with every
checked condition/expert held-out set; base train/validation overlap is zero.
No training is launched by preparation or audit commands.

The combined expert/base chemical audit passed for all six families in
`expert_inputs/combined_base_chemical_audit.json`. Before/after hashes bind 100
files, including condition files, expert text, base keys, the completed base
receipt and its source inputs. Expert validation overlaps with rebuilt-base
train/validation are zero in every family. This closes the standalone expert
audit's previously untested base-membership field. Original condition query
splits and prepared expert test files remain unchanged. Existing checkpoints
still reflect their old inputs and are not repaired by these checks.

## Downstream Reuse Protection

Review found that the old baseline-repeat runner could overwrite its experiment
configuration before skipping completed seeds, and automatically imported B1
from a fixed historical result path. This is unsafe when Stage 1 routes change;
it does not by itself prove that the original same-route baseline arithmetic
was wrong. The maintained runner now binds study inputs/settings and completed
output bytes, verifies B3/B4 export manifests, and runs B1 once on this study's
routes. All baseline inputs/results must use new versioned roots after repair.
Synthetic tests include actual B3/B4 export generation and same-path checkpoint
replacement; no new scientific baseline training has been launched.
Future B3/B4 retention preserves full prediction/candidate files as byte-verified
gzip archives before work-directory cleanup; it no longer keeps only summary
JSON. Previously pruned historical archives are not reconstructed by this change.

## Confirmed Failure

The original condition and raw route splits are canonical-reaction-disjoint,
but splitting products and selecting mapped precursors changes reaction identity.
The actual augmented expert-training tensors contain condition-held-out reactions.
The whole-corpus membership counts and six independent binary positive examples
are in `../final_release_audit_20260915/augmented_split_audit.json` and
`augmented_overlap_examples.json`.

| Family | Train / condition validation | Train / condition test | Collisions traced to mapped raw transformations |
| --- | ---: | ---: | ---: |
| Beckmann | 16 | 20 | 36 / 36 |
| Buchwald-Hartwig | 7 | 6 | 13 / 13 |
| Chan-Lam | 3 | 2 | 5 / 5 |
| Diels-Alder | 30 | 19 | 49 / 49 |
| Friedel-Crafts acylation | 13 | 10 | 23 / 23 |
| Friedel-Crafts alkylation | 34 | 27 | 61 / 61 |

Counts are distinct full canonical reactions within each family, not globally
deduplicated totals and not merely shared products. Every traced raw source lies
outside the condition held-out reactions before transformation. Matching source
chemistry does not reconstruct an absent historical row-order manifest.
`raw_transformation_trace.json` contains the summary and hashes; local detailed
lineage stays out of Git.

## Completed Base Audit

`scripts/audit_stage1_base_augmented.py` scans all USPTO augmented train/validation
rows, checks both binary tensors against text, and compares complete canonical
reactions against condition and transformed expert validation/test sets. It also
checks base train/validation overlap. No negative conclusion is based on sampling.
`base_augmented/progress.json` is progress only; `audit.json` is written only after
the full scan, counts, and before/after source hashes have been checked.
The original raw USPTO audit alone cannot establish base eligibility.

The full scan completed in 1,288.85 seconds. All 7,437,290 training and
1,859,060 validation augmented pairs match their actual binary tensors, with
zero unparseable text pairs. Base train/validation exact-reaction overlap is
zero. Neither base split contains an exact original condition-test reaction.
However, base validation contains one transformed expert-test reaction each
for Buchwald-Hartwig, Diels-Alder and Friedel-Crafts alkylation. These three
are not exact original condition-test reactions; the distinction is preserved.

A strict fresh-base option would remove 85 distinct training reactions (850
augmented pairs) and 19 distinct validation reactions (190 pairs), covering
all condition and transformed-expert held-out matches. A fixed-base option
would instead require clean downstream validation and explicit disclosure of
the three transformed-test validation overlaps. The user has been asked to
choose between these compute/scope tradeoffs. No new base training is launched.

## Repair Rules

1. Preserve original source files and the complete condition test-query manifest.
2. Remove expert training pairs that match condition validation/test or augmented
   expert validation/test reactions. Remove all corresponding augmented variants.
3. Remove expert validation pairs overlapping condition/augmented test reactions
   or actual base training/validation membership. Never move them into test.
4. Record exact source augmented row indices, canonical-reaction hashes, reasons,
   removed counts, source hashes, and output tensor equality in versioned copies.
5. Complete independent output audits and decide base eligibility before training.
   If the base itself requires retraining, update the exclusion policy accordingly;
   do not declare the present base valid simply because expert inputs are repaired.
6. Propagate any condition-validation restriction to downstream model selection,
   regenerate affected route caches and reevaluate affected mainline/comparators.
   Do not repair only the final metric denominator.

`scripts/prepare_stage1_expert_split_repair.py` implements preparatory copies only.
It requires the completed full base audit and refuses to overwrite an output root.
Its manifests explicitly withhold formal-training admission until independent
output checks, base eligibility and condition-validation propagation are settled.

## Documents and Resources

Main text and SI now qualify retained performance after the new overlap finding.
Their numerical tables remain historical observations, not corrected-input results.
Main Figure 1 is unchanged and excluded from certification as requested.
The completed full scan used 10 CPU workers; no new GPU training is admitted while
the split boundary is unresolved. Large row keys, detailed source lineages and
checkpoints stay local. Preparation requires at least 6 GiB free.

The fixed shared SPE vocabulary encodes some unseen substructure tokens as UNK.
The complete tensor count is in `vocabulary_audit.json`; this is distinct from
split leakage and does not invalidate the six exact positive overlap witnesses,
all of which have zero unknown tokens. Vocabulary/model dimensions are unchanged.
