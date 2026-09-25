# Project completion and robustness study

Status: experimental work and local document publication completed; final
requirement closure is recorded in `../final_release_50k_20260925/FINAL_CHECK.md`.
Historical promoted results are immutable controls.

Current admission supersedes the original FULL prescription in item 4 below:
the user selected a filtered USPTO-50K base trained from random initialization,
with no FULL network weights. That base and repaired expert inputs passed their
audits; the completed expert study is
`../stage1_50k_fidelity_v2_expert_multiseed_20260924/`. Original condition queries
and downstream splits stay fixed. Expert seeds 0/1/2 share that one new base;
downstream seeds use the predeclared expert seed 1, not the best test seed.
Historical requirements below document why this replacement was necessary.

The requested same-seed full Stage 2/3 repeat has now also been performed on
fresh 50K routes: Beckmann seed 0, original and independently refitted run.
All 40,520 candidates, 2,013 route embeddings, full ranking and predictions,
both neural payloads and both XGBoost models match exactly. Receipt:
`../stage23_50k_repeat_check_20260924/independent_repeat_audit.json`.
This is a one-family same-environment check, not cross-hardware determinism.

## Requirements

1. Fix reaction-side canonicalization using whole-side parsing before connected
   components; reject invalid whole sides without salvaging partial fragments.
   Add regression tests, repeat the full condition-split audit and run a
   versioned Diels-Alder downstream sensitivity comparison over seeds 0/1/2.
2. Investigate temperature reconstruction drift using controlled repeated
   fits and artifact comparisons. Separate measured causes from hypotheses.
   Define one unambiguous current temperature evidence version and update
   maintained tables/documentation, retaining historical provenance.
3. Re-audit the actual filtered USPTO base-training and validation corpus
   against all six condition validation/test sets, stripping atom maps and
   using whole-side canonical reactions. Report product overlaps separately
   from exact-reaction leakage. Verify linkage to the training inputs where
   retained artifacts allow it; disclose any missing historical linkage.
4. Run six families x seeds 0/1/2 from the same frozen USPTO-FULL base in this
   environment. Keep fixed original augmented/binarized inputs. Use each
   family's historical batch-token limit (Beckmann 8192, others 16384), fixed
   validation seed 7, validation-loss best checkpoint, patience 15, maximum
   200 epochs / 200000 updates. Other training hyperparameters stay fixed.
   Evaluate Route@1/3/5/10 on identical query identities with identical decoding.
   Report means and sample SD, and the fixed-base comparator, not three base
   pretraining replicates. Do not replace fixed Stage-1 downstream ablations.
5. Test implementations, review numerical consistency, document scope and
   resource use, and synchronize current maintenance/results/checklist files.

## Newly discovered requirements (not silently waived)

- The actual augmented expert inputs overlap condition test reactions in all
  six families, despite raw-split disjointness. The 2026-09-15 expert queue is
  suspended and invalidated. Requirement 4's original-input prescription is
  superseded by clean-input admission and a new study root. See
  `../stage1_split_repair_20260915/README.md` for evidence and repair rules.
- The USPTO raw audit found 98 condition-validation reaction overlaps. Decide
  on a clean validation subset with the fixed base or full filtered-base
  retraining before accepting new expert/selection repeats. Preview and
  implications are documented in `VALIDATION_DECISION.md`.
- Retained family prepared data contain 14610 empty training pairs and 80
  empty validation pairs. Their EOS-only binary membership is verified. Fix
  export and create versioned clean inputs before new expert fits. Do not
  reduce the fixed test query denominator. The USPTO base augmented inputs
  have no empty pairs.
- Repeat one complete corrected Stage-2/3 fit at the same seed to verify
  deterministic full-regressor predictions, not just encoder weights.

## Execution and storage

- All new studies use separate output roots; do not overwrite historical caches.
- Stage-1 study retrains all three seeds rather than mixing old-environment
  seed 1 with new seeds. Only best and last checkpoints are retained.
- Start with one Stage-1 GPU worker, inspect measured memory/utilization before
  increasing concurrency. Avoid changing token batches for throughput.
- Data volume had 31 GiB free at start. Reserve at least 5 GiB at job admission;
  do not delete original data or model artifacts to make an experiment pass.
- Three training seeds quantify conditional fine-tuning variability, not split
  variability or randomness of USPTO pretraining. External/prospective tests
  are not part of this requested completion work.
