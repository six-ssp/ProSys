# Completion audit: project goal still in progress

## User-directed freeze (2026-09-13)

The user requested no further protocol changes or training, and publication
of the completed code/documentation changes only. Retain the current datasets,
models and promoted results. The remaining items below are recorded limitations
and deferred work, not queued jobs or a selected clean-validation protocol.
Local model bundles, row-level evidence and manuscript drafts are not Git payloads.

## Proven in this phase

- Whole-side canonicalization repaired in shared features, preprocessing
  grouping and EditRetro fragment handling. Full condition split audit passes.
- Six Diels-Alder repair sensitivity fits completed and independently replayed;
  all seeds and conditional-support differences are retained in the report.
- Same-seed R-GNN nondeterminism isolated in a four-fit probe. Deterministic
  full-pipeline repetition and corrected one-product inference parity pass.
- Retained filtered USPTO train/val corpus audited: no parseable full-reaction
  condition-test overlaps, 98 condition-validation overlaps; 41 invalid raw
  rows are explicitly accounted for with historical provenance limits.
- All 39640 family validation text/bin pairs match. Actual empty family targets
  are counted and verified in binaries. Export repair and real CLI smoke pass.
- Versioned nonempty expert copies are built and independently verified across
  all retained tensors: train 851350, validation 39560, prepared test 39740.
  Original files are hash-unchanged; validation overlap removal is still pending.
- Inference rejects incompatible feature-source manifests. The 60-test suite
  and `git diff --check` pass. Historical results and inputs are unchanged.

## Not complete

1. User decision: fixed-base clean-validation subset or filtered-base retraining.
   The preview and tradeoffs are in `VALIDATION_DECISION.md`; no choice is
   inferred from the preselected option of an unanswered question.
2. Apply the selected validation policy to a new version of the verified
   nonempty expert copies. Original system test queries must stay unchanged.
3. Six-family expert fine-tuning at seeds 0/1/2 and Route@1/3/5/10 aggregation.
   The initial seed-0 pilot was stopped intentionally and is not a replicate.
4. Reconcile/promote the resulting protocol's six-family downstream temperature
   and system evidence with required matching comparisons and documentation.

No GPU training worker remains from the completed diagnostic phase. Common
preparation is complete; further formal training is blocked on the unanswered
validation-policy decision. The overall goal is not achieved. No Git push occurred.

Authoritative evidence: `FINDINGS.md`, `canonical_sensitivity/summary_audit.json`,
`deterministic_pipeline_repeat/comparison.json`, `corrected_inference_smoke/parity.json`,
`uspto_audit/audit.json`, `validation_preview/preview.json`,
`empty_stage1_pairs/audit.json`, `tests_nonempty_inputs.log`,
`stage1_nonempty_inputs/independent_verification.json`.
