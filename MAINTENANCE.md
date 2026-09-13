# Maintenance

## Current Freeze (2026-09-13)

At the user's request, retain the current data, models and promoted results;
do not launch further training or validation-policy changes. Publish completed
repairs, tests and aggregate audit evidence only. The validation overlaps and
unfinished Stage-1 expert replicates remain disclosed limitations, not fixes
already applied. Model bundles, row-level data and paper drafts stay local.

## Active Correctness Work (2026-09-13)

See `Experiment/project_completion_20260913/PLAN.md` and `FINDINGS.md`.
The shared canonicalizer and preprocessing now preserve cross-dot ring bonds;
new deterministic R-GNN caches are distinct from historical artifacts. A
paired sensitivity study has completed: Diels-Alder Sys@10 changes from
26.47 +/- 0.38% to 27.43 +/- 1.72%. This single-family diagnostic is not a new
six-family mainline, and its conditional-temperature support identities differ.
The full retained USPTO audit confirms zero exact reaction overlaps with
condition tests but identifies 98 condition-validation overlaps. Stage-1
replicate training was stopped intentionally pending a protocol decision.
Original mainline checkpoints and frozen splits remain unchanged.

The prepared-input audit additionally verifies actual EOS-only family targets:
14610 train and 80 validation augmented pairs. Export now filters and records
these pairs; separate nonempty copies now pass full retained-tensor and hash
verification. They do not yet apply the pending validation policy. Base
USPTO augmented inputs have no empty pairs. The clean-validation preview is
read-only and excludes 98 reactions / 359 condition rows. A full deterministic
Diels-Alder seed-0 repeat and a corrected cached-product inference smoke both
match their respective reference predictions exactly. Inference now rejects
incompatible feature-source manifests rather than silently mixing old weights
with the repaired implementation. Licensed traces and model bundles remain
local-only under the new study's scoped ignore rules.

## Documentation Cleanup (2026-09-13)

`CURRENT_RESULTS.md` and the maintained Stage 2/3 detail documents now describe
only the parallel mainline. Their complete pre-cleanup mixed-version text is
preserved in `Experiment/document_archive_20260913/`; historical numbers and
algorithms must not be copied back into current-method sections. Baseline
overview tables likewise use the current comparison only. No model, data
split or reported experimental metric changed during this cleanup.

Matched-control claims must distinguish equal configurations and aggregate
metrics/support counts from a per-candidate identity audit. The retained
compact records establish the former, not an independent identity-hash check.
Paper figures should be rebuilt from promoted current result CSVs, never from
the archival manuscript-package tables.

## Maintained Workflow

```text
stage1_retrosynthesis/ -> stage2_ReaFNN/ -> stage3_XGBoost/
```

The maintained Stage 2/3 entrypoint is:

```bash
bash scripts/run_stage23_non_oracle_suite.sh .
```

It runs product-Morgan KNN and ReaFNN in parallel, selects each family fusion
weight on validation routes only, then applies tabular XGB-LTR and a separate
R-GNN-assisted temperature regressor. `stage2_KNN/` is a required compatibility
shim and must remain importable.

The suite invokes `scripts/run_verified_mainline.py`, not an existence-only
cache check. Family cache manifests bind splits, Stage 1 route caches and
referenced checkpoint bytes, model-source files, package versions, configuration
and generated tables/models. Unknown or changed caches fail closed; use a new
output root or explicitly rebuild. A forced rebuild also retrains ReaFNN and
R-GNN. Family process locks prevent simultaneous writers. Pruned report bundles
are not reusable training caches.

`scripts/predict_product.py` is the read-only product-query entrypoint. It
requires a retained family bundle and matching training condition library,
checks their hashes, and never trains or reselects fusion weights. Detailed
commands and evidence are in
`Experiment/mainline_evidence_completion_20260913/README.md`.

## Result Scope

- Current maintained record:
  `Experiment/stage23_parallel_post_fusion_multiseed_20260903/`, a fixed
  Stage-1, six-family, seeds-0/1/2 parallel evaluation.
- `Experiment/stage2_parallel_post_fusion_20260901.md` remains the detailed
  seed-0 development record, not the primary uncertainty estimate.
- `Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/` is the
  paired current Stage-2 ReaFNN-removal control: it disables ReaFNN, preserves
  fixed Stage-1 routes and the common KNN settings, and re-trains XGB-LTR on
  the KNN-only pool. Use it, not serial controls, for current ReaFNN claims.
- `Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/` is the
  paired current Stage-3 control: it keeps the complete parallel Stage-2 pool
  fixed for each family and seed, then replaces XGB-LTR with deterministic
  Stage-1/2 ranking. Use it, not serial controls, for current XGB-LTR claims.
- `Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/` is
  the paired current temperature-representation control. It preserves all
  Stage 1/2/3 ranking inputs and outputs while changing the temperature XGBoost
  input from `52 + 128` R-GNN-augmented features to the same 52 tabular
  features alone. All 18 family-seed pairs pass retained protocol and aggregate
  ranking/support-count checks; no independent identity-hash audit is retained.
- `CURRENT_RESULTS.md` states the reportable boundary and distinguishes the
  current parallel study from historical serial results.
- `Experiment/stage2_reafnn_only_multiseed_20260913/` completes the strict
  KNN-removal control, including zero-KNN feature checks and replayable local
  candidate files. Its six-family Sys@10 is `36.24 +/- 0.27%`.
- `Experiment/mainline_evidence_completion_20260913/` is a separate, completed
  18-run identity-level reconstruction. Do not replace historical temperature figures
  with its predictions without an explicit numerical comparison and promotion.
  Its canonicalization note records one known training-only representation
  edge case; the frozen mainline was not silently changed during the study.
- The 2026-08-30 serial multi-seed mainline and its ablations are historical
  controls only; do not reuse their values as paired evidence for the current
  parallel candidate distribution.

## Archive Layout

- `Experiment/legacy_stage2/`: neural-V2 implementation and historical launchers.
- `Experiment/legacy_stage3/`: retired candidate-aware ranking probes.
- `Experiment/local_archive/`: ignored local-only exploratory material. Its
  README is tracked, but its contents are never GitHub deliverables.
- `Experiment/legacy_tools/`: ignored one-off legacy helpers.

## Output and Disk Policy

The `outputs/` tree is ignored because it is regenerable and can be large.
The fixed route caches (`outputs/stage1_routes/` and
`outputs/stage1_routes_validation/`) are useful Stage-2/3 rerun accelerators,
but are not irreplaceable: they can be regenerated from the retained Stage-1
checkpoints. Before deleting any completed run root, preserve a compact audited
result record under `Experiment/`; paper claims must never rely only on an
ignored `outputs/` directory.

Safe regeneration targets include `__pycache__/`, `.ipynb_checkpoints/`, and
completed smoke/probe output directories. Large historical result trees should
be reviewed against their matching `Experiment/` record before removal.

## Recommended Checks

```bash
conda activate ProSys
python -m py_compile prosys_shared/*.py stage2_ReaFNN/*.py stage3_XGBoost/*.py scripts/run_stage23_mainline_non_oracle.py
bash -n scripts/run_stage23_non_oracle_suite.sh
python data_preprocess/audit_data_splits.py --strict
```
