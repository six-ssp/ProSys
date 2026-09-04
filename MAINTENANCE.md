# Maintenance

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

## Result Scope

- Current maintained record:
  `Experiment/stage23_parallel_post_fusion_multiseed_20260903/`, a fixed
  Stage-1, six-family, seeds-0/1/2 parallel evaluation.
- `Experiment/stage2_parallel_post_fusion_20260901.md` remains the detailed
  seed-0 development record, not the primary uncertainty estimate.
- `Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/` is the
  paired current Stage-3 control: it keeps the complete parallel Stage-2 pool
  fixed for each family and seed, then replaces XGB-LTR with deterministic
  Stage-1/2 ranking. Use it, not serial controls, for current XGB-LTR claims.
- `CURRENT_RESULTS.md` states the reportable boundary and distinguishes the
  current parallel study from historical serial results.
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
Retain route caches needed by the current workflow (`outputs/stage1_routes/`
and `outputs/stage1_routes_validation/`) and any active run root. Before
deleting model/result directories, preserve a compact audited result record.

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
