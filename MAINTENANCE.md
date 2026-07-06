# Maintenance

## Active code

Current maintained mainline:

```text
stage1_retrosynthesis/ -> stage2_KNN/ -> stage3_XGBoost/
```

Supporting experiment code:

- `baseline/`
- `ablation/`
- `scripts/run_stage23_non_oracle_suite.sh`

## Legacy code

These paths are still kept because the historical baseline depends on them:

- `Experiment/legacy_stage2/`
- `save_models/`
- `baseline/legacy_models.py`
- `baseline/run_oracle_baselines.py`
- `baseline/run_non_oracle_baselines.py`
- `scripts/run_stage2_v2_family_batch.py`
- `scripts/run_stage2_v2_non_oracle.py`

They are compatibility layers now, not the preferred path for new development.

## Active outputs

- `outputs/stage1_routes/`
- `outputs/stage23_non_oracle_all10/`

## Archive policy

Move non-mainline but still useful material into `Experiment/`:

- notebooks
- old oracle / non-oracle baseline result trees
- route-budget analyses
- obsolete render scripts
- one-off helper utilities

## Delete policy

Safe to delete when regenerated:

- `__pycache__/`
- smoke outputs
- duplicated intermediate summaries

## Before changing the mainline

1. Confirm whether the change touches `stage2_KNN/` or `stage3_XGBoost/`.
2. If it changes evaluation, also refresh:
   - `baseline/render_stage23_nonoracle_reports.py`
   - `outputs/stage23_non_oracle_all10/*.md`
3. Keep `Temp@10C` / `Temp@20C` aligned with the current rule:
   - within top-10 there exists a full system hit with valid temperature
   - and the temperature error is within `+/-10C` or `+/-20C`
4. Update:
   - `README.md`
   - `ProSys_goal.md`
   - `todo.md`
   - `log.md`

## Recommended sanity checks

```bash
conda activate ProSys
python -m py_compile prosys_shared/*.py baseline/*.py stage2_KNN/*.py stage3_XGBoost/*.py
python scripts/audit_data_splits.py --strict
python baseline/render_stage23_nonoracle_reports.py --output_root outputs/stage23_non_oracle_all10
```
