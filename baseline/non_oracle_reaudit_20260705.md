# Non-Oracle Reaudit (2026-07-05)

## Scope

This note records a focused reproducibility and methodology audit for the current
Non-Oracle Stage-2/Stage-3 pipeline:

1. Stage 1 route cache is treated as fixed input.
2. Stage 2 builds candidate condition pools from training data only.
3. Stage 3 reranks Stage-2 candidates and predicts temperature.
4. Final evaluation is based on exact `(route, reagent, solvent)` matches for
   `sys@k`, with temperature hit statistics defined as top-k end-to-end hits.

The purpose is to verify that the current mainline (`KNN+XGB`) and the baseline /
ablation experiments are scientifically valid and reproducible.

## What Was Checked

### 1. Split leakage

Command:

```bash
conda run --no-capture-output -n ProSys python data_preprocess/audit_data_splits.py --strict
```

Result:

- No overlap between Stage-2 `train / val / test` splits for all 10 families.
- No overlap between Stage-1 raw `train / val`.
- No overlap between Stage-1 `train / val` reactions and Stage-2 `test`.

Interpretation:

- The current evaluation is not using test reactions as Stage-2 memory.
- The current Non-Oracle pipeline does not show obvious split leakage.

### 2. Candidate-pool construction logic

Files:

- [stage2_KNN/knn_condition_selector.py](/root/autodl-tmp/ProSys/stage2_KNN/knn_condition_selector.py:46)
- [baseline/run_non_oracle_stage23_experiments.py](/root/autodl-tmp/ProSys/baseline/run_non_oracle_stage23_experiments.py:150)

Checked behavior:

- KNN memory is built from the family `train` split only.
- Non-Oracle test candidates are generated from `outputs/stage1_routes/<family>/route_cache.json`.
- Test-route records are queried against train memory and are not written back into memory.

Interpretation:

- Stage 2 is acting as a true retrieval-style screening step.
- Test-time route hypotheses are downstream inputs, not hidden training labels.

### 3. Stage-3 feature/label boundary

Files:

- [stage3_XGBoost/xgb_reranker.py](/root/autodl-tmp/ProSys/stage3_XGBoost/xgb_reranker.py:20)
- [baseline/common.py](/root/autodl-tmp/ProSys/baseline/common.py:418)

Checked behavior:

- XGBoost uses engineered numeric features from Stage 1 + Stage 2 tables.
- Gold labels are assigned after candidate-table construction.
- `label = 1` only when the predicted route and the predicted condition pair
  exactly match the gold reaction system.

Interpretation:

- The reranker is trained on exact system relevance, not on relaxed route-only
  or context-only targets.
- `sys@k` is a strict end-to-end metric.

### 4. Temperature metrics

File:

- [baseline/common.py](/root/autodl-tmp/ProSys/baseline/common.py:511)

Checked behavior:

- `Temp@10C` and `Temp@20C` are computed as top-10 end-to-end hits:
  there must exist a top-10 candidate that
  - matches the full `(route, reagent, solvent)` system,
  - has valid temperature annotations,
  - and has temperature error within `+/-10C` or `+/-20C`.

Interpretation:

- Temperature is now counted as a joint success event, not as a separately
  conditioned accuracy on already-hit systems.
- This avoids inflating temperature results with route-only or context-only hits.

### 5. Zero-route handling in Stage 1

Observed behavior:

- Some Stage-1 `route_cache.json` files contain test reactions with zero
  predicted routes.
- Those reactions naturally produce no Stage-2 / Stage-3 candidate slate.

Interpretation:

- This is consistent with the Non-Oracle definition.
- `num_slates` in downstream result files is therefore slightly smaller than the
  raw number of test reactions for some families.

## Why Official Outputs Were Refreshed

The previous official directory:

- `outputs/stage23_non_oracle_all10`

was not guaranteed to be perfectly aligned with the current code path. During the
audit, several Beckmann shared-table files were found to differ slightly from a
fresh rerun, even though the current environment reproduced identical results
across repeated fresh runs.

Interpretation:

- The current code is reproducible.
- The old official output tree was stale relative to the current code version.

Action taken:

- Archived the old tree to:
  `Experiment/legacy_outputs/baselines/non_oracle_all10_legacy_refreshfix_20260705T115731Z`
- Rebuilt `outputs/stage23_non_oracle_all10` from the current code.

## Determinism Fixes Added

Files updated:

- [baseline/common.py](/root/autodl-tmp/ProSys/baseline/common.py:264)
- [baseline/tabular_models.py](/root/autodl-tmp/ProSys/baseline/tabular_models.py:25)
- [stage3_XGBoost/xgb_reranker.py](/root/autodl-tmp/ProSys/stage3_XGBoost/xgb_reranker.py:55)
- [baseline/run_non_oracle_stage23_experiments.py](/root/autodl-tmp/ProSys/baseline/run_non_oracle_stage23_experiments.py:11)
- [scripts/run_stage23_non_oracle_suite.sh](/root/autodl-tmp/ProSys/scripts/run_stage23_non_oracle_suite.sh:31)

What changed:

- Added canonical stable sorting before writing candidate/training tables.
- Added stable sorting before fitting / scoring XGBoost and tabular rerankers.
- Set a default `OMP_NUM_THREADS=8` in the main scripts to reduce uncontrolled
  runtime variation.

Interpretation:

- These fixes do not change the intended methodology.
- They make the regenerated official outputs easier to reproduce exactly.

## Runtime Fixes For Legacy FNN

During the refresh run, the main bottleneck was the historical FNN baseline /
FNN-pool generation path rather than KNN or XGBoost. To make the official
Non-Oracle rerun finish in a reasonable time without changing the algorithm, the
following engineering-only optimizations were added:

- enable `torch.inference_mode()` for legacy candidate generation and ranking;
- allow legacy models to use GPU via `PROSYS_LEGACY_DEVICE` when available;
- replace slow `torch.Tensor(list_of_numpy_arrays)` materialization with
  `np.stack(...)` + `torch.from_numpy(...)`;
- rank only the required top-`k` contexts before converting them back to names;
- add progress logging for long historical-FNN loops;
- sanitize invalid `OMP_NUM_THREADS` values such as `0`.

Interpretation:

- These changes are runtime / infrastructure fixes only.
- They do not redefine the candidate pool, label semantics, or evaluation
  metric.
- They were necessary to regenerate the current official result tree from the
  current code base.

## About `prototype_fnn = 0`

For Beckmann, the refreshed `prototype_fnn` result still shows:

- `cover = 0`
- `sys@10 = 0`

But the labeled table also shows:

- non-zero `route_match`
- non-zero `context_match`

Interpretation:

- The historical FNN pool is not empty.
- It can retrieve route-related or partial-context candidates.
- It does not retrieve any exact `(route, reagent, solvent)` positive for this
  family under the current strict label definition.

So this is not the same as "the evaluator dropped all candidates". It means:

- partial hits exist,
- exact system hits do not.

This is important when comparing the original FNN baseline against the current
mainline, because the mainline is being evaluated on strict end-to-end system
success rather than relaxed route-only matching.

## Current Conclusion

At this stage, the current Non-Oracle Stage-2/Stage-3 pipeline is methodologically
reasonable:

- no obvious split leakage was found,
- label semantics are strict and internally consistent,
- temperature statistics are now aligned with the top-k end-to-end definition,
- the previous official result tree was stale and has been archived,
- the refreshed `outputs/stage23_non_oracle_all10` tree has been regenerated
  from the current code.

The current single source of truth is:

- `outputs/stage23_non_oracle_all10/overview.md`
- `outputs/stage23_non_oracle_all10/baseline_historical.md`
- `outputs/stage23_non_oracle_all10/ablation_stage2.md`
- `outputs/stage23_non_oracle_all10/ablation_stage3.md`
- `outputs/stage23_non_oracle_all10/average_effect.md`
