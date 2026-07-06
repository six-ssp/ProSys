# Experiment Archive

This directory stores material that is still useful for reference, but is no longer part of the maintained ProSys mainline.

## Contents

- `notebooks/`
  - exploratory notebooks and analysis assets
- `legacy_outputs/`
  - old oracle / non-oracle baseline result trees
- `route_budget/`
  - route-budget sensitivity experiments such as `n_best=20`
- `legacy_stage2/`
  - archived neural-V2 Stage 2 code kept only for historical baseline / reproduction
- `legacy_tools/`
  - older render scripts and one-off helper utilities

## Rule

If a file is:

1. useful to keep,
2. not part of the current `stage1 -> stage2_KNN -> stage3_XGBoost` workflow,
3. and not worth maintaining as active project code,

it should live here instead of the repo root.
