# Experiment Records

This directory primarily stores material that is useful for reference but is no
longer active mainline code. One compact result artifact is an explicit
exception: it is the current official performance record.

## Promoted Mainline Artifact

- `stage23_product_morgan_reafnn_multiseed_20260830/`
  - Current six-family, fixed-Stage-1, three-seed Stage 2/3 robustness record.
  - Contains 18 compact family-seed results, CSV aggregates, run manifest, and
    retained model metadata.
  - Large candidate tables and checkpoints were pruned only after the compact
    records were written, to respect the storage budget.
  - See `../CURRENT_RESULTS.md` for reportable tables and scope.

## Archived Contents

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

it should live here instead of the repo root. Only the artifact explicitly
listed under **Promoted Mainline Artifact** may be used as a current official
result source.
