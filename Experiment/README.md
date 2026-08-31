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

## Matched Current-Mainline Ablations

- current_mainline_matched_ablation_multiseed_20260830/
  - Current three-seed component controls for the promoted product-Morgan
    mainline: KNN-only plus re-trained XGB-LTR, and full Stage 2 plus a
    deterministic no-XGB-LTR ordering.
  - Contains 36 compact family-seed records, aggregate CSVs, run metadata, and
    the completion manifest.
  - An independent check verifies the shared 3,860-identity denominator and,
    for the no-XGB-LTR arm, exact family-seed Stage 2 candidate-pool agreement
    with the promoted mainline.
  - See ablation/current_mainline_matched_ablation_results_20260830.md for
    paper-ready interpretation.

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

it should live here instead of the repo root. The artifact explicitly listed
under **Promoted Mainline Artifact** is the only source of headline current
performance; the matched-ablation artifact above is the only source of current
component-attribution figures.
