# ProSys TODO

## Active

- Validate the workspace-local `runtime/overlay_cuda121` runtime from a shell with real GPU access:
  - confirm `torch.cuda.is_available()`,
  - confirm at least one CUDA tensor allocation succeeds,
  - decide whether to standardize training on the overlay path or a cleaned base env.
- Rerun Stage 1 family finetune through the validated CUDA runtime and confirm at least one family completes real GPU training updates.
- Rebuild and verify `fairseq.libnat_cuda` when CUDA toolkit support is available.
- Launch Stage 2 V2 full-family batch training with the unified batch runner and collect stable checkpoints per family.
- Implement Oracle and Non-Oracle evaluation entries for the Stage 2 V2 pipeline.
- Profile full-split Stage 2A candidate-pool build time and optimize if it becomes the next bottleneck.

## Done

- Replace active `rxn_yield_context` runtime imports and package assumptions with the `ProSys` / `stage2` structure.
- Normalize the Stage 2 requirement entry to `stage2/stage2_detail.md`.
- Audit Stage 1 / Stage 2 split leakage and rebuild Stage 1 route exports with the corrected split logic.
- Add unified runtime helpers:
  - `scripts/check_runtime.py`
  - `scripts/setup_prosys_env.sh`
  - `scripts/audit_data_splits.py`
- Add a workspace-local CUDA overlay fallback for locked or conflicting `ProSys` environments:
  - `scripts/prepare_prosys_cuda121_overlay.sh`
  - `scripts/run_in_prosys_cuda121_overlay.sh`
  - `scripts/libittnotify_stub.c`
- Add unified family runners:
  - `stage1/scripts/run_family_finetune_one.sh`
  - `stage1/scripts/run_family_finetune_batch.sh`
  - `scripts/run_stage2_v2_family_batch.py`
