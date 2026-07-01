# ProSys Development Log

## Pre-autodl history (2026-06-30 → 07-01, condensed)

Work on the earlier `/home/six_ssp/...` host, before the autodl handover (superseded; kept for reference):

- Established the root repo as the only active repo; `stage2/stage2_detail.md` as the canonical Stage 2 spec; dropped legacy `rxn_yield_context` naming for the `stage2` package layout.
- Built the Stage 2 V2 module set (`stage2/v2/*`, `build_*` CLIs, `train_stage2_v2.py`) and smoke-tested it on Beckmann.
- Fixed a Stage 1 route-split leakage bug, rebuilt route exports (`raw_{train,val,test}.csv`), added `scripts/audit_data_splits.py` (splits verified clean).
- Patched vendored fairseq for modern NumPy (`np.float`) and a CPU reposition-target shape bug; added `ensure_fairseq_extensions.sh` + a CUDA guard in the batch runner.
- Added unified entrypoints (`check_runtime.py`, `setup_prosys_env.sh`, family finetune runners, Stage 2 batch runner) and a workspace-local CUDA overlay fallback (obsolete on autodl — the GPU works natively there).
- That host was blocked from GPU / package access, so full training was deferred to a GPU-capable host.

## 2026-07-01 — autodl host

### Environment (done)

- Host: autodl, `/root/autodl-tmp/ProSys`, RTX 3090 24 GB, driver 580, system CUDA 11.8 (`/usr/local/cuda`, nvcc 11.8). GPU directly usable.
- The `ProSys` conda env was empty; rebuilt by cloning the proven `retro_gan` stack (`conda create --clone retro_gan -n ProSys`) + `pip install xgboost openpyxl`. py3.9, torch 2.7.1+cu118, rdkit/rxnmapper/selfies/SmilesPE/textdistance/prettytable.
- Built vendored fairseq extensions in-place (`libnat`, `libnat_cuda` sm_86, `libbleu`); repointed the editable-fairseq finder from the deleted `editretro/fairseq` path to `stage1/fairseq`. `check_runtime.py` fully green.
- `audit_data_splits.py --strict`: PASS across all 10 families.
- Run any code with `OMP_NUM_THREADS` set (shell inherits `0`, which spams libgomp); set `CUDA_HOME=/usr/local/cuda` for fairseq ext rebuilds.

### Fixes (done)

- `stage1/fairseq/fairseq/checkpoint_utils.py`: force `torch.load(weights_only=False)` (torch≥2.6 flips the default to True and can't unpickle legacy fairseq checkpoints).

### Stage 1 base checkpoint

- `checkpoint_UPSTO_full_best.pt` was initially corrupted (truncated in transfer). User re-uploaded a valid 470 MB copy; verified it loads and fairseq-train restores + trains on GPU.

### Stage 2 V2 improvements (done)

- Fixed `import stage2` in `run_stage2_v2_family_batch.py` (repo-root bootstrap) so the documented command runs without a manual `PYTHONPATH`.
- Optimized Stage 2A candidate-pool build: compute one `ProductSupportContext` per product and reuse across candidates (old code recomputed the Morgan FP + full-matrix Tanimoto per candidate). Byte-identical output, **~7x faster**.
- Added the Stage 2 V2 evaluation entry (`stage2/v2/evaluate.py` + `stage2/evaluate_stage2_v2.py`) — one evaluator for Oracle/Non-Oracle: pool coverage, system/context/route top-{1,3,5,10}, temperature MAE/RMSE/±10/±20 ℃. Auto-run after training by the batch runner (`eval_oracle_test.json`).

### High-throughput end-to-end pipeline (running)

- Added Stage 1 knobs `NUM_WORKERS` (dataloader, 8), `PATIENCE` (early stop, 15), `KEEP_LAST_EPOCHS` (2 — per-epoch checkpoints are ~470 MB each; disk guard). Verified 2 families co-located on the GPU reach **96–100% util at ~20 GB** (was ~22% single-family).
- `scripts/run_full_pipeline.sh`: Stage 1 finetune (`GPU_IDS=0,0`) → Stage 2 V2 full-family (train + Oracle eval) → `scripts/summarize_hitrates.py`. Console: `stage1/results/full_pipeline_console.log`; results: `stage1/results/full_pipeline/hitrate_summary.{txt,json}`.
- Hit rates computed = Stage 2 Oracle (system/context/route top-k, coverage, temperature) for all 10 families. Stage 1 route-recall eval (EditRetro generation) is a pending follow-up; this run produces the family checkpoints it will consume.
