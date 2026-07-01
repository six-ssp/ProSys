# ProSys Development Log

## 2026-06-30

### Decision

- Confirmed root repository as the only active Git repository for ProSys.
- Confirmed `stage2/stage2_detail.md` as the canonical Stage 2 requirement document.
- Confirmed old `rxn_yield_context` naming and import paths should be fully removed from active development code.

### Work

- Initialized project log and todo tracking.
- Started repository hygiene cleanup for cache files and Windows metadata residue.
- Started Stage 2 package-path migration from legacy `rxn_yield_context` imports to the current `stage2` package layout.
- Added `stage2.preprocess_data.sort_out_data` back into the active codebase through a new local parser implementation.
- Added the first executable Stage 2 V2 module set:
  - `stage2/v2/features.py`
  - `stage2/v2/product_memory.py`
  - `stage2/build_product_memory.py`
- Validated product-memory artifact generation on `Beckmann` using a temporary output directory.
- Removed the old Git remote and kept development local-only for the current repo state.
- Audited Stage 2 split integrity across all 10 families and confirmed `train/validate/test` are mutually disjoint.
- Found and fixed a Stage 1 leakage bug in `collect_stage1_route_data()` where extra Stage 1 train data failed to exclude the Stage 2 test split.
- Rebuilt the Stage 1 route exporter to emit `raw_train.csv`, `raw_val.csv`, and `raw_test.csv` using the corrected split logic.
- Added `scripts/audit_data_splits.py` with strict failure mode and verified:
  - Stage 2 splits are clean.
  - Stage 1 route datasets no longer overlap Stage 2 test reactions.
- Patched Stage 1 preprocessing to support the current `data/editretro/datasets/...` layout instead of the historical `stage1/datasets/...` assumption.
- Fixed the Stage 1 route rebuild to call `RXNMapper.get_attention_guided_atom_maps()` instead of the nonexistent `RXNMapper.run()`.
- Patched local `fairseq` compatibility issues with modern NumPy deprecations (`np.float`).
- Added the Stage 2 V2 executable module set for neural ranking:
  - `stage2/v2/constants.py`
  - `stage2/v2/candidate_pool.py`
  - `stage2/build_candidate_pool.py`
  - `stage2/v2/training_table.py`
  - `stage2/build_training_table.py`
  - `stage2/v2/dataset.py`
  - `stage2/v2/losses.py`
  - `stage2/v2/model.py`
  - `stage2/v2/trainer.py`
  - `stage2/train_stage2_v2.py`
- Smoke-tested the Stage 2 V2 path on `Beckmann`:
  - built product memory,
  - built Stage 2A candidate pools,
  - built Stage 2B training tables,
  - trained the new neural model for 1 epoch on a small real-data subset,
  - saved a working checkpoint at `/tmp/prosys_stage2_smoke/train/best_model.pt`.
- Relaunched the full local Stage 1 pipeline in the `rxn_yield_context` environment after the above fixes.
- Assessed the terminated Stage 1 pipeline:
  - full Stage 1 route rebuild completed successfully for all 10 families,
  - split audit stayed clean after the rebuild,
  - `Beckmann` preprocessing and `data-bin` generation completed successfully,
  - batch finetune stopped at the first `fairseq-train` because `fairseq.libnat` / `libnat_cuda` had not been built.
- Built `fairseq.libnat` in-place under `stage1/fairseq` and verified it imports correctly in the active environment.
- Confirmed the rebuilt Stage 1 route CSVs now contain mapped reactions:
  - `REAXYS_Beckmann_SINGLE_CATMERGE/raw/raw_train.csv` has 6604 / 6604 non-empty `mapped_reaction_smiles`,
  - mean mapping confidence is about `0.8024`.
- Found and fixed a concrete CPU-fallback bug in both EditRetro and bundled fairseq:
  - `_get_advanced_reposition_targets_cpu()` padded reposition labels to `out_seq_len`,
  - but downstream `_apply_reposition_words()` requires `in_seq_len`,
  - causing an immediate tensor-shape crash on CPU training.
- Added `stage1/scripts/ensure_fairseq_extensions.sh` and wired it into `run_family_finetune_batch.sh` so future finetune runs build/check `fairseq.libnat` before training.
- Added a Stage 1 runtime guard in `run_family_finetune_batch.sh`:
  - if `torch.cuda.is_available()` is false, the script now exits early by default,
  - CPU-only smoke runs require explicit `ALLOW_CPU_STAGE1=1`.
- Runtime assessment after the fixes:
  - the host has an RTX 4070 and NVIDIA driver available,
  - the current `rxn_yield_context` environment uses CPU-only PyTorch (`torch.version.cuda is None`),
  - there is no system CUDA toolkit / `nvcc`,
  - CPU-only smoke training no longer crashes immediately, but is too slow for practical 10-family finetuning.

### Notes

- Stage 2 V2 development will proceed directly against the current requirement document without re-running historical reproduction work.
- Stage 1 reverse-synthesis reproduction dependencies can remain deferred until later validation work.
- Full-family Stage 2A candidate-pool construction on real splits is functional but noticeably slower than the small-sample smoke test; keep this in mind when scheduling batch runs.
- For practical Stage 1 training, the next environment step is to provide CUDA-capable PyTorch plus CUDA toolkit / `nvcc`, then rebuild `fairseq/libnat_cuda` and rerun family finetuning.

## 2026-07-01

### Work

- Switched the active runtime target to the local `ProSys` environment and audited its current package state.
- Confirmed the `ProSys` environment already contains the core chemistry and Stage 2 dependencies:
  - `rdkit`
  - `rxnmapper`
  - `textdistance`
  - `selfies`
  - `SmilesPE`
  - `fairseq.libnat`
- Found the `ProSys` environment in a mixed PyTorch state:
  - CPU-only `pytorch 2.5.1` from conda was still taking precedence,
  - CUDA runtime packages and older pip-installed CUDA wheels were also present,
  - `torch.cuda.is_available()` remained false before repair.
- Added unified runtime and batch entrypoints:
  - `scripts/check_runtime.py`
  - `scripts/setup_prosys_env.sh`
  - `stage1/scripts/run_family_finetune_one.sh`
  - upgraded `stage1/scripts/run_family_finetune_batch.sh`
  - upgraded `scripts/run_stage2_v2_family_batch.py`
- The Stage 2 batch runner now supports:
  - family filtering,
  - artifact reuse,
  - parallel candidate-pool preprocessing,
  - multi-device family training scheduling,
  - per-family training logs.
- Updated `README.md` to make the unified local workflow the default operating path.

### Notes

- A forced CUDA-wheel reinstall for `torch` / `torchvision` / `torchaudio` in `ProSys` was started; final GPU verification is pending completion of that install.
- Until the repaired `torch` state is verified, Stage 1 full-family GPU finetuning remains blocked by runtime rather than by data or code.

### Follow-up

- Mirror-based reinstall was attempted next, but the current shell had outbound package access blocked by the local proxy / sandbox boundary:
  - direct Tsinghua / USTC pip mirror checks failed before package resolution,
  - the active shell could not write back into `/home/six_ssp/miniconda3/envs/ProSys`.
- Switched to a workspace-local fallback instead of modifying the locked base env:
  - extracted cached local conda packages into `runtime/overlay_cuda121`,
  - added `scripts/prepare_prosys_cuda121_overlay.sh`,
  - added `scripts/run_in_prosys_cuda121_overlay.sh`,
  - added `scripts/libittnotify_stub.c` and built a no-op `libittnotify.so` shim for the missing `iJIT_*` symbols.
- Verified the overlay runtime can now import a consistent CUDA build stack:
  - `torch 2.5.1`
  - `torchvision 0.20.1`
  - `torchaudio 2.5.1`
  - `torch.backends.cuda.is_built() == True`
  - `torch.version.cuda == 12.1`
- Remaining blocker is no longer the Python package stack:
  - inside the current sandbox, `nvidia-smi` reports `GPU access blocked by the operating system`,
  - `torch.cuda.is_available()` stays false because the shell cannot access GPU device nodes,
  - full GPU training validation must be rerun from a shell with actual `/dev/nvidia*` visibility.
