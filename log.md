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

### Stage 1 crash fix (only Beckmann had trained)

- Symptom check: GPU looked idle and logs looked stale mid-run. Root cause: the Stage 1 batch **aborted 6 s in** (`[pipeline] Stage 1 finished exit=1`). `stage1/preprocess/preprocess_data.py` crashed on Buchwald with `AttributeError: 'float' object has no attribute 'split'` — the chosen `mapped_reaction_smiles` column had 64 NaN rows (0.4%, failed atom-mapping), and the `x.split('>')` maps hit a float NaN. Under the batch script's `set -e`, that non-zero exit killed the whole run; Beckmann (already binarized) kept training as an orphan to a clean early-stop (epoch 74), so only Beckmann got a Stage 1 model. Beckmann had 0 NaN, which is why it alone survived.
- Fix: `preprocess_data.py` now drops rows whose reaction string is missing/blank or lacks the full `reactants>reagents>product` form before the `>` splits. Verified Buchwald binarizes past the old crash point.
- Added a `FAMILIES` env override to `run_family_finetune_batch.sh` (space/comma separated dataset names; defaults to all ten) so a subset can be (re)run without editing the script.
- Relaunched Stage 1 for the 9 missing families (`FAMILIES=<9> GPU_IDS=0,0 MAX_TOKENS=12288`), co-scheduled with the still-running Stage 2 pipeline (Stage 1 = GPU, Stage 2 = mostly CPU); GPU memory stays safe (~9–20 GB of 24). Log: `stage1/results/stage1_rerun_9fam.log`.
- Note: the Stage 2 Oracle hit-rate pipeline is independent of Stage 1 models (it uses gold routes), so it kept producing per-family Oracle results throughout. Its low GPU use is expected (tiny MLP + `num_workers=0` dataloader); the "stale log" was Python stdout block-buffering, not a hang.


### Two more Stage 1 robustness fixes (second rerun)

- The 9-family rerun still aborted after 4 families: `preprocess_data.py` crashed on Friedel-CraftsAcylation with `'NoneType' object has no attribute 'GetAtoms'`. Root cause: the quality checks set `status="invalid_r"/"invalid_p"` when `MolFromSmiles` returns None, but then called `rea_mol.GetAtoms()` / `pro_mol.GetAtoms()` unconditionally on the next line — crashing before the guard mattered. Fixed by short-circuiting (`elif`) so GetAtoms is only called on parsed mols.
- Systemic fix: `run_family_finetune_batch.sh` used `set -euo pipefail` + `wait -n`, so any one family's non-zero exit aborted the whole batch AND killed its concurrent GPU sibling (this is why Buchwald had a Stage 2 model but no Stage 1 checkpoint — it was killed mid-train when Friedel-Acylation crashed). Rewrote the reaping loop to track PID→dataset, tolerate per-family failures, and print a `batch done: N ok, M failed` summary (bash-5.0 compatible). Verified with a mock (5 jobs, 1 failing → all others still complete).
- Relaunched the 6 still-missing families with both fixes; confirmed Friedel-CraftsAcylation now binarizes and trains (98% GPU). Log: `stage1/results/stage1_rerun_6fam.log`.

### Full run complete (2026-07-02)

- **Stage 1: 10/10 family route models trained** (resilient batch: "6 ok, 0 failed"). Early-stop epochs 40–74, best val ppl 5.7–20.5. Checkpoints under `stage1/results/family_finetune/<family>/<ts>/checkpoints/checkpoint_best.pt`.
- **Stage 2: Oracle hit rates computed for all 10 families.** Macro-avg: coverage 85.9%, sys@1 41.9%, sys@3 58.6%, sys@5 65.2%, sys@10 73.0%, temp MAE 27.2 °C. Full table + per-family in `stage1/results/full_pipeline/hitrate_summary.{txt,json}`.
- Both preprocess crashes (NaN reaction row; None-mol GetAtoms) and the fatal `set -e` batch-abort are fixed and committed, so the family finetune is now robust end-to-end.
- Follow-ups (not blocking): FNN candidate branch (`--fnn_checkpoint_pattern`) to lift Stage 2A coverage; Stage 1 route cache → Non-Oracle eval; investigate DielsAlder temperature (MAE 56 °C).

### End-to-end Non-Oracle run complete (2026-07-02)

- Built the full Non-Oracle chain and ran it for all 10 families: Stage 1 EditRetro route generation (fairseq-interactive TTA) → route cache → Stage 2A candidate pool (product-memory) → Stage 2B ranking with the Oracle-trained checkpoints → end-to-end hit rates. Route caches: 136–1183 reactions/family with ≥1 predicted route (most families 95–100%; DielsAlder 851/884).
- Bug found + fixed (Non-Oracle only): Stage 1 sometimes predicts an unparseable reactant SMILES (e.g. malformed organometallic `[Li]<-[c]1cccc1`). `reaction_morgan_fp` raised `ValueError`, which crashed NegishiCoupling's whole eval (it silently produced no result the first pass). Fixed to return a zero route-fp for unparseable reactants (a hallucinated route becomes a zero-signal candidate, not a fatal error). Also made `run_stage2_v2_non_oracle.py` catch/report per-family failures instead of dropping them silently.
- **Non-Oracle macro-avg**: Stage 1 route recall @1/@10 = 28.5/41.9%, pool coverage 37.5%, end-to-end system top-1/3/5/10 = 10.3/15.3/17.8/22.1%, temp MAE 26.3 °C. Full per-family Oracle vs Non-Oracle table in `stage1/results/full_pipeline/hitrate_summary.{txt,json}`.
- Reading: Non-Oracle is gated by Stage 1 route recall (coverage ≈ route recall, since Stage 2A candidates hang off predicted routes). Strong: Buchwald (sys@10 50.8), Friedel-Acylation (39.7), Friedel-Alkylation (36.9). Weak couplings: Kumada (sys@10 2.6, route recall @10 only 3.5%) and Negishi (9.2) — Stage 1 barely recovers their gold routes. This localizes the biggest end-to-end lever to Stage 1 route recall for the coupling families.
