# ProSys Development Log

## 2026-07-05 — stage23 mainline promotion + repo cleanup

### Mainline decision

- Promoted `KNN + XGBoost` to the maintained mainline for post-Stage-1 comparison.
- Code split is now explicit:
  - `stage2_KNN/` handles feasible-condition screening
  - `stage3_XGBoost/` handles reranking and temperature prediction
- The old neural-V2 path is retained only for historical compatibility and the legacy baseline.

### Non-Oracle stage23 experiment suite

- Completed the 10-family Non-Oracle Stage 2 / Stage 3 suite under:
  - `outputs/stage23_non_oracle_all10/`
- Final report roots now include:
  - `overview.md`
  - `baseline_historical.md`
  - `ablation_stage2.md`
  - `ablation_stage3.md`
  - `average_effect.md`

### Temperature metric cleanup

- Temperature reporting was changed from raw error-only headline usage to hit-rate style summaries:
  - `Temp@10C`
  - `Temp@20C`
- Then unified again so both metrics are now computed as **top-10 end-to-end temperature hits**:
  - within top-10 there exists a full system hit
  - and its temperature error is within `+/-10C` or `+/-20C`
- This makes the temperature metric a true joint route+condition+temperature success rate.

### Archive / maintenance cleanup

- Removed repo-wide `__pycache__/`.
- Retired the root `stage2/` package:
  - active shared utilities were extracted to `prosys_shared/`
  - legacy neural-V2 / FNN runtime was archived to `Experiment/legacy_stage2/`
  - current mainline and legacy entrypoints were rewired to the new locations
- Deleted redundant smoke outputs:
  - `outputs/baselines_smoke`
  - `outputs/baselines_smoke2`
  - `outputs/stage23_non_oracle_smoke_fischer`
  - `outputs/stage23_non_oracle_smoke_fischer_v2`
- Moved non-mainline but still valuable material into `Experiment/`:
  - exploratory notebooks
  - old oracle / non-oracle baseline output trees
  - route-budget sensitivity results
  - obsolete render scripts
  - one-off helper utilities
- Added:
  - `README.md` rewrite for the current structure
  - `MAINTENANCE.md`
  - `Experiment/README.md`
  - `scripts/run_stage23_non_oracle_suite.sh`

### Current status

- Active result roots:
  - `outputs/stage1_routes/`
  - `outputs/stage23_non_oracle_all10/`
- Active mainline:
  - `stage1 -> stage2_KNN -> stage3_XGBoost`
- Historical baseline:
  - `Original FNN`
- Legacy experiment branch:
  - `stage2/` + `save_models/` + old baseline wrappers

## 2026-07-05 — Non-Oracle official refresh + summary finalization

### Reaudit and official output refresh

- Rechecked the post-Stage-1 Non-Oracle pipeline with `scripts/audit_data_splits.py --strict`.
- Confirmed no obvious split leakage in the current Stage 2 / Stage 3 setup.
- Archived the stale old official Non-Oracle tree and regenerated:
  - `outputs/stage23_non_oracle_all10/`
- Added the audit note:
  - `baseline/non_oracle_reaudit_20260705.md`

### Legacy FNN runtime fixes

- The main bottleneck in the refresh run was the historical FNN / FNN-pool path.
- Added engineering-only speedups so the official 10-family rerun could finish:
  - `torch.inference_mode()`
  - GPU-enabled legacy inference via `PROSYS_LEGACY_DEVICE`
  - faster `np.stack + torch.from_numpy` tensor materialization
  - top-k-only legacy context ranking
  - long-run progress logging
  - invalid `OMP_NUM_THREADS` sanitization

These changes were made to improve reproducibility and runtime, not to change
the evaluation definition.

### Final official Non-Oracle outputs

- Regenerated and re-rendered:
  - `overview.md`
  - `baseline_historical.md`
  - `ablation_stage2.md`
  - `ablation_stage3.md`
  - `average_effect.md`
  - `results_flat.csv`
- Final result count:
  - `10 families x 7 methods = 70 result.json`

### Main numerical readout

- Full 10-family `KNN+XGB` macro:
  - `cover 36.7`
  - `sys@1 15.4`
  - `sys@5 23.6`
  - `sys@10 27.0`
  - `Temp@10C 9.2`
  - `Temp@20C 14.5`
- Filtered main table (`KNN+XGB sys@10 > 20%`) keeps 6 families:
  - `Beckmann`
  - `Buchwald-Hartwig`
  - `Chan-Lam`
  - `Diels-Alder`
  - `Friedel-Crafts Acyl.`
  - `Friedel-Crafts Alkyl.`
- On that filtered subset:
  - `Original FNN`: `cover 38.0`, `sys@10 28.3`, `Temp@10C 20.0`, `Temp@20C 24.1`
  - `KNN+XGB`: `cover 52.2`, `sys@10 38.6`, `Temp@10C 13.1`, `Temp@20C 20.9`

### Final interpretation

- `KNN+XGB` remains the chosen mainline because it improves end-to-end system
  hit over the historical baseline while keeping the staged story clear:
  - `KNN` screens feasible conditions
  - `XGBoost` reranks and predicts temperature
- `KNN+SVM` is stronger on raw filtered-subset `sys@k`, but its temperature hit
  rates collapse under the current conditioned metric, so it is not the preferred
  full-system mainline.

### Disk cleanup after the refresh

- Deleted repo-wide `__pycache__/` and `*.pyc`.
- Removed temporary per-turn summary files.
- Removed the extra stale archived Non-Oracle tree created only for the refresh.
- Disk availability recovered from near-full to a safe working margin, allowing
  the final markdown summaries to be written successfully.

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

### Route-budget experiment (2026-07-02)

- Goal: test a low-risk Stage 1-side optimization before retraining anything. Hypothesis: for weak coupling families, keeping more unique routes per product in the Stage 1 route cache might lift Non-Oracle coverage and end-to-end hits.
- Tooling change 1: `stage1/build_route_cache.py` now accepts `--generation_file` together with `--skip_generation`, so an existing `generation.txt` can be re-aggregated with a different `n_best` / ranking budget without rerunning fairseq generation.
- Tooling change 2: `scripts/run_stage2_v2_non_oracle.py` now accepts `--result_root`, so experimental Non-Oracle candidate tables / eval JSONs can be written to a separate tree without overwriting the baseline under `outputs/stage2_v2/`.
- Experiment setup: reused the existing Stage 1 `generation.txt` for `KumadaCoupling` and `NegishiCoupling`, rebuilt route caches with `n_best=20` (baseline `n_best=10`), and re-ran Non-Oracle evaluation into `outputs/stage2_v2_routebudget_n20/`.
- Stage 1 route-recall signal:
  - `KumadaCoupling`: route recall@10 unchanged at **3.47%**, route recall@20 rose to **3.71%**.
  - `NegishiCoupling`: route recall@10 unchanged at **14.10%**, route recall@20 rose to **15.75%**.
- Stage 2 / end-to-end effect:
  - `KumadaCoupling`: candidate rows **506,346 -> 632,419** (+24.9%), pool coverage **3.16% -> 3.41%**, but system top-10 stayed **2.55% -> 2.55%** and temperature MAE worsened slightly (**26.04 -> 26.79 °C**).
  - `NegishiCoupling`: candidate rows **200,769 -> 349,291** (+74.0%), pool coverage **11.17% -> 12.33%**, but system top-10 stayed **9.17% -> 9.17%**; temperature MAE improved only marginally (**26.85 -> 26.69 °C**).
- Conclusion: simply increasing the Stage 1 route-cache budget is not enough to move end-to-end top-10 on the weak coupling families. It does recover a few extra gold routes and improves pool coverage, but the gain is too small relative to the much larger candidate tables. Keep `n_best>10` as an experiment knob, not the new default. The next meaningful levers remain better Stage 1 route quality/ranking and the deferred FNN candidate branch.

### Baseline planning for post-Stage-1 comparison (2026-07-03)

- Added [`baseline/baseline.md`](/root/autodl-tmp/ProSys/baseline/baseline.md) as the working plan for the new baseline suite focused on modules **after Stage 1**.
- The plan explicitly separates comparison into:
  - Stage 2A candidate-pool screening
  - Stage 2B reranking
- Defined the main baseline lines to implement next:
  - historical legacy pipeline: `MultiTask_Evaluator + ReactionModel_LWTemp`
  - `XGBoost` reranker on unified candidate tables
  - ML `KNN` candidate-pool baseline
  - ML cluster-retrieval candidate-pool baseline
- Locked the recommended rollout order: reproduce legacy baseline first, unify candidate-table format second, then add `XGBoost`, then add the `KNN` and cluster candidate-pool baselines, and only after Oracle is stable move to Non-Oracle.

### Oracle baseline runner + first results (2026-07-03)

- Added baseline runtime files under `baseline/`:
  - `common.py`
  - `legacy_models.py`
  - `run_oracle_baselines.py`
- The new runner supports:
  - `legacy_rank`: legacy FNN pool + legacy `ReactionModel_LWTemp`
  - `v2_xgb`: current V2 candidate table + `XGBRanker`
  - `knn_xgb`: reaction-similarity KNN candidate pool + `XGBRanker`
  - `cluster_xgb`: reaction-cluster candidate pool + `XGBRanker`
- `legacy_models.py` wraps the old checkpoints with `torch.load(weights_only=False)` so they remain loadable under torch>=2.6.
- `XGBRanker` training initially failed because row-level `sample_weight` was passed in ranking mode; fixed by removing row-wise weights from the XGBoost fit path.
- First Oracle baseline runs completed:
  - 4-family `legacy_rank` vs `v2_xgb`: `outputs/baselines/oracle_rep4_lv/oracle_baseline_summary.{txt,json}`
  - `Beckmann` full smoke (`legacy_rank`, `v2_xgb`, `knn_xgb`, `cluster_xgb`): `outputs/baselines/current_turn_summary.{txt,json}`
  - `FischerIndoleSynthesis` `knn_xgb` + `cluster_xgb`: `outputs/baselines/oracle_fischer_kc/oracle_baseline_summary.{txt,json}`
- Readout so far:
  - `v2_xgb` consistently beats the legacy baseline on coverage and top-10 for `Buchwald`, `Friedel-CraftsAcylation`, and `Negishi`.
  - `legacy_rank` is unexpectedly very strong on `Kumada` Oracle (`sys@10 81.8` vs `v2_xgb 68.7`), so that family deserves a closer audit before drawing conclusions.
  - `knn_xgb` is stronger than `cluster_xgb` on both completed families (`Beckmann`, `FischerIndoleSynthesis`), matching the expectation that local-neighbor retrieval is the stronger simple screening baseline.
  - On `Beckmann`, the global legacy candidate model never retrieves the gold context (`coverage 0.0`), which highlights a domain mismatch risk of the old global-label-space baseline.
- Practical bottleneck found:
  - `knn_xgb` on large families is slow when building full train-split candidate tables.
  - Added `--max_train_routes` / `--max_val_routes` to the runner so first-pass Oracle sweeps can stay tractable while still using the full train split as retrieval memory.

### Non-Oracle baseline runner + first results (2026-07-04)

- Added `baseline/run_non_oracle_baselines.py`, which reuses Stage 1 `route_cache.json` files and supports:
  - `v2_neural_ref` (read-only current mainline reference from `outputs/stage2_v2/*/non_oracle/eval_non_oracle_test.json`)
  - `v2_xgb`
  - `legacy_rank`
  - `knn_xgb`
  - `cluster_xgb`
- The runner trains `XGBRanker` on Oracle train/val tables and scores Non-Oracle test candidate tables built from Stage 1 route caches.
- Full-family Non-Oracle `v2_neural_ref` vs `v2_xgb` completed:
  - summary: `outputs/baselines/non_oracle_v2xgb_all/non_oracle_baseline_summary.{txt,json}`
  - delta table: `outputs/baselines/non_oracle_v2xgb_all/v2xgb_vs_neural_delta.txt`
- Readout:
  - `v2_xgb` is almost identical to the current neural mainline on some families (`Beckmann`, `Grignard`, `Kumada`, `Negishi`, `Friedel-*`).
  - But it is dramatically better on some families:
    - `Chan_LamCoupling`: `sys@10 20.2 -> 57.7`
    - `DielsAlder`: `17.3 -> 26.1`
    - `FischerIndoleSynthesis`: `2.9 -> 15.4`
  - This means the “same Stage 1 routes + same candidate pool + different ranker” comparison is now clearly visible end-to-end, and in several families the tabular reranker is the stronger choice.
- Additional completed Non-Oracle baselines:
  - `NegishiCoupling` `legacy_rank` vs `v2_xgb`: `outputs/baselines/non_oracle_smoke_negishi/non_oracle_baseline_summary.{txt,json}`
    - `legacy_rank` stays clearly below `v2_xgb` on end-to-end system hit (`sys@10 4.8` vs `9.2`).
  - `FischerIndoleSynthesis` `knn_xgb` + `cluster_xgb`: `outputs/baselines/non_oracle_fischer_kc/non_oracle_baseline_summary.{txt,json}`
    - `knn_xgb` > `cluster_xgb`
    - both beat the current neural ref on end-to-end `sys@10`
- New bottleneck localized:
  - large-family `knn_xgb` / `cluster_xgb` Non-Oracle runs slow down in `write_candidate_training_table`, because Stage 1 predicts many malformed reactant routes and the label-build path repeatedly canonicalizes them through RDKit.
  - This is now a concrete optimization target for the next pass (safe route canonicalization / caching / filtering invalid routes earlier).

### Non-Oracle baseline optimization + compact tables (2026-07-04)

- Replaced the baseline label-build path with a baseline-local implementation in [`baseline/common.py`](/root/autodl-tmp/ProSys/baseline/common.py):
  - caches `product`, `reactants`, and condition canonicalization by unique value instead of by row
  - reuses any existing `product_canonical` / `route_canonical` columns
  - suppresses RDKit parse-error spam for malformed Stage 1 reactants
- Quick timing check on previously problematic Non-Oracle candidate tables:
  - `Beckmann knn_xgb` test table label-build: about **4.8 s**
  - `NegishiCoupling knn_xgb` test table label-build: about **7.0 s**
- With the faster labeler, completed the previously stuck larger-family Non-Oracle screening baselines for:
  - `Beckmann`: `knn_xgb`, `cluster_xgb`
  - `NegishiCoupling`: `knn_xgb`, `cluster_xgb`
  - output root: `outputs/baselines/non_oracle_beckmann_negishi_full/`
- Added [`baseline/render_non_oracle_tables.py`](/root/autodl-tmp/ProSys/baseline/render_non_oracle_tables.py) to merge one or more Non-Oracle result trees and render compact family tables in the same style as the end-to-end hit-rate summary.
- Final merged compact tables written to:
  - `outputs/baselines/non_oracle_final_tables/v2_neural_ref.txt`
  - `outputs/baselines/non_oracle_final_tables/v2_xgb.txt`
  - `outputs/baselines/non_oracle_final_tables/knn_xgb.txt`
  - `outputs/baselines/non_oracle_final_tables/cluster_xgb.txt`
  - `outputs/baselines/non_oracle_final_tables/legacy_rank.txt`
- Readout from the newly completed larger-family screening baselines:
  - `Beckmann`: `knn_xgb` is close to the current mainline end-to-end (`sys@10 26.4` vs `26.7`), while `cluster_xgb` drops sharply (`18.6`)
  - `NegishiCoupling`: `knn_xgb` slightly beats the current mainline (`sys@10 10.2` vs `9.2`) and clearly beats `cluster_xgb` (`3.2`)
  - Across all completed Non-Oracle screening families so far (`Beckmann`, `FischerIndoleSynthesis`, `NegishiCoupling`), `knn_xgb` is consistently the strongest simple self-built screening baseline.

### Non-Oracle baseline suite completed on all 10 families (2026-07-04)

- Finished the remaining 10-family Non-Oracle sweeps for:
  - `knn_xgb`
  - `cluster_xgb`
  - `legacy_rank`
- Main output roots:
  - `outputs/baselines/non_oracle_all10_screening/`
  - `outputs/baselines/non_oracle_all10_legacy/`
- Re-rendered the merged final family tables at:
  - `outputs/baselines/non_oracle_final_tables/v2_neural_ref.txt`
  - `outputs/baselines/non_oracle_final_tables/v2_xgb.txt`
  - `outputs/baselines/non_oracle_final_tables/knn_xgb.txt`
  - `outputs/baselines/non_oracle_final_tables/cluster_xgb.txt`
  - `outputs/baselines/non_oracle_final_tables/legacy_rank.txt`
- Final 10-family macro readout (`sys@1 / sys@5 / sys@10`):
  - `v2_neural_ref`: `10.3 / 17.8 / 22.1`
  - `v2_xgb`: `15.2 / 24.8 / 27.9`
  - `knn_xgb`: `16.1 / 24.3 / 27.7`
  - `legacy_rank`: `7.8 / 14.1 / 20.3`
  - `cluster_xgb`: `2.7 / 7.5 / 10.7`
- Key takeaways:
  - `v2_xgb` is the best overall reranking baseline on the full 10-family Non-Oracle benchmark, narrowly ahead of `knn_xgb` on macro `sys@10`.
  - `knn_xgb` is the strongest self-built screening baseline and is very competitive with the mainline `v2_xgb`; it is especially strong on `Chan_Lam`, `DielsAlder`, `Friedel-*`, and `Negishi`.
  - `cluster_xgb` is consistently weaker than `knn_xgb`, indicating that coarse cluster memory loses too much route-level specificity.
  - `legacy_rank` remains a meaningful historical reference but is clearly below both `v2_xgb` and `knn_xgb` end-to-end, and completely fails on `Beckmann`.
