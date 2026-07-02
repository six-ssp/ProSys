# ProSys TODO

## Active

- End-to-end Oracle run COMPLETE: 10/10 Stage 1 route models + 10/10 Stage 2 Oracle hit rates (`stage1/results/full_pipeline/hitrate_summary.txt`).
- Wire the family-specific Stage 2A FNN branch (`--fnn_checkpoint_pattern`) to raise candidate-pool coverage (currently product-memory only).
- Build the Stage 1 route cache from the finished family models, then run Non-Oracle Stage 2 eval (`stage2/evaluate_stage2_v2.py --mode non_oracle`) for true end-to-end hit rates.
- Wire a Stage 1 route-recall evaluator (product→reactants top-k via EditRetro generation).
- Investigate DielsAlder temperature head (MAE 56 °C, only 10% within ±20 °C).


## Done (summary)

- Environment on autodl: rebuilt `ProSys` conda env (clone of `retro_gan`), built fairseq `libnat`/`libnat_cuda` (sm_86), fixed torch≥2.6 checkpoint loading; `check_runtime.py` + split audit green.
- Stage 1: restored the base checkpoint, added throughput knobs (patience / dataloader workers / capped checkpoints), verified 2 families saturate the GPU, launched the 10-family finetune.
- Stage 2 V2: `import stage2` bootstrap fix, ~7x faster candidate-pool build (per-product context reuse, verified identical output), Oracle/Non-Oracle evaluation entry + auto-eval from the batch runner.
- Tooling: `scripts/run_full_pipeline.sh` (end-to-end orchestrator) and `scripts/summarize_hitrates.py`.
- Earlier: migrated off `rxn_yield_context` to the `stage2` package; canonical spec `stage2/stage2_detail.md`; fixed Stage 1 split leakage + rebuilt route exports; added unified runtime/audit/runner entrypoints.
