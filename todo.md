# ProSys TODO

## Active

- Full pipeline RUNNING in background (`scripts/run_full_pipeline.sh`): Stage 1 finetune (2 families co-located on the GPU) → Stage 2 V2 full-family (train + Oracle eval) → hit-rate summary. Watch `stage1/results/full_pipeline_console.log`; results at `stage1/results/full_pipeline/hitrate_summary.{txt,json}`.
- Wire Stage 1 route-recall evaluation (product→reactants top-k via EditRetro generation + aggregation); the current finetune produces the family checkpoints it needs.
- After Stage 1 route models exist: build the Stage 1 route cache, then run Non-Oracle Stage 2 eval (`stage2/evaluate_stage2_v2.py --mode non_oracle`).
- Train / wire family-specific Stage 2A FNN checkpoints and pass `--fnn_checkpoint_pattern` so candidate pools include the FNN branch (currently product-memory only).

## Done (summary)

- Environment on autodl: rebuilt `ProSys` conda env (clone of `retro_gan`), built fairseq `libnat`/`libnat_cuda` (sm_86), fixed torch≥2.6 checkpoint loading; `check_runtime.py` + split audit green.
- Stage 1: restored the base checkpoint, added throughput knobs (patience / dataloader workers / capped checkpoints), verified 2 families saturate the GPU, launched the 10-family finetune.
- Stage 2 V2: `import stage2` bootstrap fix, ~7x faster candidate-pool build (per-product context reuse, verified identical output), Oracle/Non-Oracle evaluation entry + auto-eval from the batch runner.
- Tooling: `scripts/run_full_pipeline.sh` (end-to-end orchestrator) and `scripts/summarize_hitrates.py`.
- Earlier: migrated off `rxn_yield_context` to the `stage2` package; canonical spec `stage2/stage2_detail.md`; fixed Stage 1 split leakage + rebuilt route exports; added unified runtime/audit/runner entrypoints.
