# ProSys TODO

## Active

- End-to-end run COMPLETE (Oracle + Non-Oracle) for all 10 families: `stage1/results/full_pipeline/hitrate_summary.{txt,json}`. Non-Oracle macro-avg: route recall@10 41.9%, system top-10 22.1%.
- Biggest end-to-end lever is **Stage 1 route recall for the coupling families** (Kumada route recall@10 3.5%, Negishi 14.1%) — investigate before further Stage 2 tuning.
- Optimize Stage 2 (was deferred): wire the family FNN candidate branch (`--fnn_checkpoint_pattern`) to lift pool coverage; consider Non-Oracle finetuning of the Stage 2 ranker.
- Investigate DielsAlder temperature head (Oracle MAE 56 °C; Non-Oracle 50 °C).


## Done (summary)

- Environment on autodl: rebuilt `ProSys` conda env (clone of `retro_gan`), built fairseq `libnat`/`libnat_cuda` (sm_86), fixed torch≥2.6 checkpoint loading; `check_runtime.py` + split audit green.
- Stage 1: restored the base checkpoint, added throughput knobs (patience / dataloader workers / capped checkpoints), verified 2 families saturate the GPU, launched the 10-family finetune.
- Stage 2 V2: `import stage2` bootstrap fix, ~7x faster candidate-pool build (per-product context reuse, verified identical output), Oracle/Non-Oracle evaluation entry + auto-eval from the batch runner.
- Tooling: `scripts/run_full_pipeline.sh` (end-to-end orchestrator) and `scripts/summarize_hitrates.py`.
- Earlier: migrated off `rxn_yield_context` to the `stage2` package; canonical spec `stage2/stage2_detail.md`; fixed Stage 1 split leakage + rebuilt route exports; added unified runtime/audit/runner entrypoints.
