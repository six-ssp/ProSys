# ProSys TODO

## Active

- End-to-end run COMPLETE (Oracle + Non-Oracle) for all 10 families: `stage1/results/full_pipeline/hitrate_summary.{txt,json}`. Non-Oracle macro-avg: route recall@10 41.9%, system top-10 22.1%.
- Biggest end-to-end lever is **Stage 1 route recall for the coupling families** (Kumada route recall@10 3.5%, Negishi 14.1%) — investigate before further Stage 2 tuning.
- Route-budget sweep completed for weak coupling families: reusing existing `generation.txt`, raising Stage 1 cache `n_best` from 10 -> 20 increased Non-Oracle candidate counts / pool coverage on Kumada and Negishi, but **did not improve system top-10**. Treat larger route budgets as an analysis knob, not the new default.
- Optimize Stage 2 (was deferred): wire the family FNN candidate branch (`--fnn_checkpoint_pattern`) to lift pool coverage where route recall is already acceptable; consider Non-Oracle finetuning of the Stage 2 ranker after the Stage 1 bottleneck is addressed.
- Investigate DielsAlder temperature head (Oracle MAE 56 °C; Non-Oracle 50 °C).


## Done (summary)

- Environment on autodl: rebuilt `ProSys` conda env (clone of `retro_gan`), built fairseq `libnat`/`libnat_cuda` (sm_86), fixed torch≥2.6 checkpoint loading; `check_runtime.py` + split audit green.
- Stage 1: restored the base checkpoint, added throughput knobs (patience / dataloader workers / capped checkpoints), verified 2 families saturate the GPU, launched the 10-family finetune.
- Stage 2 V2: `import stage2` bootstrap fix, ~7x faster candidate-pool build (per-product context reuse, verified identical output), Oracle/Non-Oracle evaluation entry + auto-eval from the batch runner.
- Experiment tooling: Stage 1 route-cache rebuild now supports reusing an existing `generation.txt`; Non-Oracle evaluation now supports writing experiment results to a separate `result_root` without overwriting the baseline artifacts.
- Tooling: `scripts/run_full_pipeline.sh` (end-to-end orchestrator) and `scripts/summarize_hitrates.py`.
- Earlier: migrated off `rxn_yield_context` to the `stage2` package; canonical spec `stage2/stage2_detail.md`; fixed Stage 1 split leakage + rebuilt route exports; added unified runtime/audit/runner entrypoints.
