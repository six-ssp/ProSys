# Legacy Stage 2 Launchers

These scripts reproduce the archived neural-V2 Stage 2 workflow and are not
called by the maintained parallel KNN + ReaFNN pipeline.

- `run_full_pipeline.sh`: Stage 1 family fine-tuning plus V2 oracle evaluation.
- `run_non_oracle_pipeline.sh`: V2 evaluation over persisted Stage 1 routes.
- `run_stage2_v2_family_batch.py`: V2 memory/training/evaluation batch runner.
- `run_stage2_v2_non_oracle.py`: V2 non-oracle evaluator.
- `summarize_hitrates.py`: V2 hit-rate summarizer.

Run the shell launchers from the repository root or pass the repository root
as their optional first argument.
