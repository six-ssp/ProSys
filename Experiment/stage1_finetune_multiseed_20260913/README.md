# Fixed-base expert fine-tuning: three training seeds

Status: paused pending the validation-overlap protocol decision. The first
Beckmann seed-0 run was intentionally interrupted after the epoch-22 checkpoint
when the independent USPTO audit found 98 condition-validation overlaps.
Best and last are retained; no new completed replicate or aggregate is claimed.
See `../project_completion_20260913/FINDINGS.md`.

This study repeats family expert fine-tuning, not USPTO base pretraining.
Seeds 0/1/2 all start from the same frozen base checkpoint in the current
environment. Existing Stage-1 caches remain the fixed inputs to published
Stage-2/3 controls. No downstream comparison is silently replaced.

The existing augmented and binarized data are reused byte-for-byte. Inputs,
base checkpoint, training sources and original logged hyperparameters are
hashed per job. Training seed changes; validation noise seed stays 7 and
decoding configuration stays fixed. Dataloader workers are bounded at 2 for
the current 14-core allocation. Beckmann uses 8192 tokens/batch, the other five
families 16384, matching their original runs. LR 0.0003, warmup 10000, Adam,
FP16, maximum 200 epochs / 200000 updates, validation-loss patience 15.

Each job retains best and last only. Checkpoints and licensed route-level
outputs are local-only; aggregate metrics and provenance are reportable.
Training success is checked from the terminal log and explicit argument
comparison, not the mere presence of a checkpoint. Query identities must
match the original family evaluation manifest before a result is accepted.

Run: `python scripts/run_stage1_multiseed.py` in the project environment.
Use `--families` / `--seeds` for a disjoint subset. Completed outputs are
hash-checked before reuse; changed inputs and partial failed training are
rejected rather than silently mixed into a repeat.
