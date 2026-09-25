# USPTO-50K Base From Scratch

Decision: confirmed by the user on 2026-09-24. Replace the planned FULL base
with a filtered USPTO-50K base trained from random initialization. No pretrained
network checkpoint is restored. This is a new study, not renamed FULL results.
The local `completion.json` certifies normal base training completion. Experts
and downstream studies are separately certified and are now complete; see
`../release_50k_20260925/README.md`. Raw inputs and detailed receipts remain local.

Current expert handoff supersedes the initial commands recorded below:
`../stage1_50k_fidelity_v2_expert_multiseed_20260924/` uses the additional
augmentation-fidelity repair and guarded decoding. Its 18 experts are complete;
do not launch the historical handoff commands below. The base itself is
unchanged and passed the raw-to-augmentation fidelity audit. Current expert
train/validation totals are 846,045 / 39,375 augmented pairs. Initial admission
counts and interrupted queue paths below remain historical provenance only.

## Completed Base Training

The base finished normally on 2026-09-24 at 13:32:48 UTC. The exit status is 0
and `completion.json` is present; source, log and checkpoint hashes have been
rechecked. The stop reason is the 50-epoch cap, not validation early stopping
or the 200,000-update ceiling. GPU training took 8,112.3 seconds (about 2 h 15 m),
excluding preparation and audits.

| Checkpoint | Epoch | Updates | Validation loss | Bytes |
| --- | ---: | ---: | ---: | ---: |
| best | 49 | 16,170 | 3.573 | 470,674,731 |
| last | 50 | 16,500 | 3.701 | 470,674,731 |

Values were read from the actual checkpoint metadata, not inferred only from
filenames. `training_summary.json` contains full SHA256 digests and the bound
completion receipt. Both best/last remain local; no old alias was overwritten.
Expert admission and all six-family three-seed fits have passed. Neither base
validation loss nor admission alone establishes final Route@k,
Sys@k or temperature performance. Earlier progress snapshots below are historical.

## Source and Boundary

- Source: [RetroSim Schneider corpus](https://github.com/connorcoley/retrosim/tree/0a272f0b5de833c448f41491e81e4dc00b4d85b0/retrosim/data),
  also referenced by [GLN](https://github.com/Hanjun-Dai/GLN).
- The downloaded CSV must match Git blob
  `6baad99c6d412ce06a84c6144bf70a16bb1bad89`; source SHA256 is recorded too.
- Reproduce the upstream `get_data.py` nonshuffled, per-class 80/10/10 rule.
  Source has 50,016 records: train 40,008, validation 5,001, reserved test 5,007.
  The class is used only to reconstruct splitting, never as a model feature.
- Only filtered source train/validation are used. Upstream test records are
  excluded from fitting and checkpoint selection, and retained only as source
  records/protected identities. Reaxys condition test queries stay unchanged.
- Compare both full canonical reactions and identities after product splitting
  and mapped-precursor selection. Exclude Reaxys condition/expert validation
  and test reactions, upstream test intersections, train/validation crossings,
  invalid records and within-split duplicate identities. Exclusions may share
  several reasons; raw rows and augmented pairs are reported separately.
- Keep existing SPE/ChEMBL tokenizer resources and fixed vocabulary. Reusing
  tokenizer resources is not restoring FULL-trained neural weights.

## Execution

Raw filtering is complete:

| Split | Source rows | Removed | Retained for fitting/selection |
| --- | ---: | ---: | ---: |
| Train | 40,008 | 320 | 39,688 |
| Validation | 5,001 | 13 | 4,988 |
| Upstream test | 5,007 | not processed for fitting | 0 |

Training exclusions comprise 202 within-split duplicates, 45 intersections with
upstream validation, 53 with upstream test, and 20 with Reaxys held-out identities.
Validation exclusions comprise 6 duplicates, 6 upstream-test intersections and
1 Reaxys held-out intersection. Counts concern original rows, not augmented pairs.
All are in `raw_preparation.json`; post-augmentation certification is separate.

Augmentation and the full audit are now complete. The existing preprocessor
rejected five additional train records (`small_p=4`, `small_r=1`). Actual train
input contains 396,830 augmented pairs / 39,683 distinct reactions; validation
contains 49,880 pairs / 4,988 distinct reactions. The passing receipt is
`base_audit/audit.json`: no unparseable pairs, exact text/binary agreement,
zero base train/validation intersection and zero intersections in all 48
Reaxys condition/expert held-out comparisons. These are exact canonical-reaction
checks, not evidence of scaffold-level independence.

Training started after audit approval. At 2026-09-24 11:20 UTC, epoch 1 had
completed with 330 updates and validation loss 15.311. The actual log confirms
no existing checkpoint at initialization and 47,056,896 trainable parameters.
This is a dated progress snapshot, not final performance. See
`training/USPTO_50K_FILTERED/run/train.log` and `status.json` for live state.

```bash
/root/miniconda3/envs/ProSys/bin/python -B -u \
  scripts/run_stage1_50k_from_scratch.py --workers 10 --train
```

The command refuses existing study/dataset roots and implicit checkpoint
resumption. It writes `data/editretro/datasets/USPTO_50K_FILTERED/raw`, applies
10-fold EditRetro augmentation, and binarizes using the shared dictionary.
`base_audit/audit.json` must pass a full text/tensor and reaction-membership scan
before GPU training is launched. No check uses test hit rate to choose a model.

| Setting | Value |
| --- | --- |
| Architecture / task / loss | `editretro_nat` / `translation_retro` / `nat_loss` |
| Initialization | random, no restore checkpoint, new empty save directory |
| Training seed / fixed validation seed | 1 / 7 |
| Augmentation | 10 |
| Learning rate / schedule / warmup | 0.0003 / inverse square root / 10,000 updates |
| Adam betas / weight decay | (0.9, 0.98) / 0.01 |
| Dropout / attention dropout / label smoothing | 0.2 / 0.2 / 0.1 |
| Max tokens / accumulation / precision | 16,384 / 1 / FP16 |
| Epoch / update ceilings | 50 / 200,000 |
| Validation patience | 10 checks without loss improvement |
| Dataloader workers | 4 |
| Retention | best and last only; no checkpoint alias promotion |

Data size and actual early stopping determine realized epochs/updates. A smaller
corpus alone does not prove a particular wall-clock time or final accuracy.

## Evidence and Remaining Work

`raw_preparation.json` records filtering counts, source/output hashes and the
split rule; `raw_lineage.json` remains local. `inputs.json` binds training sources
and the augmented-data audit. `training_config.json`, the actual training log and
`completion.json` bind parameters and checkpoints. Large source CSVs, key arrays,
logs, lineage and model weights remain outside Git.

Base training and expert-data/model admission have now completed. Fresh family
fine-tuning and route-cache generation are running; new mainline/baseline/
ablation evaluations remain pending. All previous FULL-based performance
remains historical.

## Expert Handoff

The repaired six-family tensors are checked against this new base with
`scripts/audit_stage1_augmented_splits.py`; the new receipt is
`expert_combined_audit.json`. The old FULL combined receipt is not substituted.
The new six-family chemical audit has passed: all checked expert internal and
condition held-out intersections, and expert validation versus 50K base
train/validation membership, are zero. All expert text pairs are parseable.
`scripts/admit_stage1_50k_experts.py --check-inputs-only` reruns independent
retained-text/tensor verification and checks all current source bindings without
admitting training. Full admission requires the normally completed base,
unchanged training inputs/logs, both best/last hashes, and the new combined audit.
An intermediate best checkpoint is explicitly rejected.

The real input-only preflight has also passed for all six families
(`expert_preflight.json`), including every retained text line and tensor byte.
Verified augmented pair totals are train 846,080, validation 39,380 and test
39,740. This receipt deliberately leaves training admission false until the
new base has completed; it is not a trained-model certificate. Subsequent
formal admission is separately recorded in `admitted_expert_inputs/`.

Once these prerequisites are complete, run:

```bash
/root/miniconda3/envs/ProSys/bin/python -B scripts/admit_stage1_50k_experts.py
/root/miniconda3/envs/ProSys/bin/python -B scripts/run_stage1_multiseed.py \
  --prepared-input-root Experiment/stage1_50k_from_scratch_20260924/admitted_expert_inputs \
  --base-checkpoint Experiment/stage1_50k_from_scratch_20260924/training/USPTO_50K_FILTERED/run/checkpoints/checkpoint_best.pt \
  --validation-protocol strict_post_augmentation \
  --study-root Experiment/stage1_50k_expert_multiseed_20260924 \
  --seeds 0,1,2
```

Admission writes new manifests and hard-links the immutable tensor/text files;
it does not modify historical manifests or duplicate large tensors. Both paths
must remain immutable during training. Original condition validation/test files
are preserved: the repaired training data, rather than evaluation denominators,
remove the diagnosed overlaps. This is three expert seeds sharing one newly
trained base, not three independent base-pretraining runs. These commands are
the next-stage handoff, not a claim that the expert runs have started.

A live automatic handoff has now admitted the completed base and started experts:
`../stage1_50k_expert_multiseed_20260924/handoff_status.json`. Do not launch the
manual commands above concurrently with that handoff. Admission has passed;
18 expert jobs and verified aggregation are not yet complete.

Admission regression checks cover interrupted training, wrong initialization,
changed checkpoints/sources/audit, missing best/last, and publication without
changing old manifests or copying tensor bytes. The complete suite passed
123 tests in 14.759 seconds (`regression_admission_tests.log`); synthetic guard
tests do not substitute for real-data audits or model evaluation.
Subsequent handoff/process-identity guard tests bring the complete suite to
129 passing tests in 14.819 seconds (`regression_handoff_tests.log`).
