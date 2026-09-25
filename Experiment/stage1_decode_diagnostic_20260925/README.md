# Diels-Alder Expert Seed-2 Decoder Diagnosis

This directory records a diagnosed failure and its explicitly admitted recovery.
The final admission is `expert_recovery_admission.json`; early trial receipts
retain their original unpromoted/pending fields as historical snapshots.
This is not a replacement downstream mainline table. Original logs remain
under `../stage1_50k_fidelity_v2_expert_multiseed_20260924/seed_2/DielsAlder/`.

## Confirmed Failure

- Training finished normally at epoch 101 / 22,119 updates, best validation loss
  4.266, after the unchanged patience of 15. Both best/last hashes still match
  `training_complete.json`; see `failure_evidence.json`.
- Original test decoding and a separate same-parameter synchronous-CUDA run
  failed. The latter locates the actual invalid lookup in the decoder's learned
  positional embedding, not in the output linear projection reported by the
  asynchronous traceback.
- `position_trace.json` captures the lookup before any CUDA assertion: capacity
  1,024 positions (1,026 embedding rows, padding index 1); generated length
  3,078; largest requested index 3,079. The offending candidate was at iteration
  10, after a previous length of 1,000. The insertion rule limits each gap, but
  its total proposed insertion length can exceed the positional capacity.
- Mapping through the active-beam and batch indices identifies frozen query
  offset/sample_index 220, reaction_id 27993108, augmentation slot 6 and beam 7
  (all zero-based). These IDs are diagnostic labels, not model features.
- Original controller 21969 and tail 212912 exited. The tail's old waiting-state
  JSON is stale; do not restart that PID-bound coordinator or claim it is live.

## Versioned Safety Recovery

`fail_closed_masks.py` and `run_length_guarded.py` operate only in a separate
inference process. The vendor code, trained weights, fixed query set, random
seed, batching and decode hyperparameters are unchanged.

The trial invalidates a hypothesis if its proposed nonpadding length exceeds
the actual decoder capacity. It emits an empty hypothesis, marks that state as
absorbing and assigns negative-infinite scores. It does not truncate a chemical
sequence into a different potentially valid route, draw a replacement hypothesis,
skip a query or choose another checkpoint. The existing route aggregator treats
an empty hypothesis as invalid while preserving its generation slot. An all-empty
token-expansion branch explicitly retains its expected beam slots.

On the no-overflow path the original insertion function is called unchanged,
including its in-place semantics. Eight guard/parser tests pass, including
30 seeded normal-input fixtures in one test, unchanged healthy rows,
absorbing failure, all-overflow batches and the vendor terminal beam branch.
These tests do not replace normal-path GPU parity or a full real-query audit.

The full 762-query run completed using the same seed-2 best checkpoint.
`length_guard_v1.json` and `recovery_generation_audit.json` verify 7,620
augmentation inputs and all 76,200 generation slots, with one invalidated
hypothesis. The result retains 5,767 routes and 43 empty queries; Route@10 is
33.858268%. Its generation log is
`/root/autodl-tmp/prosys_diels_seed2_length_guard_v1_20260925.log`.

## Completed Admission

The separate full seed-1 GPU parity run has no overflow. All 76,200 hypothesis
strings and all 762 final ranked route lists, scores and probabilities match
the original run exactly. Two EOS token scores on already invalid SMILES differ;
`normal_path_gpu_parity.json` records them explicitly. Valid-hypothesis token
scores are identical; universal raw-token bitwise equality is not claimed.

`expert_recovery_admission.json` certifies this one versioned recovery alongside
17 untouched original decodes. All-18 expert summary certification, independent
mean/sample-SD recalculation, isolated cost measurement and local manuscript
publication are complete. Original failed controller state and missing original
completion marker remain unchanged. The recovery is not silently relabelled as
the original decoder and is not installed as a new default production decoder.

All 69 frozen production-source hashes remained unchanged after diagnosis.
No completed Stage 2/3 or baseline/ablation result has been overwritten.
