# Mainline evidence completion contract

This study implements the six-item improvement plan without changing the
promoted mainline or selecting a new model on the test set.

| Requirement | Completion evidence |
| --- | --- |
| Three-way Stage 2 control | All 18 ReaFNN-only runs, replay audit and KNN-only/full comparison |
| Identity-level component controls | Reconstructed full candidate slate; no-LTR control on exactly those rows; no-R-GNN regression on the same training/validation rows; retained ranked identities, temperature support identities and labels |
| Reproduction and cache correctness | Content-bound input/config/model cache checks, mismatch tests and tested command entry points |
| Generalization diagnostics | Per-family/per-seed seen-product and unseen-product metrics; training-library condition availability; fixed-denominator accounting |
| Failure and case analysis | Exhaustive route/pool/ranking/hit partition; first exact hit rank; deterministic representative cases with intermediates |
| Cost and usable inference | Per-stage timing/memory measurements with hardware/scope; reusable models; product-query CLI and smoke test |

The reconstruction uses the existing six families and seeds 0/1/2. It retains
compact predictions and model bundles before removing feature-table scratch.
The new within-slate controls are paired to this reconstruction, not asserted
identical to deleted historical per-row files. Differences from promoted
results are reported, never silently substituted.

An independently collected external dataset would be stronger future evidence,
but is not available in this workspace and is not fabricated by re-splitting an
already inspected benchmark. The planned generalization diagnostic is an
explicitly labelled subgroup analysis of the existing reaction-group split.

Throughput policy: up to four independent family jobs, bounded CPU threads,
unchanged batch/model settings, data-disk-only temporary files, at least 8 GiB
free before a new job, immediate per-family compaction. Concurrency timings
are resource observations, not isolated latency benchmarks.
