# Evidence-plan completion report

Date: 2026-09-13. The six-item plan is complete within its declared scope.
The promoted model definition and historical result files were not replaced.
This report distinguishes fresh reconstruction evidence from historical
compact results and explicitly records the remaining limitations.

## Requirement-to-evidence matrix

| Planned item | Completed work | Primary evidence |
| --- | --- | --- |
| Strict three-way Stage 2 control | Six families, seeds 0/1/2; ReaFNN-only removes KNN proposals and evidence, retrains LTR; 18 row-level replays pass | [Three-way table](../stage2_reafnn_only_multiseed_20260913/SUMMARY.md) |
| Identity-level Stage 3 and temperature controls | 18 reconstructed candidate slates; no-LTR scores aligned by complete candidate identity; no-R-GNN shares ranked identities, eligible temperature identities and reference labels | [Replay audit](replay_audit.json), per-family exact-control audits |
| Reproduction and cache correctness | Content-bound family caches, writer locks, stale-route-manifest checks, real reuse/rejection smokes, and raw-workflow command-wiring tests | [Inference/cache audit](inference_cache_audit.json), [test log](completion_tests.log) |
| Generalization diagnostics | Per-family/per-seed condition-memory-seen/unseen-product metrics and historical-context availability | [Subgroups](subgroups_per_family_seed.csv), [macro rows](subgroups_macro_by_seed.csv) |
| Failure and detailed case analysis | Exhaustive route/pool/ranking/hit partition; three first-hit-correct cases per family with full candidate/feature traces | [Failures](failures_per_family_seed.csv), [example.md](../../example.md) |
| Cost and usable inference | Retained family bundles; fresh product-to-system CLI; measured reconstruction and cold-query resources | [Reconstruction costs](cost_per_family_seed.csv), [cold-query costs](cold_inference_cost.csv), [usage](README.md) |

The final requirement-level checker revalidated fixed denominators, subgroup
membership against training data, first-hit logic, exact sparse input values,
fusion arithmetic, graph dimensions and label-free inference outputs. Thirty
tests passed. See [completion_gate.json](completion_gate.json) for artifact
hashes and scope limits. This is not a claim that every possible project issue
has been eliminated.

## Main numerical findings

All percentages below are equal-family macro means with sample SD over seeds
0/1/2. The 3,860 query identities and Stage 1 caches remain fixed.

| System | Sys@10 | Full-minus-control |
| --- | ---: | ---: |
| Parallel KNN + ReaFNN + XGB-LTR | 43.77 +/- 0.60 | -- |
| KNN-only + retrained XGB-LTR | 39.86 +/- 2.08 | +3.91 pp |
| ReaFNN-only + retrained XGB-LTR | 36.24 +/- 0.27 | +7.53 pp |
| Full parallel pool + deterministic Stage 1/2 ranking | 36.03 +/- 0.16 | +7.74 pp |

The Stage 2 removals change candidate composition and retrain the ranker;
they are not same-pool ranking interventions and their effects are not
additive. The Stage 3 comparison is a same-candidate intervention. These
findings support complementary contributions under this benchmark, not
universal indispensability of either branch or a stable Top-1 ranking gain.

Every reconstructed family/seed Sys@1/3/5/10 value agrees with the promoted
record. Temperature predictions do not reproduce bit-for-bit:

| Temperature result | MAE (C) | Within 10 C |
| --- | ---: | ---: |
| Historical promoted R-GNN-assisted result | 11.49 +/- 0.26 | 63.09 +/- 1.76% |
| Newly reconstructed R-GNN-assisted result | 11.82 +/- 0.33 | 61.58 +/- 0.77% |
| Newly paired no-R-GNN result | 13.93 +/- 0.38 | 55.56 +/- 1.23% |

The new identity-matched comparison gives a 2.11 C MAE reduction and a 6.02 pp
within-10-C gain. Its conditional supports are 1,785/1,795/1,797. Temperature
uses the first exact-system row with finite gold and predicted temperature
over the entire slate, not only Top-10. The largest individual family/seed
MAE difference from the historical reconstruction is 2.458091 C. A cause such
as numerical nondeterminism has not been isolated experimentally; do not label
the drift a proved hardware effect or mix the two temperature result versions.

## Generalization and failures

Of the fixed test identities, 3,569 (92.46%) have products absent from the
corresponding condition-training memory. Their macro Sys@10 is
45.32 +/- 0.62%, versus 29.28 +/- 1.09% for the 291 memory-seen identities.
This rules out an explanation based solely on exact-product lookup in that
memory. It does not establish that unseen products are intrinsically easier,
that the subsets are difficulty-matched, or that products are unseen to
EditRetro's larger training corpora. It is not an external validation dataset.

At least one reference context exists in the training library for 3,729
identities; 131 (3.39%) have no such context and cannot be exactly recovered
by a historical-context-only generator. All rates retain no-slate failures.

| Exhaustive category | Macro proportion |
| --- | ---: |
| Stage 1 route miss | 36.80 +/- 0.00% |
| Route retained, exact context missing from Stage 2 pool | 8.94 +/- 0.15% |
| Exact system present, first hit below rank 10 | 10.49 +/- 0.51% |
| Exact system within rank 10 | 43.77 +/- 0.60% |

Route recovery remains the largest measured failure source. This supports
prioritizing future route-quality work rather than adding downstream network
complexity only to improve headline appearance. The examples cover true first
hits at ranks 1, 3, 5 and 10 across six families. Local JSON traces include KNN
neighbors, pre-fusion branch unions, sparse fingerprints, training-scaled
ReaFNN inputs, both token heads, all candidate scores and 128 graph coordinates.

## Measured resources and limits

On the available 14-core/90-GiB allocation and RTX 3090, the 18 Stage 2/3
reconstructions used bounded four-process scheduling. Per-family reconstruction
mean times ranged from about 150 to 1,043 seconds, including fitting,
candidate-table work, controls and evidence compaction. Maximum observed
per-process RSS was 4,870.99 MiB; maximum PyTorch-allocated GPU memory was
444.28 MiB. These are not pure model-training times or total GPU-hours.

After reconstruction workers exited, one fresh product query per family took
9.91-19.93 seconds including interpreter/model imports, cold artifact loading,
Stage 1 decoding and Stage 2/3 inference. Sampled whole-device GPU-memory peaks
were 591-735 MiB with zero baseline GPU allocation. The 0.2-second sampler is
not allocator-exact. This six-query smoke is not a population latency or
throughput benchmark; Stage 2's library loading dominates some cold queries.

Known boundaries remain explicit:

- One Diels-Alder training record has a cross-dot ring-closure representation
  issue in the frozen helper. A whole-molecule scan found no affected
  validation/test sides and zero reaction overlap between splits. See
  [the canonicalization note](CANONICALIZATION_NOTE.md).
- The first pilot predates runtime source-manifest instrumentation. Its
  model/data hashes and exact-control checks exist; no missing source trace was
  retrospectively fabricated.
- The independent external dataset, prospective synthesis validation and a
  fresh USPTO-pretraining audit were not performed or claimed in this plan.
- Local model bundles and licensed row-level traces are retained for reuse but
  excluded from ordinary Git payloads. No Git push was performed for this work.
