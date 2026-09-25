# Accepted Filtered-50K Release

The user accepted this mainline on 2026-09-25. The baseline and ablation suite
is complete. No further tuning or USPTO-FULL training is planned for this release.
Sys@k and FS@k are the same joint route/reagent-set/solvent-set Top-k metric.

## Results and Evidence

| Evidence | Public artifact |
| --- | --- |
| Mainline, four baselines and four removal controls | [150 family/model/seed rows](../50k_verified_comparisons_20260924/full/per_family_model_seed.csv), [summary](../50k_verified_comparisons_20260924/full/RESULTS.md) |
| Exact Stage 3 controls | [54 family/seed/arm rows](../stage23_50k_evidence_20260924/per_family_seed_controls.csv) |
| Base vs fixed expert seed 1 | [Paired table](../stage1_50k_paired_comparison_20260924/all_families_seed1.csv) |
| Expert seeds 0/1/2 | [18 expert rows](../stage1_50k_fidelity_v2_expert_multiseed_20260924/per_family_seed_metrics.csv), [summary](../stage1_50k_fidelity_v2_expert_multiseed_20260924/SUMMARY.md) |
| Query denominators and empty routes | [Route counts](../stage1_50k_downstream_routes_20260924/route_counts.csv) |
| Direct condition metrics | [48 auxiliary rows](../paper_auxiliary_50k_20260924/all_families/direct_condition_metrics.csv) |
| Validation-only branch fusion | [18 family/seed rows](../paper_auxiliary_50k_20260924/all_families/validation_fusion.csv) |
| Failure analysis | [Grouped failure counts](../query_diagnostics_50k_20260924/all_families/failures_per_family_seed.csv), [subgroups](../query_diagnostics_50k_20260924/all_families/subgroups_per_family_seed.csv) |
| Six fixed-query cold measurements | [Cost CSV](../inference_cost_50k_20260925/cold_inference_cost.csv) |
| Dataset/model input corrections | [Fidelity repair](../stage1_fidelity_repair_20260924/README.md), [50K base](../stage1_50k_from_scratch_20260924/README.md) |
| Final expert safety recovery | [Recovery policy and code](../stage1_decode_diagnostic_20260925/README.md) |

Mainline macro Sys@10 is **37.87 +/- 0.27%**, versus **26.16 +/- 0.61%**
for B3, the strongest macro baseline: **11.71 pp** improvement. Removal of
KNN/ReaFNN/LTR gives 31.31/34.16/29.90%. The paired temperature MAE is
11.32 +/- 0.33 C with R-GNN versus 13.63 +/- 0.29 C without it.

Expert-seed Route@10 is **55.42 +/- 0.39%**, sharing one fresh base.
The fixed-expert-seed-1 base/expert comparison is **10.08% vs 55.86%**.
All downstream seeds use expert seed 1 and the same 3,860 queries, including
66 no-route queries. These are not three full-pipeline/pretraining repetitions.

## Public Verification

Run from a clone using standard-library Python; no GPU, model weights or raw
chemical records are required for this aggregate check:

```bash
python -B Experiment/release_50k_20260925/verify_public_release.py
```

It verifies [public file hashes and statistics](public_evidence.json), exact
family/model/seed grids, fixed denominators, equal-family means and sample SD.
It does not pretend that aggregate arithmetic alone validates model training.
The local completion work additionally replayed retained candidates, checked
data/model/source bindings and passed 252 project plus eight decoder tests.

## Scientific Limits

LTR improves Top-10 but decreases macro Top-1; not every family/cutoff or seed
benefits. Stage 2 removals rebuild pools and refit their rankers, while no-LTR
uses the same pool. The temperature ablation keeps ranking and eligible support
fixed. B3 temperature has its own support and higher within-10-C accuracy;
there is no uniform temperature-baseline superiority claim.

The benchmark was used during development. Exact-reaction isolation is not
product/scaffold-disjoint or prospective evaluation. Family selection is
external. Stage 2 recommends training-library contexts, not novel combinations.
Temperature uses the first eligible exact system anywhere in the slate, not
only Top-10. Cold costs are six individual observations including model loads,
not sustained throughput or population latency. Hardware was an RTX 3090,
14 CPU-equivalents allocated, two model threads; device memory was sampled.

One final expert decode exceeded its positional capacity. The versioned
same-weight recovery invalidates one hypothesis rather than truncating chemistry,
while retaining all 762 queries and 76,200 slots. The other 17 original decodes
are unchanged. Full normal-path GPU replay gives identical hypothesis strings
and ranked routes; two EOS-score differences on already invalid strings are
disclosed. The recovery is not silently installed into the frozen vendor code.

## Local-Only Materials

Raw/licensed Reaxys records, processed datasets, checkpoints, token/context
vocabularies, per-query predictions, detailed cases and manuscript drafts are
not uploaded. Detailed local audit receipts can reference these artifacts;
their absence from a clone is deliberate, not missing experiment execution.
Full model-level reproduction requires licensed data and the bound artifacts
or rerunning the documented pipeline. Public hashes record integrity, not
automatic access to restricted inputs.

The previous local closure receipt is a time-stamped computational/document
audit. This publication only updates maintenance text, file-selection rules and
public aggregate verification; it does not retrain models or change the results.
Original maintenance snapshots and local paper documents are preserved.
