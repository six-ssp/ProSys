# Experiment Records

## Accepted 50K Mainline

The user accepted the filtered-USPTO-50K replacement on 2026-09-25.
Current macro Sys@10 is **37.87 +/- 0.27%**. The former FULL-based and serial
studies are historical controls, not competing current headline results.

- [Public release index](release_50k_20260925/README.md): scope, validation and local/public evidence boundaries.
- [Mainline and Stage 3](stage23_50k_evidence_20260924/RESULTS.md): 18 fits with same-pool ranking and support-matched temperature controls.
- [All baselines and ablations](50k_verified_comparisons_20260924/full/RESULTS.md): 150 family/model/seed rows, B1 deterministic.
- [Expert three-seed results](stage1_50k_fidelity_v2_expert_multiseed_20260924/SUMMARY.md): one shared fresh base, 18 experts.
- [Fixed-seed base/expert comparison](stage1_50k_paired_comparison_20260924/README.md): same 3,860 original query identities.
- [Base training](stage1_50k_from_scratch_20260924/README.md) and [input repair](stage1_fidelity_repair_20260924/README.md).
- [Decoder safety recovery](stage1_decode_diagnostic_20260925/README.md): one explicit same-weight invalid-hypothesis recovery, not query deletion.
- [Cold-query cost](inference_cost_50k_20260925/cold_inference_cost.csv): six fixed queries, not population latency or throughput.

The public release includes aggregate tables and source code. Local workspaces
also retain models, route caches, per-query predictions and detailed audit
receipts. These are not all uploaded; an absent local-only artifact in a fresh
clone does not mean the experiment was skipped. Full replay needs the licensed
source data and bound local models.

## Historical and Exploratory Records

Dated FULL-based, serial, earlier parallel and repair-diagnostic directories
are retained for provenance. Statements such as "current headline" inside an
original historical report refer to its own date, not this release. Never
combine their metrics or temperature supports with the accepted 50K study.

Joint Stage 2/3 training, explicit wrong-route negative training and other
retired probes are not the maintained mainline. No additional FULL training
or optimization sweep is queued. Temporary workspaces are ignored by Git;
the explicit public-file allowlist is in [.gitignore](.gitignore).
