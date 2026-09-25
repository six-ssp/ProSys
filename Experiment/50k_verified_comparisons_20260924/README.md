# Complete 50K Comparison Reports

All six families are complete. The authoritative aggregate export is
[full/RESULTS.md](full/RESULTS.md), backed by 150 family/model/seed rows.
It covers ProSys, four baselines and four removal arms. B1 is deterministic;
each other arm has seeds 0/1/2, always with fixed expert seed 1 and the
original 3,860-query denominator.

Public CSV files contain per-family/per-seed values, equal-family macro means
and sample SD. Exact values are retained before rounding. Private family
replay receipts, model artifacts and per-query predictions remain local.

ProSys macro Sys@10 is 37.87 +/- 0.27%, versus 26.16 +/- 0.61% for the
strongest macro baseline B3. No-LTR keeps the same pool; Stage 2 branch
removals rebuild pools and refit rankers. R-GNN removal affects temperature
only and preserves system order/support. These interventions are not additive.

Do not infer uniform superiority: LTR lowers macro Top-1 and some family
early-rank metrics. B3 temperature uses its own support and has higher macro
within-10-C accuracy. Three training seeds do not establish statistical
significance or independent prospective validation.

To rebuild comparisons after producing all licensed local evidence, use a
new output directory:

```bash
python -B scripts/collect_50k_comparisons.py \
  --mainline Experiment/stage23_50k_evidence_20260924 \
  --baselines Experiment/baseline_50k_multiseed_20260924 \
  --no-knn Experiment/stage2_50k_reafnn_only_20260924 \
  --knn-only Experiment/stage2_50k_knn_only_20260924 \
  --routes Experiment/stage1_50k_downstream_routes_20260924 \
  --output Experiment/my_50k_comparisons
```

The collector refuses missing or inconsistent requested evidence. The
[public-release checker](../release_50k_20260925/README.md) can independently
check the uploaded aggregate arithmetic without private models or data.
