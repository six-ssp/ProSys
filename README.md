# ProSys

**ProSys: A Product-to-System Framework for Target-Product-Driven
Reaction-System Recommendation.** Given a target product and a specified
reaction family, ProSys proposes routes, recommends reagent-solvent contexts,
ranks complete systems, and estimates temperature separately.

## Current Release

The accepted mainline uses **filtered USPTO-50K trained from random neural
initialization**, followed by family-specific expert fine-tuning. No USPTO-FULL
weights are inherited. Former FULL-based results are historical only;
they are not the current headline and no FULL retraining is planned.

**Sys@k and FS@k are the same full-system Top-k accuracy:** at least one of
the first k candidates jointly matches the reference route, complete reagent
set and complete solvent set. Temperature is not part of this hit criterion.

All 3,860 original query identities are evaluated, including 66 with no route.
Rates below are equal-family macro means +/- sample SD over downstream seeds
0/1/2 with fixed family expert seed 1.

| Model | Sys@1 (%) | Sys@10 (%) | Conditional temperature MAE (C) |
| --- | ---: | ---: | ---: |
| **ProSys** | **19.81 +/- 0.30** | **37.87 +/- 0.27** | **11.32 +/- 0.33** |
| B1 Product-Bernoulli Naive Bayes | 7.88 | 19.44 | NA |
| B2 Product-GNN | 5.06 +/- 0.51 | 18.85 +/- 0.89 | NA |
| B3 EditRetro + Sequential FNN | 13.80 +/- 0.59 | 26.16 +/- 0.61 | 12.28 +/- 0.19 |
| B4 EditRetro + Reaction-GCNN | 5.31 +/- 0.17 | 17.84 +/- 0.42 | NA |

B1 is deterministic; no seed SD is manufactured. The strongest-baseline
Sys@10 gap is **11.71 percentage points**. B3 temperature uses its own eligible
queries, not a support-matched comparison with ProSys.

| Ablation | Result | Interpretation |
| --- | --- | --- |
| Base vs expert, fixed seed 1 | Route@10: 10.08% vs 55.86% | Family adaptation |
| Without KNN | Sys@10: 31.31 +/- 0.50% | Rebuild pool and refit ranker |
| Without ReaFNN | Sys@10: 34.16 +/- 1.80% | Rebuild pool and refit ranker |
| Without XGB-LTR | Sys@10: 29.90 +/- 0.10% | Same pool, deterministic prior ranking |
| Without temperature R-GNN | MAE: 13.63 +/- 0.29 C | Same ranking and temperature support |

Separate expert fine-tuning seeds share one base and give Route@10
**55.42 +/- 0.39%**, not three independent base-pretraining runs. LTR improves
Top-10 but lowers macro Top-1 (20.22% without LTR). Contributions are neither
additive nor uniformly positive at every cutoff; no significance claim is made.

[Detailed current results](CURRENT_RESULTS.md) |
[Family/seed comparison tables](Experiment/50k_verified_comparisons_20260924/full/RESULTS.md) |
[Release scope and evidence](Experiment/release_50k_20260925/README.md)

## Workflow

```text
target product + specified reaction family
  -> Stage 1: family-tuned EditRetro, top-10 precursor routes
  -> Stage 2: parallel product-Morgan KNN and route-conditioned ReaFNN
  -> validation-selected fusion, up to 20 historical contexts per route
  -> Stage 3: 52-feature XGB-LTR system ranking
             separate temperature XGBoost with 52 + 128 R-GNN features
```

KNN uses a 4,096-bit product fingerprint and K=64. ReaFNN uses an
8,218-dimensional route input and independently proposes up to 64 historical
contexts. Current generation does not create novel context combinations.
There is no joint Stage 2/3 training or explicit wrong-route negative training.
R-GNN has no temperature inclusion gate and never changes ranking.

## Reproduction

Start from licensed source data and the audited 50K preparation/expert
entrypoints in the [script map](scripts/README.md). Stage 2/3 requires a
completed, admitted validation/test route pair; it cannot reuse old FULL caches.

```bash
conda activate ProSys
python -B scripts/run_50k_downstream_evidence.py \
  --route-study Experiment/stage1_50k_downstream_routes_20260924 \
  --study-root Experiment/my_50k_reproduction \
  --scratch-root outputs/my_50k_reproduction \
  --family Beckmann --seed 0
```

The release includes source code, tests and aggregate results, not trained
weights, raw Reaxys records or per-query predictions. Full retained-row audits
require local licensed data and bound artifacts. The final expert decode
needed one explicitly documented capacity-safe recovery; no query, seed or
checkpoint was substituted. See the release index for its scope.

## Scope and Documentation

Reaction-identity split and augmentation audits passed; this is not a
product/scaffold-disjoint or independent prospective benchmark. The benchmark
was used during development. Exact-match recovery is not experimental proof
of chemical feasibility. Family selection is external, not inferred by ProSys.
Temperature is conditional on the highest-ranked eligible exact match anywhere
in the candidate slate, not restricted to Top-10; always report its support.

- [Stage 1](stage1_retrosynthesis/stage1_detail.md), [Stage 2](stage2_ReaFNN/stage2_ReaFNN_detail.md), [Stage 3](stage3_XGBoost/stage3_XGBoost_detail.md).
- [Baselines](baseline/baseline.md), [ablations](ablation/ablation.md), [metric names](NOMENCLATURE.md).
- [Experiment index](Experiment/README.md), [maintenance](todo.md), [log](log.md).

Manuscript drafts remain local. Historical experiment records are preserved
for provenance, not promoted as current performance.
