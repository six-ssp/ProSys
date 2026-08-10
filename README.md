# ProSys

**ProSys** is a target-product-driven framework for complete reaction-system
recommendation. Given a target product and a predefined reaction family, it
proposes retrosynthetic routes, retrieves feasible reagent-solvent contexts,
ranks complete systems, and estimates temperature.

## Scope

- Evaluation: six Reaxys reaction families, strict `Non-Oracle`, end-to-end.
- Molecular input: target-product structure only.
- Family handling: the evaluation partition selects a family-specific expert,
  retrieval memory, and vocabulary; the family label is not concatenated to a
  molecular, ReaFNN, or XGBoost feature vector.

```text
target product
  -> Stage 1: family-tuned EditRetro route generation
  -> Stage 2: wide KNN retrieval + ReaFNN condition-pool selection
  -> Stage 3: tabular XGB-LTR reranking + fixed R-GNN temperature regression
  -> ranked route / reagent-set / solvent-set systems with temperature
```

The R-GNN is used only by the temperature regressor. The system ranker always
uses the 52 non-graph tabular features.

## Current Results

The maintained result is a fixed-Stage-1, three-seed Stage-2/3 reconstruction
over six families. Values are equal-family mean +/- sample standard deviation.

| Candidate coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48.52 +/- 0.59 | 28.96 +/- 0.91 | 36.61 +/- 1.05 | 39.53 +/- 1.24 | 42.90 +/- 1.00 | 33.78 +/- 0.96 | 35.13 +/- 0.97 |

- Fixed Stage-1 Route@10: `63.20%`
- Conditional temperature MAE: `10.75 +/- 0.14 C`
- Temperature within +/-5 / +/-10 / +/-20 C:
  `41.66 +/- 1.39% / 64.14 +/- 1.65% / 84.74 +/- 0.38%`

Temperature is evaluated only on the highest-ranked exact route-and-condition
match with a valid temperature label. It does not affect system ranking.
Detailed per-family and per-seed results are in
[CURRENT_RESULTS.md](CURRENT_RESULTS.md) and
`outputs/unified_rgnn_multiseed_20260809/`.

## Quick Start

```bash
conda activate ProSys
bash scripts/setup_prosys_env.sh
python data_preprocess/audit_data_splits.py --strict
```

Build a Stage-1 route cache when needed:

```bash
python stage1_retrosynthesis/build_route_cache.py --repo_root . --family Beckmann
```

Run the maintained Stage-2/3 pipeline without overwriting archived results:

```bash
OUTPUT_ROOT=outputs/stage23_mainline_current \
ROUTE_ROOT=outputs/stage1_routes \
REAFNN_DEVICE=cuda:0 \
GNN_DEVICE=cuda:0 \
bash scripts/run_stage23_non_oracle_suite.sh .
```

## Repository Map

| Path | Purpose |
| --- | --- |
| `stage1_retrosynthesis/` | EditRetro training, fine-tuning, and route caches |
| `stage2_ReaFNN/` | KNN retrieval and ReaFNN feasible-condition selection |
| `stage3_XGBoost/` | XGB-LTR reranking and fixed R-GNN temperature regression |
| `baseline/` | Reproducible comparison methods |
| `ablation/` | Controlled component analyses |
| `data_preprocess/` | Cleaning, normalization, splits, and audits |
| `Experiment/` | Historical exploratory material, not part of the mainline |

## Documentation

- [Workflow](workflow.md): complete end-to-end protocol.
- [Performance](performance.md): current results and interpretation.
- [Stage 2 details](stage2_ReaFNN/stage2_KNN_detail.md): candidate-pool construction.
- [Stage 3 details](stage3_XGBoost/stage3_XGBoost_detail.md): ranking and temperature prediction.
- [Baseline and ablation study](baseline&ablation.md): comparison scope and safeguards.
- [Metric nomenclature](NOMENCLATURE.md): public metric names and definitions.

Older Oracle analyses, gated-temperature snapshots, and direct graph-ranking
trials are historical records only and are not headline results.
