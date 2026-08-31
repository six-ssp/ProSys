# ProSys

**ProSys** is a target-product-driven framework for complete reaction-system
recommendation. Given a target product and a predefined reaction family, it
proposes retrosynthetic routes, retrieves feasible reagent-solvent contexts,
ranks complete systems, and estimates temperature.

## Scope

- Evaluation: six Reaxys reaction families, reaction-disjoint and test-time
  `Non-Oracle` with fixed Stage-1 route caches. The retained training and
  validation candidate tables use reference routes; see
  `Experiment/stage23_legality_audit_20260830.md` for the resulting
  distribution-shift disclosure.
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

The R-GNN is used only by the temperature regressor. The system ranker uses a
fixed 52-slot non-graph schema; 33 fields vary in the current historical-only
candidate configuration and 19 retained compatibility fields are constant.

## Current Mainline Result

The official record is the post-hardening fixed-Stage-1, three-seed Stage-2/3
robustness experiment in [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md). It covers
3,860 test identities from six families, with 3,833 candidate slates and 27
retained no-slate identities in every seed. It measures Stage-2/3 stochastic
variation; it does not estimate variability from retraining Stage 1.

| Candidate coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54.44 +/- 0.14 | 27.12 +/- 0.37 | 36.84 +/- 0.80 | 40.47 +/- 0.77 | 44.62 +/- 0.42 | 33.28 +/- 0.48 | 34.70 +/- 0.53 |

- Fixed Stage-1 Route@10: `63.20%`
- Conditional temperature MAE: `11.73 +/- 0.54 C`
- Temperature within +/-5 / +/-10 / +/-20 C:
  `40.59 +/- 0.56% / 61.80 +/- 2.36% / 83.04 +/- 1.10%`

Temperature is evaluated only on the highest-ranked exact route-and-condition
match with a valid temperature label. It does not affect system ranking.
Detailed per-family and per-seed results are in
[CURRENT_RESULTS.md](CURRENT_RESULTS.md) and
`Experiment/stage23_product_morgan_reafnn_multiseed_20260830/`.

The matched three-seed ablations attribute a +4.76 pp Sys@10 contribution
to ReaFNN relative to KNN-only plus a re-trained ranker, and a +8.17 pp
contribution to XGB-LTR relative to a matched-Stage-2 deterministic ordering.
See ablation/current_mainline_matched_ablation_results_20260830.md.

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
| `Experiment/` | Compact promoted result records and historical exploratory material |

## Documentation

- [Workflow](workflow.md): complete end-to-end protocol.
- [Performance](performance.md): current results and interpretation.
- [Stage 2 details](stage2_ReaFNN/stage2_KNN_detail.md): candidate-pool construction.
- [Stage 3 details](stage3_XGBoost/stage3_XGBoost_detail.md): ranking and temperature prediction.
- [Baseline and ablation study](baseline&ablation.md): comparison scope and safeguards.
- [Metric nomenclature](NOMENCLATURE.md): public metric names and definitions.

Older analyses, gated-temperature snapshots, direct graph-ranking trials, and
the pre-hardening 2026-08-09 snapshot are historical records only.
