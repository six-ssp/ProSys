# ProSys: A Product-to-System Framework for Target Product-Driven Reaction-System Recommendation

## Current Maintained Mainline: Parallel KNN + ReaFNN Post-Fusion (2026-09-03)

**Authoritative implementation.** The maintained Stage-2 procedure is the
parallel product-Morgan KNN + ReaFNN post-fusion workflow invoked by
`scripts/run_stage23_non_oracle_suite.sh`. KNN and ReaFNN independently propose
train-only historical contexts, their route-local union is fused with a
family-specific validation-selected KNN weight, and the top 20 contexts per
Stage-1 route are sent to the 52-feature XGB-LTR ranker. The R-GNN remains a
128-dimensional temperature-only representation for the separate XGBoost
regressor.

The maintained workflow uses no joint Stage-2/Stage-3 optimization, no
wrong-route negative-sample training, no route-validity auxiliary head, and no
novel reagent-solvent context generation. Test slates always originate from
persisted Stage-1 predictions. The KNN and ReaFNN libraries, token
vocabularies, feature standardizers, and model fitting use only the matching
family training split; the Stage-2 mixture weight is selected only on predicted
validation-route caches.

### Verified Six-Family Three-Seed Result

The authoritative record is
`Experiment/stage23_parallel_post_fusion_multiseed_20260903/`. It repeats the
current parallel workflow at seeds `0, 1, 2`, while retaining the persisted
split, six test manifests, and Stage-1 route caches. It measures variation in
ReaFNN, Reaction-GNN, and XGBoost only; it does not estimate variability from
retraining EditRetro Stage 1.

All values below are equal-family macro averages. Rates are percentages and
`+/-` is sample standard deviation (`ddof=1`).

| Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |

Conditional temperature MAE is `11.49 +/- 0.26 C`; conditional hit rates are
`41.41 +/- 2.16%` within `+/-5 C`, `63.09 +/- 1.76%` within `+/-10 C`, and
`83.74 +/- 0.69%` within `+/-20 C`. The temperature support is 1,785, 1,795,
and 1,797 products for seeds 0, 1, and 2, respectively, because it is defined
only for the highest-ranked exact full-system match with a valid temperature.

The fixed manifest contains 3,860 product identities. Each seed has 3,833
candidate slates and 27 retained no-slate identities, which receive zero in
the full-system denominator. The fixed Stage-1 macro Route@1/3/5/10 is
`43.46/56.69/59.97/63.20%` across all three seeds.

`Experiment/stage2_parallel_post_fusion_20260901.md` remains the detailed
seed-0 development record. The serial three-seed records below remain useful
historical controls but are not numerically interchangeable with the current
parallel candidate distribution.

## Current Parallel Stage-2 Matched ReaFNN Ablation (2026-09-05)

The paired Stage-2 control is retained in
`Experiment/stage2_parallel_post_fusion_ablation_multiseed_20260904/`. It
compares the full parallel KNN + ReaFNN post-fusion pool with a KNN-only pool
at seeds 0/1/2. Both arms use the same persisted Stage-1 routes, 3,860-product
denominator, product-Morgan KNN (radius 2, 4,096 bits, K=64), 64-context
prefilter, route-local top-20 cap, family train-only memory, and reference-split
training/validation tables. The control disables ReaFNN and post-fusion, then
re-trains its own 52-feature XGB-LTR on the changed KNN-only candidate
distribution. Temperature is skipped because it does not affect Sys@k.

| System | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full current mainline | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |
| KNN-only + XGB-LTR | 53.39 +/- 0.00 | 22.72 +/- 1.86 | 31.44 +/- 2.45 | 35.19 +/- 2.23 | 39.86 +/- 2.08 | 28.63 +/- 1.98 | 29.85 +/- 2.05 |
| Full minus KNN-only (pp) | +0.87 | +2.42 | +3.68 | +3.92 | +3.91 | +2.90 | +3.30 |

All 18 family-seed pairs pass fixed Stage-1 route-recall, manifest and
candidate-slate accounting, and common-KNN-protocol checks. Every family mean
favors the full pool at Sys@10; the paired seed-level gains are `+1.42`,
`+3.72`, and `+6.59 pp`. Because the candidate composition changes and each
arm retrains its own ranker, this is a Stage-2 availability/composition result,
not a within-identical-pool ranking attribution. See
`ablation/current_parallel_stage2_ablation_results_20260905.md` for the
family-resolved table and paper-safe interpretation.

## Current Parallel Stage-3 Matched Ablation (2026-09-04)

The paired Stage-3 control is retained in
`Experiment/stage3_parallel_post_fusion_ablation_multiseed_20260904/`. It uses
the same fixed Stage-1 route caches, family train-only condition memory,
product-Morgan KNN (`K=64`), independently generated ReaFNN top-64 contexts,
validation-only post-fusion, and 20-context cap as the current mainline at
seeds 0/1/2. The only ranking change is to replace the fitted 52-feature
XGB-LTR with the deterministic Stage-1/2 prior. Temperature is skipped because
it neither changes candidate membership nor the system-ranking score.

| System | Candidate recall | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full current mainline | 54.26 +/- 0.15 | 25.13 +/- 1.20 | 35.12 +/- 1.27 | 39.11 +/- 1.04 | 43.77 +/- 0.60 | 31.53 +/- 1.12 | 33.16 +/- 1.02 |
| Full Stage 2 + deterministic no-XGB-LTR | 54.26 +/- 0.15 | 24.48 +/- 0.09 | 32.23 +/- 0.34 | 34.05 +/- 0.02 | 36.03 +/- 0.16 | 29.24 +/- 0.07 | 29.99 +/- 0.06 |
| Full minus no-XGB-LTR (pp) | 0.00 | +0.65 | +2.89 | +5.06 | +7.74 | +2.29 | +3.16 |

All 18 family-seed records passed exact Stage-2-protocol and candidate-availability
matching against the corresponding full-mainline compact record. Thus the
`+7.74 pp` macro Sys@10 difference is a within-pool ranking effect, not a
candidate-availability effect. The Sys@1 mean difference is smaller and changes
direction across seeds (`-0.69`, `+0.81`, and `+1.83 pp`); no stable Top-1
claim is made. The effect is consistently positive in the six family-mean
Sys@10 comparisons. See
`ablation/current_parallel_stage3_ablation_results_20260904.md` for the
family-resolved table and paper-safe interpretation.

## Current Matched Temperature-Representation Ablation (2026-09-04)

The paired temperature-representation control is retained in
`Experiment/stage3_temperature_no_rgnn_ablation_multiseed_20260904/`. It holds
fixed the persisted Stage-1 routes, parallel product-Morgan KNN + ReaFNN
post-fusion candidate pool, 20-context cap, 52-feature XGB-LTR, test manifest,
and random seed. The full arm uses a 180-column temperature regressor
(`52` tabular candidate features plus `128` R-GNN route features); the control
re-trains only the temperature regressor with the identical 52 tabular features
and no `route_gnn_feat_*` column.

| Temperature arm | Conditional support | MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | ---: | ---: | ---: | ---: | ---: |
| R-GNN + XGBoost | 1,792.33 +/- 6.43 | 11.49 +/- 0.26 | 41.41 +/- 2.16 | 63.09 +/- 1.76 | 83.74 +/- 0.69 |
| Tabular XGBoost, no R-GNN | 1,792.33 +/- 6.43 | 13.93 +/- 0.38 | 35.30 +/- 0.81 | 55.56 +/- 1.23 | 78.09 +/- 1.22 |
| Full minus no-R-GNN | 0.00 | -2.43 C | +6.11 pp | +7.53 pp | +5.65 pp |

All 18 family-seed pairs passed exact matching of Stage 1 route recall, Stage 2
protocol and candidate pool, all Sys@k/MRR/nDCG ranking metrics, and conditional
temperature support. The graph branch therefore improves the conditional
temperature estimate without changing candidate membership or system ranking.
See `ablation/current_parallel_temperature_ablation_results_20260904.md` for
family-resolved results and paper-safe wording.

## Historical Serial Three-Seed Reference (2026-08-30)

**Historical record.** The former serial mainline was the post-hardening,
fixed-Stage-1, three-seed Stage 2/3 robustness experiment retained in
`Experiment/stage23_product_morgan_reafnn_multiseed_20260830/`. It evaluates
six families and 3,860 test identities. Stage 1 route caches are fixed across
seeds; ReaFNN, the R-GNN, and XGBoost are independently rebuilt at seeds 0, 1,
and 2. This measures Stage 2/3 stochastic variation, not Stage 1 retraining
variation.

The former serial configuration used product-Morgan KNN retrieval (radius 2, 4,096
bits; `K=64`; 64-context wide pool), ReaFNN 512x2 ReLU refinement with 12 KNN
anchors, and a 20-context cap per proposed route. ReaFNN selects only from the
retrieved historical wide pool; context augmentation and novel-combination
generation are disabled. XGB-LTR uses the fixed 52-column non-graph schema.
The 128-dimensional R-GNN is used only by the temperature XGBoost regressor and
does not affect system ranking.

Test candidate slates come from persisted Stage 1 predictions. Training and
validation candidate tables use their reference split routes; the distribution
shift and canonical leave-one-reaction-out KNN safeguard are recorded in
`Experiment/stage23_legality_audit_20260830.md`.

All values below are equal-family macro averages. Rates are percentages and
`+/-` denotes sample standard deviation across seeds (`ddof=1`). A full-system
hit requires an exact joint match of canonical route, normalized reagent set,
and normalized solvent set. Temperature is conditional on a valid-temperature
exact full-system match and never affects system ranking.

| Coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 | Temp. MAE (C) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54.44 +/- 0.14 | 27.12 +/- 0.37 | 36.84 +/- 0.80 | 40.47 +/- 0.77 | 44.62 +/- 0.42 | 33.28 +/- 0.48 | 34.70 +/- 0.53 | 11.73 +/- 0.54 |

Temperature within +/-5 / +/-10 / +/-20 C is
`40.59 +/- 0.56% / 61.80 +/- 2.36% / 83.04 +/- 1.10%`. Each seed has 3,833
candidate slates and 27 retained no-slate identities in the 3,860-product
denominator. The three seed-level Sys@10 values are 45.08%, 44.51%, and 44.27%,
respectively.

| Family | Coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 43.55 +/- 0.65 | 17.45 +/- 1.53 | 22.98 +/- 1.13 | 25.96 +/- 1.95 | 29.36 +/- 1.13 | 21.63 +/- 1.10 | 22.07 +/- 1.18 |
| Buchwald-Hartwig | 59.51 +/- 0.24 | 32.61 +/- 0.82 | 44.71 +/- 0.21 | 48.41 +/- 0.57 | 52.65 +/- 0.14 | 39.84 +/- 0.57 | 41.80 +/- 0.39 |
| Chan-Lam | 72.14 +/- 0.15 | 41.62 +/- 2.38 | 54.10 +/- 2.04 | 58.12 +/- 2.72 | 62.14 +/- 2.15 | 48.96 +/- 2.05 | 50.56 +/- 2.05 |
| Diels-Alder | 34.34 +/- 0.15 | 15.70 +/- 0.59 | 21.35 +/- 0.53 | 23.53 +/- 1.29 | 26.29 +/- 1.18 | 19.51 +/- 0.65 | 20.27 +/- 0.89 |
| Friedel-Crafts Acyl. | 65.89 +/- 0.00 | 28.00 +/- 3.60 | 39.79 +/- 4.06 | 44.56 +/- 3.38 | 51.79 +/- 1.64 | 35.99 +/- 3.26 | 37.94 +/- 3.11 |
| Friedel-Crafts Alkyl. | 51.24 +/- 0.17 | 27.33 +/- 1.05 | 38.12 +/- 0.64 | 42.27 +/- 0.68 | 45.49 +/- 0.69 | 33.74 +/- 0.76 | 35.54 +/- 0.75 |

| Family | Conditional MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | ---: | ---: | ---: | ---: |
| Beckmann | 11.34 +/- 0.79 | 35.95 +/- 1.49 | 57.20 +/- 8.16 | 85.26 +/- 1.48 |
| Buchwald-Hartwig | 12.20 +/- 0.24 | 34.11 +/- 0.97 | 56.79 +/- 2.36 | 81.38 +/- 0.70 |
| Chan-Lam | 5.91 +/- 0.20 | 66.29 +/- 2.73 | 84.22 +/- 1.22 | 94.57 +/- 1.87 |
| Diels-Alder | 19.48 +/- 1.48 | 24.64 +/- 1.77 | 43.80 +/- 4.75 | 65.87 +/- 3.79 |
| Friedel-Crafts Acyl. | 11.02 +/- 0.78 | 42.79 +/- 2.00 | 66.06 +/- 3.49 | 84.36 +/- 1.31 |
| Friedel-Crafts Alkyl. | 10.41 +/- 0.45 | 39.76 +/- 0.94 | 62.76 +/- 2.61 | 86.82 +/- 1.20 |

Detailed per-seed and per-family CSV tables, source records, and compact model
metadata are retained with the promoted result artifact. The strongest completed
baseline is B3 at `31.71 +/- 0.10%` Sys@10, so the former serial reference
had a descriptive `+12.91 pp` Sys@10 advantage on the fixed manifest and route cache.

## Historical Pre-Hardening Reference (2026-08-09)

**Pre-hardening reference.** Every family uses a fixed R-GNN
(128-dimensional, four-layer) plus XGBoostRegressor temperature branch. There
is no family-level no-GNN fallback and no temperature-model selection gate. The
figures below are from a completed six-family, three-seed reproduction. They
predate canonical leave-one-reaction-out KNN hardening and must be rerun before
being used as the final post-audit headline; the older gated point snapshot is
historical only.

## Shared Evaluation Scope

This is a family-stratified, target-product-driven, end-to-end specialist
evaluation. The molecular query is the target product only. A family identifier
selects separately trained specialist artifacts, condition memory, and label
vocabularies outside feature encoders; it is not concatenated to a product
fingerprint, route representation, ReaFNN input, or XGB feature row.

    target product
      -> family-specific specialist artifacts selected by the evaluation partition
      -> family-tuned EditRetro route proposals
      -> KNN wide recall plus ReaFNN condition-pool selection
      -> tabular XGB-LTR full-system reranking
      -> fixed R-GNN structural representation plus XGBoost temperature prediction
      -> ranked reaction systems

## Evaluation Protocol

- Fixed end-to-end test manifest: 3,860 product identities across six families.
- Fixed Stage 1 route caches: 3,833 products have a candidate slate in every
  seed; the remaining 27 products remain in the denominator and receive zero
  for full-system metrics.
- The fixed Stage 1 macro Route@1/3/5/10 is
  43.46/56.69/59.97/63.20%; this robustness experiment measures Stage 2/3
  stochastic variation, not Stage 1 retraining variation.
- A full-system match is an exact joint match of canonical route, normalized
  reagent set, and normalized solvent set.
- Stage 2 emits at most 20 reagent-solvent contexts per proposed route.
- ReaFNN, the R-GNN, and XGBoost are retrained independently at seeds 0, 1,
  and 2. KNN retrieval, split, route caches, vocabulary construction, and
  evaluation are held fixed.
- Temperature is a conditional statistic. For a product with an available
  valid-temperature exact system match, the highest-ranked such match supplies
  one absolute error. It is not an all-candidate regression average and it is
  not included in the full-system ranking score.

## Historical Pre-Hardening Performance

All values are unweighted six-family macro averages over three seeds. Rates are
percentages and +/- denotes sample standard deviation (ddof=1).

| Candidate coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48.52 +/- 0.59 | 28.96 +/- 0.91 | 36.61 +/- 1.05 | 39.53 +/- 1.24 | 42.90 +/- 1.00 | 33.78 +/- 0.96 | 35.13 +/- 0.97 |

| Family | Candidate coverage | Sys@1 | Sys@3 | Sys@5 | Sys@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 35.89 +/- 3.19 | 18.30 +/- 0.74 | 22.27 +/- 1.72 | 23.83 +/- 1.95 | 27.94 +/- 1.72 | 21.41 +/- 1.07 | 22.03 +/- 0.98 |
| Buchwald-Hartwig | 48.35 +/- 0.37 | 36.12 +/- 0.98 | 42.77 +/- 1.03 | 44.59 +/- 0.66 | 46.07 +/- 0.56 | 39.77 +/- 0.79 | 40.90 +/- 0.68 |
| Chan-Lam | 68.12 +/- 1.41 | 46.92 +/- 1.36 | 56.50 +/- 1.04 | 59.66 +/- 1.21 | 62.39 +/- 1.55 | 52.52 +/- 1.23 | 53.94 +/- 1.37 |
| Diels-Alder | 28.70 +/- 1.00 | 18.02 +/- 0.89 | 23.71 +/- 0.08 | 25.55 +/- 0.89 | 26.99 +/- 0.87 | 21.34 +/- 0.46 | 22.40 +/- 0.12 |
| Friedel-Crafts Acyl. | 60.63 +/- 0.42 | 25.89 +/- 5.37 | 34.88 +/- 6.23 | 40.42 +/- 5.73 | 47.86 +/- 5.16 | 32.89 +/- 5.36 | 34.71 +/- 5.68 |
| Friedel-Crafts Alkyl. | 49.46 +/- 0.23 | 28.51 +/- 1.38 | 39.53 +/- 0.84 | 43.16 +/- 0.95 | 46.16 +/- 0.22 | 34.78 +/- 0.87 | 36.78 +/- 0.79 |

## Historical Pre-Hardening Temperature Performance

The temperature support is 1,603, 1,592, and 1,576 matched products at seeds
0, 1, and 2, respectively. The support changes with the learned ranking, so
temperature results should be reported together with system accuracy rather
than as a manifest-wide score.

| Family | Conditional MAE (C) | Within +/-5 C | Within +/-10 C | Within +/-20 C |
| --- | ---: | ---: | ---: | ---: |
| Beckmann | 10.52 +/- 1.73 | 38.27 +/- 6.57 | 62.93 +/- 10.86 | 86.13 +/- 4.14 |
| Buchwald-Hartwig | 10.72 +/- 0.24 | 36.09 +/- 1.58 | 61.13 +/- 1.62 | 84.59 +/- 1.53 |
| Chan-Lam | 5.85 +/- 0.53 | 66.05 +/- 2.37 | 83.42 +/- 2.86 | 94.73 +/- 1.52 |
| Diels-Alder | 17.44 +/- 1.31 | 26.19 +/- 2.68 | 48.13 +/- 2.61 | 69.62 +/- 3.74 |
| Friedel-Crafts Acyl. | 9.94 +/- 0.33 | 42.82 +/- 4.18 | 65.75 +/- 2.44 | 86.43 +/- 0.94 |
| Friedel-Crafts Alkyl. | 10.01 +/- 0.12 | 40.57 +/- 2.49 | 63.47 +/- 0.38 | 86.93 +/- 1.62 |
| Macro average | 10.75 +/- 0.14 | 41.66 +/- 1.39 | 64.14 +/- 1.65 | 84.74 +/- 0.38 |

## Historical Reproducibility Record

The compact formal artifacts are retained at
outputs/unified_rgnn_multiseed_20260809/:

- README.md: macro and per-seed summary.
- macro_mean_std.csv, macro_by_seed.csv, and per_family_mean_std.csv:
  reportable aggregates.
- per_family_seed_metrics.csv: family-level seed records.
- compact/: audited per-family metrics and model metadata.

The raw candidate tables and checkpoints were intentionally pruned only after
the collector retained these compact records, to respect the storage budget.
All 18 family-seed temperature metadata records state always_enabled = true
and selection = none; every ranker has 52 non-graph features and every
temperature regressor has 180 features, including 128 R-GNN columns.

## Historical Context

The prior 2026-08-03 gated point snapshot used a different temperature
selection policy and an earlier R-GNN configuration. It remains useful for
traceability in its archived output directory, but is not the current
headline result and must not be compared as a gate-only ablation: the current
implementation simultaneously changed R-GNN capacity to 128 dimensions and
four message-passing layers. A matched architecture with and without a gate
would be required for a causal gate ablation.

## Historical Serial Matched Ablations

The following component controls are aligned to the former serial product-Morgan
configuration, fixed Stage 1 route caches, six-family 3,860-identity manifest,
and seeds 0/1/2. Their compact records and audit contracts are retained in
Experiment/current_mainline_matched_ablation_multiseed_20260830/ and are
described in ablation/current_mainline_matched_ablation_results_20260830.md.

| Controlled comparison | Candidate coverage | Sys@10 | Interpretation |
| --- | ---: | ---: | --- |
| Base Stage 1 vs family-tuned Stage 1 | n/a | Route@10: 14.97% -> 63.20% | Family fine-tuning improves route availability by +48.22 pp. |
| KNN-only + XGB-LTR vs full KNN + ReaFNN + XGB-LTR | 53.39 +/- 0.00% -> 54.44 +/- 0.14% | 39.86 +/- 2.08% -> 44.62 +/- 0.42% | With an XGB-LTR retrained on each arm's own candidate table, ReaFNN adds +1.06 pp candidate coverage and +4.76 pp Sys@10. |
| Full Stage 2 + deterministic no-XGB-LTR vs full historical serial reference | 54.44 +/- 0.14% -> 54.44 +/- 0.14% | 36.45 +/- 0.08% -> 44.62 +/- 0.42% | The Stage 2 generation protocol and recorded candidate-availability metrics match per family and seed; learned XGB-LTR adds +8.17 pp Sys@10 without a measured availability gain. |

For the ReaFNN control, the downstream ranker is re-trained on the changed
candidate distribution, so the comparison is not confounded by applying a
ranker trained on a different pool. For the no-XGB-LTR control, no learned
ranking parameters are fitted; rows are ordered only by the fixed Stage 1/2
prior and stable condition ties. Both controls skip temperature because the
temperature branch neither adds candidates nor affects any Sys@k metric.

The maintained R-GNN remains a fixed input to the temperature XGBoost
regressor only. It is not used by the 52-feature system ranker, and the table
above makes no causal ranking claim for the R-GNN. Earlier frequency-pool and
direct-R-GNN experiments remain archived for traceability but are not used to
attribute effects in the historical serial configuration.

## Candidate-aware GNN Update

`Experiment/legacy_stage3/condition_aware_gnn.py` preserves a route-context interaction
network on top of frozen 64-dimensional R-GNN features and learned reagent
and solvent token embeddings. Its score fusion is selected on validation
full-system Top-10 accuracy only. The completed Beckmann interaction-model pilot selected
`alpha = 0`, and the related six-family auxiliary-token residual probe selected
no nonzero residual. This implementation is preserved for further development,
but it is not part of the official model or of the result tables above.

## Baseline Scope

Four reproducible baselines are complete on the same fixed manifest. Two are
direct Product-to-Condition models, which predict contexts from the target
product before pairing them with frozen Stage 1 routes; two are downstream
route-conditioned baselines.

`Product-Bernoulli Naive Bayes` (B1) is deterministic under fixed data and
hyperparameters. B2 Product-GNN, B3 EditRetro + Sequential FNN, and B4
EditRetro + Reaction-GCNN were independently retrained at seeds `0`, `1`, and
`2` with fixed Stage 1 caches and validation-only fusion selection.

| Method | Replication | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Product-Bernoulli Naive Bayes | deterministic | 31.68 | 9.10 | 14.87 | 17.45 | 21.48 | 13.15 | 14.56 |
| Product-GNN | 3 seeds | 38.33 +/- 0.28 | 6.11 +/- 0.49 | 12.84 +/- 0.56 | 16.75 +/- 0.43 | 23.03 +/- 0.73 | 11.40 +/- 0.54 | 13.32 +/- 0.58 |
| EditRetro + Sequential FNN | 3 seeds | 45.85 +/- 0.26 | 16.75 +/- 0.27 | 24.32 +/- 0.28 | 27.68 +/- 0.40 | 31.71 +/- 0.10 | 22.01 +/- 0.22 | 23.64 +/- 0.18 |
| EditRetro + Reaction-GCNN | 3 seeds | 38.15 +/- 0.24 | 7.43 +/- 0.21 | 13.16 +/- 0.10 | 16.39 +/- 0.34 | 21.10 +/- 0.28 | 11.88 +/- 0.11 | 13.27 +/- 0.10 |
| ProSys historical serial reference | 3 seeds | 54.44 +/- 0.14 | 27.12 +/- 0.37 | 36.84 +/- 0.80 | 40.47 +/- 0.77 | 44.62 +/- 0.42 | 33.28 +/- 0.48 | 34.70 +/- 0.53 |

B2-B4 are matched to the historical serial reference in replication depth and use the same
fixed full test manifest. B1 remains a deterministic reference, not a
pseudo-replicated stochastic model. The reaction-level split is not
product-disjoint, which remains a required disclosure for every product-only
model.

The detailed comparison is documented in
[`baseline/multiseed_baseline_results_20260810.md`](baseline/multiseed_baseline_results_20260810.md).
The direct Product-to-System Transformer remains unreported because no
verified compatible checkpoint was available. It is not a zero-valued baseline.

## Audit Status

- Historical serial matched-ablation audit: PASS; 36 family-seed records cover two
  three-seed controls, every arm/seed preserves the 3,860-identity denominator
  with 3,833 candidate slates and 27 no-slate failures, and all 18 no-XGB-LTR
  records exactly match the corresponding official Stage 2 candidate-pool
  protocol and coverage fields.

- Strict canonical-reaction split audit: PASS; every reported train/validation/
  test overlap count is zero.
- Mainline candidate stability after the leave-one-reaction-out correction:
  PASS; held-out candidate keys are unchanged.
- Candidate-aware GNN residual probes: PASS as validation-only exploratory
  controls; no nonzero residual was accepted, so headline rankings are unchanged.
- Earlier historical ablation audit: PASS; fixed manifest, candidate cap,
  target-feature exclusion, and full-reference equality all pass. It is
  retained as a historical control record and is not the source of the current
  ReaFNN or XGB-LTR attribution figures.
- Multi-seed baseline audit: PASS; `54` family-level records cover B2-B4 across
  three seeds and six families, with the same `3,860`-product manifest,
  `3,833` candidate slates, and `27` retained no-route failures in every run.
- Fixed R-GNN temperature branch audit: PASS; all 18 ranker feature lists
  exclude route_gnn_feat_*, all 18 temperature models contain the 128 graph
  features, and metadata records always_enabled = true with selection = none.

See [`project_audit_20260727.md`](project_audit_20260727.md) for scope,
remaining limitations, and links to the raw audit artifacts.
