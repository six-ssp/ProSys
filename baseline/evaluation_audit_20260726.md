# Baseline Result and Legality Audit (2026-07-26)

> **Historical mainline snapshot warning.** This audit remains valid for its
> fixed-manifest, candidate-budget, and information-flow checks. Its ProSys
> numbers predate the corrected leave-one-reaction-out training retrieval and
> are not current performance values. Cite
> [`multiseed_baseline_results_20260810.md`](multiseed_baseline_results_20260810.md)
> and [`CURRENT_RESULTS.md`](../CURRENT_RESULTS.md) instead.

## Scope

This audit covers the completed six-family Non-Oracle comparison:

- ProSys mainline: Stage 1 EditRetro, Stage 2 KNN + ReaFNN, and Stage 3 Reaction-GNN + XGBoost.
- Baseline 2: the adapted Sequential FNN condition predictor.
- Baseline 3: the adapted Reaction-GCNN multi-label condition predictor.

All three methods consume the same saved Stage 1 top-10 route caches. The Direct Product-to-System Transformer remains excluded because a compatible upstream checkpoint is still unavailable.

## Corrected Evaluation Protocol

The historical mainline summary used only samples with a non-empty candidate slate as its denominator. This is not an acceptable end-to-end denominator because a Stage 1 no-route result is itself a valid failure. On 2026-07-26, the saved `test_scored.csv` files were re-evaluated with every Stage 1 test identity included.

- Test population: `3860` samples across six families.
- Non-empty candidate slates: `3833`.
- Missing Stage 1 routes: `27`; each is a zero hit for `cover`, `System@k`, `MRR`, and `nDCG@10`.
- Temperature remains conditional on the highest-ranked valid full-match candidate; its support count is reported and is not scaled by missing slates.
- The resulting official files are `outputs/stage23_mainline_reafnn_gnn_fused_20260723/fixed_manifest_reaudit_20260726/` and the refreshed `overview.md` / `result.json` summaries in both mainline output trees.

The denominator correction changes the mainline macro result only slightly: legacy candidate-slate `cover = 49.46%` and `System@10 = 40.45%` become fixed-manifest `cover = 49.18%` and `System@10 = 40.22%`.

## Results

| Method | Cover | System@1 | System@3 | System@5 | System@10 | MRR | nDCG@10 | Temp MAE (n) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProSys mainline | 49.18 | 19.85 | 29.76 | 34.54 | 40.22 | 26.42 | 28.85 | 10.35 (1603) |
| EditRetro + Sequential FNN | 45.99 | 17.06 | 24.64 | 27.98 | 31.67 | 22.27 | 23.84 | 11.87 (1464) |
| EditRetro + Reaction-GCNN | 38.01 | 7.93 | 13.16 | 16.36 | 21.03 | 12.18 | 13.50 | N/A |

Values except temperature MAE are percentages. Reaction-GCNN has no temperature head, so it is not assigned a temperature score.

### What the Baseline Says

- Sequential FNN is the strong baseline: it retains `45.99%` cover and reaches `31.67%` System@10. This is a credible result, not a failed baseline.
- ProSys improves over Sequential FNN by `+3.18` pp cover, `+2.79` pp System@1, `+5.12` pp System@3, `+6.56` pp System@5, and `+8.54` pp System@10.
- The growing gap with `k` is consistent with the staged design: the mainline has a modest candidate-recall advantage and a larger system-level reranking advantage.
- Reaction-GCNN is lower than the mainline in all six families. This supports the claim that a graph encoder plus independent multi-label heads is not by itself sufficient for structured route-condition candidate ranking.

Sequential FNN is stronger than the mainline only for Chan-Lam at System@10 (`56.15%` versus `52.56%`). The mainline wins the remaining five families:

| Family | Mainline System@10 | Sequential FNN System@10 | Mainline - FNN (pp) |
| --- | ---: | ---: | ---: |
| Beckmann | 27.66 | 23.83 | +3.83 |
| Buchwald-Hartwig | 42.86 | 19.93 | +22.93 |
| Chan-Lam | 52.56 | 56.15 | -3.59 |
| Diels-Alder | 27.17 | 20.73 | +6.43 |
| Friedel-Crafts Acyl. | 49.68 | 43.37 | +6.32 |
| Friedel-Crafts Alkyl. | 41.38 | 26.03 | +15.35 |

The mainline has the lower temperature MAE (`10.35` versus FNN `11.87`), but the two MAEs are computed on different conditional support sets: mainline `n=1603`, FNN `n=1464`. The FNN's higher `Temp±5C` and `Temp±10C` rates therefore should not be read as an unconditional temperature advantage. Temperature comparisons must always report both MAE and support size.

## Integrity Checks

| Check | Result | Interpretation |
| --- | --- | --- |
| Canonical reaction split audit | Pass | Strict audit reports zero train/validation/test overlap for every Stage 2 family and zero Stage 1 train/validation overlap with the Stage 2 test set. |
| Mainline-baseline test identities | Pass | For every family, the Stage 1 cache identity set exactly equals the Sequential FNN and Reaction-GCNN test manifests. |
| Fixed denominator | Pass | Candidate slates plus missing slates equal the family test manifest in all 12 baseline family-method runs and all six mainline runs. |
| Train-only condition memory | Pass | Baseline vocabularies and context libraries are constructed from family training rows only. |
| Validation-only selection | Pass | Early stopping and route-score fusion weights use validation data; test labels are loaded only for the final evaluator. |
| Stage 3 target-feature boundary | Pass | The mainline test candidate table carries labels for post-hoc scoring, but saved XGBoost metadata contains none of `label`, match, yield, or temperature target columns as input features. |
| Downstream gold-route access | Pass | Route-cache loaders expose only predicted reactants, product, rank, and confidence to downstream models; cached `gold_reactants` is not passed through. |
| Candidate budget | Pass | Each method permits at most 20 contexts for each of at most 10 Stage 1 routes, i.e. at most 200 candidates per product. Observed means are 184.33 mainline and 185.15 for each baseline. |

The small difference between canonical split counts and the `3860` manifest count is expected: the audit prints unique canonical reactions, whereas evaluation retains the official reaction-record identities. There is no cross-split overlap, and all compared methods use the same official identities.

## Limits to State Explicitly

- These are single-seed (`seed=0`) results. A paper-quality robustness table should eventually report at least three seeds for the mainline and both learned baselines.
- The Sequential FNN and Reaction-GCNN are faithful ProSys adapters of the linked external model families under a unified data contract. They are not claims of reproducing the original repositories' published benchmark numbers.
- A development smoke run evaluated the Beckmann test cache before the formal batch. The saved formal runner itself performs only validation-based selection, and no test-dependent selection is encoded in its artifacts; nevertheless, this means the current result should be described as a reproducible fixed-test result rather than a fully blind holdout result.
- The candidate space is history-centered. This is a deliberate and realistic setting, but it should not be described as unrestricted de novo condition generation.
- Candidate capacity is matched, but the methods generate different candidate identities. This is appropriate for a pipeline baseline; use the stage-specific ablations, rather than this table alone, to attribute gains to an individual module.

## Conclusion

The comparison is methodologically valid for an internal and paper-facing controlled baseline study once the fixed-manifest denominator is used. It supports a precise claim: under matched Stage 1 inputs and matched candidate capacity, the full KNN + ReaFNN + Reaction-GNN + XGBoost pipeline provides a clear average system-ranking gain over both the sequential fingerprint baseline and the graph-only multi-label baseline. It does not yet support a broad claim of statistically significant superiority or direct reproduction of upstream benchmark results.
