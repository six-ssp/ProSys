# Independent Audit: Current Mainline Matched Ablations

Status: PASS
Scope: six families, three seeds, two matched controls
Artifact: Experiment/current_mainline_matched_ablation_multiseed_20260830/

## Fixed Evaluation Contract

- Test manifest: 3,860 identities for every family/seed/arm.
- Candidate slates: 3,833 for every family/seed/arm.
- Retained no-slate identities: 27 for every family/seed/arm.
- Product-Morgan retrieval: radius 2, 4,096 bits, K=64, 64-context wide pool,
  and 20-context final cap.
- Test routes: persisted Stage 1 predictions, unchanged from the promoted
  current mainline.

## Completed Checks

1. Completion metadata reports 36 compact results, equal to 6 families x 3
   seeds x 2 arms.
2. Every compact result was checked against the expected family, seed, manifest,
   product-Morgan retrieval configuration, candidate cap, and split boundary.
3. The KNN-only arm has ReaFNN disabled, uses a re-trained 52-feature tabular
   XGB-LTR, and has no graph-ranking or temperature model.
4. The full-Stage-2 deterministic arm has no fitted ranking parameters and
   uses only the declared Stage 1/Stage 2 sort keys.
5. For all 18 full-Stage-2 deterministic records, Stage 2 protocol fields and
   candidate-coverage fields exactly match the corresponding official
   full-mainline family-seed record.
6. Aggregate CSV values agree with the compact result records. The reported
   macro Sys@10 values are 39.86 +/- 2.08% for KNN-only plus XGB-LTR and
   36.45 +/- 0.08% for full Stage 2 plus deterministic no-XGB-LTR.

## Interpretation Boundary

This audit supports causal claims about ReaFNN candidate-stage contribution and
XGB-LTR ranking contribution under the fixed current configuration and
matched Stage 2 generation procedure. Because raw candidate tables were
pruned after compaction, this is not a retained per-row candidate-key hash
audit.
It does not measure Stage 1 retraining variance. Temperature fitting is omitted
only from the two controls because temperature neither changes candidate
membership nor contributes to Sys@k.
