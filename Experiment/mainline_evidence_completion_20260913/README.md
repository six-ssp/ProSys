# Current-mainline evidence completion

This record supplements, rather than silently replaces, the promoted
2026-09-03 parallel mainline. See [PLAN.md](PLAN.md) for the full completion
contract. `status.json` and live process inspection distinguish running jobs
from completed results. `SUMMARY.md`, `replay_audit.json`, and
`finalization.json` are emitted only after all 18 runs finish and replay passes.

Completed: all 18 runs, independent row replay, 18 traced cases and six cold
product-query smokes. See [COMPLETION_REPORT.md](COMPLETION_REPORT.md) for
the requirement-to-evidence matrix, findings and limitations.

## Retained evidence

- `compact/seed_<seed>/<family>/`: full result, exact-control audit,
  reconstruction cost, model/data provenance, per-query failure classifications,
  candidate slates and model bundles. Row-level files and weights stay local.
- `per_family_seed_controls.csv`: reconstructed full, no-LTR, and no-R-GNN
  metrics. `reconstructed_minus_promoted.csv` explicitly records drift from
  historical scores, including temperature changes.
- `subgroups_per_family_seed.csv`: seen/unseen canonical-product subgroups and
  whether a gold context occurs in the training library. These are diagnostic
  slices of the existing split, not independently collected external data.
- `failures_per_family_seed.csv`: exhaustive route/pool/ranking/hit counts.
- `cost_per_family_seed.csv`: measured reconstruction costs, not isolated
  inference latency. Per-family loops run concurrently with bounded threads.
- `examples/`: three fully traced seed-0 examples per family. `example.md`
  uses the first exact hit to label Top-k, not a later positive candidate.

## Read-only product inference

After the corresponding bundle is complete:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/predict_product.py \
  --product 'N#CCCc1ccccc1' --family Beckmann \
  --artifact_root Experiment/mainline_evidence_completion_20260913/compact/seed_0/Beckmann \
  --output outputs/product_query_demo --device cuda:0
```

The family argument selects a deployed expert, not a learned molecular feature.
No gold reactants, conditions, temperature or yield are supplied. Model/library
checksums are verified before Stage 2 loads; the inference path never fits a
model or selects a fusion weight. `--route_cache` optionally reuses frozen
Stage 1 predictions for a product; absent that flag, EditRetro decodes freshly.
Use a new output directory for each invocation. The CLI preserves routes,
all ranked intermediate rows, Top-10 predictions and measured per-stage times.
Initial molecular-feature parity and no-training tests are in
`ablation/test_product_inference.py`; one retained product also matched all
200 candidate identities, all 52 LTR features, scores and temperatures.

## Content-bound reproduction

The maintained Stage 2/3 shell suite invokes `scripts/run_verified_mainline.py`.
It accepts the original pipeline arguments, binds caches to split bytes,
test/validation route-cache bytes, referenced Stage 1 checkpoint bytes,
configuration, model-source files and package versions, and checks all retained
table/model output hashes before reuse. Each family is protected by a process
lock. Unknown legacy or partial caches fail closed. Use a new output root or
explicit `--force_rebuild`; forced rebuild also retrains ReaFNN and R-GNN.
Pruned results are reporting artifacts, not valid inference/training caches.

The original lower-level pipeline module remains importable by frozen
experiments; call the verified entrypoint for maintained cache-safe runs.
Checksums prevent future stale reuse; they do not retrospectively prove when
an older unmanifested route cache was generated. The fixed benchmark's route
caches remain inherited inputs, rather than newly trained Stage 1 evidence.

The full raw-data shell workflow has recording-stub integration tests for
preprocessing, base training, family tuning, test and validation route caches,
verified Stage 2/3, and statistics collection. Those tests do not re-execute
expensive base training or alter frozen data.

## Known audit boundary

[CANONICALIZATION_NOTE.md](CANONICALIZATION_NOTE.md) records one Diels-Alder
training reaction whose cross-dot ring closure is mishandled by the frozen
side-splitting helper. A whole-molecule re-audit found no split overlap and no
validation/test affected sides. The representation issue is documented, not
silently fixed in the middle of a paired reconstruction.
