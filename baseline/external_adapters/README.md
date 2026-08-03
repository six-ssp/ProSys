# External Baseline Adapters

This directory keeps the third-party repositories intact and provides the
ProSys-specific contract around them. It is the only supported path for turning
their inputs and outputs into the common Non-Oracle candidate-table protocol.

## Vendor sources

The exact source URLs and commits are recorded in `source_manifest.json`.

| Reference | ProSys role | Required adaptation |
|---|---|---|
| MolecularTransformer | Direct Product-to-System Transformer | Product-only source and tagged reactant/reagent/solvent target |
| Reaction_condition_recommendation | Sequential FNN | Catalyst is merged into the reagent set; labels allow up to three reagents and two solvents |
| reaction-gcnn | Reaction-GCNN | Six family-specific vocabularies replace the upstream hard-coded cross-coupling dictionaries |

The upstream FNN code targets legacy Keras/Python 2 conventions and the upstream
GCNN code targets legacy Chainer with fixed dataset names. The vendor directories
are retained as reference implementations; ProSys-specific data and output
handling are implemented here rather than patching vendor source code.

## Data packages

Generate inputs only after choosing a fresh output directory:

```bash
cd /root/autodl-tmp/ProSys
/root/miniconda3/envs/ProSys/bin/python -m baseline.external_adapters.build_datasets \
  --families all \
  --models all \
  --output-root outputs/baseline_inputs_20260725
```

The generator creates the following artifacts per family:

- `molecular_transformer`: tokenized product-to-system train/validation pairs,
  product-only test sources, a train-only condition vocabulary, and a separate
  held-out gold-system file.
- `sequential_fnn`: aggregated route labels, the train-only context library, and
  `test_stage1_routes.jsonl` built exclusively from the Stage 1 route cache.
- `reaction_gcnn`: graph-model CSV inputs, the same train-only vocabulary and
  context library, and a Stage 1-only test route CSV.

`--include-retro-pretrain` additionally exports product-to-reactant examples
from the same filtered USPTO corpus used by the Stage 1 base model. It does not
use any Reaxys test record or condition label.

## Formal Baseline 2/3 run

For the formal Sequential FNN and Reaction-GCNN comparison, reuse the existing
mainline test route caches and create a separate validation-only route cache.
The validation cache is used only to select the relative route/condition score
weight; it is not a second test set.

```bash
cd /root/autodl-tmp/ProSys
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ProSys/bin/python -B -m baseline.external_adapters.build_datasets \
  --families all \
  --models sequential_fnn,reaction_gcnn \
  --route-root outputs/stage1_routes \
  --validation-route-root outputs/stage1_routes_validation \
  --output-root outputs/baseline23_inputs_<date>

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ProSys/bin/python -B -m baseline.external_adapters.run_baselines23 \
  --input-root outputs/baseline23_inputs_<date> \
  --output-root outputs/baselines/non_oracle_external_b23_<date> \
  --families all --methods all --device cuda \
  --max-epochs 50 --patience 7 --seed 0 --top-contexts 20
```

The runner creates one family-specific model for each method. It trains with
train rows, uses gold-route validation rows only for early stopping, selects the
route-score ratio from validation Stage 1 routes, and evaluates the untouched
mainline test cache once. `summary.csv` includes per-family results and an
equal-family `MACRO-AVG`; `run_metadata.json` and `fusion_selection.json` keep
the complete audit trail.

## Prediction contract

Sequential FNN and Reaction-GCNN predictors should emit one JSONL object per
Stage 1 route:

```json
{
  "sample_index": 12,
  "retro_rank": 3,
  "candidates": [
    {
      "reagent_ids": ["R0001", "R0042"],
      "solvent_ids": ["S0003"],
      "condition_score": -4.21,
      "temperature_pred": 55.0
    }
  ]
}
```

Direct Transformer predictors should emit one JSONL object per product input:

```json
{
  "sample_index": 12,
  "beams": [
    {
      "text": "<REACTANT> C C O <REAGENT> R0001 <SOLVENT> S0003",
      "sequence_score": -6.4
    }
  ]
}
```

Convert either format to the common candidate table with validation-selected
score weights:

```bash
/root/miniconda3/envs/ProSys/bin/python -m baseline.external_adapters.export_candidates \
  --method sequential_fnn \
  --predictions outputs/baseline_runs/example/predictions.jsonl \
  --vocabulary outputs/baseline_inputs_20260725/sequential_fnn/Beckmann/label_vocabulary.json \
  --route-manifest outputs/baseline_inputs_20260725/sequential_fnn/Beckmann/test_stage1_routes.jsonl \
  --output outputs/baseline_runs/example/candidates.csv \
  --route-weight 0.5 \
  --condition-weight 1.0
```

The candidate exporter canonicalizes condition labels, de-duplicates systems,
keeps at most 20 contexts per route, and writes `system_score` for the existing
ProSys evaluator. The two score weights must be fixed using validation data.

For Reaction-GCNN, the builder retains upstream-style CSV files and additionally
writes `train_routes.jsonl`, `val_routes.jsonl`, and
`test_stage1_routes.jsonl` for the PyTorch adapter. The package also contains
`test_manifest.jsonl`, which lists every formal test sample, including samples
for which Stage 1 produced no route.

## Fixed-denominator evaluation

Use `evaluate_candidates.py` after exporting candidates. It reuses shared ProSys
gold matching but divides all Non-Oracle all-sample metrics by the full
`test_manifest.jsonl` count. Samples without any Stage 1 route remain in the
denominator as zero-hit slates.

`/root/miniconda3/envs/ProSys/bin/python -m baseline.external_adapters.evaluate_candidates --candidates <candidates.csv> --gold-split <test_split.txt> --test-manifest <test_manifest.jsonl> --labeled-output <labeled.csv> --output <metrics.json> [--temperature-column temperature_pred]`

## MolecularTransformer weights

The upstream README points to an IBM Box collection rather than a direct model
artifact. That collection returned HTTP 404 when checked on 2026-07-25, and no
compatible official checkpoint was found in the upstream Git history, releases,
or public model registry. The code therefore does not silently substitute a
different model.

When a verified direct checkpoint URL and SHA256 are available, use:

```bash
/root/miniconda3/envs/ProSys/bin/python -m baseline.external_adapters.fetch_molecular_transformer_weights \
  --url https://example.org/model.pt \
  --sha256 <sha256> \
  --output baseline/MolecularTransformer/checkpoints/model.pt
```

The original published checkpoint is a forward reaction-prediction model. Even
if recovered, it must be treated as initialization only: ProSys needs a
product-to-system target format and requires family-specific fine-tuning.
