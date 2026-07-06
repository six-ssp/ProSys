# ProSys Goal

## Project target

`ProSys` aims to recommend a complete reaction system from a target product:

```text
product
-> reactants route
-> reagent set
-> solvent set
-> temperature
```

The project is not just a condition classifier.  
It is a staged **product-to-system** framework.

## Current maintained mainline

After the current cleanup, the maintained mainline is:

```text
Stage 1  EditRetro route generation
Stage 2  KNN candidate screening
Stage 3  XGBoost reranking + temperature prediction
```

That is:

- `stage1/`
- `stage2_KNN/`
- `stage3_XGBoost/`

## Evaluation priority

Primary setting:

- **Non-Oracle end-to-end**

Meaning:

- use Stage 1 predicted `route_cache.json`
- compare all methods under the same route noise
- report real end-to-end `sys@k`

Secondary setting:

- `Oracle`

It is kept only for upper-bound analysis and historical debugging.

## Historical reference

The original FNN pipeline is still kept as a **historical baseline**, not as the mainline:

- original FNN candidate generation
- original ranking / temperature heads

It exists to answer:

- how much the current `KNN + XGBoost` line improves over the original project

## What the codebase should support

1. Clean Stage 1 route generation by family
2. Family-specific Stage 2 candidate screening
3. Stage 3 reranking and temperature prediction
4. Historical baseline comparison
5. Stage 2 / Stage 3 ablation experiments
6. Re-renderable summary tables and logs

## Current success criteria

### Method

- `KNN` should act as the feasible-condition filter
- `XGBoost` should act as the final reranker and temperature predictor
- baseline and ablation should explain why both modules matter

### Engineering

- fixed entrypoints
- clear active vs archived directories
- reproducible Non-Oracle result root
- no scattered one-off scripts in the repo root

### Reporting

- per-family tables
- macro-average tables
- historical baseline table
- Stage 2 ablation table
- Stage 3 ablation table
- temperature hit rates under the current top-10 end-to-end definition
