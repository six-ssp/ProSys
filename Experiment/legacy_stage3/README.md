# Legacy Stage 3 Experiments

This package preserves retired Stage 3 ranking experiments that are not
imported by the maintained pipeline.

- `condition_aware_gnn.py`: route-context residual scorer tested with a
  validation-only fusion gate.
- `probes/`: candidate-aware, bottleneck, and condition-residual ranking
  probes.

All completed probes selected no useful nonzero ranking contribution. The
maintained Stage 3 remains the 52-feature XGB-LTR ranker plus the separate
R-GNN-assisted temperature regressor in `stage3_XGBoost/`.
