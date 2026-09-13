# Multi-Seed Baseline Evidence

This directory is the compact, Git-tracked evidence record for the completed
B1-B4 six-family baseline comparison. Large model states, candidate tables, and
per-sample predictions remain regenerable local `outputs/` material.

- `macro_mean_std.csv` and `macro_by_seed.csv`: reportable aggregate metrics.
- `per_family_seed_metrics.csv` and `deterministic_b1_per_family.csv`:
  family-resolved source values.
- `experiment_config.json`: fixed route-cache hashes and run protocol.
- `seed_<n>/external_compact/`: per-family model metadata, validation fusion
  selections, run metadata, and compact result summaries for B3/B4.
