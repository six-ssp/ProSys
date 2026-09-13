# Multi-Seed Baseline Robustness

B2 Product-GNN, B3 EditRetro + Sequential FNN, and B4 EditRetro + Reaction-GCNN were independently retrained at seeds 0, 1, and 2 with fixed Stage 1 route caches, formal family splits, validation-only fusion selection, and the fixed full test manifest.

B1 Product-Bernoulli Naive Bayes is deterministic under its fixed training data and hyperparameters, so it is retained as one deterministic reference rather than pseudo-replicated.

| Method | Seeds | Candidate recall | Full-system Top-1 | Full-system Top-10 | MRR | nDCG@10 | Conditional temp. MAE (C) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 Product-Bernoulli Naive Bayes | deterministic | 31.68 | 9.10 | 21.48 | 13.15 | 14.56 | N/A |
| B2 Product-GNN | 0,1,2 | 38.33 +/- 0.28 | 6.11 +/- 0.49 | 23.03 +/- 0.73 | 11.40 +/- 0.54 | 13.32 +/- 0.58 | N/A |
| B4 EditRetro + Reaction-GCNN | 0,1,2 | 38.15 +/- 0.24 | 7.43 +/- 0.21 | 21.10 +/- 0.28 | 11.88 +/- 0.11 | 13.27 +/- 0.10 | N/A |
| B3 EditRetro + Sequential FNN | 0,1,2 | 45.85 +/- 0.26 | 16.75 +/- 0.27 | 31.71 +/- 0.10 | 22.01 +/- 0.22 | 23.64 +/- 0.18 | 12.20 +/- 0.29 |

External B3/B4 work directories were pruned only after their per-family run metadata, validation fusion selections, and model metadata had been copied into `seed_<n>/external_compact/`. Product-GNN retains its compact model and top-10 audit outputs directly.
