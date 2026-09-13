# Diels-Alder canonicalization sensitivity

Status: complete (six fits, three seeds per arm).

Raw data, splits and Stage-1 predictions are fixed. Both arms use deterministic R-GNN.
The legacy arm explicitly restores the old helper in an isolated process; model source hashes otherwise agree.
This is a complete-pipeline repair intervention, not an isolated temperature-head ablation.

| mode      |   seed |   n_queries |   sys1_percent |   sys3_percent |   sys5_percent |   sys10_percent |   temperature_n |   temperature_mae_c |   temperature_within10_percent |
|:----------|-------:|------------:|---------------:|---------------:|---------------:|----------------:|----------------:|--------------------:|-------------------------------:|
| legacy    |      0 |         762 |        14.3045 |        20.4724 |        22.9659 |         26.2467 |             208 |             18.8373 |                        46.1538 |
| corrected |      0 |         762 |        13.9108 |        20.8661 |        23.0971 |         25.5906 |             208 |             17.7030 |                        46.1538 |
| legacy    |      1 |         762 |        17.1916 |        21.6535 |        22.9659 |         26.2467 |             208 |             18.9986 |                        44.2308 |
| corrected |      1 |         762 |        18.2415 |        23.0971 |        25.4593 |         29.0026 |             208 |             18.2140 |                        46.1538 |
| legacy    |      2 |         762 |        16.0105 |        22.1785 |        24.1470 |         26.9029 |             208 |             19.1207 |                        45.6731 |
| corrected |      2 |         762 |        17.1916 |        22.5722 |        24.5407 |         27.6903 |             208 |             18.4077 |                        46.1538 |

Candidate and conditional-temperature identity checks:

|   seed | same_raw_splits_and_routes   | same_recorded_model_source_hashes   | same_candidate_identities   | same_selected_temperature_identities   |   sys1_delta_pp |   sys3_delta_pp |   sys5_delta_pp |   sys10_delta_pp |
|-------:|:-----------------------------|:------------------------------------|:----------------------------|:---------------------------------------|----------------:|----------------:|----------------:|-----------------:|
|      0 | True                         | True                                | False                       | False                                  |         -0.3937 |          0.3937 |          0.1312 |          -0.6562 |
|      1 | True                         | True                                | False                       | False                                  |          1.0499 |          1.4436 |          2.4934 |           2.7559 |
|      2 | True                         | True                                | False                       | True                                   |          1.1811 |          0.3937 |          0.3937 |           0.7874 |

Different selected temperature identities, where present, preclude treating the two aggregate MAEs as an exact same-support temperature-only comparison.
No result here replaces the promoted six-family mainline; clean-validation selection and expert repeats remain separate work.
