"""Shared field definitions for ProSys Stage 2 V2."""

from __future__ import annotations

STAGE2A_CANDIDATE_COLUMNS = [
    'family',
    'sample_index',
    'reaction_id',
    'product',
    'reactants',
    'retro_rank',
    'retro_score',
    'retro_probability',
    'reagent_norm',
    'solvent_norm',
    'from_fnn',
    'from_product_exact',
    'from_product_scaffold',
    'from_product_knn',
    'product_exact_pair_support',
    'product_exact_reagent_support',
    'product_exact_solvent_support',
    'product_scaffold_pair_support',
    'product_scaffold_reagent_support',
    'product_scaffold_solvent_support',
    'product_knn_pair_support',
    'product_knn_reagent_support',
    'product_knn_solvent_support',
    'product_pair_freq',
    'product_pair_mean_yield',
]

SUPPORT_FEATURE_COLUMNS_V2 = [
    'from_fnn',
    'from_product_exact',
    'from_product_scaffold',
    'from_product_knn',
    'product_exact_pair_support',
    'product_exact_reagent_support',
    'product_exact_solvent_support',
    'product_scaffold_pair_support',
    'product_scaffold_reagent_support',
    'product_scaffold_solvent_support',
    'product_knn_pair_support',
    'product_knn_reagent_support',
    'product_knn_solvent_support',
    'product_pair_freq',
    'product_pair_mean_yield',
]

ROUTE_DENSE_COLUMNS_V2 = [
    'retro_rank',
    'retro_score',
    'retro_probability',
    'route_component_count',
    'reactants_length',
]

CONTEXT_DENSE_COLUMNS_V2 = [
    'num_reagents',
    'num_solvents',
]

PRODUCT_DESCRIPTOR_COLUMNS_V2 = [f'product_feat_{idx}' for idx in range(8)]
ROUTE_GRAPH_FEATURE_COLUMNS_V2 = [f'route_graph_feat_{idx}' for idx in range(24)]

LABEL_COLUMNS_V2 = [
    'label',
    'label_type',
    'rank_relevance',
    'sample_weight',
    'route_match',
    'context_match',
    'temperature_gold',
]

FEATURE_COLUMNS_V2 = (
    ROUTE_DENSE_COLUMNS_V2
    + CONTEXT_DENSE_COLUMNS_V2
    + PRODUCT_DESCRIPTOR_COLUMNS_V2
    + SUPPORT_FEATURE_COLUMNS_V2
)

LEARNED_FEATURE_COLUMNS_V2 = [
    'route_fp',
    'route_graph_features',
    'product_fp',
    'reagent_multi_hot',
    'solvent_multi_hot',
]

PRODUCT_DESCRIPTOR_DIM_V2 = 8
ROUTE_GRAPH_DIM_V2 = 24

