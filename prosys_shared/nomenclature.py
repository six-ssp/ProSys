"""Canonical public terminology for ProSys reports.

Persisted metric keys intentionally remain stable for backwards-compatible
CSV/JSON parsing.  This module contains only reader-facing names used in
generated reports and documentation.
"""

from __future__ import annotations

from typing import Final


PUBLIC_METRIC_LABELS: Final[dict[str, str]] = {
    'route_recall_top1': 'Route@1',
    'route_recall_top3': 'Route@3',
    'route_recall_top5': 'Route@5',
    'route_recall_top10': 'Route@10',
    'pool_route_coverage': 'Route recall',
    'pool_context_coverage': 'Condition recall',
    'pool_coverage': 'Candidate recall',
    'system_top1_all': 'Full-system Top-1 accuracy',
    'system_top3_all': 'Full-system Top-3 accuracy',
    'system_top5_all': 'Full-system Top-5 accuracy',
    'system_top10_all': 'Full-system Top-10 accuracy',
    'system_mrr': 'MRR',
    'system_ndcg10': 'nDCG@10',
    'temperature.mae': 'MAE (deg C)',
    'temperature.rmse': 'RMSE (deg C)',
    'temperature.within_5c': 'Within +/-5 deg C',
    'temperature.within_10c': 'Within +/-10 deg C',
    'temperature.within_20c': 'Within +/-20 deg C',
}


DIRECT_BASELINE_DISPLAY_NAMES: Final[dict[str, str]] = {
    'product_naive_bayes': 'Product-Bernoulli Naive Bayes',
    'product_gnn': 'Product-GNN',
}


def direct_baseline_display_name(method: str) -> str:
    """Return the paper-facing method name without changing method keys."""

    return DIRECT_BASELINE_DISPLAY_NAMES.get(method, method)
