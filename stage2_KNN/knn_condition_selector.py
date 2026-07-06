"""KNN candidate screening for ProSys Stage 2.

This module turns a route `(reactants, product)` into a feasible condition
candidate pool by retrieving similar training reactions and aggregating their
historical `(reagent_norm, solvent_norm)` contexts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.common import (
    RouteRecordLite,
    base_candidate_row,
    load_route_records,
    load_split_rows,
    split_file_for_family,
)
from prosys_shared.features import normalize_fp, reaction_morgan_fp
from prosys_shared.product_memory import normalize_condition_labels, safe_float
from prosys_shared.route_cache import load_route_records_from_cache


class KNNContextPoolBuilder:
    """Family-specific KNN context retriever.

    Memory is built strictly from the family train split. During inference, each
    query route is encoded with a reaction Morgan fingerprint and matched
    against the train-memory matrix using cosine similarity over L2-normalized
    fingerprints.
    """

    def __init__(self, repo_root: Path, family: str, *, top_k: int, max_contexts: int, fpsize: int, radius: int):
        self.repo_root = Path(repo_root)
        self.family = family
        self.top_k = int(top_k)
        self.max_contexts = int(max_contexts)
        self.fpsize = int(fpsize)
        self.radius = int(radius)

        train_rows = load_split_rows(split_file_for_family(self.repo_root, self.family, 'train'))
        self.route_matrix, self.route_contexts = self._build_route_memory(train_rows)
        self.global_contexts = self._global_contexts(train_rows)

    def _build_route_memory(self, rows: list[dict]) -> tuple[np.ndarray, list[list[dict]]]:
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row['reaction_id'], row['reactants'], row['product'])].append(row)

        matrix_rows: list[np.ndarray] = []
        route_contexts: list[list[dict]] = []
        for (_reaction_id, reactants, product), bucket in grouped.items():
            fp = normalize_fp(reaction_morgan_fp(reactants, product, fpsize=self.fpsize, radius=self.radius))
            matrix_rows.append(fp)
            route_contexts.append(bucket)

        if not matrix_rows:
            return np.zeros((0, self.fpsize * 2), dtype=np.float32), []
        return np.stack(matrix_rows).astype(np.float32), route_contexts

    def _global_contexts(self, rows: list[dict]) -> list[dict]:
        agg: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            key = (normalize_condition_labels(row['reagent_norm']), normalize_condition_labels(row['solvent_norm']))
            stats = agg.setdefault(key, {'count': 0.0, 'yield_sum': 0.0, 'yield_n': 0.0})
            stats['count'] += 1.0
            y = safe_float(row['yield'])
            if not np.isnan(y):
                stats['yield_sum'] += y
                stats['yield_n'] += 1.0

        total = sum(stats['count'] for stats in agg.values()) or 1.0
        rows_out = []
        for (reagent_norm, solvent_norm), stats in agg.items():
            rows_out.append(
                {
                    'reagent_norm': reagent_norm,
                    'solvent_norm': solvent_norm,
                    'context_count': stats['count'],
                    'context_support': stats['count'] / total,
                    'mean_yield': (stats['yield_sum'] / stats['yield_n'] if stats['yield_n'] else 0.0),
                }
            )
        rows_out.sort(key=lambda row: (-row['context_count'], row['reagent_norm'], row['solvent_norm']))
        return rows_out

    def candidate_rows(self, record: RouteRecordLite) -> list[dict]:
        """Return top candidate contexts for one query route."""

        if self.route_matrix.shape[0] == 0:
            return self.global_contexts[: self.max_contexts]

        query_fp = normalize_fp(
            reaction_morgan_fp(record.reactants, record.product, fpsize=self.fpsize, radius=self.radius)
        )
        similarities = np.clip(self.route_matrix @ query_fp, 0.0, None)
        if similarities.size == 0 or float(np.max(similarities)) <= 0.0:
            return self.global_contexts[: self.max_contexts]

        top_indices = np.argsort(similarities)[::-1][: self.top_k]
        agg: dict[tuple[str, str], dict[str, float]] = {}
        for index in top_indices:
            sim = float(similarities[index])
            if sim <= 0.0:
                continue
            for row in self.route_contexts[index]:
                key = (normalize_condition_labels(row['reagent_norm']), normalize_condition_labels(row['solvent_norm']))
                stats = agg.setdefault(
                    key,
                    {
                        'knn_similarity_sum': 0.0,
                        'knn_similarity_max': 0.0,
                        'knn_neighbor_count': 0.0,
                        'knn_weighted_yield': 0.0,
                        'knn_yield_weight': 0.0,
                    },
                )
                stats['knn_similarity_sum'] += sim
                stats['knn_similarity_max'] = max(stats['knn_similarity_max'], sim)
                stats['knn_neighbor_count'] += 1.0
                y = safe_float(row['yield'])
                if not np.isnan(y):
                    stats['knn_weighted_yield'] += sim * y
                    stats['knn_yield_weight'] += sim

        if not agg:
            return self.global_contexts[: self.max_contexts]

        candidate_rows = []
        for (reagent_norm, solvent_norm), stats in agg.items():
            mean_yield = stats['knn_weighted_yield'] / stats['knn_yield_weight'] if stats['knn_yield_weight'] > 0 else 0.0
            candidate_rows.append(
                {
                    'reagent_norm': reagent_norm,
                    'solvent_norm': solvent_norm,
                    'knn_similarity_sum': stats['knn_similarity_sum'],
                    'knn_similarity_max': stats['knn_similarity_max'],
                    'knn_neighbor_count': stats['knn_neighbor_count'],
                    'knn_weighted_mean_yield': mean_yield,
                }
            )

        candidate_rows.sort(
            key=lambda row: (
                -row['knn_similarity_sum'],
                -row['knn_similarity_max'],
                -row['knn_neighbor_count'],
                row['reagent_norm'],
                row['solvent_norm'],
            )
        )
        return candidate_rows[: self.max_contexts]

    def _candidate_rows(self, record: RouteRecordLite) -> list[dict]:
        """Backward-compatible alias for older baseline scripts."""

        return self.candidate_rows(record)

    def _records_to_frame(self, records: list[RouteRecordLite], *, max_routes: int | None = None) -> pd.DataFrame:
        if max_routes is not None:
            records = records[:max_routes]

        rows: list[dict] = []
        for record in records:
            base = base_candidate_row(record)
            for candidate in self.candidate_rows(record):
                rows.append(
                    {
                        **base,
                        'reagent_norm': candidate['reagent_norm'],
                        'solvent_norm': candidate['solvent_norm'],
                        'from_baseline_knn': 1,
                        'knn_similarity_sum': float(candidate.get('knn_similarity_sum', 0.0)),
                        'knn_similarity_max': float(candidate.get('knn_similarity_max', 0.0)),
                        'knn_neighbor_count': float(candidate.get('knn_neighbor_count', 0.0)),
                        'knn_weighted_mean_yield': float(
                            candidate.get('knn_weighted_mean_yield', candidate.get('mean_yield', 0.0))
                        ),
                        'cluster_id': -1,
                        'cluster_context_count': 0.0,
                        'cluster_context_support': 0.0,
                        'cluster_context_mean_yield': 0.0,
                    }
                )
        return pd.DataFrame(rows)

    def build_table(self, split: str, output_file: Path, *, max_routes: int | None = None) -> Path:
        records = load_route_records(split_file_for_family(self.repo_root, self.family, split), family=self.family)
        return self.build_table_from_records(records, output_file, max_routes=max_routes)

    def build_table_from_records(
        self,
        records: list[RouteRecordLite],
        output_file: Path,
        *,
        max_routes: int | None = None,
    ) -> Path:
        frame = self._records_to_frame(records, max_routes=max_routes)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_file, index=False)
        return output_file

    def build_non_oracle_table(self, route_cache_file: str | Path, output_file: Path) -> Path:
        records = load_route_records_from_cache(route_cache_file, family=self.family)
        return self.build_table_from_records(records, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build KNN Stage 2 candidate pools for ProSys.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--family', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--split', type=str, default=None, help='train/val/test for Oracle-style build')
    parser.add_argument('--route_cache', type=str, default=None, help='Non-Oracle route_cache.json path')
    parser.add_argument('--fpsize', type=int, default=4096)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--max_routes', type=int, default=None)
    args = parser.parse_args()

    if bool(args.split) == bool(args.route_cache):
        raise ValueError('Provide exactly one of --split or --route_cache')

    builder = KNNContextPoolBuilder(
        repo_root=Path(args.repo_root).resolve(),
        family=args.family,
        top_k=args.top_k,
        max_contexts=args.max_contexts,
        fpsize=args.fpsize,
        radius=args.radius,
    )
    output_file = Path(args.output_file).resolve()

    if args.route_cache:
        builder.build_non_oracle_table(args.route_cache, output_file)
    else:
        builder.build_table(args.split, output_file, max_routes=args.max_routes)


if __name__ == '__main__':
    main()
