"""KNN candidate screening for ProSys Stage 2.

This module turns a route `(reactants, product)` into a feasible condition
candidate pool by retrieving similar training reactions and aggregating their
historical `(reagent_norm, solvent_norm)` contexts.
"""

from __future__ import annotations

import multiprocessing as mp
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from prosys_shared.features import normalize_fp, reaction_morgan_fp
from prosys_shared.mainline import (
    base_candidate_row,
    load_route_records,
    load_split_rows,
    split_file_for_family,
)
from prosys_shared.product_memory import normalize_condition_labels, safe_float
from prosys_shared.route_cache import RouteRecord, load_route_records_from_cache
from .reafnn_selector import (
    ReaFNNConfig,
    ReaFNNSelector,
    build_default_context_library,
    train_reafnn_selector,
)


_PARALLEL_BUILDER = None


def _parallel_candidate_rows(payload: tuple[RouteRecord, bool, bool]) -> list[dict]:
    """Fork-worker helper used only for retrieval-only KNN table construction."""

    if _PARALLEL_BUILDER is None:
        raise RuntimeError('KNN parallel worker was not initialized.')
    record, allow_novel, leave_one_reaction_out = payload
    return _PARALLEL_BUILDER._record_candidate_rows(
        record,
        allow_novel=allow_novel,
        leave_one_reaction_out=leave_one_reaction_out,
    )


class KNNContextPoolBuilder:
    """Family-specific KNN context retriever.

    Memory is built strictly from the family train split. During inference, each
    query route is encoded with a reaction Morgan fingerprint and matched
    against the train-memory matrix using cosine similarity over L2-normalized
    fingerprints.
    """

    def __init__(
        self,
        repo_root: Path,
        family: str,
        *,
        top_k: int,
        max_contexts: int,
        fpsize: int,
        radius: int,
        prefilter_contexts: int | None = None,
        reaffn_artifact_dir: Path | None = None,
        reaffn_device: str = 'cpu',
        reaffn_force_retrain: bool = False,
        reaffn_config: ReaFNNConfig | None = None,
        parallel_workers: int = 1,
        sparse_similarity: bool = False,
    ):
        self.repo_root = Path(repo_root)
        self.family = family
        self.top_k = int(top_k)
        self.max_contexts = int(max_contexts)
        self.prefilter_contexts = int(prefilter_contexts or max_contexts)
        self.fpsize = int(fpsize)
        self.radius = int(radius)
        self.reaffn_artifact_dir = Path(reaffn_artifact_dir) if reaffn_artifact_dir is not None else None
        self.sparse_similarity = bool(sparse_similarity)
        self._feature_postings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.parallel_workers = max(1, int(parallel_workers))

        train_split = split_file_for_family(self.repo_root, self.family, 'train')
        val_split = split_file_for_family(self.repo_root, self.family, 'val')
        train_rows = load_split_rows(train_split)
        self.route_matrix, self.route_contexts, self.route_keys = self._build_route_memory(train_rows)
        self.route_index_by_key = {
            route_key: index
            for index, route_key in enumerate(self.route_keys)
        }
        self.global_contexts = self._global_contexts(train_rows)
        self.global_context_by_key = {
            (str(row['reagent_norm']), str(row['solvent_norm'])): row
            for row in self.global_contexts
        }
        self.global_context_total = float(sum(float(row['context_count']) for row in self.global_contexts))
        if self.sparse_similarity:
            # Reaction fingerprints are very sparse; this preserves the cosine
            # dot product while avoiding a dense 8192-dimensional scan per row.
            self._feature_postings = self._build_feature_postings(self.route_matrix)


        self.reaffn_selector: ReaFNNSelector | None = None
        if self.reaffn_artifact_dir is not None:
            config = reaffn_config or ReaFNNConfig(
                fpsize=self.fpsize,
                radius=self.radius,
                device=reaffn_device,
            )
            train_reafnn_selector(
                train_split_file=train_split,
                val_split_file=val_split,
                output_dir=self.reaffn_artifact_dir,
                config=config,
                force_retrain=reaffn_force_retrain,
            )
            self.reaffn_selector = ReaFNNSelector(
                artifact_dir=self.reaffn_artifact_dir,
                context_library=build_default_context_library(train_split),
                device=reaffn_device,
            )

    def _build_route_memory(
        self,
        rows: list[dict],
    ) -> tuple[np.ndarray, list[list[dict]], list[tuple[str, str, str]]]:
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row['reaction_id'], row['reactants'], row['product'])].append(row)

        matrix_rows: list[np.ndarray] = []
        route_contexts: list[list[dict]] = []
        route_keys: list[tuple[str, str, str]] = []
        for route_key, bucket in grouped.items():
            _reaction_id, reactants, product = route_key
            fp = normalize_fp(reaction_morgan_fp(reactants, product, fpsize=self.fpsize, radius=self.radius))
            matrix_rows.append(fp)
            route_contexts.append(bucket)

            route_keys.append(route_key)
        if not matrix_rows:
            return np.zeros((0, self.fpsize * 2), dtype=np.float32), [], []
        return np.stack(matrix_rows).astype(np.float32), route_contexts, route_keys

    @staticmethod
    def _build_feature_postings(matrix: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Index non-zero matrix entries by feature for exact sparse dot products."""

        if matrix.size == 0:
            return {}
        row_indices, column_indices = np.nonzero(matrix)
        if row_indices.size == 0:
            return {}
        values = matrix[row_indices, column_indices].astype(np.float32, copy=False)
        order = np.argsort(column_indices, kind='stable')
        sorted_rows = row_indices[order].astype(np.int32, copy=False)
        sorted_columns = column_indices[order]
        sorted_values = values[order]
        starts = np.flatnonzero(
            np.r_[True, sorted_columns[1:] != sorted_columns[:-1]]
        )
        ends = np.r_[starts[1:], sorted_columns.size]
        return {
            int(sorted_columns[start]): (
                sorted_rows[start:end],
                sorted_values[start:end],
            )
            for start, end in zip(starts, ends)
        }

    def _route_similarities(self, query_fp: np.ndarray) -> np.ndarray:
        if not self.sparse_similarity:
            return np.clip(self.route_matrix @ query_fp, 0.0, None)
        similarities = np.zeros((self.route_matrix.shape[0],), dtype=np.float32)
        for feature_index in np.flatnonzero(query_fp):
            posting = self._feature_postings.get(int(feature_index))
            if posting is None:
                continue
            route_indices, values = posting
            similarities[route_indices] += values * float(query_fp[feature_index])
        return np.clip(similarities, 0.0, None)

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
                    '_yield_sum': stats['yield_sum'],
                    '_yield_n': stats['yield_n'],
                }
            )
        rows_out.sort(key=lambda row: (-row['context_count'], row['reagent_norm'], row['solvent_norm']))
        return rows_out

    @staticmethod
    def _record_key(record: RouteRecord) -> tuple[str, str, str]:
        return (str(record.reaction_id), str(record.reactants), str(record.product))

    def _excluded_context_stats(
        self,
        record: RouteRecord,
        *,
        leave_one_reaction_out: bool,
    ) -> tuple[dict[tuple[str, str], dict[str, float]], float]:
        """Remove the query reaction from train-memory summary statistics."""

        if not leave_one_reaction_out:
            return {}, self.global_context_total
        route_index = self.route_index_by_key.get(self._record_key(record))
        if route_index is None:
            return {}, self.global_context_total

        excluded: dict[tuple[str, str], dict[str, float]] = {}
        for row in self.route_contexts[route_index]:
            key = (
                normalize_condition_labels(row['reagent_norm']),
                normalize_condition_labels(row['solvent_norm']),
            )
            stats = excluded.setdefault(key, {'count': 0.0, 'yield_sum': 0.0, 'yield_n': 0.0})
            stats['count'] += 1.0
            yield_value = safe_float(row['yield'])
            if not np.isnan(yield_value):
                stats['yield_sum'] += yield_value
                stats['yield_n'] += 1.0

        removed_count = sum(float(stats['count']) for stats in excluded.values())
        return excluded, max(self.global_context_total - removed_count, 1.0)

    def _adjusted_global_context(
        self,
        context: dict,
        *,
        excluded: dict[tuple[str, str], dict[str, float]],
        remaining_total: float,
    ) -> dict | None:
        key = (str(context['reagent_norm']), str(context['solvent_norm']))
        removed = excluded.get(key)
        removed_count = float(removed['count']) if removed is not None else 0.0
        count = float(context['context_count']) - removed_count
        if count <= 0.0:
            return None

        removed_yield_sum = float(removed['yield_sum']) if removed is not None else 0.0
        removed_yield_n = float(removed['yield_n']) if removed is not None else 0.0
        yield_sum = float(context.get('_yield_sum', 0.0)) - removed_yield_sum
        yield_n = float(context.get('_yield_n', 0.0)) - removed_yield_n
        return {
            'reagent_norm': key[0],
            'solvent_norm': key[1],
            'context_count': count,
            'context_support': count / max(remaining_total, 1.0),
            'mean_yield': yield_sum / yield_n if yield_n > 0.0 else 0.0,
        }

    def _fallback_contexts(
        self,
        *,
        excluded: dict[tuple[str, str], dict[str, float]],
        remaining_total: float,
        limit: int,
    ) -> list[dict]:
        rows = []
        for context in self.global_contexts:
            adjusted = self._adjusted_global_context(
                context,
                excluded=excluded,
                remaining_total=remaining_total,
            )
            if adjusted is not None:
                rows.append(adjusted)
        rows.sort(key=lambda row: (-row['context_count'], row['reagent_norm'], row['solvent_norm']))
        return rows[:limit]

    def _aggregate_knn_contexts(
        self,
        record: RouteRecord,
        *,
        limit: int,
        leave_one_reaction_out: bool = False,
    ) -> list[dict]:
        excluded, remaining_total = self._excluded_context_stats(
            record,
            leave_one_reaction_out=leave_one_reaction_out,
        )
        fallback_contexts = self._fallback_contexts(
            excluded=excluded,
            remaining_total=remaining_total,
            limit=limit,
        )
        if self.route_matrix.shape[0] == 0:
            return fallback_contexts

        query_fp = normalize_fp(
            reaction_morgan_fp(record.reactants, record.product, fpsize=self.fpsize, radius=self.radius)
        )
        similarities = self._route_similarities(query_fp)
        if similarities.size == 0 or float(np.max(similarities)) <= 0.0:
            return fallback_contexts

        agg: dict[tuple[str, str], dict[str, float]] = {}
        selected_neighbors = 0
        query_key = self._record_key(record)
        for index in np.argsort(similarities)[::-1]:
            sim = float(similarities[index])
            if sim <= 0.0:
                break
            if leave_one_reaction_out and self.route_keys[index] == query_key:
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
            selected_neighbors += 1
            if selected_neighbors >= self.top_k:
                break

        if not agg:
            return fallback_contexts

        candidate_rows = []
        for (reagent_norm, solvent_norm), stats in agg.items():
            mean_yield = stats['knn_weighted_yield'] / stats['knn_yield_weight'] if stats['knn_yield_weight'] > 0 else 0.0
            global_context = self._adjusted_global_context(
                self.global_context_by_key.get((reagent_norm, solvent_norm), {}),
                excluded=excluded,
                remaining_total=remaining_total,
            )
            if global_context is None:
                continue
            candidate_rows.append(
                {
                    'reagent_norm': reagent_norm,
                    'solvent_norm': solvent_norm,
                    'knn_similarity_sum': stats['knn_similarity_sum'],
                    'knn_similarity_max': stats['knn_similarity_max'],
                    'knn_neighbor_count': stats['knn_neighbor_count'],
                    'knn_weighted_mean_yield': mean_yield,
                    'context_count': float(global_context.get('context_count', 0.0)),
                    'context_support': float(global_context.get('context_support', 0.0)),
                    'mean_yield': float(global_context.get('mean_yield', mean_yield)),
                    'from_reafnn_generated': 0,
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
        return candidate_rows[:limit]

    def candidate_rows(
        self,
        record: RouteRecord,
        *,
        allow_novel: bool = True,
        leave_one_reaction_out: bool = False,
    ) -> list[dict]:
        """Return top candidate contexts for one query route."""

        wide_contexts = self._aggregate_knn_contexts(
            record,
            limit=self.prefilter_contexts,
            leave_one_reaction_out=leave_one_reaction_out,
        )
        if self.reaffn_selector is None:
            return wide_contexts[: self.max_contexts]

        existing_keys = {(str(row['reagent_norm']), str(row['solvent_norm'])) for row in wide_contexts}
        merged: dict[tuple[str, str], dict] = {
            (str(row['reagent_norm']), str(row['solvent_norm'])): dict(row)
            for row in wide_contexts
        }
        for generated in self.reaffn_selector.generate_contexts(
            record.reactants,
            record.product,
            existing_contexts=existing_keys,
        ):
            if not allow_novel and int(generated.get('from_reafnn_novel', 0)) == 1:
                continue
            key = (str(generated['reagent_norm']), str(generated['solvent_norm']))
            merged[key] = {
                **generated,
                'knn_similarity_sum': 0.0,
                'knn_similarity_max': 0.0,
                'knn_neighbor_count': 0.0,
                'knn_weighted_mean_yield': float(generated.get('reafnn_mean_yield', 0.0)),
                'context_count': float(generated.get('reafnn_context_count', 0.0)),
                'context_support': float(generated.get('reafnn_context_support', 0.0)),
                'mean_yield': float(generated.get('reafnn_mean_yield', 0.0)),
            }

        rescored = self.reaffn_selector.rescore_contexts(
            record.reactants,
            record.product,
            list(merged.values()),
        )
        rescored.sort(
            key=lambda row: (
                -float(row.get('reafnn_context_score', 0.0)),
                -int(row.get('reafnn_is_historical', 1)),
                -float(row.get('knn_similarity_sum', 0.0)),
                -float(row.get('reafnn_context_support', row.get('context_support', 0.0))),
                -float(row.get('knn_neighbor_count', 0.0)),
                row['reagent_norm'],
                row['solvent_norm'],
            )
        )
        if self.reaffn_selector is None:
            return rescored[: self.max_contexts]

        novel_cap = min(
            int(self.reaffn_selector.config.max_selected_novel_contexts),
            self.max_contexts,
        )
        if novel_cap < 0:
            novel_cap = 0

        selected: list[dict] = []
        selected_novel = 0
        for row in rescored:
            is_novel = int(row.get('from_reafnn_novel', 0)) == 1
            if is_novel and selected_novel >= novel_cap:
                continue
            selected.append(row)
            if is_novel:
                selected_novel += 1
            if len(selected) >= self.max_contexts:
                break
        return selected

    def _candidate_rows(
        self,
        record: RouteRecord,
        *,
        allow_novel: bool = True,
        leave_one_reaction_out: bool = False,
    ) -> list[dict]:
        """Backward-compatible alias for older baseline scripts."""

        return self.candidate_rows(
            record,
            allow_novel=allow_novel,
            leave_one_reaction_out=leave_one_reaction_out,
        )

    def _record_candidate_rows(
        self,
        record: RouteRecord,
        *,
        allow_novel: bool,
        leave_one_reaction_out: bool,
    ) -> list[dict]:
        base = base_candidate_row(record)
        return [
            {
                **base,
                'reagent_norm': candidate['reagent_norm'],
                'solvent_norm': candidate['solvent_norm'],
                'from_baseline_knn': int(candidate.get('from_reafnn_generated', 0) == 0),
                'knn_similarity_sum': float(candidate.get('knn_similarity_sum', 0.0)),
                'knn_similarity_max': float(candidate.get('knn_similarity_max', 0.0)),
                'knn_neighbor_count': float(candidate.get('knn_neighbor_count', 0.0)),
                'knn_weighted_mean_yield': float(candidate.get('knn_weighted_mean_yield', candidate.get('mean_yield', 0.0))),
                'reafnn_reagent_score': float(candidate.get('reafnn_reagent_score', 0.0)),
                'reafnn_solvent_score': float(candidate.get('reafnn_solvent_score', 0.0)),
                'reafnn_token_score': float(candidate.get('reafnn_token_score', 0.0)),
                'reafnn_prior_score': float(candidate.get('reafnn_prior_score', 0.0)),
                'reafnn_historical_bonus': float(candidate.get('reafnn_historical_bonus', 0.0)),
                'reafnn_novelty_penalty': float(candidate.get('reafnn_novelty_penalty', 0.0)),
                'reafnn_context_score': float(candidate.get('reafnn_context_score', 0.0)),
                'reafnn_context_count': float(candidate.get('reafnn_context_count', candidate.get('context_count', 0.0))),
                'reafnn_context_support': float(candidate.get('reafnn_context_support', candidate.get('context_support', 0.0))),
                'reafnn_mean_yield': float(candidate.get('reafnn_mean_yield', candidate.get('mean_yield', 0.0))),
                'from_reafnn_generated': int(candidate.get('from_reafnn_generated', 0)),
                'from_reafnn_novel': int(candidate.get('from_reafnn_novel', 0)),
                'reafnn_is_historical': int(candidate.get('reafnn_is_historical', 1)),
                'cluster_id': -1,
                'cluster_context_count': 0.0,
                'cluster_context_support': 0.0,
                'cluster_context_mean_yield': 0.0,
            }
            for candidate in self.candidate_rows(
                record,
                allow_novel=allow_novel,
                leave_one_reaction_out=leave_one_reaction_out,
            )
        ]

    def _records_to_frame(
        self,
        records: list[RouteRecord],
        *,
        max_routes: int | None = None,
        allow_novel: bool = True,
        leave_one_reaction_out: bool = False,
    ) -> pd.DataFrame:
        if max_routes is not None:
            records = records[:max_routes]

        rows: list[dict] = []
        use_parallel = self.parallel_workers > 1 and self.reaffn_selector is None and len(records) > 1
        if use_parallel and 'fork' in mp.get_all_start_methods():
            global _PARALLEL_BUILDER
            _PARALLEL_BUILDER = self
            context = mp.get_context('fork')
            with context.Pool(processes=min(self.parallel_workers, len(records))) as pool:
                for candidate_rows in pool.imap(
                    _parallel_candidate_rows,
                    ((record, allow_novel, leave_one_reaction_out) for record in records),
                    chunksize=8,
                ):
                    rows.extend(candidate_rows)
            _PARALLEL_BUILDER = None
        else:
            for record in records:
                rows.extend(
                    self._record_candidate_rows(
                        record,
                        allow_novel=allow_novel,
                        leave_one_reaction_out=leave_one_reaction_out,
                    )
                )
        return pd.DataFrame(rows)

    def build_table(self, split: str, output_file: Path, *, max_routes: int | None = None) -> Path:
        records = load_route_records(split_file_for_family(self.repo_root, self.family, split), family=self.family)
        return self.build_table_from_records(
            records,
            output_file,
            max_routes=max_routes,
            allow_novel=False,
            leave_one_reaction_out=(split == 'train'),
        )

    def build_table_from_records(
        self,
        records: list[RouteRecord],
        output_file: Path,
        *,
        max_routes: int | None = None,
        allow_novel: bool = True,
        leave_one_reaction_out: bool = False,
    ) -> Path:
        frame = self._records_to_frame(
            records,
            max_routes=max_routes,
            allow_novel=allow_novel,
            leave_one_reaction_out=leave_one_reaction_out,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_file, index=False)
        return output_file

    def build_non_oracle_table(
        self,
        route_cache_file: str | Path,
        output_file: Path,
        *,
        max_routes: int | None = None,
        leave_one_reaction_out: bool = False,
    ) -> Path:
        records = load_route_records_from_cache(route_cache_file, family=self.family)
        return self.build_table_from_records(
            records,
            output_file,
            max_routes=max_routes,
            allow_novel=True,
            leave_one_reaction_out=leave_one_reaction_out,
        )

def main() -> None:
    parser = argparse.ArgumentParser(description='Build KNN Stage 2 candidate pools for ProSys.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--family', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--split', type=str, default=None, help='train/val/test for Oracle-style build')
    parser.add_argument('--route_cache', type=str, default=None, help='Non-Oracle route_cache.json path')
    parser.add_argument('--fpsize', type=int, default=4096)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=64)
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--prefilter_contexts', type=int, default=64)
    parser.add_argument('--max_routes', type=int, default=None)
    parser.add_argument('--reafnn_artifact_dir', type=str, default=None)
    parser.add_argument('--reafnn_device', type=str, default='cpu')
    parser.add_argument('--reafnn_force_retrain', action='store_true')
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
        prefilter_contexts=args.prefilter_contexts,
        reaffn_artifact_dir=Path(args.reafnn_artifact_dir).resolve() if args.reafnn_artifact_dir else None,
        reaffn_device=args.reafnn_device,
        reaffn_force_retrain=args.reafnn_force_retrain,
    )
    output_file = Path(args.output_file).resolve()

    if args.route_cache:
        builder.build_non_oracle_table(args.route_cache, output_file)
    else:
        builder.build_table(args.split, output_file, max_routes=args.max_routes)


if __name__ == '__main__':
    main()
