"""Run Oracle baselines for ProSys Stage 2.

Baselines:
- legacy_rank: legacy FNN candidate generation + legacy ranker (B1 test-only)
- v2_xgb: current Stage 2 V2 candidate tables + XGBoost reranker (B3)
- knn_xgb: KNN candidate pool + XGBoost reranker (B4)
- cluster_xgb: cluster candidate pool + XGBoost reranker (B5)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from baseline.common import (
    RouteRecordLite,
    base_candidate_row,
    evaluate_scored_frame,
    family_dir,
    label_candidate_table,
    load_route_records,
    load_split_rows,
    parse_families_arg,
    score_table_with_xgb,
    split_file_for_family,
    train_xgb_ranker,
    write_summary_table,
)
from baseline.legacy_models import LegacyMultiTaskEvaluator, LegacyRankingEvaluator
from stage2_KNN import KNNContextPoolBuilder as Stage2KNNContextPoolBuilder
from Experiment.legacy_stage2.train_multilabel.data_utils import create_rxn_Morgan2FP_concatenate, get_classes
from prosys_shared.features import normalize_fp, reaction_morgan_fp
from prosys_shared.product_memory import normalize_condition_labels, safe_float

LEGACY_CANDIDATE_CKPT = 'save_models/test_10R_first_local_10/multitask_model_epoch-80.checkpoint'
LEGACY_RANK_CKPT = 'save_models/test_10R_second_7/rxn_model_relevance_listwise_morgan_epoch-80.checkpoint'
LEGACY_LABEL_DIR = 'data/reaxys_output/label_processed'
LEGACY_CUTOFF_SOLV = 0.25
LEGACY_CUTOFF_REAG = 0.3
LEGACY_MAX_SOLV = 11
LEGACY_MAX_REAG = 11


def _legacy_device_from_env() -> torch.device:
    requested = os.environ.get('PROSYS_LEGACY_DEVICE')
    if requested:
        return torch.device(requested)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _legacy_reaction_fp(reactants: str, product: str, fpsize: int, radius: int) -> np.ndarray:
    fp = create_rxn_Morgan2FP_concatenate(reactants, product, fpsize=fpsize, radius=radius)
    if fp is None:
        return reaction_morgan_fp(reactants, product, fpsize=fpsize, radius=radius).astype(np.float32)
    return np.asarray(fp, dtype=np.float32)


def _load_legacy_evaluators(repo_root: Path, *, with_ranker: bool) -> tuple[LegacyMultiTaskEvaluator, LegacyRankingEvaluator | None]:
    label_dir = repo_root / LEGACY_LABEL_DIR
    solvent_classes = get_classes(label_dir / 'class_names_solvent_labels_processed.pkl')
    reagent_classes = get_classes(label_dir / 'class_names_reagent_labels_processed.pkl')
    legacy_device = _legacy_device_from_env()

    # Match the historical evaluator defaults from Experiment/legacy_stage2/evaluate_model/evaluate_overall.py.
    mt = LegacyMultiTaskEvaluator(
        solvent_classes,
        reagent_classes,
        cutoff_solv=LEGACY_CUTOFF_SOLV,
        cutoff_reag=LEGACY_CUTOFF_REAG,
    )
    mt.max_solv = LEGACY_MAX_SOLV
    mt.max_reag = LEGACY_MAX_REAG
    mt.load_model(repo_root / LEGACY_CANDIDATE_CKPT, device=legacy_device)

    if not with_ranker:
        return mt, None

    rk = LegacyRankingEvaluator(solvent_classes, reagent_classes)
    rk.load_model(repo_root / LEGACY_RANK_CKPT, device=legacy_device)
    return mt, rk


def build_legacy_candidate_table(
    repo_root: Path,
    family: str,
    split: str,
    output_file: Path,
    *,
    score_with_ranker: bool,
    max_contexts: int = 500,
) -> Path:
    records = load_route_records(split_file_for_family(repo_root, family, split), family=family)
    mt, rk = _load_legacy_evaluators(repo_root, with_ranker=score_with_ranker)
    context_builder = LegacyRankingEvaluator(mt.solvent_classes, mt.reagent_classes)

    rows: list[dict] = []
    with torch.inference_mode():
        for record in records:
            base = base_candidate_row(record)
            base['from_fnn'] = 1
            rxn_fp_np = _legacy_reaction_fp(
                record.reactants,
                record.product,
                fpsize=mt.args_MT.fpsize,
                radius=mt.args_MT.radius,
            )
            rxn_fp = torch.as_tensor(rxn_fp_np, dtype=torch.float32)
            input_solvents, input_reagents = mt.make_input_rxn_condition(rxn_fp)

            if score_with_ranker and rk is not None:
                ranked = rk.rank_top_contexts(rxn_fp, input_solvents, input_reagents, top_k=max_contexts)
                for rank, (solvent_norm, reagent_norm, temp_pred, score) in enumerate(ranked, start=1):
                    rows.append(
                        {
                            **base,
                            'reagent_norm': normalize_condition_labels(reagent_norm),
                            'solvent_norm': normalize_condition_labels(solvent_norm),
                            'legacy_score': float(score),
                            'legacy_temperature_pred': float(temp_pred),
                            'legacy_rank': rank,
                        }
                    )
            else:
                contexts = context_builder.make_contexts(input_solvents, input_reagents)
                seen = set()
                for rank, (solvent_norm, reagent_norm) in enumerate(contexts, start=1):
                    key = (normalize_condition_labels(reagent_norm), normalize_condition_labels(solvent_norm))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            **base,
                            'reagent_norm': key[0],
                            'solvent_norm': key[1],
                            'legacy_rank': rank,
                        }
                    )
                    if len(seen) >= max_contexts:
                        break

    frame = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)
    return output_file


class KNNContextPoolBuilder(Stage2KNNContextPoolBuilder):
    """Backward-compatible alias to the standalone Stage 2 KNN module."""


class ClusterContextPoolBuilder:
    def __init__(
        self,
        repo_root: Path,
        family: str,
        *,
        cluster_num: int,
        max_contexts: int,
        fpsize: int,
        radius: int,
        svd_dim: int,
    ):
        self.repo_root = repo_root
        self.family = family
        self.cluster_num = cluster_num
        self.max_contexts = max_contexts
        self.fpsize = fpsize
        self.radius = radius
        self.svd_dim = svd_dim

        train_rows = load_split_rows(split_file_for_family(repo_root, family, 'train'))
        self.route_matrix, self.route_contexts = self._build_route_memory(train_rows)
        self.global_contexts = self._global_contexts(train_rows)
        self.svd, self.kmeans, self.cluster_contexts = self._fit_cluster_memory()

    def _build_route_memory(self, rows: list[dict]) -> tuple[np.ndarray, list[list[dict]]]:
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row['reaction_id'], row['reactants'], row['product'])].append(row)

        matrix_rows: list[np.ndarray] = []
        route_contexts: list[list[dict]] = []
        for (_reaction_id, reactants, product), bucket in grouped.items():
            matrix_rows.append(normalize_fp(reaction_morgan_fp(reactants, product, fpsize=self.fpsize, radius=self.radius)))
            route_contexts.append(bucket)
        matrix = np.stack(matrix_rows).astype(np.float32) if matrix_rows else np.zeros((0, self.fpsize * 2), dtype=np.float32)
        return matrix, route_contexts

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
        results = []
        total = sum(stats['count'] for stats in agg.values()) or 1.0
        for (reagent_norm, solvent_norm), stats in agg.items():
            results.append(
                {
                    'reagent_norm': reagent_norm,
                    'solvent_norm': solvent_norm,
                    'cluster_context_count': stats['count'],
                    'cluster_context_support': stats['count'] / total,
                    'cluster_context_mean_yield': (stats['yield_sum'] / stats['yield_n'] if stats['yield_n'] else 0.0),
                    'cluster_id': -1,
                }
            )
        results.sort(key=lambda row: (-row['cluster_context_count'], row['reagent_norm'], row['solvent_norm']))
        return results

    def _fit_cluster_memory(self):
        if self.route_matrix.shape[0] == 0:
            return None, None, {}

        reduced = self.route_matrix
        svd = None
        n_samples, n_features = reduced.shape
        max_components = min(self.svd_dim, n_samples - 1, n_features - 1)
        if max_components >= 2:
            svd = TruncatedSVD(n_components=max_components, random_state=0)
            reduced = svd.fit_transform(reduced)

        n_clusters = max(1, min(self.cluster_num, reduced.shape[0]))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=min(1024, reduced.shape[0]))
        labels = kmeans.fit_predict(reduced)

        cluster_aggs: dict[int, dict[tuple[str, str], dict[str, float]]] = defaultdict(dict)
        cluster_totals: dict[int, float] = defaultdict(float)
        for route_idx, cluster_id in enumerate(labels):
            for row in self.route_contexts[route_idx]:
                key = (normalize_condition_labels(row['reagent_norm']), normalize_condition_labels(row['solvent_norm']))
                stats = cluster_aggs[cluster_id].setdefault(key, {'count': 0.0, 'yield_sum': 0.0, 'yield_n': 0.0})
                stats['count'] += 1.0
                cluster_totals[cluster_id] += 1.0
                y = safe_float(row['yield'])
                if not np.isnan(y):
                    stats['yield_sum'] += y
                    stats['yield_n'] += 1.0

        cluster_contexts = {}
        for cluster_id, agg in cluster_aggs.items():
            rows = []
            total = cluster_totals[cluster_id] or 1.0
            for (reagent_norm, solvent_norm), stats in agg.items():
                rows.append(
                    {
                        'reagent_norm': reagent_norm,
                        'solvent_norm': solvent_norm,
                        'cluster_id': int(cluster_id),
                        'cluster_context_count': stats['count'],
                        'cluster_context_support': stats['count'] / total,
                        'cluster_context_mean_yield': (stats['yield_sum'] / stats['yield_n'] if stats['yield_n'] else 0.0),
                    }
                )
            rows.sort(key=lambda row: (-row['cluster_context_count'], -row['cluster_context_support'], row['reagent_norm'], row['solvent_norm']))
            cluster_contexts[int(cluster_id)] = rows[: self.max_contexts]

        return svd, kmeans, cluster_contexts

    def _candidate_rows(self, record: RouteRecordLite) -> list[dict]:
        if self.kmeans is None:
            return self.global_contexts[: self.max_contexts]

        query_fp = normalize_fp(reaction_morgan_fp(record.reactants, record.product, fpsize=self.fpsize, radius=self.radius)).reshape(1, -1)
        query_vec = self.svd.transform(query_fp) if self.svd is not None else query_fp
        cluster_id = int(self.kmeans.predict(query_vec)[0])
        return self.cluster_contexts.get(cluster_id, self.global_contexts)[: self.max_contexts]

    def build_table(self, split: str, output_file: Path, *, max_routes: int | None = None) -> Path:
        records = load_route_records(split_file_for_family(self.repo_root, self.family, split), family=self.family)
        if max_routes is not None:
            records = records[:max_routes]
        rows: list[dict] = []
        for record in records:
            base = base_candidate_row(record)
            for candidate in self._candidate_rows(record):
                rows.append(
                    {
                        **base,
                        'reagent_norm': candidate['reagent_norm'],
                        'solvent_norm': candidate['solvent_norm'],
                        'from_baseline_cluster': 1,
                        'knn_similarity_sum': 0.0,
                        'knn_similarity_max': 0.0,
                        'knn_neighbor_count': 0.0,
                        'knn_weighted_mean_yield': 0.0,
                        'cluster_id': int(candidate.get('cluster_id', -1)),
                        'cluster_context_count': float(candidate.get('cluster_context_count', 0.0)),
                        'cluster_context_support': float(candidate.get('cluster_context_support', 0.0)),
                        'cluster_context_mean_yield': float(candidate.get('cluster_context_mean_yield', 0.0)),
                    }
                )
        frame = pd.DataFrame(rows)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_file, index=False)
        return output_file


def _write_json(data: dict, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output_file


def _run_legacy_rank(repo_root: Path, family: str, family_root: Path, *, legacy_max_contexts: int) -> dict:
    candidate_file = family_root / 'legacy_rank' / 'candidate_pool_test_scored.csv'
    build_legacy_candidate_table(
        repo_root,
        family,
        'test',
        candidate_file,
        score_with_ranker=True,
        max_contexts=legacy_max_contexts,
    )
    labeled_file = family_root / 'legacy_rank' / 'test_scored_labeled.csv'
    label_candidate_table(candidate_file, split_file_for_family(repo_root, family, 'test'), labeled_file)
    frame = pd.read_csv(labeled_file)
    metrics = evaluate_scored_frame(
        frame,
        score_column='legacy_score',
        temperature_column='legacy_temperature_pred',
    )
    result = {
        'baseline': 'legacy_rank',
        'family': family,
        'candidate_table': str(labeled_file),
        'metrics': metrics,
    }
    _write_json(result, family_root / 'legacy_rank' / 'result.json')
    return result


def _run_v2_xgb(repo_root: Path, family: str, family_root: Path) -> dict:
    base = repo_root / 'outputs' / 'stage2_v2' / family / 'training_tables'
    output_dir = family_root / 'v2_xgb'
    artifacts = train_xgb_ranker(base / 'train.csv', base / 'val.csv', output_dir / 'model')
    scored_test = score_table_with_xgb(base / 'test.csv', artifacts.model_file, artifacts.metadata_file)
    scored_test_file = output_dir / 'test_scored.csv'
    scored_test.to_csv(scored_test_file, index=False)
    metrics = evaluate_scored_frame(scored_test, score_column='xgb_score', temperature_column='xgb_temperature_pred')
    result = {
        'baseline': 'v2_xgb',
        'family': family,
        'candidate_table': str(base / 'test.csv'),
        'scored_test_file': str(scored_test_file),
        'model': artifacts.to_dict(),
        'metrics': metrics,
    }
    _write_json(result, output_dir / 'result.json')
    return result


def _run_knn_xgb(
    repo_root: Path,
    family: str,
    family_root: Path,
    *,
    top_k: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    max_train_routes: int | None,
    max_val_routes: int | None,
) -> dict:
    output_dir = family_root / 'knn_xgb'
    builder = KNNContextPoolBuilder(
        repo_root,
        family,
        top_k=top_k,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
    )
    candidate_dir = output_dir / 'candidate_pool'
    table_dir = output_dir / 'training_tables'
    for split in ('train', 'val', 'test'):
        candidate_file = candidate_dir / f'{split}.csv'
        route_limit = None
        if split == 'train':
            route_limit = max_train_routes
        elif split == 'val':
            route_limit = max_val_routes
        builder.build_table(split, candidate_file, max_routes=route_limit)
        label_candidate_table(candidate_file, split_file_for_family(repo_root, family, split), table_dir / f'{split}.csv')

    artifacts = train_xgb_ranker(table_dir / 'train.csv', table_dir / 'val.csv', output_dir / 'model')
    scored_test = score_table_with_xgb(table_dir / 'test.csv', artifacts.model_file, artifacts.metadata_file)
    scored_test_file = output_dir / 'test_scored.csv'
    scored_test.to_csv(scored_test_file, index=False)
    metrics = evaluate_scored_frame(scored_test, score_column='xgb_score', temperature_column='xgb_temperature_pred')
    result = {
        'baseline': 'knn_xgb',
        'family': family,
        'candidate_table': str(table_dir / 'test.csv'),
        'scored_test_file': str(scored_test_file),
        'model': artifacts.to_dict(),
        'metrics': metrics,
    }
    _write_json(result, output_dir / 'result.json')
    return result


def _run_cluster_xgb(
    repo_root: Path,
    family: str,
    family_root: Path,
    *,
    cluster_num: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    svd_dim: int,
    max_train_routes: int | None,
    max_val_routes: int | None,
) -> dict:
    output_dir = family_root / 'cluster_xgb'
    builder = ClusterContextPoolBuilder(
        repo_root,
        family,
        cluster_num=cluster_num,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        svd_dim=svd_dim,
    )
    candidate_dir = output_dir / 'candidate_pool'
    table_dir = output_dir / 'training_tables'
    for split in ('train', 'val', 'test'):
        candidate_file = candidate_dir / f'{split}.csv'
        route_limit = None
        if split == 'train':
            route_limit = max_train_routes
        elif split == 'val':
            route_limit = max_val_routes
        builder.build_table(split, candidate_file, max_routes=route_limit)
        label_candidate_table(candidate_file, split_file_for_family(repo_root, family, split), table_dir / f'{split}.csv')

    artifacts = train_xgb_ranker(table_dir / 'train.csv', table_dir / 'val.csv', output_dir / 'model')
    scored_test = score_table_with_xgb(table_dir / 'test.csv', artifacts.model_file, artifacts.metadata_file)
    scored_test_file = output_dir / 'test_scored.csv'
    scored_test.to_csv(scored_test_file, index=False)
    metrics = evaluate_scored_frame(scored_test, score_column='xgb_score', temperature_column='xgb_temperature_pred')
    result = {
        'baseline': 'cluster_xgb',
        'family': family,
        'candidate_table': str(table_dir / 'test.csv'),
        'scored_test_file': str(scored_test_file),
        'model': artifacts.to_dict(),
        'metrics': metrics,
    }
    _write_json(result, output_dir / 'result.json')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ProSys Oracle baselines under baseline/.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument(
        '--baselines',
        type=str,
        default='legacy_rank,v2_xgb,knn_xgb,cluster_xgb',
        help='Comma-separated baselines: legacy_rank,v2_xgb,knn_xgb,cluster_xgb',
    )
    parser.add_argument('--output_root', type=str, default='outputs/baselines/oracle')
    parser.add_argument('--fpsize', type=int, default=4096)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--knn_top_k', type=int, default=20)
    parser.add_argument('--max_contexts', type=int, default=50)
    parser.add_argument('--cluster_num', type=int, default=64)
    parser.add_argument('--svd_dim', type=int, default=128)
    parser.add_argument('--legacy_max_contexts', type=int, default=500)
    parser.add_argument('--max_train_routes', type=int, default=2000)
    parser.add_argument('--max_val_routes', type=int, default=500)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    families = parse_families_arg(args.families)
    baselines = [item.strip() for item in args.baselines.split(',') if item.strip()]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for family in families:
        family_root = output_root / family
        if 'legacy_rank' in baselines:
            summary_rows.append(
                _run_legacy_rank(
                    repo_root,
                    family,
                    family_root,
                    legacy_max_contexts=args.legacy_max_contexts,
                )
            )
        if 'v2_xgb' in baselines:
            summary_rows.append(_run_v2_xgb(repo_root, family, family_root))
        if 'knn_xgb' in baselines:
            summary_rows.append(
                _run_knn_xgb(
                    repo_root,
                    family,
                    family_root,
                    top_k=args.knn_top_k,
                    max_contexts=args.max_contexts,
                    fpsize=args.fpsize,
                    radius=args.radius,
                    max_train_routes=args.max_train_routes,
                    max_val_routes=args.max_val_routes,
                )
            )
        if 'cluster_xgb' in baselines:
            summary_rows.append(
                _run_cluster_xgb(
                    repo_root,
                    family,
                    family_root,
                    cluster_num=args.cluster_num,
                    max_contexts=args.max_contexts,
                    fpsize=args.fpsize,
                    radius=args.radius,
                    svd_dim=args.svd_dim,
                    max_train_routes=args.max_train_routes,
                    max_val_routes=args.max_val_routes,
                )
            )

    summary_json = output_root / 'oracle_baseline_summary.json'
    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_summary_table(summary_rows, output_root / 'oracle_baseline_summary.txt')
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
