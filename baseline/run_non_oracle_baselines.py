"""Run Non-Oracle baselines for ProSys Stage 2 using Stage 1 route caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from baseline.common import (
    base_candidate_row,
    evaluate_scored_frame,
    label_candidate_table,
    parse_families_arg,
    score_table_with_xgb,
    split_file_for_family,
    train_xgb_ranker,
    write_non_oracle_compact_tables,
)
from baseline.run_oracle_baselines import (
    ClusterContextPoolBuilder,
    _legacy_reaction_fp,
    _load_legacy_evaluators,
)
from stage2_KNN import KNNContextPoolBuilder
from stage3_XGBoost import TEMPERATURE_METADATA_FILE_NAME
from prosys_shared.features import canonicalize_reaction_side
from prosys_shared.product_memory import normalize_condition_labels
from prosys_shared.route_cache import load_route_records_from_cache

TOPKS = (1, 3, 5, 10)


def stage1_route_recall(route_cache_file: Path, topks: tuple[int, ...] = TOPKS) -> dict:
    cache = json.loads(route_cache_file.read_text(encoding='utf-8'))
    hits = {k: 0 for k in topks}
    n = 0
    for reaction in cache.get('reactions', []):
        gold_key = canonicalize_reaction_side(reaction.get('gold_reactants', ''))
        if not gold_key:
            continue
        n += 1
        routes = sorted(reaction.get('routes', []), key=lambda row: row.get('retro_rank', 1))
        pred_keys = [canonicalize_reaction_side(row['reactants']) for row in routes]
        for k in topks:
            if gold_key in pred_keys[:k]:
                hits[k] += 1
    return {
        'n': int(n),
        **{f'route_recall_top{k}': (hits[k] / n if n else 0.0) for k in topks},
    }


def _write_json(data: dict, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output_file


def _write_non_oracle_summary(rows: list[dict], output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('=' * 150)
    lines.append('Non-Oracle baseline summary')
    lines.append('=' * 150)
    lines.append(
        f"{'family':<34} {'baseline':<16} {'rr@10':>7} {'cover':>7} {'sys@1':>7} {'sys@3':>7} {'sys@5':>7} {'sys@10':>8} {'temp_MAE':>10}"
    )
    for row in rows:
        m = row['metrics']
        rr = row.get('stage1_route_recall', {})
        temp = m.get('temperature', {})
        temp_mae = temp.get('mae')
        lines.append(
            f"{row['family']:<34} "
            f"{row['baseline']:<16} "
            f"{rr.get('route_recall_top10', 0.0) * 100:>6.1f} "
            f"{m.get('pool_coverage', 0.0) * 100:>6.1f} "
            f"{m.get('system_top1_all', 0.0) * 100:>6.1f} "
            f"{m.get('system_top3_all', 0.0) * 100:>6.1f} "
            f"{m.get('system_top5_all', 0.0) * 100:>6.1f} "
            f"{m.get('system_top10_all', 0.0) * 100:>7.1f} "
            f"{(f'{temp_mae:.1f}' if temp_mae is not None else 'NA'):>10}"
        )
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_file


def _load_existing_v2_neural_ref(repo_root: Path, family: str) -> dict | None:
    path = repo_root / 'outputs' / 'stage2_v2' / family / 'non_oracle' / 'eval_non_oracle_test.json'
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding='utf-8'))
    return {
        'baseline': 'v2_neural_ref',
        'family': family,
        'candidate_table': obj.get('candidate_table'),
        'metrics': obj['metrics'],
        'stage1_route_recall': obj.get('stage1_route_recall', {}),
        'reference_file': str(path),
    }


def _ensure_v2_xgb_model(repo_root: Path, family: str, output_dir: Path) -> tuple[Path, Path]:
    model_dir = output_dir / 'model'
    model_file = model_dir / 'xgb_ranker.json'
    meta_file = model_dir / 'xgb_ranker_meta.json'
    temp_meta_file = model_dir / TEMPERATURE_METADATA_FILE_NAME
    if model_file.exists() and meta_file.exists() and temp_meta_file.exists():
        return model_file, meta_file

    oracle_tables = repo_root / 'outputs' / 'stage2_v2' / family / 'training_tables'
    train_xgb_ranker(oracle_tables / 'train.csv', oracle_tables / 'val.csv', model_dir)
    return model_file, meta_file


def _build_legacy_non_oracle_table(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_file: Path,
    *,
    legacy_max_contexts: int,
) -> Path:
    routes = load_route_records_from_cache(route_cache, family=family)
    mt, rk = _load_legacy_evaluators(repo_root, with_ranker=True)

    rows: list[dict] = []
    with torch.inference_mode():
        for record in routes:
            base = base_candidate_row(record)
            base['from_fnn'] = 1
            rxn_fp = torch.as_tensor(
                _legacy_reaction_fp(
                    record.reactants,
                    record.product,
                    fpsize=mt.args_MT.fpsize,
                    radius=mt.args_MT.radius,
                ),
                dtype=torch.float32,
            )
            input_solvents, input_reagents = mt.make_input_rxn_condition(rxn_fp)
            ranked = rk.rank_top_contexts(rxn_fp, input_solvents, input_reagents, top_k=legacy_max_contexts)
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
    frame = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)
    return output_file


def _build_knn_non_oracle_table(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_file: Path,
    *,
    top_k: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
) -> Path:
    routes = load_route_records_from_cache(route_cache, family=family)
    builder = KNNContextPoolBuilder(
        repo_root,
        family,
        top_k=top_k,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
    )
    rows: list[dict] = []
    for record in routes:
        base = base_candidate_row(record)
        for candidate in builder.candidate_rows(record):
            rows.append(
                {
                    **base,
                    'reagent_norm': candidate['reagent_norm'],
                    'solvent_norm': candidate['solvent_norm'],
                    'from_baseline_knn': 1,
                    'knn_similarity_sum': float(candidate.get('knn_similarity_sum', 0.0)),
                    'knn_similarity_max': float(candidate.get('knn_similarity_max', 0.0)),
                    'knn_neighbor_count': float(candidate.get('knn_neighbor_count', 0.0)),
                    'knn_weighted_mean_yield': float(candidate.get('knn_weighted_mean_yield', candidate.get('mean_yield', 0.0))),
                    'cluster_id': -1,
                    'cluster_context_count': 0.0,
                    'cluster_context_support': 0.0,
                    'cluster_context_mean_yield': 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)
    return output_file


def _build_cluster_non_oracle_table(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_file: Path,
    *,
    cluster_num: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    svd_dim: int,
) -> Path:
    routes = load_route_records_from_cache(route_cache, family=family)
    builder = ClusterContextPoolBuilder(
        repo_root,
        family,
        cluster_num=cluster_num,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        svd_dim=svd_dim,
    )
    rows: list[dict] = []
    for record in routes:
        base = base_candidate_row(record)
        for candidate in builder._candidate_rows(record):
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


def _ensure_knn_xgb_model(
    repo_root: Path,
    family: str,
    output_dir: Path,
    *,
    top_k: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    max_train_routes: int,
    max_val_routes: int,
) -> tuple[Path, Path]:
    model_dir = output_dir / 'model'
    model_file = model_dir / 'xgb_ranker.json'
    meta_file = model_dir / 'xgb_ranker_meta.json'
    temp_meta_file = model_dir / TEMPERATURE_METADATA_FILE_NAME
    if model_file.exists() and meta_file.exists() and temp_meta_file.exists():
        return model_file, meta_file

    builder = KNNContextPoolBuilder(
        repo_root,
        family,
        top_k=top_k,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
    )
    candidate_dir = output_dir / 'oracle_candidate_pool'
    table_dir = output_dir / 'oracle_training_tables'
    builder.build_table('train', candidate_dir / 'train.csv', max_routes=max_train_routes)
    builder.build_table('val', candidate_dir / 'val.csv', max_routes=max_val_routes)
    label_candidate_table(candidate_dir / 'train.csv', split_file_for_family(repo_root, family, 'train'), table_dir / 'train.csv')
    label_candidate_table(candidate_dir / 'val.csv', split_file_for_family(repo_root, family, 'val'), table_dir / 'val.csv')
    train_xgb_ranker(table_dir / 'train.csv', table_dir / 'val.csv', model_dir)
    return model_file, meta_file


def _ensure_cluster_xgb_model(
    repo_root: Path,
    family: str,
    output_dir: Path,
    *,
    cluster_num: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    svd_dim: int,
    max_train_routes: int,
    max_val_routes: int,
) -> tuple[Path, Path]:
    model_dir = output_dir / 'model'
    model_file = model_dir / 'xgb_ranker.json'
    meta_file = model_dir / 'xgb_ranker_meta.json'
    temp_meta_file = model_dir / TEMPERATURE_METADATA_FILE_NAME
    if model_file.exists() and meta_file.exists() and temp_meta_file.exists():
        return model_file, meta_file

    builder = ClusterContextPoolBuilder(
        repo_root,
        family,
        cluster_num=cluster_num,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        svd_dim=svd_dim,
    )
    candidate_dir = output_dir / 'oracle_candidate_pool'
    table_dir = output_dir / 'oracle_training_tables'
    builder.build_table('train', candidate_dir / 'train.csv', max_routes=max_train_routes)
    builder.build_table('val', candidate_dir / 'val.csv', max_routes=max_val_routes)
    label_candidate_table(candidate_dir / 'train.csv', split_file_for_family(repo_root, family, 'train'), table_dir / 'train.csv')
    label_candidate_table(candidate_dir / 'val.csv', split_file_for_family(repo_root, family, 'val'), table_dir / 'val.csv')
    train_xgb_ranker(table_dir / 'train.csv', table_dir / 'val.csv', model_dir)
    return model_file, meta_file


def _run_v2_xgb_non_oracle(repo_root: Path, family: str, route_cache: Path, output_dir: Path) -> dict:
    model_file, meta_file = _ensure_v2_xgb_model(repo_root, family, output_dir)
    test_table = repo_root / 'outputs' / 'stage2_v2' / family / 'non_oracle' / 'test.csv'
    scored = score_table_with_xgb(test_table, model_file, meta_file)
    scored_file = output_dir / 'non_oracle' / 'test_scored.csv'
    scored_file.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_file, index=False)
    result = {
        'baseline': 'v2_xgb',
        'family': family,
        'candidate_table': str(test_table),
        'scored_test_file': str(scored_file),
        'metrics': evaluate_scored_frame(scored, score_column='xgb_score', temperature_column='xgb_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, output_dir / 'non_oracle' / 'result.json')
    return result


def _run_legacy_non_oracle(repo_root: Path, family: str, route_cache: Path, output_dir: Path, *, legacy_max_contexts: int) -> dict:
    candidate_file = output_dir / 'non_oracle' / 'candidate_pool_test_scored.csv'
    table_file = output_dir / 'non_oracle' / 'test_scored_labeled.csv'
    _build_legacy_non_oracle_table(repo_root, family, route_cache, candidate_file, legacy_max_contexts=legacy_max_contexts)
    label_candidate_table(candidate_file, split_file_for_family(repo_root, family, 'test'), table_file)
    frame = pd.read_csv(table_file)
    result = {
        'baseline': 'legacy_rank',
        'family': family,
        'candidate_table': str(table_file),
        'metrics': evaluate_scored_frame(frame, score_column='legacy_score', temperature_column='legacy_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, output_dir / 'non_oracle' / 'result.json')
    return result


def _run_knn_non_oracle(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_dir: Path,
    *,
    top_k: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    max_train_routes: int,
    max_val_routes: int,
) -> dict:
    model_file, meta_file = _ensure_knn_xgb_model(
        repo_root,
        family,
        output_dir,
        top_k=top_k,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        max_train_routes=max_train_routes,
        max_val_routes=max_val_routes,
    )
    candidate_file = output_dir / 'non_oracle' / 'candidate_pool_test.csv'
    test_table = output_dir / 'non_oracle' / 'test.csv'
    _build_knn_non_oracle_table(
        repo_root,
        family,
        route_cache,
        candidate_file,
        top_k=top_k,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
    )
    label_candidate_table(candidate_file, split_file_for_family(repo_root, family, 'test'), test_table)
    scored = score_table_with_xgb(test_table, model_file, meta_file)
    scored_file = output_dir / 'non_oracle' / 'test_scored.csv'
    scored.to_csv(scored_file, index=False)
    result = {
        'baseline': 'knn_xgb',
        'family': family,
        'candidate_table': str(test_table),
        'scored_test_file': str(scored_file),
        'metrics': evaluate_scored_frame(scored, score_column='xgb_score', temperature_column='xgb_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, output_dir / 'non_oracle' / 'result.json')
    return result


def _run_cluster_non_oracle(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_dir: Path,
    *,
    cluster_num: int,
    max_contexts: int,
    fpsize: int,
    radius: int,
    svd_dim: int,
    max_train_routes: int,
    max_val_routes: int,
) -> dict:
    model_file, meta_file = _ensure_cluster_xgb_model(
        repo_root,
        family,
        output_dir,
        cluster_num=cluster_num,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        svd_dim=svd_dim,
        max_train_routes=max_train_routes,
        max_val_routes=max_val_routes,
    )
    candidate_file = output_dir / 'non_oracle' / 'candidate_pool_test.csv'
    test_table = output_dir / 'non_oracle' / 'test.csv'
    _build_cluster_non_oracle_table(
        repo_root,
        family,
        route_cache,
        candidate_file,
        cluster_num=cluster_num,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        svd_dim=svd_dim,
    )
    label_candidate_table(candidate_file, split_file_for_family(repo_root, family, 'test'), test_table)
    scored = score_table_with_xgb(test_table, model_file, meta_file)
    scored_file = output_dir / 'non_oracle' / 'test_scored.csv'
    scored.to_csv(scored_file, index=False)
    result = {
        'baseline': 'cluster_xgb',
        'family': family,
        'candidate_table': str(test_table),
        'scored_test_file': str(scored_file),
        'metrics': evaluate_scored_frame(scored, score_column='xgb_score', temperature_column='xgb_temperature_pred'),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, output_dir / 'non_oracle' / 'result.json')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ProSys Non-Oracle baselines under baseline/.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument(
        '--baselines',
        type=str,
        default='v2_neural_ref,v2_xgb,legacy_rank',
        help='Comma-separated baselines: v2_neural_ref,v2_xgb,legacy_rank,knn_xgb,cluster_xgb',
    )
    parser.add_argument('--output_root', type=str, default='outputs/baselines/non_oracle')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--fpsize', type=int, default=4096)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--knn_top_k', type=int, default=20)
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--cluster_num', type=int, default=32)
    parser.add_argument('--svd_dim', type=int, default=64)
    parser.add_argument('--legacy_max_contexts', type=int, default=200)
    parser.add_argument('--max_train_routes', type=int, default=2000)
    parser.add_argument('--max_val_routes', type=int, default=500)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    families = parse_families_arg(args.families)
    baselines = [item.strip() for item in args.baselines.split(',') if item.strip()]

    summary_rows: list[dict] = []
    for family in families:
        route_cache = route_root / family / 'route_cache.json'
        if not route_cache.exists():
            print(f'[baseline-nonoracle] skip {family}: missing {route_cache}')
            continue
        family_root = output_root / family
        if 'v2_neural_ref' in baselines:
            ref = _load_existing_v2_neural_ref(repo_root, family)
            if ref is not None:
                summary_rows.append(ref)
        if 'v2_xgb' in baselines:
            summary_rows.append(_run_v2_xgb_non_oracle(repo_root, family, route_cache, family_root / 'v2_xgb'))
        if 'legacy_rank' in baselines:
            summary_rows.append(
                _run_legacy_non_oracle(
                    repo_root,
                    family,
                    route_cache,
                    family_root / 'legacy_rank',
                    legacy_max_contexts=args.legacy_max_contexts,
                )
            )
        if 'knn_xgb' in baselines:
            summary_rows.append(
                _run_knn_non_oracle(
                    repo_root,
                    family,
                    route_cache,
                    family_root / 'knn_xgb',
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
                _run_cluster_non_oracle(
                    repo_root,
                    family,
                    route_cache,
                    family_root / 'cluster_xgb',
                    cluster_num=args.cluster_num,
                    max_contexts=args.max_contexts,
                    fpsize=args.fpsize,
                    radius=args.radius,
                    svd_dim=args.svd_dim,
                    max_train_routes=args.max_train_routes,
                    max_val_routes=args.max_val_routes,
                )
            )

    _write_json(summary_rows, output_root / 'non_oracle_baseline_summary.json')
    _write_non_oracle_summary(summary_rows, output_root / 'non_oracle_baseline_summary.txt')
    write_non_oracle_compact_tables(summary_rows, output_root / 'compact_tables')
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
