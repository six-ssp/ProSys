"""Run the maintained ProSys Non-Oracle mainline: Stage 2 + tabular XGB-LTR."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosys_shared.mainline import (
    FAMILY_ORDER,
    display_family_name,
    evaluate_scored_frame_with_manifest,
    label_candidate_table,
    parse_families_arg,
    split_file_for_family,
    stage1_route_recall,
)
from prosys_shared.nomenclature import PUBLIC_METRIC_LABELS
from prosys_shared.route_cache import load_route_cache_sample_indices
from stage2_KNN import KNNContextPoolBuilder
from stage2_KNN.reafnn_selector import ReaFNNConfig
from stage3_XGBoost import score_table_with_xgb, train_xgb_ranker_and_temperature, train_xgb_temperature_regressor
from stage3_XGBoost.xgb_reranker import HEURISTIC_STAGE3_SORT_SPECS
from stage3_XGBoost.reaction_gnn_features import (
    ReactionGNNConfig,
    augment_table_with_reaction_gnn_features,
    train_reaction_gnn_feature_model,
)


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(FAMILY_ORDER), family)


def _write_json(data: dict | list, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output_file


def _shared_paths(shared_root: Path) -> dict[str, Path]:
    return {
        'candidate_train': shared_root / 'candidate_pool' / 'train.csv',
        'candidate_val': shared_root / 'candidate_pool' / 'val.csv',
        'candidate_test': shared_root / 'candidate_pool' / 'test.csv',
        'table_train': shared_root / 'training_tables' / 'train.csv',
        'table_val': shared_root / 'training_tables' / 'val.csv',
        'table_test': shared_root / 'training_tables' / 'test.csv',
    }


def _shared_augmented_table_paths(shared_root: Path) -> dict[str, Path]:
    return {
        'table_train': shared_root / 'training_tables' / 'train.csv',
        'table_val': shared_root / 'training_tables' / 'val.csv',
        'table_test': shared_root / 'training_tables' / 'test.csv',
    }


def _reaction_gnn_cache_name(config: ReactionGNNConfig) -> str:
    """Keep graph tables from incompatible R-GNN capacities separate."""
    dropout = f'{config.dropout:.3f}'.rstrip('0').rstrip('.').replace('.', 'p')
    return (
        f'_shared_reaction_gnn_h{config.hidden_dim}_e{config.embedding_dim}'
        f'_mp{config.message_passing_steps}_d{dropout}_s{config.random_state}'
    )


def _reafnn_cache_name(
    config: ReaFNNConfig,
    *,
    knn_retrieval_mode: str,
    use_reafnn: bool,
    top_k: int,
    max_contexts: int,
    prefilter_contexts: int,
) -> str:
    """Keep Stage 2 tables tied to retrieval, selector capacity, and seed."""

    if use_reafnn:
        anchors = 'all' if config.knn_anchor_contexts <= 0 else str(config.knn_anchor_contexts)
        if config.enable_independent_post_fusion:
            policy = 'postfusion'
        elif config.enable_context_augmentation:
            policy = 'augment'
        elif config.enable_knn_wide_refinement:
            policy = 'refine'
        else:
            policy = 'checked'
        config_json = json.dumps(config.to_dict(), sort_keys=True, separators=(',', ':'))
        config_signature = hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:12]
        selector = f'reafnn_{policy}_a{anchors}_{config_signature}'
    else:
        selector = 'knnonly'
    return (
        f'_shared_knn_{knn_retrieval_mode}_k{top_k}_p{prefilter_contexts}_m{max_contexts}'
        f'_fp{config.fpsize}_r{config.radius}_{selector}'
    )


def _maybe_unlink(paths: dict[str, Path]) -> None:
    for path in paths.values():
        if path.exists():
            path.unlink()


def _ensure_knn_tables(
    repo_root: Path,
    family: str,
    route_cache: Path,
    shared_root: Path,
    *,
    top_k: int,
    max_contexts: int,
    prefilter_contexts: int,
    fpsize: int,
    radius: int,
    knn_retrieval_mode: str,
    max_train_routes: int | None,
    max_val_routes: int | None,
    force_rebuild: bool,
    reafnn_config: ReaFNNConfig,
    use_reafnn: bool,
    reaffn_force_retrain: bool,
    post_fusion_validation_route_cache: Path | None,
) -> dict[str, Path]:
    """Build reference-route training tables and predicted-route test tables."""

    paths = _shared_paths(shared_root)
    if force_rebuild:
        _maybe_unlink(paths)
    if all(path.exists() for path in paths.values()):
        return paths

    builder = KNNContextPoolBuilder(
        repo_root=repo_root,
        family=family,
        top_k=top_k,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        retrieval_mode=knn_retrieval_mode,
        prefilter_contexts=prefilter_contexts,
        reaffn_artifact_dir=(shared_root / 'reafnn') if use_reafnn else None,
        reaffn_device=reafnn_config.device,
        reaffn_force_retrain=reaffn_force_retrain,
        reaffn_config=reafnn_config if use_reafnn else None,
        post_fusion_validation_route_cache=post_fusion_validation_route_cache,
        sparse_similarity=True,
    )
    train_split_file = split_file_for_family(repo_root, family, 'train')
    val_split_file = split_file_for_family(repo_root, family, 'val')
    test_split_file = split_file_for_family(repo_root, family, 'test')

    # Training and validation candidate rows use their reference split routes;
    # test rows are emitted solely from persisted Stage 1 predictions.
    builder.build_table('train', paths['candidate_train'], max_routes=max_train_routes)
    builder.build_table('val', paths['candidate_val'], max_routes=max_val_routes)
    builder.build_non_oracle_table(route_cache, paths['candidate_test'])
    label_candidate_table(paths['candidate_train'], train_split_file, paths['table_train'])
    label_candidate_table(paths['candidate_val'], val_split_file, paths['table_val'])
    label_candidate_table(paths['candidate_test'], test_split_file, paths['table_test'])
    return paths


def _ensure_gnn_augmented_tables(
    repo_root: Path,
    family: str,
    stage2_root: Path,
    table_paths: dict[str, Path],
    *,
    force_rebuild: bool,
    gnn_config: ReactionGNNConfig,
    gnn_force_retrain: bool,
) -> dict[str, Path]:
    # Bind graph features to the precise Stage 2 candidate-table cache. This
    # prevents a retrieval-policy change from accidentally reusing graph
    # tables created from different candidate rows.
    gnn_root = stage2_root / _reaction_gnn_cache_name(gnn_config)
    paths = _shared_augmented_table_paths(gnn_root)
    if force_rebuild:
        _maybe_unlink(paths)
    if all(path.exists() for path in paths.values()):
        return paths

    split_train = split_file_for_family(repo_root, family, 'train')
    split_val = split_file_for_family(repo_root, family, 'val')
    model_dir = gnn_root / 'model'
    train_reaction_gnn_feature_model(
        train_split_file=split_train,
        val_split_file=split_val,
        output_dir=model_dir,
        config=gnn_config,
        force_retrain=gnn_force_retrain,
    )
    augment_table_with_reaction_gnn_features(
        table_file=table_paths['table_train'],
        artifact_dir=model_dir,
        output_file=paths['table_train'],
        device=gnn_config.device,
    )
    augment_table_with_reaction_gnn_features(
        table_file=table_paths['table_val'],
        artifact_dir=model_dir,
        output_file=paths['table_val'],
        device=gnn_config.device,
    )
    augment_table_with_reaction_gnn_features(
        table_file=table_paths['table_test'],
        artifact_dir=model_dir,
        output_file=paths['table_test'],
        device=gnn_config.device,
    )
    return paths


def _score_table_with_stage2_heuristic(table_file: Path) -> pd.DataFrame:
    """Apply the fixed Stage 1/2 prior used by the no-XGB-LTR control.

    This arm has no fitted parameters. It uses only candidate fields emitted by
    Stages 1 and 2 and resolves ties stably, so it cannot learn from labels.
    """

    frame = pd.read_csv(table_file)
    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, is_ascending in HEURISTIC_STAGE3_SORT_SPECS:
        if column in frame.columns:
            sort_columns.append(column)
            ascending.append(is_ascending)
    if sort_columns:
        ranked = frame.sort_values(
            sort_columns,
            ascending=ascending,
            kind='mergesort',
        ).reset_index(drop=True)
    else:
        ranked = frame.reset_index(drop=True)
    ranked['stage2_prior_rank'] = ranked.groupby('sample_index', sort=False).cumcount() + 1
    ranked['stage2_prior_score'] = -ranked['stage2_prior_rank'].astype(np.float32)
    return ranked


def _baseline_identifier(*, ranking_mode: str, enable_temperature: bool) -> str:
    if ranking_mode == 'stage2_heuristic':
        return 'stage2_heuristic_no_xgb_ltr'
    return (
        'knn_xgb_reaction_gnn_temperature'
        if enable_temperature
        else 'knn_xgb_stage2_ablation_ranking_only'
    )


def _run_family(
    repo_root: Path,
    family: str,
    route_cache: Path,
    output_root: Path,
    *,
    top_k: int,
    max_contexts: int,
    prefilter_contexts: int,
    fpsize: int,
    radius: int,
    knn_retrieval_mode: str,
    max_train_routes: int | None,
    max_val_routes: int | None,
    force_rebuild: bool,
    reafnn_config: ReaFNNConfig,
    use_reafnn: bool,
    reaffn_force_retrain: bool,
    post_fusion_validation_route_root: Path | None,
    gnn_config: ReactionGNNConfig,
    reuse_candidate_tables_root: Path | None,
    gnn_force_retrain: bool,
    enable_temperature: bool,
    ranking_mode: str,
    seed: int,
) -> dict:
    family_root = output_root / family
    shared_root = family_root / _reafnn_cache_name(
        reafnn_config,
        knn_retrieval_mode=knn_retrieval_mode,
        use_reafnn=use_reafnn,
        top_k=top_k,
        max_contexts=max_contexts,
        prefilter_contexts=prefilter_contexts,
    )
    post_fusion_validation_route_cache = (
        post_fusion_validation_route_root / family / 'route_cache.json'
        if post_fusion_validation_route_root is not None else None
    )
    post_fusion_calibration_file = shared_root / 'reafnn' / 'post_fusion_calibration.json'
    post_fusion_calibration = (
        json.loads(post_fusion_calibration_file.read_text(encoding='utf-8'))
        if use_reafnn and reafnn_config.enable_independent_post_fusion and post_fusion_calibration_file.exists()
        else None
    )
    stage2_protocol = {
        'architecture': 'knn_reafnn' if use_reafnn else 'knn_only',
        # Test slates always come from persisted Stage-1 predictions. Training
        # and validation tables use only their matching reference split routes.
        'training_candidate_table_mode': 'reference_split_routes',
        'training_candidate_route_source': 'reference_split_routes',
        'knn_retrieval_mode': knn_retrieval_mode,
        'knn_feature_space': (
            'product_morgan_fingerprint' if knn_retrieval_mode == 'product_morgan'
            else 'reactant_product_delta_morgan_fingerprint'
        ),
        'knn_top_k': int(top_k),
        'prefilter_contexts': int(prefilter_contexts),
        'max_contexts': int(max_contexts),
        'candidate_source': (
            'historical_contexts_from_family_train_split'
            if use_reafnn and reafnn_config.enable_independent_post_fusion
            else 'historical_contexts_retrieved_from_family_train_split'
        ),
        'reafnn_enabled': bool(use_reafnn),
        'reafnn_feature_space': (
            'product_fp_plus_delta_fp_plus_route_descriptors' if use_reafnn else None
        ),
        'reafnn_candidate_policy': (
            'independent_knn_reafnn_post_fusion' if use_reafnn and reafnn_config.enable_independent_post_fusion
            else 'knn_wide_pool_refinement' if use_reafnn and reafnn_config.enable_knn_wide_refinement
            else 'context_augmentation' if use_reafnn and reafnn_config.enable_context_augmentation
            else 'knn_core_rank_correction' if use_reafnn
            else 'not_used'
        ),
        'reafnn_config': reafnn_config.to_dict() if use_reafnn else None,
    }
    if use_reafnn and reafnn_config.enable_independent_post_fusion:
        stage2_protocol['reafnn_post_fusion_calibration'] = post_fusion_calibration
    result_dir = family_root / (
        'knn_xgb' if ranking_mode == 'xgb_ltr' else 'stage2_heuristic'
    ) / 'non_oracle'
    result_file = result_dir / 'result.json'
    if result_file.exists() and not force_rebuild:
        existing = json.loads(result_file.read_text(encoding='utf-8'))
        existing_model = existing.get('model') or {}
        existing_temperature_protocol = existing_model.get('temperature_protocol') or {}
        existing_ranker_features = existing_model.get('feature_columns') or []
        existing_ranking_protocol = existing_model.get('ranking_protocol') or {}
        existing_stage2_protocol = existing_model.get('stage2_protocol') or {}
        existing_ranker_is_non_graph = bool(existing_ranker_features) and all(
            not str(column).startswith('route_gnn_feat_') for column in existing_ranker_features
        )
        ranker_matches = (
            existing_ranker_is_non_graph
            if ranking_mode == 'xgb_ltr'
            else existing_ranking_protocol.get('architecture') == 'deterministic_stage1_stage2_prior'
        )
        expected_baseline = _baseline_identifier(
            ranking_mode=ranking_mode,
            enable_temperature=enable_temperature,
        )
        temperature_matches = (
            existing_temperature_protocol.get('always_enabled') is True
            and existing_temperature_protocol.get('selection') == 'none'
            and existing_temperature_protocol.get('reaction_gnn_config') == gnn_config.to_dict()
        ) if enable_temperature else (
            existing_temperature_protocol.get('always_enabled') is False
            and existing_temperature_protocol.get('selection') == 'not_run_for_stage2_ablation'
        )
        if (
            existing.get('baseline') == expected_baseline
            and temperature_matches
            and ranker_matches
            and existing_stage2_protocol == stage2_protocol
        ):
            return existing

    if reuse_candidate_tables_root is not None:
        source_family_root = reuse_candidate_tables_root / family
        source_shared_root = source_family_root / _reafnn_cache_name(
            reafnn_config,
            knn_retrieval_mode=knn_retrieval_mode,
            use_reafnn=use_reafnn,
            top_k=top_k,
            max_contexts=max_contexts,
            prefilter_contexts=prefilter_contexts,
        )
        if use_reafnn:
            source_reafnn_meta = source_shared_root / 'reafnn' / 'reafnn_meta.json'
            if not source_reafnn_meta.exists():
                raise FileNotFoundError(
                    f'{family} is missing a ReaFNN cache matching the requested configuration: {source_reafnn_meta}'
                )
            source_reafnn_config = json.loads(source_reafnn_meta.read_text(encoding='utf-8')).get('config')
            if source_reafnn_config != reafnn_config.to_dict():
                raise ValueError(f'{family} has an incompatible reusable ReaFNN configuration.')
        table_paths = _shared_paths(source_shared_root)
        missing_tables = [str(path) for path in table_paths.values() if not path.exists()]
        if missing_tables:
            raise FileNotFoundError(f'{family} is missing reusable candidate tables: {missing_tables[:3]}')
    else:
        table_paths = _ensure_knn_tables(
            repo_root=repo_root,
            family=family,
            route_cache=route_cache,
            shared_root=shared_root,
            top_k=top_k,
            max_contexts=max_contexts,
            prefilter_contexts=prefilter_contexts,
            fpsize=fpsize,
            radius=radius,
            knn_retrieval_mode=knn_retrieval_mode,
            max_train_routes=max_train_routes,
            max_val_routes=max_val_routes,
            force_rebuild=force_rebuild,
            reafnn_config=reafnn_config,
            use_reafnn=use_reafnn,
            reaffn_force_retrain=reaffn_force_retrain,
            post_fusion_validation_route_cache=post_fusion_validation_route_cache,
        )

    if use_reafnn and reafnn_config.enable_independent_post_fusion:
        calibration_file = table_paths['candidate_test'].parents[1] / 'reafnn' / 'post_fusion_calibration.json'
        if not calibration_file.exists():
            raise FileNotFoundError(f'{family} is missing post-fusion calibration: {calibration_file}')
        stage2_protocol['reafnn_post_fusion_calibration'] = json.loads(
            calibration_file.read_text(encoding='utf-8')
        )

    model_dir = result_dir / 'model'
    gnn_temperature_model_dir = result_dir / 'gnn_temperature_model'
    if force_rebuild:
        for directory in (model_dir, gnn_temperature_model_dir):
            if directory.exists():
                for path in sorted(directory.glob('*')):
                    if path.is_file():
                        path.unlink()

    rank_artifacts: dict = {}
    if ranking_mode == 'xgb_ltr':
        rank_artifacts = train_xgb_ranker_and_temperature(
            train_table_file=table_paths['table_train'],
            val_table_file=table_paths['table_val'],
            output_dir=model_dir,
            random_state=seed,
            train_temperature=False,
        )

    if enable_temperature:
        # Reusing frozen Stage 2 tables intentionally does not reuse graph
        # tables: the R-GNN cache remains tied to the current Stage 2 cache.
        gnn_table_paths = _ensure_gnn_augmented_tables(
            repo_root=repo_root,
            family=family,
            stage2_root=shared_root,
            table_paths=table_paths,
            force_rebuild=force_rebuild,
            gnn_config=gnn_config,
            gnn_force_retrain=gnn_force_retrain,
        )
        gnn_temperature_artifacts = train_xgb_temperature_regressor(
            train_table_file=gnn_table_paths['table_train'],
            val_table_file=gnn_table_paths['table_val'],
            output_dir=gnn_temperature_model_dir,
            random_state=seed,
        )
        selected_temperature_model_file = gnn_temperature_artifacts.get('model_file')
        selected_temperature_metadata_file = gnn_temperature_artifacts.get('metadata_file')
        if selected_temperature_model_file is None or selected_temperature_metadata_file is None:
            raise RuntimeError(f'{family}: the mandatory R-GNN temperature model could not be trained.')
        scoring_table = gnn_table_paths['table_test']
        temperature_model = {
            'architecture': 'reaction_gnn_augmented_xgboost_regressor',
            'always_enabled': True,
            'reaction_gnn_config': gnn_config.to_dict(),
            **gnn_temperature_artifacts,
        }
        temperature_protocol = {
            'architecture': 'reaction_gnn_augmented_xgboost_regressor',
            'always_enabled': True,
            'reaction_gnn_config': gnn_config.to_dict(),
            'selection': 'none',
        }
        scoring_kwargs = {
            'temperature_model_file': selected_temperature_model_file,
            'temperature_metadata_file': selected_temperature_metadata_file,
        }
    else:
        gnn_temperature_artifacts = {}
        selected_temperature_model_file = None
        selected_temperature_metadata_file = None
        scoring_table = table_paths['table_test']
        temperature_model = {
            'architecture': 'not_run_for_stage2_ablation',
            'always_enabled': False,
            'reason': 'Sys@k-only controlled Stage 2 ablation',
        }
        temperature_protocol = {
            'architecture': 'not_run_for_stage2_ablation',
            'always_enabled': False,
            'selection': 'not_run_for_stage2_ablation',
        }
        scoring_kwargs = {}

    if ranking_mode == 'xgb_ltr':
        scored = score_table_with_xgb(
            table_file=scoring_table,
            model_file=rank_artifacts['model_file'],
            metadata_file=rank_artifacts['metadata_file'],
            **scoring_kwargs,
        )
        ranker_model = {
            'architecture': 'xgb_ranker',
            'feature_space': 'tabular_non_graph',
            **rank_artifacts,
        }
        ranking_protocol = {
            'architecture': 'xgb_ranker',
            'feature_space': 'tabular_non_graph',
            'feature_count': len(rank_artifacts.get('feature_columns') or []),
        }
        score_column = 'xgb_score'
    else:
        scored = _score_table_with_stage2_heuristic(scoring_table)
        ranker_model = {
            'architecture': 'deterministic_stage1_stage2_prior',
            'feature_space': 'stage1_stage2_prior_only',
            'feature_columns': [],
            'sort_specs': [list(spec) for spec in HEURISTIC_STAGE3_SORT_SPECS],
            'learned_parameters': False,
        }
        ranking_protocol = {
            'architecture': 'deterministic_stage1_stage2_prior',
            'feature_space': 'stage1_stage2_prior_only',
            'score_column': 'stage2_prior_score',
            'sort_specs': [list(spec) for spec in HEURISTIC_STAGE3_SORT_SPECS],
            'learned_parameters': False,
        }
        score_column = 'stage2_prior_score'
    scored_file = result_dir / 'test_scored.csv'
    scored_file.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_file, index=False)

    result = {
        'family': family,
        'baseline': _baseline_identifier(
            ranking_mode=ranking_mode,
            enable_temperature=enable_temperature,
        ),
        'seed': seed,
        'candidate_table': str(scoring_table),
        'scored_test_file': str(scored_file),
        'model': {
            'stage2_protocol': stage2_protocol,
            'ranker': ranker_model,
            'temperature': temperature_model,
            # Preserve flat artifact keys for downstream audit scripts.
            'output_dir': rank_artifacts.get('output_dir'),
            'model_file': rank_artifacts.get('model_file'),
            'metadata_file': rank_artifacts.get('metadata_file'),
            'feature_columns': rank_artifacts.get('feature_columns'),
            'best_iteration': rank_artifacts.get('best_iteration'),
            'temperature_model_file': selected_temperature_model_file,
            'temperature_metadata_file': selected_temperature_metadata_file,
            'temperature_num_train': gnn_temperature_artifacts.get('temperature_num_train'),
            'temperature_gnn': gnn_temperature_artifacts if enable_temperature else None,
            'ranking_protocol': ranking_protocol,
            'temperature_protocol': temperature_protocol,
        },
        'metrics': evaluate_scored_frame_with_manifest(
            scored,
            expected_sample_indices=load_route_cache_sample_indices(route_cache),
            score_column=score_column,
            temperature_column='xgb_temperature_pred' if enable_temperature else None,
        ),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, result_file)
    return result


def _prune_completed_family_intermediates(
    *,
    family_root: Path,
    result: dict,
    ranking_mode: str,
) -> dict:
    """Retain the auditable result record while releasing regenerable tables."""

    result_dir = family_root / (
        'knn_xgb' if ranking_mode == 'xgb_ltr' else 'stage2_heuristic'
    ) / 'non_oracle'
    result_file = result_dir / 'result.json'
    if not result_file.exists():
        return result

    retained_file = result_file.resolve()
    removed_files = 0
    removed_bytes = 0
    for path in sorted(
        family_root.rglob('*'),
        key=lambda value: (len(value.parts), str(value)),
        reverse=True,
    ):
        if not path.is_file():
            continue
        if path.resolve() == retained_file:
            continue
        removed_bytes += int(path.stat().st_size)
        path.unlink()
        removed_files += 1

    for path in sorted(
        family_root.rglob('*'),
        key=lambda value: (len(value.parts), str(value)),
        reverse=True,
    ):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    compact_result = dict(result)
    compact_result['intermediate_artifacts'] = {
        'status': 'pruned_after_completed_family',
        'retained': ['result.json', 'intermediate_cleanup.json'],
        'removed_file_count': int(removed_files),
        'removed_bytes': int(removed_bytes),
        'note': 'Candidate pools, labeled tables, scored CSVs, and model binaries are regenerable from the recorded protocol.',
    }
    _write_json(compact_result, result_file)
    _write_json(
        compact_result['intermediate_artifacts'],
        family_root / 'intermediate_cleanup.json',
    )
    return compact_result


def _flatten_rows(rows: list[dict]) -> pd.DataFrame:
    flat_rows: list[dict] = []
    for row in rows:
        metrics = row.get('metrics', {})
        temp = metrics.get('temperature', {})
        recall = row.get('stage1_route_recall', {})
        flat_rows.append(
            {
                'family': row.get('family'),
                'rr1': recall.get('route_recall_top1'),
                'rr3': recall.get('route_recall_top3'),
                'rr5': recall.get('route_recall_top5'),
                'rr10': recall.get('route_recall_top10'),
                'pool_route_coverage': metrics.get('pool_route_coverage'),
                'pool_context_coverage': metrics.get('pool_context_coverage'),
                'pool_coverage': metrics.get('pool_coverage'),
                'sys1': metrics.get('system_top1_all'),
                'sys3': metrics.get('system_top3_all'),
                'sys5': metrics.get('system_top5_all'),
                'sys10': metrics.get('system_top10_all'),
                'ndcg10': metrics.get('system_ndcg10'),
                'mrr': metrics.get('system_mrr'),
                'temp_n': temp.get('n'),
                'temp_mae': temp.get('mae'),
                'temp_mse': temp.get('mse'),
                'temp_rmse': temp.get('rmse'),
                'temp_within_5c': temp.get('within_5c'),
                'temp_within_10c': temp.get('within_10c'),
                'temp_within_20c': temp.get('within_20c'),
            }
        )
    return pd.DataFrame(flat_rows)


def _mean_metric(values: list[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return 'NA'
    return f'{value * 100.0:.1f}'


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return 'NA'
    return f'{value:.{digits}f}'


def _write_overview(flat: pd.DataFrame, output_root: Path) -> list[Path]:
    outputs: list[Path] = []
    if flat.empty:
        return outputs

    flat = flat.sort_values('family', key=lambda s: s.map(lambda v: _family_sort_key(str(v)))).reset_index(drop=True)

    macro = {
        'family': 'MACRO-AVG',
        'rr10': _mean_metric(flat['rr10'].tolist()),
        'pool_coverage': _mean_metric(flat['pool_coverage'].tolist()),
        'sys1': _mean_metric(flat['sys1'].tolist()),
        'sys3': _mean_metric(flat['sys3'].tolist()),
        'sys5': _mean_metric(flat['sys5'].tolist()),
        'sys10': _mean_metric(flat['sys10'].tolist()),
        'temp_n': int(pd.to_numeric(flat['temp_n'], errors='coerce').fillna(0).sum()),
        'temp_mae': _mean_metric(flat['temp_mae'].dropna().tolist()),
        'temp_within_5c': _mean_metric(flat['temp_within_5c'].dropna().tolist()),
        'temp_within_10c': _mean_metric(flat['temp_within_10c'].dropna().tolist()),
        'temp_within_20c': _mean_metric(flat['temp_within_20c'].dropna().tolist()),
    }

    display_rows = flat.to_dict(orient='records') + [macro]

    md_lines = ['# ProSys Non-Oracle Mainline Results', '', '## Route and Full-System Metrics', '']
    md_lines.append('| Family | Route@10 | Candidate recall | Full-system Top-1 accuracy | Full-system Top-3 accuracy | Full-system Top-5 accuracy | Full-system Top-10 accuracy |')
    md_lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in display_rows:
        md_lines.append(
            '| '
            + ' | '.join(
                [
                    display_family_name(str(row['family'])),
                    _format_percent(row.get('rr10')),
                    _format_percent(row.get('pool_coverage')),
                    _format_percent(row.get('sys1')),
                    _format_percent(row.get('sys3')),
                    _format_percent(row.get('sys5')),
                    _format_percent(row.get('sys10')),
                ]
            )
            + ' |'
        )
    md_lines.extend(['', '## Conditional Temperature Metrics', ''])
    md_lines.append('| Family | N_temp | MAE (deg C) | Within +/-5 deg C | Within +/-10 deg C | Within +/-20 deg C |')
    md_lines.append('| --- | ---: | ---: | ---: | ---: | ---: |')
    for row in display_rows:
        md_lines.append(
            '| '
            + ' | '.join(
                [
                    display_family_name(str(row['family'])),
                    _format_number(row.get('temp_n'), 0),
                    _format_number(row.get('temp_mae')),
                    _format_percent(row.get('temp_within_5c')),
                    _format_percent(row.get('temp_within_10c')),
                    _format_percent(row.get('temp_within_20c')),
                ]
            )
            + ' |'
        )
    overview_md = output_root / 'overview.md'
    overview_md.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    outputs.append(overview_md)

    family_width = max(len(display_family_name(str(row['family']))) for row in display_rows)
    route_label = PUBLIC_METRIC_LABELS['route_recall_top10']
    cr_label = PUBLIC_METRIC_LABELS['pool_coverage']
    fs1_label = PUBLIC_METRIC_LABELS['system_top1_all']
    fs3_label = PUBLIC_METRIC_LABELS['system_top3_all']
    fs5_label = PUBLIC_METRIC_LABELS['system_top5_all']
    fs10_label = PUBLIC_METRIC_LABELS['system_top10_all']
    txt_lines = [
        'ProSys Non-Oracle Mainline Results',
        f'{"family":<{family_width}}  {route_label:>8}  {cr_label:>5}  {fs1_label:>5}  {fs3_label:>5}  {fs5_label:>5}  {fs10_label:>5}',
    ]
    for row in display_rows:
        txt_lines.append(
            f'{display_family_name(str(row["family"])):<{family_width}}  '
            f'{_format_percent(row.get("rr10")):>8}  '
            f'{_format_percent(row.get("pool_coverage")):>5}  '
            f'{_format_percent(row.get("sys1")):>5}  '
            f'{_format_percent(row.get("sys3")):>5}  '
            f'{_format_percent(row.get("sys5")):>5}  '
            f'{_format_percent(row.get("sys10")):>5}'
        )
    overview_txt = output_root / 'overview.txt'
    overview_txt.write_text('\n'.join(txt_lines) + '\n', encoding='utf-8')
    outputs.append(overview_txt)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the maintained ProSys Non-Oracle mainline.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--output_root', type=str, default='outputs/stage23_mainline')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--fpsize', type=int, default=4096)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument(
        '--knn_retrieval_mode',
        type=str,
        default='product_morgan',
        choices=['reaction_morgan', 'product_morgan'],
        help='KNN representation; product_morgan intentionally omits reactants.',
    )
    parser.add_argument('--knn_top_k', type=int, default=64)
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--prefilter_contexts', type=int, default=64)
    parser.add_argument('--max_train_routes', type=int, default=0, help='0 means use all train routes')
    parser.add_argument('--max_val_routes', type=int, default=0, help='0 means use all val routes')
    parser.add_argument('--force_rebuild', action='store_true')
    parser.add_argument(
        '--cleanup_family_intermediates',
        action='store_true',
        help='After each completed family, retain only its result record and cleanup manifest to limit disk use.',
    )
    parser.add_argument('--reafnn_device', type=str, default='cpu')
    parser.add_argument('--reafnn_force_retrain', action='store_true')
    parser.add_argument('--disable_reafnn', action='store_true')
    parser.add_argument('--reafnn_hidden_dim', type=int, default=512)
    parser.add_argument('--reafnn_hidden_layers', type=int, default=2)
    parser.add_argument('--reafnn_dropout', type=float, default=0.10)
    parser.add_argument('--reafnn_activation', type=str, default='relu', choices=['relu', 'gelu'])
    parser.add_argument('--reafnn_use_layer_norm', action='store_true')
    parser.add_argument('--reafnn_learning_rate', type=float, default=1e-3)
    parser.add_argument('--reafnn_weight_decay', type=float, default=1e-5)
    parser.add_argument('--reafnn_batch_size', type=int, default=64)
    parser.add_argument('--reafnn_max_epochs', type=int, default=30)
    parser.add_argument('--reafnn_patience', type=int, default=8)
    parser.add_argument('--reafnn_knn_anchor_contexts', type=int, default=12, help='Number of KNN-core anchors retained during wide-pool refinement.')
    parser.add_argument('--reafnn_correction_weight', type=float, default=0.65)
    parser.add_argument('--reafnn_correction_clip', type=float, default=0.35)
    parser.add_argument('--reafnn_enable_context_augmentation', action='store_true')
    parser.add_argument(
        '--reafnn_enable_knn_wide_refinement',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--reafnn_enable_independent_post_fusion',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Maintained Stage 2 mode: independently predict historical contexts with KNN and ReaFNN, then fuse validation-calibrated scores.',
    )
    parser.add_argument(
        '--reafnn_independent_contexts',
        type=int,
        default=64,
        help='Historical contexts retained independently from ReaFNN before KNN/ReaFNN post-fusion.',
    )
    parser.add_argument(
        '--reafnn_post_fusion_weights',
        type=str,
        default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
        help='Validation-only grid for w in w*KNN + (1-w)*ReaFNN.',
    )
    parser.add_argument(
        '--reafnn_post_fusion_validation_route_root',
        type=str,
        default='outputs/stage1_routes_validation',
        help='Held-out Stage 1 validation route caches used only to select the Stage 2 fusion weight; use an empty string to fall back to reference validation routes.',
    )
    parser.add_argument('--gnn_device', type=str, default='cpu')
    parser.add_argument('--gnn_force_retrain', action='store_true')
    parser.add_argument('--skip_temperature', action='store_true', help='For Sys@k-only ablations; leaves the full mainline unchanged.')
    parser.add_argument(
        '--ranking_mode',
        type=str,
        default='xgb_ltr',
        choices=['xgb_ltr', 'stage2_heuristic'],
        help='Stage 3 ranking policy; stage2_heuristic is the no-XGB-LTR ablation only.',
    )
    parser.add_argument('--reuse_candidate_tables_root', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='Shared random seed for ReaFNN, R-GNN, and XGB-LTR.')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    families = parse_families_arg(args.families)
    max_train_routes = args.max_train_routes if args.max_train_routes > 0 else None
    max_val_routes = args.max_val_routes if args.max_val_routes > 0 else None
    reuse_candidate_tables_root = (repo_root / args.reuse_candidate_tables_root).resolve() if args.reuse_candidate_tables_root else None
    post_fusion_validation_route_root = (
        (repo_root / args.reafnn_post_fusion_validation_route_root).resolve()
        if args.reafnn_post_fusion_validation_route_root else None
    )
    if args.prefilter_contexts < args.max_contexts:
        parser.error('--prefilter_contexts must be at least --max_contexts.')
    if args.reafnn_knn_anchor_contexts < 0 or args.reafnn_knn_anchor_contexts > args.max_contexts:
        parser.error('--reafnn_knn_anchor_contexts must be between 0 and --max_contexts.')
    if args.reafnn_enable_context_augmentation and args.reafnn_enable_knn_wide_refinement:
        parser.error('ReaFNN context augmentation and KNN-wide refinement are mutually exclusive.')
    if args.disable_reafnn and args.reafnn_enable_context_augmentation:
        parser.error('--disable_reafnn cannot be combined with context augmentation.')
    if args.disable_reafnn and args.reafnn_enable_independent_post_fusion:
        parser.error('--disable_reafnn cannot be combined with independent post-fusion.')
    if args.reafnn_enable_independent_post_fusion and args.reafnn_enable_context_augmentation:
        parser.error('Independent post-fusion uses historical contexts only and cannot be combined with context augmentation.')
    if args.reafnn_enable_independent_post_fusion and args.reafnn_independent_contexts < args.max_contexts:
        parser.error('--reafnn_independent_contexts must be at least --max_contexts.')
    if args.ranking_mode == 'stage2_heuristic' and not args.skip_temperature:
        parser.error('--ranking_mode stage2_heuristic requires --skip_temperature.')
    reafnn_config = ReaFNNConfig(
        fpsize=args.fpsize,
        radius=args.radius,
        hidden_dim=args.reafnn_hidden_dim,
        hidden_layers=args.reafnn_hidden_layers,
        dropout=args.reafnn_dropout,
        activation=args.reafnn_activation,
        use_layer_norm=args.reafnn_use_layer_norm,
        learning_rate=args.reafnn_learning_rate,
        weight_decay=args.reafnn_weight_decay,
        batch_size=args.reafnn_batch_size,
        max_epochs=args.reafnn_max_epochs,
        patience=args.reafnn_patience,
        device=args.reafnn_device,
        random_state=args.seed,
        knn_anchor_contexts=(0 if args.reafnn_enable_independent_post_fusion else args.reafnn_knn_anchor_contexts),
        correction_weight=(0.0 if args.reafnn_enable_independent_post_fusion else args.reafnn_correction_weight),
        correction_clip=(0.0 if args.reafnn_enable_independent_post_fusion else args.reafnn_correction_clip),
        enable_context_augmentation=args.reafnn_enable_context_augmentation,
        enable_knn_wide_refinement=(
            False if args.reafnn_enable_independent_post_fusion
            else args.reafnn_enable_knn_wide_refinement
        ),
        enable_independent_post_fusion=args.reafnn_enable_independent_post_fusion,
        independent_contexts=args.reafnn_independent_contexts,
        post_fusion_weight_grid=args.reafnn_post_fusion_weights,
        post_fusion_validation_source=(
            'predicted_stage1_validation_routes'
            if args.reafnn_enable_independent_post_fusion and post_fusion_validation_route_root is not None
            else 'reference_split_routes'
        ),
    )
    gnn_config = ReactionGNNConfig(
        device=args.gnn_device,
        random_state=args.seed,
    )

    summary_rows: list[dict] = []
    for family in families:
        route_cache = route_root / family / 'route_cache.json'
        if not route_cache.exists():
            print(f'[mainline] skip {family}: missing {route_cache}', flush=True)
            continue
        print(f'[mainline] running {family}', flush=True)
        family_result = _run_family(
            repo_root=repo_root,
            family=family,
            route_cache=route_cache,
            output_root=output_root,
            top_k=args.knn_top_k,
            max_contexts=args.max_contexts,
            prefilter_contexts=args.prefilter_contexts,
            fpsize=args.fpsize,
            radius=args.radius,
            knn_retrieval_mode=args.knn_retrieval_mode,
            max_train_routes=max_train_routes,
            max_val_routes=max_val_routes,
            force_rebuild=args.force_rebuild,
            reafnn_config=reafnn_config,
            use_reafnn=not args.disable_reafnn,
            reaffn_force_retrain=args.reafnn_force_retrain,
            post_fusion_validation_route_root=post_fusion_validation_route_root,
            gnn_config=gnn_config,
            gnn_force_retrain=args.gnn_force_retrain,
            enable_temperature=not args.skip_temperature,
            ranking_mode=args.ranking_mode,
            reuse_candidate_tables_root=reuse_candidate_tables_root,
            seed=args.seed,
        )
        if args.cleanup_family_intermediates:
            family_result = _prune_completed_family_intermediates(
                family_root=output_root / family,
                result=family_result,
                ranking_mode=args.ranking_mode,
            )
        summary_rows.append(family_result)

    flat = _flatten_rows(summary_rows)
    if not flat.empty:
        flat = flat.sort_values('family', key=lambda s: s.map(lambda v: _family_sort_key(str(v)))).reset_index(drop=True)
    flat.to_csv(output_root / 'results_flat.csv', index=False)
    _write_json(summary_rows, output_root / 'all_results.json')
    _write_overview(flat, output_root)


if __name__ == '__main__':
    main()
