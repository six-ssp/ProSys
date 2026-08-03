"""Run the maintained ProSys Non-Oracle mainline: KNN pool + XGBoost rerank."""

from __future__ import annotations

import argparse
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
    stable_sort_candidate_frame,
)
from prosys_shared.route_cache import load_route_cache_sample_indices
from stage2_KNN import KNNContextPoolBuilder
from stage2_KNN.reafnn_selector import ReaFNNConfig
from stage3_XGBoost import score_table_with_xgb, train_xgb_ranker_and_temperature, train_xgb_temperature_regressor
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


def _aux_training_paths(shared_root: Path) -> dict[str, Path]:
    aux_root = shared_root / 'non_oracle_training'
    return {
        'candidate_train_non_oracle': aux_root / 'candidate_pool' / 'train.csv',
        'candidate_val_non_oracle': aux_root / 'candidate_pool' / 'val.csv',
        'table_train_non_oracle': aux_root / 'training_tables' / 'train.csv',
        'table_val_non_oracle': aux_root / 'training_tables' / 'val.csv',
        'table_train_oracle': aux_root / 'training_tables' / 'train_oracle.csv',
        'table_val_oracle': aux_root / 'training_tables' / 'val_oracle.csv',
    }


def _shared_augmented_table_paths(shared_root: Path) -> dict[str, Path]:
    return {
        'table_train': shared_root / 'training_tables' / 'train.csv',
        'table_val': shared_root / 'training_tables' / 'val.csv',
        'table_test': shared_root / 'training_tables' / 'test.csv',
    }


def _maybe_unlink(paths: dict[str, Path]) -> None:
    for path in paths.values():
        if path.exists():
            path.unlink()


def _resolve_route_cache(route_root: Path | None, family: str, split: str) -> Path:
    if route_root is None:
        raise ValueError(f'{split} route root is required for this train_table_mode')
    cache_file = route_root / family / 'route_cache.json'
    if not cache_file.exists():
        raise FileNotFoundError(f'missing {split} route cache for {family}: {cache_file}')
    return cache_file


def _dedup_key_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ['sample_index', 'reaction_id', 'route_canonical', 'reagent_norm', 'solvent_norm']
    columns = [column for column in preferred if column in frame.columns]
    if columns:
        return columns
    return [
        column
        for column in ['sample_index', 'reaction_id', 'reactants', 'product', 'reagent_norm', 'solvent_norm']
        if column in frame.columns
    ]


def _merge_oracle_with_non_oracle_hard_negatives(
    *,
    oracle_table_file: Path,
    non_oracle_table_file: Path,
    output_file: Path,
    hard_negative_per_sample: int,
) -> Path:
    oracle = pd.read_csv(oracle_table_file)
    non_oracle = pd.read_csv(non_oracle_table_file)
    if oracle.empty or non_oracle.empty:
        merged = stable_sort_candidate_frame(oracle.copy())
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_file, index=False)
        return output_file

    route_match = pd.to_numeric(non_oracle.get('route_match', 0.0), errors='coerce').fillna(0.0)
    hard = non_oracle.loc[route_match < 0.5].copy()
    if hard.empty:
        merged = stable_sort_candidate_frame(oracle.copy())
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_file, index=False)
        return output_file

    key_columns = _dedup_key_columns(oracle)
    if key_columns:
        oracle_keys = oracle.loc[:, key_columns].drop_duplicates().copy()
        oracle_keys['_in_oracle'] = 1
        hard = hard.merge(oracle_keys, on=key_columns, how='left')
        hard = hard.loc[hard['_in_oracle'].isna()].drop(columns=['_in_oracle'])

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, is_ascending in [
        ('sample_index', True),
        ('context_match', False),
        ('retro_rank', True),
        ('retro_probability', False),
        ('retro_score', False),
        ('reafnn_context_score', False),
        ('knn_similarity_sum', False),
        ('knn_neighbor_count', False),
        ('reagent_norm', True),
        ('solvent_norm', True),
    ]:
        if column in hard.columns:
            sort_columns.append(column)
            ascending.append(is_ascending)
    if sort_columns:
        hard = hard.sort_values(sort_columns, ascending=ascending, kind='mergesort').reset_index(drop=True)

    if hard_negative_per_sample > 0 and 'sample_index' in hard.columns:
        hard = hard.groupby('sample_index', sort=False).head(hard_negative_per_sample).reset_index(drop=True)

    oracle = oracle.copy()
    hard = hard.copy()
    oracle['training_source'] = 'oracle'
    hard['training_source'] = 'non_oracle_hard_negative'

    merged = stable_sort_candidate_frame(pd.concat([oracle, hard], ignore_index=True))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    return output_file


def _build_labeled_non_oracle_table(
    *,
    builder: KNNContextPoolBuilder,
    route_cache_file: Path,
    gold_split_file: Path,
    candidate_output_file: Path,
    table_output_file: Path,
    max_routes: int | None,
) -> Path:
    builder.build_non_oracle_table(route_cache_file, candidate_output_file, max_routes=max_routes)
    return label_candidate_table(candidate_output_file, gold_split_file, table_output_file)


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
    max_train_routes: int | None,
    max_val_routes: int | None,
    train_table_mode: str,
    train_route_root: Path | None,
    val_route_root: Path | None,
    hard_negative_per_sample: int,
    force_rebuild: bool,
    reaffn_device: str,
    reaffn_force_retrain: bool,
    reafnn_seed: int,
) -> dict[str, Path]:
    paths = _shared_paths(shared_root)
    aux_paths = _aux_training_paths(shared_root)
    if force_rebuild:
        _maybe_unlink(paths)
        _maybe_unlink(aux_paths)
    if all(path.exists() for path in paths.values()):
        return paths

    builder = KNNContextPoolBuilder(
        repo_root=repo_root,
        family=family,
        top_k=top_k,
        max_contexts=max_contexts,
        fpsize=fpsize,
        radius=radius,
        prefilter_contexts=prefilter_contexts,
        reaffn_artifact_dir=shared_root / 'reafnn',
        reaffn_device=reaffn_device,
        reaffn_force_retrain=reaffn_force_retrain,
        reaffn_config=ReaFNNConfig(
            fpsize=fpsize,
            radius=radius,
            device=reaffn_device,
            random_state=reafnn_seed,
        ),
    )
    builder.build_non_oracle_table(route_cache, paths['candidate_test'])
    label_candidate_table(paths['candidate_test'], split_file_for_family(repo_root, family, 'test'), paths['table_test'])

    train_split_file = split_file_for_family(repo_root, family, 'train')
    val_split_file = split_file_for_family(repo_root, family, 'val')

    if train_table_mode == 'oracle':
        builder.build_table('train', paths['candidate_train'], max_routes=max_train_routes)
        builder.build_table('val', paths['candidate_val'], max_routes=max_val_routes)
        label_candidate_table(paths['candidate_train'], train_split_file, paths['table_train'])
        label_candidate_table(paths['candidate_val'], val_split_file, paths['table_val'])
        return paths

    if train_table_mode == 'non_oracle':
        train_route_cache = _resolve_route_cache(train_route_root, family, 'train')
        val_route_cache = _resolve_route_cache(val_route_root, family, 'val')
        _build_labeled_non_oracle_table(
            builder=builder,
            route_cache_file=train_route_cache,
            gold_split_file=train_split_file,
            candidate_output_file=paths['candidate_train'],
            table_output_file=paths['table_train'],
            max_routes=max_train_routes,
        )
        _build_labeled_non_oracle_table(
            builder=builder,
            route_cache_file=val_route_cache,
            gold_split_file=val_split_file,
            candidate_output_file=paths['candidate_val'],
            table_output_file=paths['table_val'],
            max_routes=max_val_routes,
        )
        return paths

    if train_table_mode != 'mixed_hard_negative':
        raise ValueError(f'Unsupported train_table_mode: {train_table_mode}')

    train_route_cache = _resolve_route_cache(train_route_root, family, 'train')
    val_route_cache = _resolve_route_cache(val_route_root, family, 'val') if val_route_root is not None else None

    builder.build_table('train', paths['candidate_train'], max_routes=max_train_routes)
    builder.build_table('val', paths['candidate_val'], max_routes=max_val_routes)
    label_candidate_table(paths['candidate_train'], train_split_file, aux_paths['table_train_oracle'])
    label_candidate_table(paths['candidate_val'], val_split_file, aux_paths['table_val_oracle'])

    _build_labeled_non_oracle_table(
        builder=builder,
        route_cache_file=train_route_cache,
        gold_split_file=train_split_file,
        candidate_output_file=aux_paths['candidate_train_non_oracle'],
        table_output_file=aux_paths['table_train_non_oracle'],
        max_routes=max_train_routes,
    )
    _merge_oracle_with_non_oracle_hard_negatives(
        oracle_table_file=aux_paths['table_train_oracle'],
        non_oracle_table_file=aux_paths['table_train_non_oracle'],
        output_file=paths['table_train'],
        hard_negative_per_sample=hard_negative_per_sample,
    )

    if val_route_cache is None:
        val_oracle = pd.read_csv(aux_paths['table_val_oracle'])
        val_oracle = stable_sort_candidate_frame(val_oracle)
        val_oracle.to_csv(paths['table_val'], index=False)
    else:
        _build_labeled_non_oracle_table(
            builder=builder,
            route_cache_file=val_route_cache,
            gold_split_file=val_split_file,
            candidate_output_file=aux_paths['candidate_val_non_oracle'],
            table_output_file=aux_paths['table_val_non_oracle'],
            max_routes=max_val_routes,
        )
        _merge_oracle_with_non_oracle_hard_negatives(
            oracle_table_file=aux_paths['table_val_oracle'],
            non_oracle_table_file=aux_paths['table_val_non_oracle'],
            output_file=paths['table_val'],
            hard_negative_per_sample=hard_negative_per_sample,
        )
    return paths


def _ensure_gnn_augmented_tables(
    repo_root: Path,
    family: str,
    family_root: Path,
    table_paths: dict[str, Path],
    *,
    force_rebuild: bool,
    gnn_device: str,
    gnn_force_retrain: bool,
    gnn_seed: int,
) -> dict[str, Path]:
    paths = _shared_augmented_table_paths(family_root / '_shared_reaction_gnn')
    if force_rebuild:
        _maybe_unlink(paths)
    if all(path.exists() for path in paths.values()):
        return paths

    split_train = split_file_for_family(repo_root, family, 'train')
    split_val = split_file_for_family(repo_root, family, 'val')
    model_dir = family_root / '_shared_reaction_gnn' / 'model'
    train_reaction_gnn_feature_model(
        train_split_file=split_train,
        val_split_file=split_val,
        output_dir=model_dir,
        config=ReactionGNNConfig(device=gnn_device, random_state=gnn_seed),
        force_retrain=gnn_force_retrain,
    )
    augment_table_with_reaction_gnn_features(
        table_file=table_paths['table_train'],
        artifact_dir=model_dir,
        output_file=paths['table_train'],
        device=gnn_device,
    )
    augment_table_with_reaction_gnn_features(
        table_file=table_paths['table_val'],
        artifact_dir=model_dir,
        output_file=paths['table_val'],
        device=gnn_device,
    )
    augment_table_with_reaction_gnn_features(
        table_file=table_paths['table_test'],
        artifact_dir=model_dir,
        output_file=paths['table_test'],
        device=gnn_device,
    )
    return paths


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
    max_train_routes: int | None,
    max_val_routes: int | None,
    train_table_mode: str,
    train_route_root: Path | None,
    val_route_root: Path | None,
    hard_negative_per_sample: int,
    force_rebuild: bool,
    reaffn_device: str,
    reaffn_force_retrain: bool,
    gnn_device: str,
    reuse_candidate_tables_root: Path | None,
    gnn_temperature_min_val_mae_improvement: float,
    gnn_force_retrain: bool,
    seed: int,
) -> dict:
    family_root = output_root / family
    shared_root = family_root / '_shared_knn'
    result_dir = family_root / 'knn_xgb' / 'non_oracle'
    result_file = result_dir / 'result.json'
    if result_file.exists() and not force_rebuild:
        return json.loads(result_file.read_text(encoding='utf-8'))

    if reuse_candidate_tables_root is not None:
        source_family_root = reuse_candidate_tables_root / family
        table_paths = _shared_paths(source_family_root / '_shared_knn')
        gnn_table_paths = _shared_augmented_table_paths(source_family_root / '_shared_reaction_gnn')
        missing_tables = [str(path) for path in (*table_paths.values(), *gnn_table_paths.values()) if not path.exists()]
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
            max_train_routes=max_train_routes,
            max_val_routes=max_val_routes,
            train_table_mode=train_table_mode,
            train_route_root=train_route_root,
            val_route_root=val_route_root,
            hard_negative_per_sample=hard_negative_per_sample,
            force_rebuild=force_rebuild,
            reaffn_device=reaffn_device,
            reaffn_force_retrain=reaffn_force_retrain,
            reafnn_seed=seed,
        )
        gnn_table_paths = _ensure_gnn_augmented_tables(
            repo_root=repo_root,
            family=family,
            family_root=family_root,
            table_paths=table_paths,
            force_rebuild=force_rebuild,
            gnn_device=gnn_device,
            gnn_force_retrain=gnn_force_retrain,
            gnn_seed=seed,
        )

    model_dir = result_dir / 'model'
    gnn_temperature_model_dir = result_dir / 'gnn_temperature_model'
    if force_rebuild:
        for directory in (model_dir, gnn_temperature_model_dir):
            if directory.exists():
                for path in sorted(directory.glob('*')):
                    if path.is_file():
                        path.unlink()

    # Ranking deliberately excludes route-GNN columns. The structural GNN is
    # retained in a separately validated temperature branch below.
    rank_artifacts = train_xgb_ranker_and_temperature(
        train_table_file=table_paths['table_train'],
        val_table_file=table_paths['table_val'],
        output_dir=model_dir,
        random_state=seed,
    )
    gnn_temperature_artifacts = train_xgb_temperature_regressor(
        train_table_file=gnn_table_paths['table_train'],
        val_table_file=gnn_table_paths['table_val'],
        output_dir=gnn_temperature_model_dir,
        random_state=seed,
    )

    val_route_cache = (
        val_route_root / family / 'route_cache.json'
        if val_route_root is not None and (val_route_root / family / 'route_cache.json').exists()
        else None
    )
    temperature_selection: dict[str, object] = {
        'enabled': False,
        'selection_split': 'validation',
        'selection_metric': 'temperature.mae',
        'minimum_mae_improvement_c': float(gnn_temperature_min_val_mae_improvement),
        'reason': 'missing_validation_route_cache' if val_route_cache is None else None,
    }
    selected_temperature_model_file = rank_artifacts.get('temperature_model_file')
    selected_temperature_metadata_file = rank_artifacts.get('temperature_metadata_file')
    if val_route_cache is not None and gnn_temperature_artifacts.get('model_file') is not None:
        val_base_scored = score_table_with_xgb(
            table_file=gnn_table_paths['table_val'],
            model_file=rank_artifacts['model_file'],
            metadata_file=rank_artifacts['metadata_file'],
            temperature_model_file=rank_artifacts.get('temperature_model_file'),
            temperature_metadata_file=rank_artifacts.get('temperature_metadata_file'),
        )
        val_gnn_scored = score_table_with_xgb(
            table_file=gnn_table_paths['table_val'],
            model_file=rank_artifacts['model_file'],
            metadata_file=rank_artifacts['metadata_file'],
            temperature_model_file=gnn_temperature_artifacts.get('model_file'),
            temperature_metadata_file=gnn_temperature_artifacts.get('metadata_file'),
        )
        val_expected = load_route_cache_sample_indices(val_route_cache)
        val_base_metrics = evaluate_scored_frame_with_manifest(
            val_base_scored,
            expected_sample_indices=val_expected,
            score_column='xgb_score',
            temperature_column='xgb_temperature_pred',
        )
        val_gnn_metrics = evaluate_scored_frame_with_manifest(
            val_gnn_scored,
            expected_sample_indices=val_expected,
            score_column='xgb_score',
            temperature_column='xgb_temperature_pred',
        )
        base_mae = (val_base_metrics.get('temperature') or {}).get('mae')
        gnn_mae = (val_gnn_metrics.get('temperature') or {}).get('mae')
        use_gnn_temperature = bool(
            base_mae is not None
            and gnn_mae is not None
            and float(gnn_mae) <= float(base_mae) - float(gnn_temperature_min_val_mae_improvement)
        )
        temperature_selection.update({
            'enabled': use_gnn_temperature,
            'base_validation_mae': base_mae,
            'gnn_validation_mae': gnn_mae,
        })
        if use_gnn_temperature:
            selected_temperature_model_file = gnn_temperature_artifacts.get('model_file')
            selected_temperature_metadata_file = gnn_temperature_artifacts.get('metadata_file')

    scored = score_table_with_xgb(
        table_file=gnn_table_paths['table_test'],
        model_file=rank_artifacts['model_file'],
        metadata_file=rank_artifacts['metadata_file'],
        temperature_model_file=selected_temperature_model_file,
        temperature_metadata_file=selected_temperature_metadata_file,
    )
    scored_file = result_dir / 'test_scored.csv'
    scored_file.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_file, index=False)

    result = {
        'family': family,
        'baseline': 'knn_xgb_gnn_temperature_gated',
        'seed': seed,
        'candidate_table': str(gnn_table_paths['table_test']),
        'scored_test_file': str(scored_file),
        'model': {
            'ranker_no_gnn': rank_artifacts,
            # Preserve the established flat artifact keys for downstream audit
            # scripts while retaining explicit ranker/temperature sub-artifacts.
            'output_dir': rank_artifacts.get('output_dir'),
            'model_file': rank_artifacts.get('model_file'),
            'metadata_file': rank_artifacts.get('metadata_file'),
            'feature_columns': rank_artifacts.get('feature_columns'),
            'best_iteration': rank_artifacts.get('best_iteration'),
            'temperature_model_file': selected_temperature_model_file,
            'temperature_metadata_file': selected_temperature_metadata_file,
            'temperature_num_train': (
                gnn_temperature_artifacts.get('temperature_num_train')
                if temperature_selection.get('enabled') else rank_artifacts.get('temperature_num_train')
            ),
            'temperature_gnn': gnn_temperature_artifacts,
            'temperature_selection': temperature_selection,
        },
        'metrics': evaluate_scored_frame_with_manifest(
            scored,
            expected_sample_indices=load_route_cache_sample_indices(route_cache),
            score_column='xgb_score',
            temperature_column='xgb_temperature_pred',
        ),
        'stage1_route_recall': stage1_route_recall(route_cache),
    }
    _write_json(result, result_file)
    return result


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


def _format_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return 'NA'
    return f'{value:.2f}'


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
        'sys5': _mean_metric(flat['sys5'].tolist()),
        'sys10': _mean_metric(flat['sys10'].tolist()),
        'temp_mae': _mean_metric(flat['temp_mae'].dropna().tolist()),
        'temp_within_5c': _mean_metric(flat['temp_within_5c'].dropna().tolist()),
        'temp_within_10c': _mean_metric(flat['temp_within_10c'].dropna().tolist()),
        'temp_within_20c': _mean_metric(flat['temp_within_20c'].dropna().tolist()),
    }

    display_rows = flat.to_dict(orient='records') + [macro]

    md_lines = ['# Non-Oracle mainline results', '', '## Route + System', '']
    md_lines.append('| Family | rr@10 | cover | sys@1 | sys@5 | sys@10 |')
    md_lines.append('| --- | ---: | ---: | ---: | ---: | ---: |')
    for row in display_rows:
        md_lines.append(
            '| '
            + ' | '.join(
                [
                    display_family_name(str(row['family'])),
                    _format_percent(row.get('rr10')),
                    _format_percent(row.get('pool_coverage')),
                    _format_percent(row.get('sys1')),
                    _format_percent(row.get('sys5')),
                    _format_percent(row.get('sys10')),
                ]
            )
            + ' |'
        )
    md_lines.extend(['', '## Temperature', ''])
    md_lines.append('| Family | Temp MAE | Temp±5C | Temp±10C | Temp±20C |')
    md_lines.append('| --- | ---: | ---: | ---: | ---: |')
    for row in display_rows:
        md_lines.append(
            '| '
            + ' | '.join(
                [
                    display_family_name(str(row['family'])),
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
    txt_lines = ['Non-Oracle mainline results', f'{"family":<{family_width}}  {"rr@10":>5}  {"cover":>5}  {"sys@1":>5}  {"sys@5":>5}  {"sys@10":>6}']
    for row in display_rows:
        txt_lines.append(
            f'{display_family_name(str(row["family"])):<{family_width}}  '
            f'{_format_percent(row.get("rr10")):>5}  '
            f'{_format_percent(row.get("pool_coverage")):>5}  '
            f'{_format_percent(row.get("sys1")):>5}  '
            f'{_format_percent(row.get("sys5")):>5}  '
            f'{_format_percent(row.get("sys10")):>6}'
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
    parser.add_argument('--knn_top_k', type=int, default=64)
    parser.add_argument('--max_contexts', type=int, default=20)
    parser.add_argument('--prefilter_contexts', type=int, default=64)
    parser.add_argument('--max_train_routes', type=int, default=0, help='0 means use all train routes')
    parser.add_argument('--max_val_routes', type=int, default=0, help='0 means use all val routes')
    parser.add_argument(
        '--train_table_mode',
        type=str,
        default='oracle',
        choices=['oracle', 'mixed_hard_negative', 'non_oracle'],
    )
    parser.add_argument('--train_route_root', type=str, default=None)
    parser.add_argument('--val_route_root', type=str, default=None)
    parser.add_argument('--hard_negative_per_sample', type=int, default=8)
    parser.add_argument('--force_rebuild', action='store_true')
    parser.add_argument('--reafnn_device', type=str, default='cpu')
    parser.add_argument('--reafnn_force_retrain', action='store_true')
    parser.add_argument('--gnn_device', type=str, default='cpu')
    parser.add_argument('--gnn_force_retrain', action='store_true')
    parser.add_argument('--reuse_candidate_tables_root', type=str, default=None)
    parser.add_argument('--gnn_temperature_min_val_mae_improvement', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0, help='Shared random seed for ReaFNN, Reaction-GNN, and XGBoost.')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    route_root = (repo_root / args.route_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    families = parse_families_arg(args.families)
    max_train_routes = args.max_train_routes if args.max_train_routes > 0 else None
    max_val_routes = args.max_val_routes if args.max_val_routes > 0 else None
    train_route_root = (repo_root / args.train_route_root).resolve() if args.train_route_root else None
    val_route_root = (repo_root / args.val_route_root).resolve() if args.val_route_root else None
    reuse_candidate_tables_root = (repo_root / args.reuse_candidate_tables_root).resolve() if args.reuse_candidate_tables_root else None

    summary_rows: list[dict] = []
    for family in families:
        route_cache = route_root / family / 'route_cache.json'
        if not route_cache.exists():
            print(f'[mainline] skip {family}: missing {route_cache}', flush=True)
            continue
        print(f'[mainline] running {family}', flush=True)
        summary_rows.append(
            _run_family(
                repo_root=repo_root,
                family=family,
                route_cache=route_cache,
                output_root=output_root,
                top_k=args.knn_top_k,
                max_contexts=args.max_contexts,
                prefilter_contexts=args.prefilter_contexts,
                fpsize=args.fpsize,
                radius=args.radius,
                max_train_routes=max_train_routes,
                max_val_routes=max_val_routes,
                train_table_mode=args.train_table_mode,
                train_route_root=train_route_root,
                val_route_root=val_route_root,
                hard_negative_per_sample=args.hard_negative_per_sample,
                force_rebuild=args.force_rebuild,
                reaffn_device=args.reafnn_device,
                reaffn_force_retrain=args.reafnn_force_retrain,
                gnn_device=args.gnn_device,
                gnn_force_retrain=args.gnn_force_retrain,
                reuse_candidate_tables_root=reuse_candidate_tables_root,
                gnn_temperature_min_val_mae_improvement=args.gnn_temperature_min_val_mae_improvement,
                seed=args.seed,
            )
        )

    flat = _flatten_rows(summary_rows)
    if not flat.empty:
        flat = flat.sort_values('family', key=lambda s: s.map(lambda v: _family_sort_key(str(v)))).reset_index(drop=True)
    flat.to_csv(output_root / 'results_flat.csv', index=False)
    _write_json(summary_rows, output_root / 'all_results.json')
    _write_overview(flat, output_root)


if __name__ == '__main__':
    main()
