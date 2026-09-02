"""XGBoost reranking and temperature prediction for ProSys Stage 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from prosys_shared.constants import (
    CONTEXT_DENSE_COLUMNS_V2,
    PRODUCT_DESCRIPTOR_COLUMNS_V2,
    ROUTE_DENSE_COLUMNS_V2,
    ROUTE_GNN_FEATURE_COLUMNS_V2,
    SUPPORT_FEATURE_COLUMNS_V2,
)
from prosys_shared.mainline import evaluate_scored_frame

TEXT_COLUMNS = {
    'family',
    'sample_index',
    'reaction_id',
    'product',
    'reactants',
    'reagent_norm',
    'solvent_norm',
    'product_canonical',
    'route_canonical',
    'label_type',
}

TARGET_COLUMNS = {
    'label',
    'route_match',
    'context_match',
    'rank_relevance',
    'sample_weight',
    'temperature_gold',
    'yield_gold',
}

# These fields describe the bounded Stage 2 KNN/ReaFNN correction. They are
# intentionally kept out of the learned ranker: their only ranking role is the
# validation-gated Stage 2 prior below, whose fusion weight can fall back to 0.
STAGE2_CORRECTION_COLUMNS = {
    'stage2_knn_rank',
    'stage2_knn_prior',
    'stage2_reafnn_check_score',
    'stage2_reafnn_residual',
    'stage2_reafnn_correction',
    'stage2_reafnn_correction_clip',
    'stage2_initial_score',
}

STAGE2_CORRECTION_WEIGHT_GRID = np.asarray(
    (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40), dtype=np.float32
)
STAGE2_DEFAULT_CORRECTION_CLIP = 0.10

STAGE2_CONTEXT_FEATURE_COLUMNS = [
    'from_baseline_knn',
    'knn_similarity_sum',
    'knn_similarity_max',
    'knn_neighbor_count',
    'knn_weighted_mean_yield',
    'reafnn_reagent_score',
    'reafnn_solvent_score',
    'reafnn_token_score',
    'reafnn_prior_score',
    'reafnn_historical_bonus',
    'reafnn_novelty_penalty',
    'reafnn_context_score',
    'reafnn_context_count',
    'reafnn_context_support',
    'reafnn_mean_yield',
    'from_reafnn_generated',
    'from_reafnn_novel',
    'reafnn_is_historical',
    'cluster_id',
    'cluster_context_count',
    'cluster_context_support',
    'cluster_context_mean_yield',
]

# The ranker is deliberately kept in this fixed 52-column non-graph space.
# A strict allowlist prevents bookkeeping fields from silently becoming learned
# features and keeps all Stage 2 ablation arms directly comparable.
TABULAR_FEATURE_COLUMNS = (
    ROUTE_DENSE_COLUMNS_V2
    + CONTEXT_DENSE_COLUMNS_V2
    + PRODUCT_DESCRIPTOR_COLUMNS_V2
    + SUPPORT_FEATURE_COLUMNS_V2
    + STAGE2_CONTEXT_FEATURE_COLUMNS
)
TEMPERATURE_FEATURE_COLUMNS = TABULAR_FEATURE_COLUMNS + ROUTE_GNN_FEATURE_COLUMNS_V2

RANKER_MODEL_FILE_NAME = 'xgb_ranker.json'
RANKER_METADATA_FILE_NAME = 'xgb_ranker_meta.json'
TEMPERATURE_MODEL_FILE_NAME = 'xgb_temperature.json'
TEMPERATURE_METADATA_FILE_NAME = 'xgb_temperature_meta.json'

RANK_FRAME_SORT_SPECS = [
    ('sample_index', True),
    ('reaction_id', True),
    ('retro_rank', True),
    ('retro_score', False),
    ('retro_probability', False),
    ('product', True),
    ('reactants', True),
    ('reagent_norm', True),
    ('solvent_norm', True),
]

HEURISTIC_STAGE3_SORT_SPECS = [
    ('sample_index', True),
    ('retro_rank', True),
    ('retro_probability', False),
    ('stage2_initial_score', False),
    ('stage2_knn_rank', True),
    ('knn_similarity_sum', False),
    ('knn_similarity_max', False),
    ('knn_neighbor_count', False),
    ('knn_weighted_mean_yield', False),
    ('reagent_norm', True),
    ('solvent_norm', True),
]


def infer_xgb_feature_columns(
    frame: pd.DataFrame,
    *,
    include_route_gnn: bool = False,
    include_stage2_correction: bool = False,
) -> list[str]:
    """Return a fixed, audited feature allowlist for each XGBoost branch."""

    allowed = list(TEMPERATURE_FEATURE_COLUMNS if include_route_gnn else TABULAR_FEATURE_COLUMNS)
    if include_stage2_correction:
        allowed.extend(sorted(STAGE2_CORRECTION_COLUMNS))
    return [column for column in allowed if column in frame.columns]


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby('sample_index', sort=True).size().astype(int).tolist()


def _prepare_rank_frame(table_file: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(table_file)
    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, is_ascending in RANK_FRAME_SORT_SPECS:
        if column in frame.columns:
            sort_columns.append(column)
            ascending.append(is_ascending)
    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, ascending=ascending, kind='mergesort').reset_index(drop=True)


def _feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    work = frame.copy()
    for column in feature_columns:
        if column not in work.columns:
            work[column] = 0.0
    work.loc[:, feature_columns] = work.loc[:, feature_columns].fillna(0.0)
    return work


def _feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    work = _feature_frame(frame, feature_columns)
    return work.loc[:, feature_columns].to_numpy(dtype=np.float32)


def _sort_frame(
    frame: pd.DataFrame,
    sort_specs: list[tuple[str, bool]],
) -> pd.DataFrame:
    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, is_ascending in sort_specs:
        if column in frame.columns:
            sort_columns.append(column)
            ascending.append(is_ascending)
    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, ascending=ascending, kind='mergesort').reset_index(drop=True)


def _attach_stage2_heuristic_prior(frame: pd.DataFrame) -> pd.DataFrame:
    work = _sort_frame(frame, HEURISTIC_STAGE3_SORT_SPECS).copy()
    work['stage2_heuristic_rank'] = work.groupby('sample_index', sort=False).cumcount() + 1
    group_size = work.groupby('sample_index', sort=False)['stage2_heuristic_rank'].transform('max').astype(np.float32)
    denom = np.maximum(group_size.to_numpy(dtype=np.float32) - 1.0, 1.0)
    rank = work['stage2_heuristic_rank'].to_numpy(dtype=np.float32)
    work['stage2_heuristic_prior'] = 1.0 - ((rank - 1.0) / denom)
    return work


def _attach_groupwise_score_columns(frame: pd.DataFrame, raw_score_column: str) -> pd.DataFrame:
    work = frame.copy()
    group_mean = work.groupby('sample_index', sort=False)[raw_score_column].transform('mean')
    group_std = work.groupby('sample_index', sort=False)[raw_score_column].transform('std').fillna(0.0)
    denom = group_std.replace(0.0, 1.0)
    work['xgb_score_z'] = ((work[raw_score_column] - group_mean) / denom).astype(np.float32)
    return work


def _score_fusion_grid() -> np.ndarray:
    return np.linspace(0.0, 2.0, 41, dtype=np.float32)


def _has_stage2_correction_fields(frame: pd.DataFrame) -> bool:
    return {
        'stage2_knn_prior',
        'stage2_reafnn_residual',
    }.issubset(frame.columns)


def _has_fixed_stage2_post_fusion(frame: pd.DataFrame) -> bool:
    """Whether Stage 2 already chose a validation-calibrated fusion score."""

    if 'stage2_post_fusion_enabled' not in frame.columns:
        return False
    marker = pd.to_numeric(frame['stage2_post_fusion_enabled'], errors='coerce').fillna(0.0)
    return bool((marker > 0.5).any())


def _apply_stage2_correction_weight(
    frame: pd.DataFrame,
    correction_weight: float | None,
) -> pd.DataFrame:
    """Recompute the bounded KNN/ReaFNN initial score without changing rows."""

    work = frame.copy()
    if correction_weight is None or not _has_stage2_correction_fields(work):
        return work

    knn_prior = pd.to_numeric(work['stage2_knn_prior'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    residual = pd.to_numeric(work['stage2_reafnn_residual'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    if 'stage2_reafnn_correction_clip' in work.columns:
        clip = pd.to_numeric(work['stage2_reafnn_correction_clip'], errors='coerce').to_numpy(dtype=np.float32)
        clip = np.where(np.isfinite(clip) & (clip > 0.0), clip, STAGE2_DEFAULT_CORRECTION_CLIP)
    else:
        clip = np.full(residual.shape, STAGE2_DEFAULT_CORRECTION_CLIP, dtype=np.float32)
    correction = np.clip(float(correction_weight) * residual, -clip, clip).astype(np.float32)
    work['stage2_reafnn_correction'] = correction
    work['stage2_initial_score'] = (knn_prior + correction).astype(np.float32)
    return work


def _select_stage2_correction_calibration(val_frame: pd.DataFrame) -> dict:
    """Select a conservative correction strength using validation Sys@10 only."""

    if val_frame.empty or 'label' not in val_frame.columns or 'xgb_score_raw' not in val_frame.columns:
        return {
            'enabled': False,
            'selected_weight': None,
            'selection_metric': 'system_top10_all',
            'tie_break_metric': 'system_top1_all',
            'reason': 'missing_validation_scores',
        }
    if _has_fixed_stage2_post_fusion(val_frame):
        return {
            'enabled': False,
            'selected_weight': None,
            'selection_metric': 'system_top10_all',
            'tie_break_metric': 'system_top1_all',
            'reason': 'fixed_validation_selected_stage2_post_fusion',
        }
    if not _has_stage2_correction_fields(val_frame):
        return {
            'enabled': False,
            'selected_weight': None,
            'selection_metric': 'system_top10_all',
            'tie_break_metric': 'system_top1_all',
            'reason': 'missing_stage2_correction_fields',
        }

    best_weight = 0.0
    best_sys10 = float('-inf')
    best_sys1 = float('-inf')
    best_fusion: dict | None = None
    candidates: list[dict] = []
    for weight in STAGE2_CORRECTION_WEIGHT_GRID:
        work = _apply_stage2_correction_weight(val_frame, float(weight))
        work = _attach_stage2_heuristic_prior(work)
        work = _attach_groupwise_score_columns(work, 'xgb_score_raw')
        fusion = _select_stage3_fusion_weight(work)
        sys10 = float(fusion.get('val_system_top10_all', 0.0))
        sys1 = float(fusion.get('val_system_top1_all', 0.0))
        candidates.append({
            'weight': float(weight),
            'val_system_top10_all': sys10,
            'val_system_top1_all': sys1,
            'fusion_weight': float(fusion.get('heuristic_weight', 0.0)),
        })
        if sys10 > best_sys10 + 1e-12 or (abs(sys10 - best_sys10) <= 1e-12 and sys1 > best_sys1):
            best_weight = float(weight)
            best_sys10 = sys10
            best_sys1 = sys1
            best_fusion = fusion

    return {
        'enabled': True,
        'selected_weight': best_weight,
        'selection_metric': 'system_top10_all',
        'tie_break_metric': 'system_top1_all',
        'val_system_top10_all': best_sys10,
        'val_system_top1_all': best_sys1,
        'candidate_weights': candidates,
        'score_fusion': best_fusion,
    }


def _select_stage3_fusion_weight(val_frame: pd.DataFrame) -> dict:
    if val_frame.empty or 'label' not in val_frame.columns:
        return {
            'enabled': False,
            'heuristic_weight': 0.0,
            'selection_metric': 'system_top10_all',
            'tie_break_metric': 'system_top1_all',
        }

    best_weight = 0.0
    best_sys10 = float('-inf')
    best_sys1 = float('-inf')
    for weight in _score_fusion_grid():
        work = val_frame.copy()
        work['stage3_score_fused'] = (
            work['xgb_score_z'].to_numpy(dtype=np.float32)
            + (float(weight) * work['stage2_heuristic_prior'].to_numpy(dtype=np.float32))
        )
        metrics = evaluate_scored_frame(work, score_column='stage3_score_fused')
        sys10 = float(metrics.get('system_top10_all', 0.0))
        sys1 = float(metrics.get('system_top1_all', 0.0))
        if sys10 > best_sys10 + 1e-12 or (abs(sys10 - best_sys10) <= 1e-12 and sys1 > best_sys1):
            best_weight = float(weight)
            best_sys10 = sys10
            best_sys1 = sys1

    return {
        'enabled': True,
        'heuristic_weight': best_weight,
        'selection_metric': 'system_top10_all',
        'tie_break_metric': 'system_top1_all',
        'val_system_top10_all': best_sys10,
        'val_system_top1_all': best_sys1,
        'score_definition': 'xgb_score_z + heuristic_weight * stage2_heuristic_prior',
    }


def _positive_temperature_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if 'temperature_gold' not in frame.columns or 'label' not in frame.columns:
        return frame.iloc[0:0].copy()
    mask = (frame['label'].to_numpy(dtype=np.float32) > 0.5) & np.isfinite(
        frame['temperature_gold'].to_numpy(dtype=np.float32)
    )
    return frame.loc[mask].copy()


def _temperature_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / TEMPERATURE_MODEL_FILE_NAME, output_dir / TEMPERATURE_METADATA_FILE_NAME


def _train_temperature_regressor(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    *,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    n_jobs: int,
) -> tuple[str | None, str, int]:
    train_temp = _positive_temperature_rows(train_frame)
    val_temp = _positive_temperature_rows(val_frame)
    model_file, metadata_file = _temperature_paths(output_dir)

    metadata = {
        'trained': False,
        'feature_columns': feature_columns,
        'train_rows': int(len(train_temp)),
        'val_rows': int(len(val_temp)),
        'params': {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
        },
    }

    if train_temp.empty:
        metadata['reason'] = 'no_positive_temperature_rows'
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        return None, str(metadata_file), 0

    x_train = _feature_matrix(train_temp, feature_columns)
    y_train = train_temp['temperature_gold'].to_numpy(dtype=np.float32)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='mae',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        tree_method='hist',
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if val_temp.empty:
        model.fit(x_train, y_train)
    else:
        x_val = _feature_matrix(val_temp, feature_columns)
        y_val = val_temp['temperature_gold'].to_numpy(dtype=np.float32)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    model.save_model(model_file)
    metadata['trained'] = True
    metadata['model_file'] = model_file.name
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return str(model_file), str(metadata_file), int(len(train_temp))


def train_xgb_temperature_regressor(
    train_table_file: str | Path,
    val_table_file: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 0,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    n_jobs: int = 1,
) -> dict:
    """Train only the Stage 3 temperature regressor on a candidate table.

    This supports architectures where ranking and temperature intentionally use
    different feature sets, while retaining the same train/validation protocol
    as :func:`train_xgb_ranker_and_temperature`.
    """
    train_frame = _prepare_rank_frame(train_table_file)
    val_frame = _prepare_rank_frame(val_table_file)
    feature_columns = infer_xgb_feature_columns(
        train_frame,
        include_route_gnn=True,
        include_stage2_correction=False,
    )
    if not feature_columns:
        raise ValueError(f'No feature columns inferred from {train_table_file}')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_file, metadata_file, num_train = _train_temperature_regressor(
        train_frame=train_frame,
        val_frame=val_frame,
        feature_columns=feature_columns,
        output_dir=output_path,
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,

        n_jobs=n_jobs,
    )
    return {
        'output_dir': str(output_path),
        'model_file': model_file,
        'metadata_file': metadata_file,
        'feature_columns': feature_columns,
        'temperature_num_train': num_train,
    }

def train_xgb_ranker_and_temperature(
    train_table_file: str | Path,
    val_table_file: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 0,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    temperature_n_estimators: int | None = None,
    temperature_learning_rate: float | None = None,
    temperature_max_depth: int | None = None,
    train_temperature: bool = True,
    n_jobs: int = 1,
) -> dict:
    train_frame = _prepare_rank_frame(train_table_file)
    val_frame = _prepare_rank_frame(val_table_file)
    # Route-GNN vectors are temperature-only. The ReaFNN correction is instead
    # used through a validation-gated initial-score prior, preserving a stable
    # learned ranking feature space.
    feature_columns = infer_xgb_feature_columns(
        train_frame,
        include_route_gnn=False,
        include_stage2_correction=False,
    )
    if not feature_columns:
        raise ValueError(f'No ranker feature columns inferred from {train_table_file}')

    x_train = _feature_matrix(train_frame, feature_columns)
    y_train = train_frame['rank_relevance'].to_numpy(dtype=np.float32)
    g_train = _group_sizes(train_frame)

    x_val = _feature_matrix(val_frame, feature_columns)
    y_val = val_frame['rank_relevance'].to_numpy(dtype=np.float32)
    g_val = _group_sizes(val_frame)

    model = xgb.XGBRanker(
        objective='rank:ndcg',
        eval_metric='ndcg@10',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        tree_method='hist',
        random_state=random_state,
        early_stopping_rounds=30,
        n_jobs=n_jobs,
    )
    model.fit(
        x_train,
        y_train,
        group=g_train,
        eval_set=[(x_val, y_val)],
        eval_group=[g_val],
        verbose=False,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / RANKER_MODEL_FILE_NAME
    metadata_file = output_path / RANKER_METADATA_FILE_NAME
    model.save_model(model_file)

    val_scored = val_frame.copy()
    val_scored['xgb_score_raw'] = model.predict(x_val).astype(np.float32)
    stage2_correction_calibration = _select_stage2_correction_calibration(val_scored)
    selected_correction_weight = stage2_correction_calibration.get('selected_weight')
    val_scored = _apply_stage2_correction_weight(val_scored, selected_correction_weight)
    val_scored = _attach_stage2_heuristic_prior(val_scored)
    val_scored = _attach_groupwise_score_columns(val_scored, 'xgb_score_raw')
    score_fusion = stage2_correction_calibration.get('score_fusion')
    if not isinstance(score_fusion, dict):
        score_fusion = _select_stage3_fusion_weight(val_scored)

    metadata = {
        'feature_columns': feature_columns,
        'ranker_feature_space': 'tabular_non_graph',
        'stage2_correction_policy': 'validation_gated_heuristic_prior_only',
        'stage2_correction_calibration': stage2_correction_calibration,
        'best_iteration': int(model.best_iteration) if model.best_iteration is not None else None,
        'score_fusion': score_fusion,
        'params': {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@10',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    if train_temperature:
        temperature_feature_columns = infer_xgb_feature_columns(
            train_frame,
            include_route_gnn=True,
            include_stage2_correction=False,
        )
        temperature_model_file, temperature_metadata_file, temperature_num_train = _train_temperature_regressor(
            train_frame=train_frame,
            val_frame=val_frame,
            feature_columns=temperature_feature_columns,
            output_dir=output_path,
            random_state=random_state,
            n_estimators=temperature_n_estimators or n_estimators,
            learning_rate=temperature_learning_rate or learning_rate,
            max_depth=temperature_max_depth or max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
        )
    else:
        temperature_model_file = None
        temperature_metadata_file = None
        temperature_num_train = 0

    return {
        'output_dir': str(output_path),
        'model_file': str(model_file),
        'metadata_file': str(metadata_file),
        'feature_columns': feature_columns,
        'best_iteration': metadata['best_iteration'],
        'temperature_model_file': temperature_model_file,
        'temperature_metadata_file': temperature_metadata_file,
        'temperature_num_train': temperature_num_train,
    }


def load_xgb_ranker(model_file: str | Path) -> xgb.XGBRanker:
    model = xgb.XGBRanker()
    model.load_model(model_file)
    return model


def load_xgb_regressor(model_file: str | Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    return model


def _resolve_temperature_artifacts(
    model_file: str | Path,
    metadata_file: str | Path,
    temperature_model_file: str | Path | None,
    temperature_metadata_file: str | Path | None,
) -> tuple[Path | None, dict | None]:
    rank_model_path = Path(model_file)
    rank_meta_path = Path(metadata_file)

    if temperature_metadata_file is None:
        temperature_metadata_file = rank_meta_path.with_name(TEMPERATURE_METADATA_FILE_NAME)
    temp_meta_path = Path(temperature_metadata_file)
    if not temp_meta_path.exists():
        return None, None

    temp_meta = json.loads(temp_meta_path.read_text(encoding='utf-8'))
    if not temp_meta.get('trained'):
        return None, temp_meta

    if temperature_model_file is None:
        model_name = temp_meta.get('model_file') or TEMPERATURE_MODEL_FILE_NAME
        temperature_model_file = rank_model_path.with_name(model_name)

    temp_model_path = Path(temperature_model_file)
    if not temp_model_path.exists():
        return None, temp_meta
    return temp_model_path, temp_meta


def score_table_with_xgb(
    table_file: str | Path,
    model_file: str | Path,
    metadata_file: str | Path,
    *,
    temperature_model_file: str | Path | None = None,
    temperature_metadata_file: str | Path | None = None,
) -> pd.DataFrame:
    frame = _prepare_rank_frame(table_file)
    metadata = json.loads(Path(metadata_file).read_text(encoding='utf-8'))
    feature_columns = list(metadata['feature_columns'])

    work = _feature_frame(frame, feature_columns)
    ranker = load_xgb_ranker(model_file)
    work['xgb_score_raw'] = ranker.predict(work.loc[:, feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)
    correction_calibration = metadata.get('stage2_correction_calibration') or {}
    if bool(correction_calibration.get('enabled')):
        work = _apply_stage2_correction_weight(
            work,
            correction_calibration.get('selected_weight'),
        )
    work = _attach_stage2_heuristic_prior(work)
    work = _attach_groupwise_score_columns(work, 'xgb_score_raw')

    score_fusion = metadata.get('score_fusion') or {}
    if bool(score_fusion.get('enabled')):
        weight = float(score_fusion.get('heuristic_weight', 0.0))
        work['xgb_score'] = (
            work['xgb_score_z'].to_numpy(dtype=np.float32)
            + (weight * work['stage2_heuristic_prior'].to_numpy(dtype=np.float32))
        ).astype(np.float32)
    else:
        work['xgb_score'] = work['xgb_score_raw'].to_numpy(dtype=np.float32)

    temp_model_path, temp_meta = _resolve_temperature_artifacts(
        model_file=model_file,
        metadata_file=metadata_file,
        temperature_model_file=temperature_model_file,
        temperature_metadata_file=temperature_metadata_file,
    )
    if temp_model_path is not None and temp_meta is not None:
        temp_feature_columns = list(temp_meta.get('feature_columns') or feature_columns)
        temp_work = _feature_frame(work, temp_feature_columns)
        regressor = load_xgb_regressor(temp_model_path)
        temp_work['xgb_temperature_pred'] = regressor.predict(
            temp_work.loc[:, temp_feature_columns].to_numpy(dtype=np.float32)
        )
        work['xgb_temperature_pred'] = temp_work['xgb_temperature_pred'].to_numpy(dtype=np.float32)

    return work


def main() -> None:
    parser = argparse.ArgumentParser(description='Train or score ProSys Stage 3 XGBoost models.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train ranker and temperature regressor')
    train_parser.add_argument('--train_table', type=str, required=True)
    train_parser.add_argument('--val_table', type=str, required=True)
    train_parser.add_argument('--output_dir', type=str, required=True)
    train_parser.add_argument('--random_state', type=int, default=0)
    train_parser.add_argument('--n_estimators', type=int, default=300)
    train_parser.add_argument('--learning_rate', type=float, default=0.05)
    train_parser.add_argument('--max_depth', type=int, default=6)
    train_parser.add_argument('--subsample', type=float, default=0.8)
    train_parser.add_argument('--colsample_bytree', type=float, default=0.8)
    train_parser.add_argument('--reg_lambda', type=float, default=1.0)
    score_parser = subparsers.add_parser('score', help='Score a candidate table with trained models')
    score_parser.add_argument('--table_file', type=str, required=True)
    score_parser.add_argument('--model_file', type=str, required=True)
    score_parser.add_argument('--metadata_file', type=str, required=True)
    score_parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()

    if args.command == 'train':
        result = train_xgb_ranker_and_temperature(
            train_table_file=Path(args.train_table).resolve(),
            val_table_file=Path(args.val_table).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    scored = score_table_with_xgb(
        table_file=Path(args.table_file).resolve(),
        model_file=Path(args.model_file).resolve(),
        metadata_file=Path(args.metadata_file).resolve(),
    )
    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_file, index=False)
    print(str(output_file))


if __name__ == '__main__':
    main()
