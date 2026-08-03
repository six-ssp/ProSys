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

STANDARD_FEATURE_COLUMNS = (
    ROUTE_DENSE_COLUMNS_V2
    + CONTEXT_DENSE_COLUMNS_V2
    + PRODUCT_DESCRIPTOR_COLUMNS_V2
    + ROUTE_GNN_FEATURE_COLUMNS_V2
    + SUPPORT_FEATURE_COLUMNS_V2
)

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
    ('knn_similarity_sum', False),
    ('knn_similarity_max', False),
    ('knn_neighbor_count', False),
    ('knn_weighted_mean_yield', False),
    ('reagent_norm', True),
    ('solvent_norm', True),
]


def infer_xgb_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in STANDARD_FEATURE_COLUMNS if column in frame.columns]
    used = set(columns)
    for column in frame.columns:
        if column in used or column in TEXT_COLUMNS or column in TARGET_COLUMNS:
            continue
        if column.startswith('legacy_'):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
            used.add(column)
    return columns


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
    feature_columns = infer_xgb_feature_columns(train_frame)
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
    n_jobs: int = 1,
) -> dict:
    train_frame = _prepare_rank_frame(train_table_file)
    val_frame = _prepare_rank_frame(val_table_file)
    feature_columns = infer_xgb_feature_columns(train_frame)
    if not feature_columns:
        raise ValueError(f'No feature columns inferred from {train_table_file}')

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
    val_scored = _attach_stage2_heuristic_prior(val_scored)
    val_scored = _attach_groupwise_score_columns(val_scored, 'xgb_score_raw')
    score_fusion = _select_stage3_fusion_weight(val_scored)

    metadata = {
        'feature_columns': feature_columns,
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

    temperature_model_file, temperature_metadata_file, temperature_num_train = _train_temperature_regressor(
        train_frame=train_frame,
        val_frame=val_frame,
        feature_columns=feature_columns,
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
