"""Shared sklearn-based ranker and temperature helpers for Non-Oracle experiments."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

from stage3_XGBoost import infer_xgb_feature_columns

RANKER_MODEL_FILE_NAME = 'tabular_ranker.pkl'
RANKER_METADATA_FILE_NAME = 'tabular_ranker_meta.json'
TEMPERATURE_MODEL_FILE_NAME = 'tabular_temperature.pkl'
TEMPERATURE_METADATA_FILE_NAME = 'tabular_temperature_meta.json'

FRAME_SORT_SPECS = [
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


def _prepare_frame(table_file: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(table_file)
    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, is_ascending in FRAME_SORT_SPECS:
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


def _positive_temperature_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if 'temperature_gold' not in frame.columns or 'label' not in frame.columns:
        return frame.iloc[0:0].copy()
    mask = (frame['label'].to_numpy(dtype=np.float32) > 0.5) & np.isfinite(
        frame['temperature_gold'].to_numpy(dtype=np.float32)
    )
    return frame.loc[mask].copy()


def _build_ranker(kind: str, random_state: int):
    if kind == 'rf':
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=random_state,
        )
    if kind == 'svm':
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('model', LinearSVC(C=1.0, class_weight='balanced', max_iter=10000, random_state=random_state)),
            ]
        )
    if kind == 'bayes':
        return GaussianNB()
    raise ValueError(f'Unsupported ranker kind: {kind}')


def _build_temperature_model(kind: str, random_state: int):
    if kind == 'rf':
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
    if kind == 'svm':
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('model', LinearSVR(C=1.0, epsilon=0.0, max_iter=10000, random_state=random_state)),
            ]
        )
    if kind == 'bayes':
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('model', BayesianRidge()),
            ]
        )
    raise ValueError(f'Unsupported temperature model kind: {kind}')


def _rank_scores(model, x: np.ndarray, *, kind: str) -> np.ndarray:
    if kind in {'rf', 'bayes'}:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1], dtype=np.float32)
    if kind == 'svm':
        if hasattr(model, 'decision_function'):
            return np.asarray(model.decision_function(x), dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def train_tabular_ranker_and_temperature(
    train_table_file: str | Path,
    val_table_file: str | Path,
    output_dir: str | Path,
    *,
    kind: str,
    random_state: int = 0,
) -> dict:
    train_frame = _prepare_frame(train_table_file)
    val_frame = _prepare_frame(val_table_file)
    feature_columns = infer_xgb_feature_columns(train_frame)
    if not feature_columns:
        raise ValueError(f'No feature columns inferred from {train_table_file}')

    x_train = _feature_matrix(train_frame, feature_columns)
    y_train = (train_frame['label'].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
    if len(np.unique(y_train)) < 2:
        raise ValueError(f'Need at least two classes to train ranker: {train_table_file}')

    ranker = _build_ranker(kind, random_state=random_state)
    ranker.fit(x_train, y_train)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / RANKER_MODEL_FILE_NAME
    metadata_file = output_path / RANKER_METADATA_FILE_NAME
    with open(model_file, 'wb') as handle:
        pickle.dump(ranker, handle)
    metadata = {
        'kind': kind,
        'feature_columns': feature_columns,
        'train_rows': int(len(train_frame)),
        'val_rows': int(len(val_frame)),
        'target': 'label',
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    train_temp = _positive_temperature_rows(train_frame)
    val_temp = _positive_temperature_rows(val_frame)
    temp_model_file = output_path / TEMPERATURE_MODEL_FILE_NAME
    temp_metadata_file = output_path / TEMPERATURE_METADATA_FILE_NAME
    temp_metadata = {
        'kind': kind,
        'feature_columns': feature_columns,
        'train_rows': int(len(train_temp)),
        'val_rows': int(len(val_temp)),
        'trained': False,
        'target_min': None,
        'target_max': None,
    }
    if not train_temp.empty:
        y_temp = train_temp['temperature_gold'].to_numpy(dtype=np.float32)
        temp_model = _build_temperature_model(kind, random_state=random_state)
        temp_model.fit(
            _feature_matrix(train_temp, feature_columns),
            y_temp,
        )
        with open(temp_model_file, 'wb') as handle:
            pickle.dump(temp_model, handle)
        temp_metadata['trained'] = True
        temp_metadata['model_file'] = temp_model_file.name
        temp_metadata['target_min'] = float(np.min(y_temp))
        temp_metadata['target_max'] = float(np.max(y_temp))
    else:
        temp_metadata['reason'] = 'no_positive_temperature_rows'

    temp_metadata_file.write_text(json.dumps(temp_metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {
        'kind': kind,
        'output_dir': str(output_path),
        'model_file': str(model_file),
        'metadata_file': str(metadata_file),
        'feature_columns': feature_columns,
        'temperature_model_file': (str(temp_model_file) if temp_metadata['trained'] else None),
        'temperature_metadata_file': str(temp_metadata_file),
        'temperature_num_train': int(len(train_temp)),
    }


def score_table_with_tabular_model(
    table_file: str | Path,
    model_file: str | Path,
    metadata_file: str | Path,
    *,
    score_column: str = 'model_score',
    temperature_column: str = 'model_temperature_pred',
) -> pd.DataFrame:
    frame = _prepare_frame(table_file)
    metadata = json.loads(Path(metadata_file).read_text(encoding='utf-8'))
    feature_columns = list(metadata['feature_columns'])
    kind = str(metadata['kind'])
    x = _feature_matrix(frame, feature_columns)

    with open(model_file, 'rb') as handle:
        ranker = pickle.load(handle)
    frame[score_column] = _rank_scores(ranker, x, kind=kind)

    temp_meta_file = Path(metadata_file).with_name(TEMPERATURE_METADATA_FILE_NAME)
    if temp_meta_file.exists():
        temp_meta = json.loads(temp_meta_file.read_text(encoding='utf-8'))
        if temp_meta.get('trained'):
            temp_model_file = Path(metadata_file).with_name(temp_meta.get('model_file', TEMPERATURE_MODEL_FILE_NAME))
            with open(temp_model_file, 'rb') as handle:
                temp_model = pickle.load(handle)
            temp_feature_columns = list(temp_meta.get('feature_columns') or feature_columns)
            preds = temp_model.predict(_feature_matrix(frame, temp_feature_columns)).astype(np.float32)
            target_min = temp_meta.get('target_min')
            target_max = temp_meta.get('target_max')
            if target_min is not None and target_max is not None:
                preds = np.clip(preds, float(target_min), float(target_max))
            frame[temperature_column] = preds

    return frame
