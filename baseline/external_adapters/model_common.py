"""Shared model-side helpers for the Sequential FNN and Reaction-GCNN baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from prosys_shared.features import reaction_morgan_fp

from .contracts import read_jsonl, write_jsonl


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def load_vocabulary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def token_id_index(vocabulary: dict[str, Any], kind: str) -> tuple[list[str], dict[str, int]]:
    identifiers = sorted(vocabulary[kind]['id_to_label'])
    return identifiers, {identifier: index for index, identifier in enumerate(identifiers)}


def multi_hot(rows: Iterable[dict[str, Any]], field: str, index: dict[str, int]) -> np.ndarray:
    rows = list(rows)
    matrix = np.zeros((len(rows), len(index)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for identifier in row.get(field, []):
            token_index = index.get(str(identifier))
            if token_index is not None:
                matrix[row_index, token_index] = 1.0
    return matrix


def route_feature_matrix(rows: Iterable[dict[str, Any]], *, fp_size: int = 4096, radius: int = 2) -> np.ndarray:
    vectors = [
        reaction_morgan_fp(str(row['reactants']), str(row['product']), fpsize=fp_size, radius=radius)
        for row in rows
    ]
    return np.asarray(vectors, dtype=np.float32)


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> 'FeatureStandardizer':
        mean = values.mean(axis=0).astype(np.float32)
        scale = values.std(axis=0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, list[float]]:
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'FeatureStandardizer':
        return cls(
            mean=np.asarray(payload['mean'], dtype=np.float32),
            scale=np.asarray(payload['scale'], dtype=np.float32),
        )


@dataclass
class ContextLibrary:
    contexts: list[dict[str, Any]]
    reagent_matrix: np.ndarray
    solvent_matrix: np.ndarray
    prior: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path, vocabulary: dict[str, Any]) -> 'ContextLibrary':
        contexts = load_jsonl_rows(path)
        _reagent_ids, reagent_index = token_id_index(vocabulary, 'reagent')
        _solvent_ids, solvent_index = token_id_index(vocabulary, 'solvent')
        reagent_matrix = multi_hot(contexts, 'reagent_ids', reagent_index)
        solvent_matrix = multi_hot(contexts, 'solvent_ids', solvent_index)
        counts = np.asarray([float(context.get('count', 1.0)) for context in contexts], dtype=np.float32)
        prior = np.log((counts + 1.0) / (counts.sum() + len(counts))) if len(counts) else counts
        return cls(contexts=contexts, reagent_matrix=reagent_matrix, solvent_matrix=solvent_matrix, prior=prior)

    def score(
        self,
        reagent_probabilities: np.ndarray,
        solvent_probabilities: np.ndarray,
        *,
        prior_weight: float = 0.05,
    ) -> np.ndarray:
        """Score full historical contexts with independent Bernoulli likelihoods."""

        reagent_probabilities = np.clip(np.asarray(reagent_probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        solvent_probabilities = np.clip(np.asarray(solvent_probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        reagent_score = self.reagent_matrix @ np.log(reagent_probabilities) + (1.0 - self.reagent_matrix) @ np.log1p(-reagent_probabilities)
        solvent_score = self.solvent_matrix @ np.log(solvent_probabilities) + (1.0 - self.solvent_matrix) @ np.log1p(-solvent_probabilities)
        return (reagent_score + solvent_score + prior_weight * self.prior).astype(np.float64, copy=False)

    def top_candidates(
        self,
        reagent_probabilities: np.ndarray,
        solvent_probabilities: np.ndarray,
        *,
        limit: int,
        temperature_pred: float | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.score(reagent_probabilities, solvent_probabilities)
        order = np.argsort(-scores, kind='mergesort')[:limit]
        rows = []
        for index in order:
            context = self.contexts[int(index)]
            rows.append(
                {
                    'reagent_ids': context['reagent_ids'],
                    'solvent_ids': context['solvent_ids'],
                    'condition_score': float(scores[int(index)]),
                    'temperature_pred': temperature_pred,
                }
            )
        return rows


def positive_class_weight(targets: np.ndarray) -> np.ndarray:
    positives = np.maximum(targets.sum(axis=0, dtype=np.float64), 1.0)
    negatives = np.maximum(float(targets.shape[0]) - positives, 1.0)
    return np.clip(negatives / positives, 1.0, 20.0).astype(np.float32)


def temperature_targets(rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    mask = []
    for row in rows:
        value = row.get('temperature_mean')
        if value is None:
            values.append(0.0)
            mask.append(False)
        else:
            values.append(float(value))
            mask.append(np.isfinite(float(value)))
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=bool)


def write_prediction_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    return write_jsonl(path, rows)


def config_dict(config: Any) -> dict[str, Any]:
    return asdict(config)
