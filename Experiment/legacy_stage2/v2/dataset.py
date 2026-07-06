"""Dataset and dataloader helpers for ProSys Stage 2 V2."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from stage2.train_multilabel.data_utils.data import get_classes

from .constants import (
    CONTEXT_DENSE_COLUMNS_V2,
    PRODUCT_DESCRIPTOR_COLUMNS_V2,
    ROUTE_DENSE_COLUMNS_V2,
    SUPPORT_FEATURE_COLUMNS_V2,
)
from .features import product_morgan_fp, reaction_graph_descriptors, reaction_morgan_fp
from .product_memory import normalize_condition_labels


@dataclass(frozen=True)
class Stage2CandidateDatapointV2:
    family: str
    sample_index: int
    reaction_id: str
    reactants: str
    product: str
    reagent_norm: str
    solvent_norm: str
    label: float
    rank_relevance: float
    sample_weight: float
    temperature_gold: float


def _multi_hot(labels: str, class_names: Sequence[tuple[str, int]]) -> np.ndarray:
    mapping = {name: idx for idx, (name, _) in enumerate(class_names)}
    vector = np.zeros((len(class_names),), dtype=np.float32)
    for token in normalize_condition_labels(labels).split('; '):
        if token and token in mapping:
            vector[mapping[token]] = 1.0
    return vector


class Stage2CandidateDatasetV2(Dataset):
    def __init__(
        self,
        candidate_table_file: str | Path,
        family_dir: str | Path,
        *,
        route_fpsize: int = 4096,
        product_fpsize: int = 2048,
        radius: int = 2,
        max_slates: int | None = None,
    ):
        self.candidate_table_file = Path(candidate_table_file)
        self.family_dir = Path(family_dir)
        self.route_fpsize = route_fpsize
        self.product_fpsize = product_fpsize
        self.radius = radius

        frame = pd.read_csv(self.candidate_table_file)
        if frame.empty:
            raise ValueError(f'Empty candidate table: {self.candidate_table_file}')

        if max_slates is not None:
            keep_ids = list(dict.fromkeys(frame['sample_index'].tolist()))[:max_slates]
            frame = frame[frame['sample_index'].isin(keep_ids)].copy()

        self.frame = frame.reset_index(drop=True)
        self.datapoints = [
            Stage2CandidateDatapointV2(
                family=str(row.family),
                sample_index=int(row.sample_index),
                reaction_id=str(row.reaction_id),
                reactants=str(row.reactants),
                product=str(row.product),
                reagent_norm=str(row.reagent_norm),
                solvent_norm=str(row.solvent_norm),
                label=float(row.label),
                rank_relevance=float(row.rank_relevance),
                sample_weight=float(row.sample_weight),
                temperature_gold=float(row.temperature_gold),
            )
            for row in self.frame.itertuples(index=False)
        ]

        self.solvent_classes = get_classes(self.family_dir / 'label_processed' / 'class_names_solvent_labels_processed.pkl')
        self.reagent_classes = get_classes(self.family_dir / 'label_processed' / 'class_names_reagent_labels_processed.pkl')

        self.route_feature_cache = self._build_route_feature_cache()
        self.product_feature_cache = self._build_product_feature_cache()
        self.context_feature_cache = self._build_context_feature_cache()

        self.group_indices = [list(indices) for _, indices in self.frame.groupby('sample_index').indices.items()]

    def _build_route_feature_cache(self) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        unique_routes = self.frame[['reactants', 'product']].drop_duplicates()
        for row in unique_routes.itertuples(index=False):
            key = (str(row.reactants), str(row.product))
            cache[key] = {
                'route_fp': reaction_morgan_fp(
                    reactants=key[0],
                    product=key[1],
                    fpsize=self.route_fpsize,
                    radius=self.radius,
                ).astype(np.float32),
                'route_graph_features': reaction_graph_descriptors(key[0], key[1]).astype(np.float32),
            }
        return cache

    def _build_product_feature_cache(self) -> dict[str, dict[str, np.ndarray]]:
        cache: dict[str, dict[str, np.ndarray]] = {}
        unique_products = self.frame[['product'] + PRODUCT_DESCRIPTOR_COLUMNS_V2].drop_duplicates(subset=['product'])
        for row in unique_products.itertuples(index=False):
            product = str(row.product)
            cache[product] = {
                'product_fp': product_morgan_fp(product, n_bits=self.product_fpsize, radius=self.radius).astype(np.float32),
                'product_features': np.asarray(
                    [getattr(row, column) for column in PRODUCT_DESCRIPTOR_COLUMNS_V2],
                    dtype=np.float32,
                ),
            }
        return cache

    def _build_context_feature_cache(self) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        unique_contexts = self.frame[['reagent_norm', 'solvent_norm']].drop_duplicates()
        for row in unique_contexts.itertuples(index=False):
            key = (str(row.reagent_norm), str(row.solvent_norm))
            cache[key] = {
                'reagent_features': _multi_hot(key[0], self.reagent_classes),
                'solvent_features': _multi_hot(key[1], self.solvent_classes),
            }
        return cache

    @property
    def family(self) -> str:
        return str(self.frame['family'].iloc[0])

    @property
    def reagent_dim(self) -> int:
        return len(self.reagent_classes)

    @property
    def solvent_dim(self) -> int:
        return len(self.solvent_classes)

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, index: int) -> dict:
        datapoint = self.datapoints[index]
        row = self.frame.iloc[index]
        route_features = self.route_feature_cache[(datapoint.reactants, datapoint.product)]
        product_features = self.product_feature_cache[datapoint.product]
        context_features = self.context_feature_cache[(datapoint.reagent_norm, datapoint.solvent_norm)]

        return {
            'family': datapoint.family,
            'sample_index': datapoint.sample_index,
            'reaction_id': datapoint.reaction_id,
            'route_fp': route_features['route_fp'],
            'route_graph_features': route_features['route_graph_features'],
            'route_dense_features': np.asarray(
                [row[column] for column in ROUTE_DENSE_COLUMNS_V2],
                dtype=np.float32,
            ),
            'context_dense_features': np.asarray(
                [row[column] for column in CONTEXT_DENSE_COLUMNS_V2],
                dtype=np.float32,
            ),
            'product_fp': product_features['product_fp'],
            'product_features': product_features['product_features'],
            'reagent_features': context_features['reagent_features'],
            'solvent_features': context_features['solvent_features'],
            'support_features': np.asarray(
                [row[column] for column in SUPPORT_FEATURE_COLUMNS_V2],
                dtype=np.float32,
            ),
            'label': np.float32(datapoint.label),
            'rank_relevance': np.float32(datapoint.rank_relevance),
            'sample_weight': np.float32(datapoint.sample_weight),
            'route_match': np.float32(row['route_match']),
            'context_match': np.float32(row['context_match']),
            'temperature_gold': np.float32(datapoint.temperature_gold),
        }


def collate_stage2_candidates_v2(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    tensor_keys = [
        'sample_index',
        'route_fp',
        'route_graph_features',
        'route_dense_features',
        'context_dense_features',
        'product_fp',
        'product_features',
        'reagent_features',
        'solvent_features',
        'support_features',
        'label',
        'rank_relevance',
        'sample_weight',
        'route_match',
        'context_match',
        'temperature_gold',
    ]
    collated: dict[str, torch.Tensor | list[str]] = {
        'family': [item['family'] for item in batch],
        'reaction_id': [item['reaction_id'] for item in batch],
    }
    for key in tensor_keys:
        values = [item[key] for item in batch]
        collated[key] = torch.as_tensor(np.stack(values), dtype=torch.float32)

    collated['sample_index'] = collated['sample_index'].to(torch.long)
    return collated


class SlateBatchSampler(Sampler[list[int]]):
    def __init__(self, groups: list[list[int]], slates_per_batch: int, shuffle: bool):
        self.groups = groups
        self.slates_per_batch = slates_per_batch
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.groups)))
        if self.shuffle:
            random.shuffle(order)

        batch: list[int] = []
        batch_count = 0
        for group_idx in order:
            batch.extend(self.groups[group_idx])
            batch_count += 1
            if batch_count == self.slates_per_batch:
                yield batch
                batch = []
                batch_count = 0

        if batch:
            yield batch

    def __len__(self) -> int:
        return (len(self.groups) + self.slates_per_batch - 1) // self.slates_per_batch


class Stage2CandidateDataLoaderV2(DataLoader):
    def __init__(
        self,
        dataset: Stage2CandidateDatasetV2,
        *,
        slates_per_batch: int = 8,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        batch_sampler = SlateBatchSampler(
            groups=dataset.group_indices,
            slates_per_batch=slates_per_batch,
            shuffle=shuffle,
        )
        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_stage2_candidates_v2,
            num_workers=num_workers,
        )
