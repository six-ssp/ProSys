"""Stage 2A candidate-pool builders for ProSys Stage 2 V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from stage2.evaluate_model.eval_utils import MultiTask_Evaluator

from .features import canonicalize_smiles, product_morgan_fp, product_scaffold_smiles, tanimoto_similarity_from_bitvect
from .product_memory import normalize_condition_labels, safe_float


@dataclass(frozen=True)
class RouteRecord:
    sample_index: int
    reaction_id: str
    reactants: str
    product: str
    family: str
    retro_rank: int = 1
    retro_score: float = 1.0
    retro_probability: float = 1.0


def load_route_records_from_split(split_file: str | Path, family: str) -> list[RouteRecord]:
    records: list[RouteRecord] = []
    seen = set()
    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            reaction_id, reactants, product = parts[:3]
            dedup_key = (reaction_id, reactants, product)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            records.append(
                RouteRecord(
                    sample_index=len(records),
                    reaction_id=reaction_id,
                    reactants=reactants,
                    product=product,
                    family=family,
                )
            )
    return records


@dataclass
class ProductSupportContext:
    """Per-product memory lookups computed once and reused across all candidates.

    Building this once per product (instead of recomputing inside every
    candidate's support-feature call) removes the O(candidates x train_products)
    Tanimoto blow-up that dominated Stage 2A build time.
    """

    product: str
    canonical: str
    scaffold: str
    exact_rows: list[dict]
    exact_total: float
    scaffold_rows: list[dict]
    scaffold_total: float
    knn_neighbor_rows: list[tuple[float, list[dict]]] = field(default_factory=list)
    knn_total_weight: float = 0.0


def _single_side_support(rows: Iterable[dict], total_count: float, candidate_value: str, field_name: str) -> float:
    if total_count <= 0 or not candidate_value:
        return 0.0
    support = 0.0
    token = set(candidate_value.split('; '))
    for row in rows:
        values = set(str(row[field_name]).split('; ')) if row[field_name] else set()
        if token & values:
            support += float(row['count'])
    return support / total_count


class ProductMemoryLookup:
    def __init__(self, memory_dir: str | Path):
        memory_dir = Path(memory_dir)
        self.exact_df = pd.read_csv(memory_dir / 'exact_product_memory.csv', keep_default_na=False)
        self.scaffold_df = pd.read_csv(memory_dir / 'scaffold_product_memory.csv', keep_default_na=False)

        knn = np.load(memory_dir / 'product_knn_index.npz', allow_pickle=True)
        self.knn_products = [str(x) for x in knn['product_smiles'].tolist()]
        self.knn_packed = knn['packed_fps']
        self.knn_n_bits = int(knn['n_bits'][0])
        self.knn_radius = int(knn['radius'][0])
        self.knn_matrix = np.unpackbits(self.knn_packed, axis=1)[:, :self.knn_n_bits].astype(np.uint8)

        self.exact_by_product: dict[str, list[dict]] = {}
        self.scaffold_by_scaffold: dict[str, list[dict]] = {}
        self.exact_totals: dict[str, float] = {}
        self.scaffold_totals: dict[str, float] = {}

        for row in self.exact_df.to_dict('records'):
            product = str(row['product_canonical'])
            row['reagent_norm'] = str(row['reagent_norm']) if pd.notna(row['reagent_norm']) else ''
            row['solvent_norm'] = str(row['solvent_norm']) if pd.notna(row['solvent_norm']) else ''
            row['count'] = float(row['count'])
            row['mean_yield'] = safe_float(row['mean_yield'])
            self.exact_by_product.setdefault(product, []).append(row)
            self.exact_totals[product] = self.exact_totals.get(product, 0.0) + row['count']

        for row in self.scaffold_df.to_dict('records'):
            scaffold = str(row['product_scaffold'])
            row['reagent_norm'] = str(row['reagent_norm']) if pd.notna(row['reagent_norm']) else ''
            row['solvent_norm'] = str(row['solvent_norm']) if pd.notna(row['solvent_norm']) else ''
            row['count'] = float(row['count'])
            row['mean_yield'] = safe_float(row['mean_yield'])
            self.scaffold_by_scaffold.setdefault(scaffold, []).append(row)
            self.scaffold_totals[scaffold] = self.scaffold_totals.get(scaffold, 0.0) + row['count']

    def build_support_context(self, product: str, *, knn_top_k: int = 10) -> ProductSupportContext:
        """Precompute all per-product memory lookups exactly once.

        The Morgan fingerprint and full-matrix Tanimoto are the expensive parts;
        computing them here (rather than per candidate) is the core optimization.
        """
        canonical = canonicalize_smiles(product)
        scaffold = product_scaffold_smiles(product)
        exact_rows = self.exact_by_product.get(canonical, [])
        exact_total = self.exact_totals.get(canonical, 0.0)
        scaffold_rows = self.scaffold_by_scaffold.get(scaffold, [])
        scaffold_total = self.scaffold_totals.get(scaffold, 0.0)

        knn_neighbor_rows: list[tuple[float, list[dict]]] = []
        knn_total_weight = 0.0
        if self.knn_matrix.shape[0] > 0:
            query_fp = product_morgan_fp(product, n_bits=self.knn_n_bits, radius=self.knn_radius)
            similarities = tanimoto_similarity_from_bitvect(query_fp, self.knn_matrix)
            top_indices = np.argsort(similarities)[::-1][:knn_top_k]
            for index in top_indices:
                sim = float(similarities[index])
                if sim <= 0:
                    continue
                neighbor_rows = self.exact_by_product.get(self.knn_products[index], [])
                knn_neighbor_rows.append((sim, neighbor_rows))
                knn_total_weight += sim

        return ProductSupportContext(
            product=product,
            canonical=canonical,
            scaffold=scaffold,
            exact_rows=exact_rows,
            exact_total=exact_total,
            scaffold_rows=scaffold_rows,
            scaffold_total=scaffold_total,
            knn_neighbor_rows=knn_neighbor_rows,
            knn_total_weight=knn_total_weight,
        )

    # --- candidate enumeration (uses the cached context) ---

    @staticmethod
    def exact_candidate_rows(ctx: ProductSupportContext) -> list[dict]:
        return sorted(ctx.exact_rows, key=lambda row: (-row['count'], row['reagent_norm'], row['solvent_norm']))

    @staticmethod
    def scaffold_candidate_rows(ctx: ProductSupportContext) -> list[dict]:
        return sorted(ctx.scaffold_rows, key=lambda row: (-row['count'], row['reagent_norm'], row['solvent_norm']))

    @staticmethod
    def knn_candidate_rows(ctx: ProductSupportContext, max_contexts: int = 50) -> list[dict]:
        aggregated: dict[tuple[str, str], dict[str, float]] = {}
        for sim, neighbor_rows in ctx.knn_neighbor_rows:
            for row in neighbor_rows:
                key = (str(row['reagent_norm']), str(row['solvent_norm']))
                stats = aggregated.setdefault(
                    key,
                    {'knn_weight': 0.0, 'weighted_yield': 0.0, 'support_products': 0.0},
                )
                stats['knn_weight'] += sim
                if not np.isnan(row['mean_yield']):
                    stats['weighted_yield'] += sim * float(row['mean_yield'])
                stats['support_products'] += 1.0

        rows = []
        for (reagent_norm, solvent_norm), stats in aggregated.items():
            mean_yield = (
                stats['weighted_yield'] / stats['knn_weight']
                if stats['knn_weight'] > 0 else float('nan')
            )
            rows.append(
                {
                    'reagent_norm': reagent_norm,
                    'solvent_norm': solvent_norm,
                    'knn_weight': stats['knn_weight'],
                    'mean_yield': mean_yield,
                }
            )
        rows.sort(key=lambda row: (-row['knn_weight'], row['reagent_norm'], row['solvent_norm']))
        return rows[:max_contexts]

    # --- support features (uses the cached context) ---

    @staticmethod
    def exact_support_from_context(ctx: ProductSupportContext, reagent_norm: str, solvent_norm: str) -> dict[str, float]:
        rows = ctx.exact_rows
        total_count = ctx.exact_total
        pair_row = next(
            (
                row for row in rows
                if str(row['reagent_norm']) == reagent_norm and str(row['solvent_norm']) == solvent_norm
            ),
            None,
        )
        pair_count = float(pair_row['count']) if pair_row else 0.0
        pair_mean_yield = float(pair_row['mean_yield']) if pair_row else 0.0
        pair_support = pair_count / total_count if total_count > 0 else 0.0
        return {
            'product_exact_pair_support': pair_support,
            'product_exact_reagent_support': _single_side_support(rows, total_count, reagent_norm, 'reagent_norm'),
            'product_exact_solvent_support': _single_side_support(rows, total_count, solvent_norm, 'solvent_norm'),
            'product_pair_freq': pair_count,
            'product_pair_mean_yield': pair_mean_yield,
        }

    @staticmethod
    def scaffold_support_from_context(ctx: ProductSupportContext, reagent_norm: str, solvent_norm: str) -> dict[str, float]:
        rows = ctx.scaffold_rows
        total_count = ctx.scaffold_total
        pair_row = next(
            (
                row for row in rows
                if str(row['reagent_norm']) == reagent_norm and str(row['solvent_norm']) == solvent_norm
            ),
            None,
        )
        pair_support = float(pair_row['count']) / total_count if pair_row and total_count > 0 else 0.0
        return {
            'product_scaffold_pair_support': pair_support,
            'product_scaffold_reagent_support': _single_side_support(rows, total_count, reagent_norm, 'reagent_norm'),
            'product_scaffold_solvent_support': _single_side_support(rows, total_count, solvent_norm, 'solvent_norm'),
        }

    @staticmethod
    def knn_support_from_context(ctx: ProductSupportContext, reagent_norm: str, solvent_norm: str) -> dict[str, float]:
        total_weight = ctx.knn_total_weight
        if total_weight <= 0:
            return {
                'product_knn_pair_support': 0.0,
                'product_knn_reagent_support': 0.0,
                'product_knn_solvent_support': 0.0,
            }

        reagent_tokens = set(reagent_norm.split('; ')) if reagent_norm else set()
        solvent_tokens = set(solvent_norm.split('; ')) if solvent_norm else set()

        pair_weight = 0.0
        reagent_weight = 0.0
        solvent_weight = 0.0
        for sim, neighbor_rows in ctx.knn_neighbor_rows:
            if any(
                str(row['reagent_norm']) == reagent_norm and str(row['solvent_norm']) == solvent_norm
                for row in neighbor_rows
            ):
                pair_weight += sim
            if any(reagent_tokens & set(str(row['reagent_norm']).split('; ')) for row in neighbor_rows):
                reagent_weight += sim
            if any(solvent_tokens & set(str(row['solvent_norm']).split('; ')) for row in neighbor_rows):
                solvent_weight += sim

        return {
            'product_knn_pair_support': pair_weight / total_weight,
            'product_knn_reagent_support': reagent_weight / total_weight,
            'product_knn_solvent_support': solvent_weight / total_weight,
        }


class FNNCandidateGenerator:
    def __init__(
        self,
        family_dir: str | Path,
        checkpoint_path: str | Path,
        *,
        cutoff_solv: float = 0.3,
        cutoff_reag: float = 0.3,
        max_solv: int = 11,
        max_reag: int = 11,
        device: Optional[str] = None,
    ):
        family_dir = Path(family_dir)
        solvent_classes = pd.read_pickle(family_dir / 'label_processed' / 'class_names_solvent_labels_processed.pkl')
        reagent_classes = pd.read_pickle(family_dir / 'label_processed' / 'class_names_reagent_labels_processed.pkl')
        solvent_classes = sorted(solvent_classes.items(), key=lambda item: item[1], reverse=True)
        reagent_classes = sorted(reagent_classes.items(), key=lambda item: item[1], reverse=True)

        self.evaluator = MultiTask_Evaluator(
            solvent_classes,
            reagent_classes,
            cutoff_solv=cutoff_solv,
            cutoff_reag=cutoff_reag,
        )
        self.evaluator.max_solv = max_solv
        self.evaluator.max_reag = max_reag
        torch_device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.evaluator.load_model(str(checkpoint_path), device=torch_device)
        self.fpsize = self.evaluator.args_MT.fpsize
        self.radius = self.evaluator.args_MT.radius

    def generate(self, reactants: str, product: str) -> list[tuple[str, str]]:
        from stage2.train_multilabel.data_utils import create_rxn_Morgan2FP_concatenate

        rxn_fp = torch.Tensor(
            create_rxn_Morgan2FP_concatenate(
                reactants,
                product,
                fpsize=self.fpsize,
                radius=self.radius,
            )
        )
        named_contexts = self.evaluator.make_input_rxn_conditionBYnames(rxn_fp)
        return [
            (normalize_condition_labels(reagent), normalize_condition_labels(solvent))
            for solvent, reagent in named_contexts
        ]


def merge_candidate_rows(
    base_row: dict,
    candidates: dict[tuple[str, str], dict],
    ctx: ProductSupportContext,
) -> list[dict]:
    rows = []
    for (reagent_norm, solvent_norm), candidate in candidates.items():
        support = {}
        support.update(ProductMemoryLookup.exact_support_from_context(ctx, reagent_norm, solvent_norm))
        support.update(ProductMemoryLookup.scaffold_support_from_context(ctx, reagent_norm, solvent_norm))
        support.update(ProductMemoryLookup.knn_support_from_context(ctx, reagent_norm, solvent_norm))

        rows.append(
            {
                **base_row,
                'reagent_norm': reagent_norm,
                'solvent_norm': solvent_norm,
                'from_fnn': int(candidate.get('from_fnn', 0)),
                'from_product_exact': int(candidate.get('from_product_exact', 0)),
                'from_product_scaffold': int(candidate.get('from_product_scaffold', 0)),
                'from_product_knn': int(candidate.get('from_product_knn', 0)),
                **support,
            }
        )
    return rows


def build_candidate_pool_for_routes(
    route_records: list[RouteRecord],
    product_lookup: ProductMemoryLookup,
    *,
    fnn_generator: Optional[FNNCandidateGenerator] = None,
    knn_top_k: int = 10,
    knn_max_contexts: int = 50,
) -> pd.DataFrame:
    rows = []
    context_cache: dict[str, ProductSupportContext] = {}
    for record in route_records:
        base_row = {
            'family': record.family,
            'sample_index': record.sample_index,
            'reaction_id': record.reaction_id,
            'product': record.product,
            'reactants': record.reactants,
            'retro_rank': record.retro_rank,
            'retro_score': record.retro_score,
            'retro_probability': record.retro_probability,
        }

        ctx = context_cache.get(record.product)
        if ctx is None:
            ctx = product_lookup.build_support_context(record.product, knn_top_k=knn_top_k)
            context_cache[record.product] = ctx

        candidates: dict[tuple[str, str], dict] = {}

        if fnn_generator is not None:
            for reagent_norm, solvent_norm in fnn_generator.generate(record.reactants, record.product):
                key = (reagent_norm, solvent_norm)
                candidates.setdefault(key, {})['from_fnn'] = 1

        for row in ProductMemoryLookup.exact_candidate_rows(ctx):
            key = (str(row['reagent_norm']), str(row['solvent_norm']))
            candidates.setdefault(key, {})['from_product_exact'] = 1

        for row in ProductMemoryLookup.scaffold_candidate_rows(ctx):
            key = (str(row['reagent_norm']), str(row['solvent_norm']))
            candidates.setdefault(key, {})['from_product_scaffold'] = 1

        for row in ProductMemoryLookup.knn_candidate_rows(ctx, max_contexts=knn_max_contexts):
            key = (str(row['reagent_norm']), str(row['solvent_norm']))
            candidates.setdefault(key, {})['from_product_knn'] = 1

        rows.extend(merge_candidate_rows(base_row, candidates, ctx))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    fill_zero_cols = [
        'product_exact_pair_support',
        'product_exact_reagent_support',
        'product_exact_solvent_support',
        'product_scaffold_pair_support',
        'product_scaffold_reagent_support',
        'product_scaffold_solvent_support',
        'product_knn_pair_support',
        'product_knn_reagent_support',
        'product_knn_solvent_support',
        'product_pair_freq',
        'product_pair_mean_yield',
    ]
    for column in fill_zero_cols:
        if column in frame.columns:
            frame[column] = frame[column].fillna(0.0)

    return frame.sort_values(
        ['sample_index', 'from_fnn', 'from_product_exact', 'from_product_scaffold', 'from_product_knn'],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
