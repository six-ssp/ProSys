#!/usr/bin/env python3
"""Reconcile manuscript counts with persisted splits without rewriting data."""

from functools import lru_cache
from pathlib import Path
import concurrent.futures
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'Experiment/final_release_audit_20260915'


def family_counts(family):
    from rdkit import RDLogger
    from prosys_shared.mainline import load_split_rows, split_file_for_family, normalize_condition_labels
    from prosys_shared.features import canonicalize_reaction_side
    from prosys_shared.cache_integrity import file_sha256
    RDLogger.DisableLog('rdApp.*')
    canonical = lru_cache(maxsize=None)(canonicalize_reaction_side)
    counts, rows, hashes, condition_sets = {}, [], {}, {}
    for split in ('train', 'val', 'test'):
        path = split_file_for_family(ROOT, family, split)
        current = load_split_rows(path)
        rows += current
        counts[split] = len(current)
        hashes[split] = file_sha256(path)
        condition_sets[split] = {(canonical(r['reactants']), canonical(r['product'])) for r in current}
        assert all(a and b for a, b in condition_sets[split])
    assert not condition_sets['train'] & condition_sets['val']
    assert not condition_sets['train'] & condition_sets['test']
    assert not condition_sets['val'] & condition_sets['test']
    condition_keys = set.union(*condition_sets.values())
    result = {'family': family, 'condition_records': len(rows), 'split_rows': counts,
        'unique_canonical_reactions': len(condition_keys), 'split_sha256': hashes,
        'condition_split_reaction_overlap': 0}
    for column in ('reagent_norm', 'solvent_norm'):
        result[column + '_labels'] = len({t.strip() for r in rows for t in
            normalize_condition_labels(r[column]).split(';') if t.strip()})
    result['condition_contexts'] = len({(normalize_condition_labels(r['reagent_norm']),
        normalize_condition_labels(r['solvent_norm'])) for r in rows})
    raw_sets = {}
    route = {}
    for split in ('train', 'val', 'test'):
        path = ROOT / 'data/editretro/datasets' / ('REAXYS_' + family + '_SINGLE_CATMERGE') / 'raw' / ('raw_' + split + '.csv')
        with path.open() as handle:
            records = list(csv.DictReader(handle))
        parts = [r['reactants>reagents>production'].split('>') for r in records]
        keys = [(canonical(r[0]), canonical(r[-1])) for r in parts]
        assert all(a and b for a, b in keys)
        assert len(set(keys)) == len(keys)
        associated = sum(k in condition_keys for k in keys)
        assert split == 'train' or associated == len(keys)
        route[split] = {'records': len(keys), 'condition_associated': associated,
            'additional': len(keys) - associated, 'sha256': file_sha256(path)}
        raw_sets[split] = set(keys)
    assert not raw_sets['train'] & raw_sets['val']
    assert not raw_sets['train'] & raw_sets['test']
    assert not raw_sets['val'] & raw_sets['test']
    assert not raw_sets['train'] & condition_sets['test']
    result['stage1_raw'] = route
    return result


def main():
    import numpy as np
    import pandas as pd
    from prosys_shared.mainline import FAMILY_ORDER
    from stage3_XGBoost.xgb_reranker import TABULAR_FEATURE_COLUMNS, TEMPERATURE_FEATURE_COLUMNS, TEXT_COLUMNS, TARGET_COLUMNS
    assert len(TABULAR_FEATURE_COLUMNS) == len(set(TABULAR_FEATURE_COLUMNS)) == 52
    assert len(TEMPERATURE_FEATURE_COLUMNS) == 180
    assert not set(TABULAR_FEATURE_COLUMNS) & (TEXT_COLUMNS | TARGET_COLUMNS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(family_counts, FAMILY_ORDER))
    directory = ROOT / 'baseline/results/multiseed_20260810'
    seed = pd.read_csv(directory / 'per_family_seed_metrics.csv')
    macro = pd.read_csv(directory / 'macro_by_seed.csv').set_index(['method', 'seed'])
    summary = pd.read_csv(directory / 'macro_mean_std.csv').set_index('method')
    fields = ['cover', 'sys1', 'sys3', 'sys5', 'sys10', 'mrr', 'ndcg10']
    assert len(seed) == 54 and not seed.duplicated(['method', 'seed', 'family']).any()
    for key, group in seed.groupby(['method', 'seed']):
        assert set(group.family) == set(FAMILY_ORDER)
        assert group.n_test_manifest.sum() == 3860
        for field in fields:
            np.testing.assert_allclose(group[field].mean(), macro.loc[key, field], atol=1e-12)
    for method, group in macro.groupby(level='method'):
        for field in fields:
            np.testing.assert_allclose(group[field].mean(), summary.loc[method, field + '_mean'], atol=1e-12)
            np.testing.assert_allclose(group[field].std(ddof=1), summary.loc[method, field + '_std'], atol=1e-12)
    payload = {'pass': True, 'families': rows, 'ltr_dimensions': 52,
        'temperature_dimensions': 180, 'no_id_or_gold_features': True,
        'baseline_aggregate_records_verified': 54,
        'baseline_scope': 'aggregate arithmetic only; no new baseline model training or unavailable row-level replay'}
    (OUT / 'data_and_schema_audit.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
