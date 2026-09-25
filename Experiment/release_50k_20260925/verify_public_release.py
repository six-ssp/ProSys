#!/usr/bin/env python3
"""Verify public hashes and recompute aggregate statistics without private data."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAMILIES = {'Beckmann': 235, 'Buchwald-HartwigCross-Coupling': 1099,
            'Chan_LamCoupling': 390, 'DielsAlder': 762,
            'Friedel-CraftsAcylation': 475, 'Friedel-CraftsAlkylation': 899}
FIELDS = ['route10', 'cover', 'sys1', 'sys3', 'sys5', 'sys10', 'mrr', 'ndcg10',
          'temp_mae', 'temp_within_5c', 'temp_within_10c', 'temp_within_20c']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open() as handle:
        reader = csv.DictReader(handle)
        forbidden = {'product', 'reactants', 'reaction_id', 'sample_index',
                     'gold_reactants', 'temperature_gold', 'yield_gold'}
        if forbidden.intersection(reader.fieldnames):
            raise ValueError('Row-level chemistry/identity columns in public aggregate: ' + str(path))
        return list(reader)


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'Aggregate mismatch: {a} != {b}')


def recompute():
    directory = ROOT / 'Experiment/50k_verified_comparisons_20260924/full'
    rows = read_csv(directory / 'per_family_model_seed.csv')
    if len(rows) != 150 or any(int(r['n_queries']) != FAMILIES[r['family']] for r in rows):
        raise ValueError('Wrong comparison grid or query denominator')
    models = sorted({r['model'] for r in rows})
    if len(models) != 9:
        raise ValueError('Expected full model, four baselines and four removals')
    values = {}
    for model in models:
        selected = [r for r in rows if r['model'] == model]
        seeds = [0] if model == 'B1_ProductNB' else [0, 1, 2]
        expected = Counter((family, seed) for family in FAMILIES for seed in seeds)
        if Counter((r['family'], int(r['seed'])) for r in selected) != expected:
            raise ValueError('Missing, duplicated or unexpected model/seed/family row')
        values[model] = {}
        for field in FIELDS:
            macro = []
            for seed in seeds:
                current = [r[field] for r in selected if int(r['seed']) == seed and r[field] != '']
                if current:
                    if len(current) != 6:
                        raise ValueError('Partial-family metric cannot be a six-family macro')
                    macro.append(statistics.mean(map(float, current)))
            values[model][field] = dict(mean=statistics.mean(macro) if macro else None,
                sample_sd=statistics.stdev(macro) if len(macro) > 1 else None, count=len(macro))

    # The published CSV has a two-level metric/statistic header and an index-name row.
    with (directory / 'macro_mean_std.csv').open() as handle:
        published = list(csv.reader(handle))
    for row in published[3:]:
        model = row[0]
        for field in FIELDS:
            for statistic, key in [('mean', 'mean'), ('std', 'sample_sd'), ('count', 'count')]:
                column = next(i for i, pair in enumerate(zip(published[0], published[1]))
                              if pair == (field, statistic))
                expected = values[model][field][key]
                if expected is None:
                    if row[column] not in ('', 'nan'):
                        raise ValueError('Missing SD/support must remain NA')
                else:
                    close(expected, row[column])

    experts = read_csv(ROOT / 'Experiment/stage1_50k_fidelity_v2_expert_multiseed_20260924/per_family_seed_metrics.csv')
    expected = Counter((family, seed) for family in FAMILIES for seed in range(3))
    if Counter((r['family'], int(r['seed'])) for r in experts) != expected:
        raise ValueError('Expert grid is not six families by three seeds')
    if any(int(r['n']) != FAMILIES[r['family']] for r in experts):
        raise ValueError('Expert denominator changed')
    expert_stats = {}
    for k in (1, 3, 5, 10):
        seed_values = [statistics.mean(float(r[f'route_recall_top{k}']) for r in experts
                                      if int(r['seed']) == seed) for seed in range(3)]
        expert_stats[str(k)] = dict(mean=statistics.mean(seed_values), sample_sd=statistics.stdev(seed_values))
    return dict(comparison_rows=150, expert_rows=18, queries_per_seed=sum(FAMILIES.values()),
                macro_fraction_or_celsius=values, expert_route_fraction=expert_stats)


def main():
    manifest = json.loads((HERE / 'public_evidence.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Missing or changed release file: ' + name)
    actual = recompute()
    if actual != manifest['aggregate_recalculation']:
        raise ValueError('Release statistics differ from the aggregate CSVs')
    print(json.dumps(dict(pass_checks=True, files_checked=len(manifest['files_sha256']),
        comparison_rows=150, expert_rows=18, queries_per_seed=3860,
        scope='Public-file integrity and independent aggregate arithmetic; not raw-data or model-level replay')))


if __name__ == '__main__':
    main()
