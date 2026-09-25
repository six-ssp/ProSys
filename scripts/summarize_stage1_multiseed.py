#!/usr/bin/env python3
"""Verify completed expert replicates before producing equal-family tables."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIELDS = [f'route_recall_top{k}' for k in (1, 3, 5, 10)]


def check_invalidation(study):
    for name in ('SCIENTIFIC_INVALIDATION.json', 'PAUSED_FOR_SPLIT_AUDIT.json'):
        if (Path(study) / name).exists():
            raise ValueError('Study has a split-integrity invalidation; scientific summary withheld: ' + name)


def aggregate_rows(rows, families, seeds):
    import numpy as np
    import pandas as pd

    expected = Counter((f, s) for f in families for s in seeds)
    observed = Counter((r['family'], r['seed']) for r in rows)
    if observed != expected:
        raise ValueError('Incomplete or duplicate family/seed grid; summary withheld')
    frame = pd.DataFrame(rows)
    for family in families:
        support = frame.loc[frame.family == family, 'n']
        if support.nunique() != 1 or support.iloc[0] <= 0:
            raise ValueError('Evaluation support differs across seeds: ' + family)
    values = frame[FIELDS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Non-finite or out-of-range route recall')
    if (np.diff(values, axis=1) < -1e-12).any():
        raise ValueError('Route recall must be nondecreasing with k')
    macro = frame.groupby('seed')[FIELDS].mean().reindex(seeds)
    family_stats = frame.groupby('family')[FIELDS].agg(['mean', 'std']).reindex(families)
    family_stats.columns = ['_'.join(c) for c in family_stats.columns]
    macro_stats = {k + '_' + stat: getattr(macro[k], stat)()
                   for k in FIELDS for stat in ('mean', 'std')}
    return frame, macro, family_stats, pd.DataFrame([macro_stats])


def verify_job(study, family, seed):
    import numpy as np
    from prosys_shared.mainline import stage1_route_recall
    from scripts.run_stage1_multiseed import sha, require_training_admission

    check_invalidation(study)
    job = study / f'seed_{seed}' / family
    protocol = json.loads((study / 'protocol.json').read_text())
    inputs = json.loads((job / 'inputs.json').read_text())
    for name, expected in inputs['input_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Recorded training input changed: ' + name)
    completion = json.loads((job / 'completion.json').read_text())
    if (completion.get('family'), completion.get('seed'), completion.get('status')) != (family, seed, 'complete'):
        raise ValueError('Invalid completion record: ' + str(job))
    for name, expected in completion['output_sha256'].items():
        if sha(job / name) != expected:
            raise ValueError('Completed output changed: ' + str(job / name))
    training = json.loads((job / 'training_complete.json').read_text())
    actual = training['actual_config']
    base = Path(inputs['base_checkpoint'])
    if (inputs['validation_protocol'] != protocol['validation_protocol'] or
            base.resolve() != Path(protocol['base_checkpoint']).resolve() or
            sha(base) != protocol['base_checkpoint_sha256'] or
            Path(actual['restore_file']).resolve() != base.resolve()):
        raise ValueError('Training protocol or selected base differs from study admission')
    require_training_admission(Path(actual['data']), base, protocol['validation_protocol'])
    checkpoint_dir = job / 'training' / ('REAXYS_' + family + '_SINGLE_CATMERGE') / 'run/checkpoints'
    if set(training['checkpoints']) != {'checkpoint_best.pt', 'checkpoint_last.pt'}:
        raise ValueError('Expected exactly best/last checkpoints')
    for name, expected in training['checkpoints'].items():
        if sha(checkpoint_dir / name) != expected:
            raise ValueError('Checkpoint changed: ' + str(checkpoint_dir / name))
    if training['actual_config']['seed'] != seed:
        raise ValueError('Actual training seed differs')
    if 'done training in' not in (checkpoint_dir.parent / 'train.log').read_text():
        raise ValueError('Training did not terminate normally')
    cache_path = job / 'routes/route_cache.json'
    cache = json.loads(cache_path.read_text())
    if protocol.get('inference_augmentation') == 'product_identity_fallback_v1':
        from scripts.stage1_route_admission import verify_guard
        if inputs.get('inference_augmentation') != protocol['inference_augmentation']:
            raise ValueError('Training and study inference protocols differ')
        verify_guard(cache_path)
    if Path(cache['checkpoint']).resolve() != (checkpoint_dir / 'checkpoint_best.pt').resolve():
        raise ValueError('Route cache was not decoded with the selected best checkpoint')
    original_path = ROOT / 'outputs/stage1_routes' / family / 'route_cache.json'
    original = json.loads(original_path.read_text())
    identity = lambda c: [(r['sample_index'], r['reaction_id'], r['product'], r['gold_reactants'])
                          for r in c['reactions']]
    if identity(cache) != identity(original):
        raise ValueError('Frozen test query identities changed')
    replay = stage1_route_recall(cache_path)
    if replay['n'] != len(original['reactions']):
        raise ValueError('Evaluation silently dropped a test query')
    saved = json.loads((job / 'metrics.json').read_text())
    for key in ['n'] + FIELDS:
        np.testing.assert_allclose(replay[key], saved[key], rtol=0, atol=1e-12)
    return {'family': family, 'seed': seed, **replay,
            'best_sha256': sha(checkpoint_dir / 'checkpoint_best.pt'),
            'route_cache_sha256': sha(cache_path), 'original_query_cache_sha256': sha(original_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study-root', type=Path, required=True)
    p.add_argument('--status-only', action='store_true')
    args = p.parse_args()
    study = args.study_root.resolve()
    protocol = json.loads((study / 'protocol.json').read_text())
    families, seeds = protocol['families'], protocol['seeds']
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or len(set(families)) != len(families):
        raise ValueError('Replicate study requires unique families and at least two seeds')
    missing = [(f, s) for f in families for s in seeds
               if not (study / f'seed_{s}' / f / 'completion.json').exists()]
    print(json.dumps({'expected_jobs': len(families) * len(seeds),
                      'completed_receipts': len(families) * len(seeds) - len(missing),
                      'missing': missing, 'receipt_count_is_not_verification': True}), flush=True)
    if args.status_only:
        return
    check_invalidation(study)
    if missing:
        raise RuntimeError('Incomplete study; no final tables written')
    rows = [verify_job(study, f, s) for f in families for s in seeds]
    frame, macro, family_stats, stats = aggregate_rows(rows, families, seeds)
    outputs = {'per_family_seed_metrics.csv': frame,
               'macro_by_seed.csv': macro.reset_index(),
               'per_family_mean_std.csv': family_stats.reset_index(), 'macro_mean_std.csv': stats}
    for name, table in outputs.items():
        temporary = study / (name + '.tmp')
        table.to_csv(temporary, index=False)
        temporary.replace(study / name)
    strict = protocol['validation_protocol'] == 'strict_post_augmentation'
    protocol_text = ('Fixed freshly retrained base; post-augmentation overlap exclusions; unchanged original condition test queries. '
                     if strict else 'Fixed base and explicitly recorded input/validation admission. ')
    lines = ['# Stage 1 Expert Fine-Tuning: Verified Seed Results', '',
             protocol_text +
             'Values are percentages, mean +/- sample SD across expert fine-tuning seeds. '
             'This is not base-pretraining or full-pipeline variability.', '',
             '| Family | Route@1 | Route@3 | Route@5 | Route@10 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for label, row in list(family_stats.iterrows()) + [('MACRO-AVG', stats.iloc[0])]:
        values = [f'{row[k + "_mean"] * 100:.2f} +/- {row[k + "_std"] * 100:.2f}' for k in FIELDS]
        lines.append('| ' + ' | '.join([label] + values) + ' |')
    lines += ['', 'Macro values weight each family equally before calculating seed SD. '
              'Admission evidence binds the base checkpoint, expert inputs and condition splits. '
              'These results do not silently replace the historical mainline.', '']
    (study / 'SUMMARY.md').write_text('\n'.join(lines))
    from scripts.run_stage1_multiseed import sha, write
    write(study / 'summary_verification.json', {'pass': True, 'verified_jobs': len(rows),
          'protocol_sha256': sha(study / 'protocol.json'),
          'output_sha256': {name: sha(study / name) for name in list(outputs) + ['SUMMARY.md']},
          'scope': 'checkpoint hashes, frozen query identity, replay and equal-family aggregation'})
    print('Verified', len(rows), 'jobs; wrote final tables', flush=True)


if __name__ == '__main__':
    main()
