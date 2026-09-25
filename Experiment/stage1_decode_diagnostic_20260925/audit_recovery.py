"""Independently recount recovered generation and verify no-overflow GPU parity."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / 'Experiment/stage1_50k_fidelity_v2_expert_multiseed_20260924'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), 'Preserve existing audit: ' + str(path)
    path.write_text(json.dumps(value, indent=2) + '\n')


def verify_sources(record):
    for name, digest in record['source_sha256'].items():
        assert sha(ROOT / name) == digest, name


def generation(path):
    rows, scores, ids = [], [], []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('H-'):
                identifier, score, prediction = line.rstrip('\n').split('\t', 2)
                identifier = int(identifier[2:])
                assert not math.isnan(float(score))
                ids.append(identifier)
                rows.append((identifier, prediction))
            elif line.startswith('P-'):
                identifier, value = line.rstrip('\n').split('\t', 1)
                numbers = tuple(float(v) for v in value.split())
                assert numbers and not any(math.isnan(v) for v in numbers)
                scores.append((int(identifier[2:]), numbers))
    assert len(rows) == len(scores) == 76200
    assert ids == [i for i in range(7620) for _ in range(10)], 'Missing, duplicated or reordered augmentation/beam slots'
    assert ids == [i for i, _ in scores]
    return rows, scores


def aggregate():
    target = OUT / 'recovery_generation_audit.json'
    assert not target.exists()
    record_path = OUT / 'length_guard_v1.json'
    record = json.loads(record_path.read_text())
    assert record['pass_execution'] and not record['promoted']
    assert (record['query_count'], record['augmentation_count']) == (762, 7620)
    verify_sources(record)
    augment_path = OUT / 'length_guard_augmentation_v1.json'
    assert sha(augment_path) == record['augmentation_receipt_sha256']
    augmentation = json.loads(augment_path.read_text())
    assert augmentation['pass'] and len(augmentation['events']) == 762
    argv = record['argv']
    options = {key: argv[argv.index(key) + 1] for key in
               ('--path', '--input', '--batch-size', '--buffer-size', '--max-tokens', '--TOPK', '--aug',
                '--repos-beam', '--mask-beam', '--token-beam', '--iter-decode-max-iter')}
    assert {k: options[k] for k in options if k not in ('--path', '--input')} == {
        '--batch-size': '64', '--buffer-size': '2000', '--max-tokens': '4000', '--TOPK': '10',
        '--aug': '10', '--repos-beam': '5', '--mask-beam': '1', '--token-beam': '2', '--iter-decode-max-iter': '10'}
    job = STUDY / 'seed_2/DielsAlder'
    training = json.loads((job / 'training_complete.json').read_text())
    checkpoint = job / 'training/REAXYS_DielsAlder_SINGLE_CATMERGE/run/checkpoints/checkpoint_best.pt'
    assert Path(options['--path']).resolve() == checkpoint
    assert sha(checkpoint) == training['checkpoints']['checkpoint_best.pt']
    assert sha(Path(options['--input'])) == sha(job / 'routes/input_products.txt')
    raw_path = Path('/root/autodl-tmp/prosys_diels_seed2_length_guard_v1_20260925.log')
    rows, scores = generation(raw_path)
    failures = []
    for event in record['events']:
        assert event['capacity'] == 1024 and event['method'] == 'forward_decoder' and event['step'] > 0
        for j, row in enumerate(event['rows']):
            active = event['insertion_active_indices'][row]
            beam_id = event['active_beam_ids'][active]
            aug_id = event['batch_augmented_ids'][beam_id // 10]
            flat_index = aug_id * 10 + beam_id % 10
            assert rows[flat_index][1] == '', 'Invalidated beam emitted chemistry'
            assert all(math.isinf(v) and v < 0 for v in scores[flat_index][1])
            failures.append(dict(query_offset=aug_id // 10, augmentation_slot=aug_id % 10,
                beam=beam_id % 10, augmented_id=aug_id, generation_slot=flat_index,
                step=event['step'], before_insertion_length=event['before_lengths'][j],
                proposed_length=event['proposed_lengths'][j]))
    assert len(failures) == record['newly_invalid_hypotheses'] == 1
    assert len({f['generation_slot'] for f in failures}) == len(failures)
    assert (failures[0]['query_offset'], failures[0]['augmentation_slot'], failures[0]['beam']) == (220, 6, 7)
    destination = OUT / 'recovered_routes'
    assert not destination.exists()
    command = [sys.executable, '-B', str(ROOT / 'stage1_retrosynthesis/build_route_cache.py'),
        '--repo_root', str(ROOT), '--family', 'DielsAlder', '--checkpoint', str(checkpoint),
        '--databin', training['actual_config']['data'], '--output', str(destination),
        '--skip_generation', '--generation_file', str(raw_path), '--processes', '2']
    with (OUT / 'recovery_aggregation.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    path = destination / 'route_cache.json'
    cache = json.loads(path.read_text())
    original = json.loads((ROOT / 'outputs/stage1_routes/DielsAlder/route_cache.json').read_text())
    fields = ('sample_index', 'reaction_id', 'product', 'gold_reactants')
    identity = lambda c: [[r[k] for k in fields] for r in c['reactions']]
    assert identity(cache) == identity(original) and len(cache['reactions']) == 762
    assert sha(destination / 'input_products.txt') == sha(Path(options['--input']))
    from prosys_shared.mainline import stage1_route_recall
    metrics = stage1_route_recall(path)
    assert metrics['n'] == 762
    write(destination / 'metrics.json', metrics)
    bindings = [record_path, augment_path, raw_path, path, destination / 'metrics.json',
                job / 'training_complete.json', checkpoint, Path(options['--input']),
                ROOT / 'stage1_retrosynthesis/build_route_cache.py',
                ROOT / 'stage1_retrosynthesis/utils/get_ranked_topk.py', Path(__file__)]
    write(target, dict(pass_checked_scope=True, promoted=False, queries=762, hypotheses=76200,
        invalidated_hypotheses=failures, original_query_identities_unchanged=True,
        route_count=sum(len(r['routes']) for r in cache['reactions']),
        empty_queries=sum(not r['routes'] for r in cache['reactions']), metrics=metrics,
        input_sha256={str(p): sha(p) for p in bindings}, source_sha256=record['source_sha256'],
        pending=['no-overflow GPU parity', 'explicit expert recovery admission and complete seed-summary verification']))
    print(json.dumps(dict(queries=762, hypotheses=76200, invalidated=failures, metrics=metrics)))


def parity():
    target = OUT / 'normal_path_gpu_parity.json'
    assert not target.exists()
    record_path = OUT / 'seed1_length_guard_parity.json'
    record = json.loads(record_path.read_text())
    assert record['pass_execution'] and record['newly_invalid_hypotheses'] == 0 and not record['events']
    verify_sources(record)
    previous_guard = STUDY / 'seed_1/DielsAlder/routes/augmentation_guard.json'
    current_guard = OUT / 'seed1_parity_augmentation.json'
    a, b = (json.loads(p.read_text()) for p in (previous_guard, current_guard))
    assert a['events'] == b['events'] and a['source_sha256'] == b['source_sha256']
    assert sha(current_guard) == record['augmentation_receipt_sha256']
    old = STUDY / 'seed_1/DielsAlder/routes/generation.txt'
    new = Path('/root/autodl-tmp/prosys_diels_seed1_length_guard_parity_20260925.log')
    old_rows, old_scores = generation(old)
    new_rows, new_scores = generation(new)
    assert old_rows == new_rows, 'Normal-path hypotheses changed'
    assert old_scores == new_scores, 'Normal-path retained token scores changed'
    bindings = [record_path, previous_guard, current_guard, old, new, Path(__file__)]
    write(target, dict(pass_checks=True, family='DielsAlder', expert_seed=1,
        query_count=762, hypotheses_checked=76200, hypotheses_exactly_equal=True,
        retained_token_scores_exactly_equal=True, augmentation_events_exactly_equal=True,
        length_guard_events=0, input_sha256={str(p): sha(p) for p in bindings},
        source_sha256=record['source_sha256'],
        scope='Complete same-checkpoint, same-query, same-batching no-overflow GPU replay; not a new training seed'))
    print('GPU parity passed for all 76,200 hypotheses and retained token scores')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity', action='store_true')
    args = parser.parse_args()
    parity() if args.parity else aggregate()
