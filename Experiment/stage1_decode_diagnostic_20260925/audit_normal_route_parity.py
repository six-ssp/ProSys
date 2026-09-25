"""Verify exact route-level parity while exposing any invalid-hypothesis EOS drift."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
from audit_recovery import STUDY, generation, sha, verify_sources, write


def main():
    target = OUT / 'normal_path_gpu_parity.json'
    assert not target.exists()
    record_path = OUT / 'seed1_length_guard_parity.json'
    record = json.loads(record_path.read_text())
    assert record['pass_execution'] and record['newly_invalid_hypotheses'] == 0 and not record['events']
    verify_sources(record)
    old_guard = STUDY / 'seed_1/DielsAlder/routes/augmentation_guard.json'
    new_guard = OUT / 'seed1_parity_augmentation.json'
    a, b = (json.loads(p.read_text()) for p in (old_guard, new_guard))
    assert a['events'] == b['events'] and a['source_sha256'] == b['source_sha256']
    assert sha(new_guard) == record['augmentation_receipt_sha256']
    old_raw = STUDY / 'seed_1/DielsAlder/routes/generation.txt'
    new_raw = Path('/root/autodl-tmp/prosys_diels_seed1_length_guard_parity_20260925.log')
    old_rows, old_scores = generation(old_raw)
    new_rows, new_scores = generation(new_raw)
    assert old_rows == new_rows
    vendor_path = ROOT / 'stage1_retrosynthesis/utils/get_ranked_topk.py'
    spec = importlib.util.spec_from_file_location('vendor_rank', vendor_path)
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    differences = []
    for slot, ((old_id, previous), (new_id, current)) in enumerate(zip(old_scores, new_scores)):
        assert old_id == new_id and len(previous) == len(current)
        changed = [(i, x, y) for i, (x, y) in enumerate(zip(previous, current)) if x != y]
        if changed:
            assert len(changed) == 1 and changed[0][0] == len(previous) - 1, 'Nonterminal token score changed'
            assert vendor.canonicalize_smiles_clear_map(old_rows[slot][1].replace(' ', ''))[0] == '', 'A valid chemical route score changed'
            differences.append(dict(generation_slot=slot, augmented_id=old_id, beam=slot % 10,
                token_index=changed[0][0], old=changed[0][1], new=changed[0][2],
                same_hypothesis=old_rows[slot][1], scope='EOS score on already invalid SMILES; retained in both raw logs'))
    assert len(differences) == 2
    original_path = STUDY / 'seed_1/DielsAlder/routes/route_cache.json'
    original = json.loads(original_path.read_text())
    training_path = STUDY / 'seed_1/DielsAlder/training_complete.json'
    training = json.loads(training_path.read_text())
    argv = record['argv']
    checkpoint = Path(argv[argv.index('--path') + 1])
    assert checkpoint.resolve() == Path(original['checkpoint']).resolve()
    assert sha(checkpoint) == training['checkpoints']['checkpoint_best.pt']
    destination = OUT / 'seed1_parity_routes'
    assert not destination.exists()
    command = [sys.executable, '-B', str(ROOT / 'stage1_retrosynthesis/build_route_cache.py'),
        '--repo_root', str(ROOT), '--family', 'DielsAlder', '--checkpoint', str(checkpoint),
        '--databin', training['actual_config']['data'], '--output', str(destination),
        '--skip_generation', '--generation_file', str(new_raw), '--processes', '2']
    with (OUT / 'seed1_parity_aggregation.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    current_path = destination / 'route_cache.json'
    current = json.loads(current_path.read_text())
    assert current['reactions'] == original['reactions'], 'Query identities or complete ranked route lists/scores differ'
    assert len(current['reactions']) == 762
    bindings = [record_path, old_guard, new_guard, old_raw, new_raw, vendor_path,
                original_path, current_path, training_path, checkpoint,
                ROOT / 'stage1_retrosynthesis/build_route_cache.py', OUT / 'audit_recovery.py', Path(__file__)]
    write(target, dict(pass_checks=True, family='DielsAlder', expert_seed=1,
        query_count=762, hypotheses_checked=76200, hypotheses_exactly_equal=True,
        retained_token_scores_exactly_equal=False, valid_hypothesis_token_scores_exactly_equal=True,
        route_lists_and_scores_exactly_equal=True, augmentation_events_exactly_equal=True,
        length_guard_events=0, invalid_hypothesis_eos_score_differences=differences,
        input_sha256={str(p): sha(p) for p in bindings}, source_sha256=record['source_sha256'],
        scope='Full GPU same-query replay: exact hypothesis strings and final ranked routes. Two EOS-score differences on already invalid strings are explicitly retained; not universal token-score bitwise determinism.'))
    print('All 76,200 hypothesis strings and all 762 final ranked route lists/scores agree; 2 invalid-string EOS-score differences recorded')


if __name__ == '__main__':
    main()
