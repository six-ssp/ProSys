"""Separate, unpromoted inference run with fail-closed hypothesis-length protection."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis'))
sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_interactive_guarded as guard
from editretro.models import editretro_nat
from fail_closed_masks import install


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Events(list):
    def append(self, value):
        for info in inspect.stack():
            local = info.frame.f_locals
            if info.function == 'generate' and 'sent_idxs' in local:
                value['step'] = local['step']
                value['active_beam_ids'] = local['sent_idxs'].cpu().tolist()
            if info.function in ('forward_decoder', 'forward_decoder_mask') and 'can_ins_mask' in local:
                value['method'] = info.function
                value['insertion_active_indices'] = local['can_ins_mask'].nonzero().flatten().cpu().tolist()
            if info.function == 'main' and 'batch' in local:
                value['batch_augmented_ids'] = local['batch'].ids.cpu().tolist()
        super().append(value)


def main():
    target = Path(os.environ['PROSYS_LENGTH_GUARD_RECEIPT'])
    assert not target.exists()
    paths = [Path(__file__), Path(__file__).with_name('fail_closed_masks.py'),
             ROOT / 'scripts/stage1_interactive_guarded.py',
             ROOT / 'scripts/stage1_identity_guard.py',
             ROOT / 'stage1_retrosynthesis/editretro/models/editretro_nat.py',
             ROOT / 'stage1_retrosynthesis/editretro/models/levenshtein_utils.py',
             ROOT / 'stage1_retrosynthesis/editretro/models/iterative_refinement_generator.py']
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    events = Events()
    install(editretro_nat, events)
    guard.main()
    assert all(sha(ROOT / name) == digest for name, digest in sources.items())
    augmentation = Path(os.environ['PROSYS_AUGMENTATION_GUARD_RECEIPT'])
    record = json.loads(augmentation.read_text())
    assert record['pass'] and record['query_count'] == 762
    target.write_text(json.dumps(dict(pass_execution=True, promoted=False,
        policy='Unsupported insertion length becomes an empty invalid hypothesis; absorbing failure marker; no truncation, resampling or query deletion',
        threshold='actual decoder.max_positions(), not a tuned hyperparameter',
        query_count=record['query_count'], augmentation_count=record['augmented_input_count'],
        newly_invalid_hypotheses=sum(len(e['rows']) for e in events), events=events,
        source_sha256=sources, augmentation_receipt_sha256=sha(augmentation),
        argv=sys.argv, pending=['route aggregation and query/metric audit', 'normal-path GPU parity', 'scientific provenance admission']), indent=2) + '\n')


if __name__ == '__main__':
    main()
