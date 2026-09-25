#!/usr/bin/env python3
"""Count unknown tokens in actual base/expert tensors and inspect overlap witnesses."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import sha, write


def main():
    import numpy as np
    from prosys_shared.mainline import FAMILY_ORDER
    sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
    from fairseq.data import Dictionary, indexed_dataset
    prepared = ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs/artifacts'
    sources = {'USPTO_STAGE2_FILTERED': ROOT / 'data/editretro/datasets/USPTO_STAGE2_FILTERED/aug10/data-bin'}
    sources.update({family: prepared / ('REAXYS_' + family + '_SINGLE_CATMERGE') / 'data-bin' for family in FAMILY_ORDER})
    witnesses_path = ROOT / 'Experiment/final_release_audit_20260915/augmented_overlap_examples.json'
    witnesses = {r['family']: r['augmented_training_row'] for r in json.loads(witnesses_path.read_text())['examples']}
    rows, checks, hashes = [], [], {}
    for family, folder in sources.items():
        for side in ('src', 'tgt'):
            dictionary_path = folder / ('dict.' + side + '.txt')
            dictionary = Dictionary.load(str(dictionary_path))
            hashes[str(dictionary_path.relative_to(ROOT))] = sha(dictionary_path)
            for split in ('train', 'valid', 'test'):
                prefix = folder / (split + '.src-tgt.' + side)
                if not Path(str(prefix) + '.idx').exists():
                    continue
                data = indexed_dataset.make_dataset(str(prefix), 'mmap')
                tokens = np.frombuffer(data._bin_buffer, dtype=data._index.dtype)
                unknown = np.flatnonzero(tokens == dictionary.unk())
                positions = unknown * np.dtype(data._index.dtype).itemsize
                unknown_rows = np.unique(np.searchsorted(data._index._pointers, positions, side='right') - 1)
                rows.append({'dataset': family, 'split': split, 'side': side, 'sequences': len(data),
                             'unknown_tokens': len(unknown), 'sequences_with_unknown_tokens': len(unknown_rows)})
                if family in witnesses and split == 'train':
                    count = int((data[witnesses[family]] == dictionary.unk()).sum())
                    checks.append({'family': family, 'side': side, 'augmented_row': witnesses[family], 'unknown_tokens': count})
                for suffix in ('.idx', '.bin'):
                    path = Path(str(prefix) + suffix)
                    hashes[str(path.relative_to(ROOT))] = sha(path)
                del tokens, data
    payload = {'audit_complete': True, 'all_tokens_in_vocabulary': all(row['unknown_tokens'] == 0 for row in rows), 'rows': rows,
        'six_positive_overlap_witnesses_have_no_unknown_tokens': len(checks) == 12 and all(r['unknown_tokens'] == 0 for r in checks),
        'witnesses': checks, 'source_sha256': hashes, 'positive_receipt_sha256': sha(witnesses_path),
        'interpretation': 'Fixed shared SPE vocabulary permits UNK; nonzero corpus counts are a representation limitation, not a split-audit failure.',
        'scope': 'complete actual tensor unknown-token counts; positive overlap witness tokens are not hidden by UNK encoding'}
    write(ROOT / 'Experiment/stage1_split_repair_20260915/vocabulary_audit.json', payload)
    print(json.dumps({'audit_complete': True, 'all_tokens_in_vocabulary': payload['all_tokens_in_vocabulary'], 'positive_witnesses_no_unknown': payload['six_positive_overlap_witnesses_have_no_unknown_tokens'],
                      'unknown_tokens': sum(r['unknown_tokens'] for r in rows)}))
    if not payload['six_positive_overlap_witnesses_have_no_unknown_tokens']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
