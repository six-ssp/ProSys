#!/usr/bin/env python3
"""Independently verify prepared-copy lineage, hashes and every retained tensor."""

from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stage1_multiseed import sha, write
STUDY = ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs'


def verify_family(row):
    import torch
    sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
    from fairseq.data import indexed_dataset
    destination = Path(row['path'])
    assert sha(destination / 'manifest.json') == row['manifest_sha256']
    manifest = json.loads((destination / 'manifest.json').read_text())
    source = Path(manifest['source'])
    assert manifest['protocol_selection_pending'] is True
    assert manifest['ready_for_formal_three_seed_training'] is False
    for relative, digest in manifest['source_sha256'].items():
        assert sha(source / relative) == digest, relative
    for relative, digest in manifest['output_sha256'].items():
        assert sha(destination / relative) == digest, relative
    audit = json.loads((ROOT / 'Experiment/project_completion_20260913/empty_stage1_pairs/audit.json').read_text())
    expected = {r['split']: r for r in audit['rows'] if r['dataset'] == row['dataset']}
    counts = {}
    for split, reference in expected.items():
        binary_split = 'valid' if split == 'val' else split
        with gzip.open(destination / f'{split}.lineage.json.gz', 'rt') as handle:
            lineage = json.load(handle)
        removed = set(reference['empty_pair_indices'])
        retained = [i for i in range(reference['rows']) if i not in removed]
        assert lineage['source_index_by_output_row'] == retained
        assert lineage['removed_empty_pair_indices'] == reference['empty_pair_indices']
        for side in ('src', 'tgt'):
            prefix = f'{binary_split}.src-tgt.{side}'
            old = indexed_dataset.make_dataset(str(source / 'data-bin' / prefix), 'mmap')
            new = indexed_dataset.make_dataset(str(destination / 'data-bin' / prefix), 'mmap')
            assert len(new) == len(retained)
            assert (new.sizes > 1).all()
            for i, source_index in enumerate(retained):
                assert torch.equal(new[i], old[source_index]), (split, side, i)
            with (destination / f'{split}.{side}').open() as handle:
                lines = 0
                for line in handle:
                    assert line.strip()
                    lines += 1
                assert lines == len(retained)
        counts[split] = len(retained)
    return {'dataset': row['dataset'], 'verified_pairs': counts,
            'hashes_and_full_retained_tensor_lineage_pass': True}


def main():
    summary = json.loads((STUDY / 'summary.json').read_text())
    assert len(summary['families']) == 6 and summary['preparatory_copy_complete']
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn')) as pool:
        rows = list(pool.map(verify_family, summary['families']))
    snapshot = json.loads((STUDY / 'builder_source_at_creation.json').read_text())
    import hashlib
    assert hashlib.sha256(snapshot['source_utf8'].encode()).hexdigest() == snapshot['sha256']
    for family in summary['families']:
        manifest = json.loads((Path(family['path']) / 'manifest.json').read_text())
        assert manifest['builder_sha256'] == snapshot['sha256']
    write(STUDY / 'independent_verification.json', {'pass': True, 'families': rows,
          'summary_sha256': sha(STUDY / 'summary.json'), 'creation_source_snapshot_verified': True,
          'scope': 'nonempty-copy correctness only; validation protocol not selected; no trained-model claim'})
    print(json.dumps({'pass': True, 'families': len(rows), 'protocol_selection_pending': True}))


if __name__ == '__main__':
    main()
