#!/usr/bin/env python3
"""Independently verify every retained text line and tensor in filtered copies.

Contiguous retained runs permit bytewise tensor comparison without millions of
Python/Torch indexing calls. This verifies the complete copy, not a sample.
Chemical split admission and trained-model provenance are separate checks.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_lineage(lineage, source_count):
    retained = lineage['source_index_by_output_row']
    empty = lineage['removed_empty_pair_indices']
    excluded = lineage.get('excluded_by_external_plan_indices', [])
    if any(type(index) is not int or index < 0 or index >= source_count
           for items in (retained, empty, excluded) for index in items):
        raise ValueError('Invalid lineage index')
    for items in (retained, empty, excluded):
        if any(a >= b for a, b in zip(items, items[1:])):
            raise ValueError('Lineage must be strictly increasing without duplicates')
    dropped = set(empty) | set(excluded)
    if len(dropped) != len(empty) + len(excluded):
        raise ValueError('Overlapping exclusion reasons in lineage')
    if len(retained) + len(dropped) != source_count or any(index in dropped for index in retained):
        raise ValueError('Lineage is not an exhaustive disjoint source partition')
    return retained, dropped


def contiguous_runs(indices):
    if not indices:
        return
    start, previous = indices[0], indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            yield start, previous + 1
            start = value
        previous = value
    yield start, previous + 1


def compare_binary(old, new, retained):
    import numpy as np
    if len(new) != len(retained) or not retained:
        raise ValueError('Invalid rebuilt binary length')
    if old._index.dtype != new._index.dtype:
        raise ValueError('Unexpected tensor storage dtype change')
    itemsize = np.dtype(new._index.dtype).itemsize
    for data in (old, new):
        if data._index._pointers[0] != 0 or not np.array_equal(
                np.diff(data._index._pointers), data.sizes[:-1].astype(np.int64) * itemsize):
            raise ValueError('Noncontiguous binary tensor pointer index')
        if len(data._bin_buffer) != int(data.sizes.sum(dtype=np.int64)) * itemsize:
            raise ValueError('Binary token storage length differs from pointer index')
    if not np.array_equal(new.sizes, old.sizes[np.asarray(retained, dtype=np.int64)]) or (new.sizes <= 1).any():
        raise ValueError('Retained tensor sizes differ or include empty sequences')
    cursor = 0
    for start, end in contiguous_runs(retained):
        old_start = int(old._index._pointers[start])
        old_end = int(old._index._pointers[end - 1]) + int(old.sizes[end - 1]) * itemsize
        new_start = int(new._index._pointers[cursor])
        size = old_end - old_start
        for offset in range(0, size, 8 * 1024**2):
            count = min(size - offset, 8 * 1024**2)
            left = np.frombuffer(old._bin_buffer, dtype=np.uint8, count=count, offset=old_start + offset)
            right = np.frombuffer(new._bin_buffer, dtype=np.uint8, count=count, offset=new_start + offset)
            if not np.array_equal(left, right):
                raise ValueError('A retained tensor changed')
        cursor += end - start
    if cursor != len(new):
        raise ValueError('Retained tensor scan is incomplete')


def verify_dataset(entry):
    import sys
    sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
    from fairseq.data import indexed_dataset
    destination = Path(entry['path'])
    manifest_path = destination / 'manifest.json'
    if sha(manifest_path) != entry['manifest_sha256']:
        raise ValueError('Copy manifest changed')
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['source'])
    checked = {}
    for folder, hashes in ((source, manifest['source_sha256']), (destination, manifest['output_sha256'])):
        for name, expected in hashes.items():
            path = (folder / name).resolve()
            path.relative_to(folder.resolve())
            if sha(path) != expected:
                raise ValueError('Copy verification hash mismatch: ' + str(path))
            checked[path] = expected
    results = []
    for row in manifest['splits']:
        split = row['split']
        count = row['original_pairs']
        with gzip.open(destination / (split + '.lineage.json.gz'), 'rt') as handle:
            lineage = json.load(handle)
        retained, dropped = check_lineage(lineage, count)
        if len(retained) != row['retained_pairs'] or \
                len(lineage['removed_empty_pair_indices']) != row['removed_empty_pairs'] or \
                len(lineage.get('excluded_by_external_plan_indices', [])) != row.get('excluded_by_external_plan_pairs', 0):
            raise ValueError('Lineage counts differ from copy manifest')
        if split == 'test' and dropped:
            raise ValueError('Repaired copy changed prepared test membership')
        for side in ('src', 'tgt'):
            dictionary = 'data-bin/dict.' + side + '.txt'
            if sha(source / dictionary) != sha(destination / dictionary):
                raise ValueError('Vocabulary changed in an exclusion-only copy')
            with (source / (split + '.' + side)).open('rb') as old_text, \
                 (destination / (split + '.' + side)).open('rb') as new_text:
                n = 0
                for index, line in enumerate(old_text):
                    if index not in dropped and (not line.strip() or line != new_text.readline()):
                        raise ValueError('Retained text changed or is empty')
                    n += 1
                if n != count or new_text.read(1):
                    raise ValueError('Text row count differs from exhaustive lineage')
            prefix = ('valid' if split == 'val' else split) + '.src-tgt.' + side
            old = indexed_dataset.make_dataset(str(source / 'data-bin' / prefix), 'mmap')
            new = indexed_dataset.make_dataset(str(destination / 'data-bin' / prefix), 'mmap')
            if old is None or new is None or len(old) != count:
                raise ValueError('Missing or inconsistent source binary')
            compare_binary(old, new, retained)
        results.append({'split': split, 'source_pairs': count, 'retained_pairs': len(retained),
                        'removed_pairs': len(dropped), 'all_retained_text_and_tensors_identical': True})
    if any(sha(path) != expected for path, expected in checked.items()) or sha(manifest_path) != entry['manifest_sha256']:
        raise ValueError('Artifacts changed during independent verification')
    return {'dataset': manifest['dataset'], 'manifest_sha256': entry['manifest_sha256'], 'splits': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    summary_path = args.input_root / 'summary.json'
    summary_hash = sha(summary_path)
    summary = json.loads(summary_path.read_text())
    if summary.get('preparatory_copy_complete') is not True:
        raise ValueError('Copy is incomplete')
    entries = summary.get('families') or [summary['dataset']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(verify_dataset, entries))
    if sha(summary_path) != summary_hash:
        raise ValueError('Copy summary changed during verification')
    output = args.input_root / 'independent_copy_verification.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({'pass': True, 'datasets': rows, 'summary_sha256': summary_hash,
        'verifier_sha256': sha(Path(__file__)), 'training_admission_eligible': False,
        'scope': 'exhaustive row partition, unchanged dictionaries, every retained text line and tensor byte',
        'remaining_scope': 'post-transformation chemical membership audits and trained-model admission'}, indent=2) + '\n')
    print(json.dumps({'pass': True, 'verified_datasets': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
