#!/usr/bin/env python3
"""Create unpromoted, nonempty expert inputs without altering frozen datasets."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import gzip
from itertools import zip_longest
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stage1_multiseed import sha, write


def fairseq_data():
    sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
    from fairseq.data import Dictionary, indexed_dataset
    return Dictionary, indexed_dataset


def check_databin(path):
    _, indexed = fairseq_data()
    rows = {}
    for split in ('train', 'valid'):
        counts = []
        for side in ('src', 'tgt'):
            dataset = indexed.make_dataset(str(Path(path) / f'{split}.src-tgt.{side}'), 'mmap')
            if dataset is None or len(dataset) == 0:
                raise ValueError(f'Missing/empty {split} {side} dataset')
            if (dataset.sizes <= 1).any():
                raise ValueError(f'EOS-only/empty {split} {side} sequences; use versioned clean inputs')
            counts.append(len(dataset))
        if counts[0] != counts[1]:
            raise ValueError('Source/target binary lengths differ')
        rows[split] = counts[0]
    return rows


def build_one(source, destination, expected_hashes=None, expected_empty=None):
    with ExitStack() as stack:
        return _build_one(source, destination, expected_hashes, expected_empty, stack)


def _build_one(source, destination, expected_hashes, expected_empty, stack):
    import torch
    Dictionary, indexed = fairseq_data()
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    working = destination.with_name(destination.name + '.partial')
    working.mkdir()
    output_bin = working / 'data-bin'
    output_bin.mkdir()
    splits = [s for s in ('train', 'val', 'test') if (source / f'{s}.src').exists()]
    files = [source / f'{s}.{side}' for s in splits for side in ('src', 'tgt')]
    files += sorted(p for p in (source / 'data-bin').iterdir() if p.is_file())
    original = {str(p.relative_to(source)): sha(p) for p in files}
    if expected_hashes is not None:
        for p in files:
            if expected_hashes.get(str(p.relative_to(ROOT))) != original[str(p.relative_to(source))]:
                raise ValueError('Input no longer matches independent empty-pair audit: ' + str(p))
    dictionaries = {}
    for side in ('src', 'tgt'):
        path = source / 'data-bin' / f'dict.{side}.txt'
        dictionaries[side] = Dictionary.load(str(path))
        shutil.copy2(path, output_bin / path.name)
    statistics = []
    for split in splits:
        binary_split = 'valid' if split == 'val' else split
        prefixes = {s: f'{binary_split}.src-tgt.{s}' for s in ('src', 'tgt')}
        inputs = {s: indexed.make_dataset(str(source / 'data-bin' / prefixes[s]), 'mmap') for s in prefixes}
        if any(value is None for value in inputs.values()):
            raise FileNotFoundError('Input binary missing')
        builders = {s: indexed.make_builder(str(output_bin / (prefixes[s] + '.bin')), 'mmap',
                                           vocab_size=len(dictionaries[s])) for s in prefixes}
        # Close without finalizing/publishing a partial dataset if a check fails.
        for builder in builders.values():
            stack.callback(builder._data_file.close)
        retained, removed, n = [], [], 0
        with (source / f'{split}.src').open() as src, (source / f'{split}.tgt').open() as tgt, \
             (working / f'{split}.src').open('w') as out_src, (working / f'{split}.tgt').open('w') as out_tgt:
            for index, pair in enumerate(zip_longest(src, tgt)):
                if any(line is None for line in pair):
                    raise ValueError('Input paired text lengths differ')
                for side, line in zip(('src', 'tgt'), pair):
                    encoded = dictionaries[side].encode_line(line, add_if_not_exist=False)
                    if index >= len(inputs[side]) or not torch.equal(encoded.long(), inputs[side][index].long()):
                        raise ValueError(f'Input text/bin mismatch: {split}/{side}/{index}')
                if all(line.strip() for line in pair):
                    retained.append(index)
                    out_src.write(pair[0])
                    out_tgt.write(pair[1])
                    for side in prefixes:
                        builders[side].add_item(inputs[side][index])
                else:
                    removed.append(index)
                n += 1
                if n % 20000 == 0:
                    write(working / 'progress.json', {'split': split, 'input_rows_checked': n})
        if any(len(value) != n for value in inputs.values()):
            raise ValueError('Input binary contains extra rows')
        if expected_empty is not None and removed != expected_empty[split]:
            raise ValueError('Exclusion indices differ from independent audit')
        if not retained:
            raise ValueError('No nonempty paired sequences remain')
        for side in prefixes:
            builders[side].finalize(str(output_bin / (prefixes[side] + '.idx')))
            rebuilt = indexed.make_dataset(str(output_bin / prefixes[side]), 'mmap')
            if len(rebuilt) != len(retained):
                raise ValueError('Rebuilt binary length mismatch')
            for output_index, input_index in enumerate(retained):
                if not torch.equal(rebuilt[output_index], inputs[side][input_index]):
                    raise ValueError('A retained token tensor changed')
        with gzip.open(working / f'{split}.lineage.json.gz', 'wt') as handle:
            json.dump({'source_index_by_output_row': retained, 'removed_empty_pair_indices': removed,
                       'index_space': 'retained augmented text/bin rows, not original raw reactions'}, handle)
        statistics.append({'split': split, 'original_pairs': n, 'removed_empty_pairs': len(removed),
                           'retained_pairs': len(retained), 'all_input_text_bin_pairs_equal': True,
                           'all_retained_token_tensors_unchanged': True})
    check_databin(output_bin)
    for name, digest in original.items():
        if sha(source / name) != digest:
            raise ValueError('Source changed while building: ' + name)
    outputs = {str(p.relative_to(working)): sha(p) for p in working.rglob('*')
               if p.is_file() and p.name != 'progress.json'}
    manifest = {'source': str(source), 'dataset': source.parent.name, 'splits': statistics,
                'source_sha256': original, 'output_sha256': outputs,
                'validation_policy': 'original reaction membership, empty augmented pairs removed only',
                'protocol_selection_pending': True, 'ready_for_formal_three_seed_training': False,
                'raw_splits_and_test_query_manifest_modified': False, 'builder_sha256': sha(Path(__file__))}
    write(working / 'manifest.json', manifest)
    working.rename(destination)
    return {'dataset': source.parent.name, 'path': str(destination), 'splits': statistics,
            'manifest_sha256': sha(destination / 'manifest.json')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check-databin', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'Experiment/project_completion_20260913/stage1_nonempty_inputs')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if args.check_databin:
        print(json.dumps(check_databin(args.check_databin)))
        return
    if args.workers < 1:
        p.error('--workers must be positive')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; never overwrite prepared inputs')
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError('Less than 5 GiB free')
    args.output.mkdir(parents=True)
    audit_path = ROOT / 'Experiment/project_completion_20260913/empty_stage1_pairs/audit.json'
    audit = json.loads(audit_path.read_text())
    sources = sorted((ROOT / 'data/editretro/datasets').glob('REAXYS*/aug10'))
    expected = {source.parent.name: {r['split']: r['empty_pair_indices'] for r in audit['rows']
                                    if r['dataset'] == source.parent.name} for source in sources}
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(build_one, source, args.output / 'artifacts' / source.parent.name,
                               audit['input_sha256'], expected[source.parent.name]) for source in sources]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            write(args.output / 'progress.json', {'completed_families': [r['dataset'] for r in results]})
            print('Verified', row['dataset'], flush=True)
    write(args.output / 'summary.json', {'preparatory_copy_complete': len(results) == 6,
          'protocol_selection_pending': True, 'families': sorted(results, key=lambda r: r['dataset']),
          'independent_source_audit_sha256': sha(audit_path), 'workers': args.workers,
          'training_launched': False, 'original_inputs_modified': False})


if __name__ == '__main__':
    main()
