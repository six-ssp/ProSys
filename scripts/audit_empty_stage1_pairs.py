#!/usr/bin/env python3
"""Identify EOS-only Stage-1 targets and verify them in retained binary inputs."""

from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'Experiment/project_completion_20260913/empty_stage1_pairs'


def main():
    from scripts.run_stage1_multiseed import write, sha
    sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
    from fairseq.data import Dictionary, indexed_dataset
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'audit.json').exists():
        raise FileExistsError(OUTPUT / 'audit.json')
    roots = sorted((ROOT / 'data/editretro/datasets').glob('REAXYS*/aug10'))
    roots.append(ROOT / 'data/editretro/datasets/USPTO_STAGE2_FILTERED/aug10')
    results, paths = [], [Path(__file__)]
    for root in roots:
        dictionaries = {s: Dictionary.load(str(root / 'data-bin' / f'dict.{s}.txt'))
                        for s in ('src', 'tgt')}
        for split in ('train', 'val', 'test'):
            source, target = root / f'{split}.src', root / f'{split}.tgt'
            if not source.exists():
                continue
            bin_split = 'valid' if split == 'val' else split
            bins = {s: indexed_dataset.make_dataset(str(root / 'data-bin' / f'{bin_split}.src-tgt.{s}'), 'mmap')
                    for s in ('src', 'tgt')}
            if any(b is None for b in bins.values()):
                raise FileNotFoundError('Missing binaries under ' + str(root))
            bad, counts, n = [], {'src': 0, 'tgt': 0}, 0
            with source.open() as a, target.open() as b:
                for index, (src, tgt) in enumerate(zip_longest(a, b)):
                    if src is None or tgt is None:
                        raise ValueError('Paired text lengths differ')
                    empty = [s for s, line in (('src', src), ('tgt', tgt)) if not line.strip()]
                    if empty:
                        for side in empty:
                            tensor = bins[side][index]
                            if len(tensor) != 1 or int(tensor[0]) != dictionaries[side].eos():
                                raise ValueError('Empty text does not match EOS-only binary target')
                            counts[side] += 1
                        bad.append(index)
                    n += 1
            if any(len(b) != n for b in bins.values()):
                raise ValueError('Text/bin lengths differ')
            name = root.parent.name
            row = {'dataset': name, 'split': split, 'rows': n, 'empty_pair_indices': bad,
                   'empty_pairs': len(bad), 'empty_sides': counts, 'remaining_rows': n - len(bad),
                   'empty_side_binary_eos_verified': True}
            results.append(row)
            write(OUTPUT / f'{name}_{split}.json', row)
            paths += [source, target]
            paths += sorted(p for p in (root / 'data-bin').iterdir()
                            if p.name.startswith((bin_split + '.', 'dict.')))
            print(name, split, len(bad), '/', n, flush=True)
    write(OUTPUT / 'audit.json', {'scope': 'retained augmented text plus binary EOS-only membership checks',
          'rows': results, 'data_modified': False,
          'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
          'cause': 'Root alignment may select no reactant component for an expanded product; original export accepted empty token targets.',
          'historical_training_log_linkage': 'Paths agree with retained training logs; no absent historical input hash is reconstructed.'})


if __name__ == '__main__':
    main()
