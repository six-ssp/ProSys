#!/usr/bin/env python3
"""Check every original validation/test product before guarded GPU decoding."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_stage1_base_augmented import sha, write
from scripts.stage1_identity_guard import augment, identity
from stage1_retrosynthesis.build_route_cache import load_test_reactions


def inspect(task):
    from rdkit import RDLogger
    from SmilesPE.tokenizer import SPE_Tokenizer
    RDLogger.DisableLog('rdApp.*')
    family, split, path = task
    with (ROOT / 'stage1_retrosynthesis/preprocess/SPE_ChEMBL.txt').open() as handle:
        tokenizer = SPE_Tokenizer(handle)
    vocabulary = {line.rsplit(' ', 1)[0] for line in (ROOT / 'data/editretro/datasets/USPTO_50K_FILTERED/aug10/data-bin/dict.src.txt').read_text().splitlines()}
    vocabulary.update(('<s>', '</s>', '<pad>', '<unk>'))
    queries = load_test_reactions(path)
    changed, normalized, unknown = [], [], 0
    for row in queries:
        tokens, slots = augment(row['product'], 10, tokenizer.tokenize)
        expected = identity(row['product'])
        if len(tokens) != 10 or any(identity(''.join(text.split())) != expected for text in tokens):
            raise ValueError('Guarded product identity mismatch')
        if slots:
            changed.append({'query_index': row['sample_index'], 'replaced_slots': slots})
        if ''.join(tokens[0].split()) != row['product']:
            normalized.append(row['sample_index'])
        unknown += sum(any(token not in vocabulary for token in text.split()) for text in tokens)
    return {'family': family, 'split': split, 'queries': len(queries),
            'augmentation_count': 10 * len(queries), 'replaced_variants': changed,
            'normalized_query_indices': normalized, 'variants_containing_unk': unknown,
            'all_guarded_variants_preserve_identity': True}


def main():
    from prosys_shared.mainline import FAMILY_ORDER, split_file_for_family
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = [(f, s, split_file_for_family(ROOT, f, s)) for f in FAMILY_ORDER for s in ('val', 'test')]
    paths = [Path(__file__), ROOT / 'scripts/stage1_identity_guard.py',
             ROOT / 'stage1_retrosynthesis/preprocess/SPE_ChEMBL.txt',
             ROOT / 'data/editretro/datasets/USPTO_50K_FILTERED/aug10/data-bin/dict.src.txt']
    paths += [path for _, _, path in tasks]
    before = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(inspect, tasks))
    if {name: sha(ROOT / name) for name in before} != before:
        raise ValueError('Preflight inputs changed')
    write(args.output, {'pass': True, 'source_sha256': before, 'results': rows,
        'scope': 'Full original validation/test query lists; ten CPU random variants per query, not all possible serializations. Fixed-vocabulary UNK disclosed, not repaired; no route accuracy computed.'})
    print('Full guarded-query preflight passed:', sum(r['queries'] for r in rows), 'queries', flush=True)


if __name__ == '__main__':
    main()
