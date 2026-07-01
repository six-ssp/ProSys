"""Backfill atom-mapped reactions for Stage 1 family route datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rxnmapper import RXNMapper


def map_reactions(rows: list[dict[str, str]], batch_size: int) -> list[dict[str, str]]:
    mapper = RXNMapper()
    output_rows: list[dict[str, str]] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        reactions = [row['reactants>reagents>production'] for row in batch]
        mapped = mapper.get_attention_guided_atom_maps(reactions)
        for row, result in zip(batch, mapped):
            out = dict(row)
            out['mapped_reaction_smiles'] = result.get('mapped_rxn', '')
            out['mapping_confidence'] = str(result.get('confidence', 0.0))
            output_rows.append(out)

    return output_rows


def remap_csv(path: Path, batch_size: int) -> None:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if 'mapped_reaction_smiles' not in fieldnames:
        fieldnames.append('mapped_reaction_smiles')
    if 'mapping_confidence' not in fieldnames:
        fieldnames.append('mapping_confidence')

    mapped_rows = map_reactions(rows, batch_size=batch_size)

    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapped_rows)

    print(f'remapped {path} ({len(mapped_rows)} rows)')


def main() -> None:
    parser = argparse.ArgumentParser(description='Backfill RXNMapper outputs into REAXYS Stage 1 route CSV files.')
    parser.add_argument(
        '--datasets_root',
        type=str,
        default='data/editretro/datasets',
        help='Root directory containing REAXYS_*_SINGLE_CATMERGE datasets.',
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument(
        '--family',
        type=str,
        default='',
        help='Optional family dataset directory name, e.g. REAXYS_Beckmann_SINGLE_CATMERGE.',
    )
    args = parser.parse_args()

    root = Path(args.datasets_root)
    if args.family:
        dataset_dirs = [root / args.family]
    else:
        dataset_dirs = sorted(root.glob('REAXYS_*_SINGLE_CATMERGE'))

    for dataset_dir in dataset_dirs:
        for split in ['train', 'val']:
            path = dataset_dir / 'raw' / f'raw_{split}.csv'
            if path.exists():
                remap_csv(path, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
