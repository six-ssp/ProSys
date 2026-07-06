"""CLI entry for building ProSys Stage 2 V2 product-memory artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from stage2.v2.product_memory import build_product_memory_artifacts, infer_family_name


def default_train_file(family_dir: str | Path) -> Path:
    return Path(family_dir) / 'For_second_part_model' / 'Splitted_second_train_labels_processed.txt'


def default_output_dir(family_dir: str | Path) -> Path:
    return Path(family_dir) / 'stage2_v2' / 'product_memory'


def main() -> None:
    parser = argparse.ArgumentParser(description='Build ProSys Stage 2 V2 product-memory artifacts.')
    parser.add_argument(
        '--family_dir',
        type=str,
        required=True,
        help='Path to data/reaction_processed_{family}_catmerge',
    )
    parser.add_argument(
        '--train_file',
        type=str,
        default=None,
        help='Optional override for the family train split file.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Optional override for the product-memory output directory.',
    )
    parser.add_argument('--n_bits', type=int, default=2048, help='Morgan fingerprint length.')
    parser.add_argument('--radius', type=int, default=2, help='Morgan fingerprint radius.')
    args = parser.parse_args()

    family_dir = Path(args.family_dir)
    train_file = Path(args.train_file) if args.train_file else default_train_file(family_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(family_dir)

    artifacts = build_product_memory_artifacts(
        train_file=train_file,
        output_dir=output_dir,
        n_bits=args.n_bits,
        radius=args.radius,
        family=infer_family_name(train_file),
    )

    for name, path in artifacts.items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
