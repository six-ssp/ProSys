"""Rebuild Stage 1 family route datasets from existing Stage 2 splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.preprocess import REACTION_TYPES, collect_stage1_route_data


def main() -> None:
    parser = argparse.ArgumentParser(description='Rebuild Stage 1 family route datasets.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to data/reaxys_input')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to data')
    parser.add_argument(
        '--families',
        type=str,
        default='all',
        help='Comma-separated family names or "all".',
    )
    args = parser.parse_args()

    if args.families == 'all':
        families = REACTION_TYPES
    else:
        families = [family.strip() for family in args.families.split(',') if family.strip()]

    for family in families:
        stage2_root = Path(args.output_dir) / f'reaction_processed_{family}_catmerge'
        print(f'=== rebuild Stage 1 route data: {family} ===')
        collect_stage1_route_data(
            rtype=family,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            stage2_root=str(stage2_root),
        )


if __name__ == '__main__':
    main()
