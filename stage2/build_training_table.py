"""CLI for building ProSys Stage 2 V2 route-context training tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from stage2.v2.training_table import write_candidate_training_table


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a ProSys Stage 2 V2 training table from candidate-pool rows.')
    parser.add_argument('--candidate_pool_file', type=str, required=True, help='Stage 2A candidate pool CSV')
    parser.add_argument('--gold_split_file', type=str, required=True, help='Gold split TXT with route/context labels')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV path')
    args = parser.parse_args()

    output_path = write_candidate_training_table(
        candidate_pool_file=args.candidate_pool_file,
        gold_split_file=args.gold_split_file,
        output_file=args.output_file,
    )
    print(f'output: {Path(output_path)}')


if __name__ == '__main__':
    main()
