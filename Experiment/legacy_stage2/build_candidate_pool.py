"""CLI for building Stage 2A candidate tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from stage2.v2.candidate_pool import (
    FNNCandidateGenerator,
    ProductMemoryLookup,
    build_candidate_pool_for_routes,
    load_route_records_from_split,
)


def infer_family_name(family_dir: Path) -> str:
    return family_dir.name.replace('reaction_processed_', '').replace('_catmerge', '')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build ProSys Stage 2A candidate pool table.')
    parser.add_argument('--family_dir', type=str, required=True, help='Path to data/reaction_processed_{family}_catmerge')
    parser.add_argument('--memory_dir', type=str, required=True, help='Path to stage2_v2/product_memory output')
    parser.add_argument('--split_file', type=str, default=None, help='Route or split table to convert into candidate table')
    parser.add_argument('--output_file', type=str, required=True, help='Candidate table CSV output path')
    parser.add_argument('--fnn_checkpoint', type=str, default=None, help='Optional family-specific Stage 2A FNN checkpoint')
    parser.add_argument('--cutoff_solv', type=float, default=0.3)
    parser.add_argument('--cutoff_reag', type=float, default=0.3)
    parser.add_argument('--max_solv', type=int, default=11)
    parser.add_argument('--max_reag', type=int, default=11)
    parser.add_argument('--knn_top_k', type=int, default=10)
    parser.add_argument('--knn_max_contexts', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    family_dir = Path(args.family_dir)
    family = infer_family_name(family_dir)
    split_file = Path(args.split_file) if args.split_file else family_dir / 'For_second_part_model' / 'Splitted_second_validate_labels_processed.txt'

    routes = load_route_records_from_split(split_file, family=family)
    product_lookup = ProductMemoryLookup(args.memory_dir)

    fnn_generator = None
    if args.fnn_checkpoint:
        fnn_generator = FNNCandidateGenerator(
            family_dir=family_dir,
            checkpoint_path=args.fnn_checkpoint,
            cutoff_solv=args.cutoff_solv,
            cutoff_reag=args.cutoff_reag,
            max_solv=args.max_solv,
            max_reag=args.max_reag,
            device=args.device,
        )

    frame = build_candidate_pool_for_routes(
        routes,
        product_lookup,
        fnn_generator=fnn_generator,
        knn_top_k=args.knn_top_k,
        knn_max_contexts=args.knn_max_contexts,
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f'rows: {len(frame)}')
    print(f'output: {output_path}')


if __name__ == '__main__':
    main()
