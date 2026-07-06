"""CLI for evaluating the ProSys Stage 2 V2 neural ranker (Oracle / Non-Oracle).

Oracle:      pass a candidate table built from the gold route
             (e.g. outputs/stage2_v2/<family>/training_tables/test.csv).
Non-Oracle:  pass a candidate table built from a Stage 1 route cache and use
             --mode non_oracle. The metric computation is identical; only the
             candidate table differs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2.v2.evaluate import run_stage2_v2_eval


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate the ProSys Stage 2 V2 neural ranker.')
    parser.add_argument('--family_dir', type=str, required=True, help='Path to data/reaction_processed_{family}_catmerge')
    parser.add_argument('--candidate_table', type=str, required=True, help='Candidate table CSV to evaluate (test split)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Trained Stage 2 V2 checkpoint (best_model.pt)')
    parser.add_argument('--mode', type=str, default='oracle', choices=['oracle', 'non_oracle'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--slates_per_batch', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--route_fpsize', type=int, default=4096)
    parser.add_argument('--product_fpsize', type=int, default=2048)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--topks', type=str, default='1,3,5,10')
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    topks = tuple(int(value) for value in args.topks.split(',') if value.strip())

    result = run_stage2_v2_eval(
        family_dir=args.family_dir,
        candidate_table=args.candidate_table,
        checkpoint_path=args.checkpoint,
        device=args.device,
        mode=args.mode,
        slates_per_batch=args.slates_per_batch,
        num_workers=args.num_workers,
        route_fpsize=args.route_fpsize,
        product_fpsize=args.product_fpsize,
        radius=args.radius,
        topks=topks,
        output_file=args.output_file,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
