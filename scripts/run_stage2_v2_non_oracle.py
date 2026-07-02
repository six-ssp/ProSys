"""Run Stage 2 V2 Non-Oracle evaluation from Stage 1 route caches.

For each family this reuses the already-built product memory and the
Oracle-trained Stage 2 checkpoint:

  route_cache.json
    -> load_route_records_from_cache (multiple predicted routes per product)
    -> build_candidate_pool_for_routes (product-memory candidates per route)
    -> write_candidate_training_table (route/context/system labels vs gold split)
    -> run_stage2_v2_eval(mode='non_oracle')  [+ Stage 1 route-recall from the cache]

Prereqs: stage1/build_route_cache.py has produced outputs/stage1_routes/<family>/route_cache.json,
and scripts/run_stage2_v2_family_batch.py has produced the family's memory/ and train/best_model.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stage2.v2.candidate_pool import (
    ProductMemoryLookup,
    build_candidate_pool_for_routes,
    load_route_records_from_cache,
)
from stage2.v2.evaluate import run_stage2_v2_eval
from stage2.v2.features import canonicalize_reaction_side
from stage2.v2.training_table import write_candidate_training_table

TOPKS = (1, 3, 5, 10)


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def stage1_route_recall(route_cache_file: Path, topks=TOPKS) -> dict:
    """Pure Stage 1 route recall: is the gold route among the predicted top-k
    (by retro rank), independent of Stage 2 ranking."""
    cache = json.loads(route_cache_file.read_text(encoding='utf-8'))
    hits = {k: 0 for k in topks}
    n = 0
    for reaction in cache.get('reactions', []):
        gold_key = canonicalize_reaction_side(reaction.get('gold_reactants', ''))
        if not gold_key:
            continue
        n += 1
        routes = sorted(reaction.get('routes', []), key=lambda r: r.get('retro_rank', 1))
        pred_keys = [canonicalize_reaction_side(r['reactants']) for r in routes]
        for k in topks:
            if gold_key in pred_keys[:k]:
                hits[k] += 1
    return {
        'n': n,
        **{f'route_recall_top{k}': (hits[k] / n if n else 0.0) for k in topks},
    }


def run_family(
    family: str,
    repo_root: Path,
    artifact_root: Path,
    result_root: Path,
    route_root: Path,
    device: str,
) -> dict | None:
    family_dir = repo_root / 'data' / f'reaction_processed_{family}_catmerge'
    memory_dir = artifact_root / family / 'memory'
    checkpoint = artifact_root / family / 'train' / 'best_model.pt'
    route_cache = route_root / family / 'route_cache.json'
    gold_split = family_dir / 'For_second_part_model' / 'Splitted_second_test_labels_processed.txt'

    for required in (memory_dir / 'exact_product_memory.csv', checkpoint, route_cache, gold_split):
        if not required.exists():
            print(f'[non_oracle] skip {family}: missing {required}')
            return None

    out_dir = result_root / family / 'non_oracle'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[non_oracle] {family}: building candidate pool from route cache')
    routes = load_route_records_from_cache(route_cache, family=family)
    lookup = ProductMemoryLookup(memory_dir)
    pool = build_candidate_pool_for_routes(routes, lookup, fnn_generator=None)
    candidate_file = out_dir / 'candidate_pool_test.csv'
    pool.to_csv(candidate_file, index=False)

    table_file = out_dir / 'test.csv'
    write_candidate_training_table(candidate_file, gold_split, table_file)

    print(f'[non_oracle] {family}: evaluating with {checkpoint.name}')
    result = run_stage2_v2_eval(
        family_dir=family_dir,
        candidate_table=table_file,
        checkpoint_path=checkpoint,
        device=device,
        mode='non_oracle',
        output_file=out_dir / 'eval_non_oracle_test.json',
    )
    result['stage1_route_recall'] = stage1_route_recall(route_cache)
    (out_dir / 'eval_non_oracle_test.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )

    m = result['metrics']
    rr = result['stage1_route_recall']
    print(
        f"[non_oracle] {family}: route_recall@10={rr['route_recall_top10']:.3f} "
        f"sys@1={m['system_top1_all']:.3f} sys@5={m['system_top5_all']:.3f} "
        f"sys@10={m['system_top10_all']:.3f} coverage={m['pool_coverage']:.3f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ProSys Stage 2 V2 Non-Oracle evaluation.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--output_root', type=str, default='outputs/stage2_v2')
    parser.add_argument(
        '--result_root',
        type=str,
        default=None,
        help='optional output root for experiment results; defaults to --output_root',
    )
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    result_root = (repo_root / args.result_root).resolve() if args.result_root else output_root
    route_root = (repo_root / args.route_root).resolve()

    if args.families == 'all':
        families = sorted(p.name for p in route_root.glob('*') if (p / 'route_cache.json').exists())
    else:
        families = parse_csv_arg(args.families)
    if not families:
        raise ValueError('no families with a route_cache.json found')

    results = {}
    failed = []
    for family in families:
        try:
            result = run_family(family, repo_root, output_root, result_root, route_root, args.device)
        except Exception as exc:  # keep going, but surface the failure loudly
            print(f'[non_oracle] ERROR {family}: {type(exc).__name__}: {exc}')
            failed.append(family)
            continue
        if result is not None:
            results[family] = result

    print(f'[non_oracle] done: {len(results)}/{len(families)} families evaluated')
    if failed:
        print(f'[non_oracle] FAILED families: {", ".join(failed)}')


if __name__ == '__main__':
    main()
