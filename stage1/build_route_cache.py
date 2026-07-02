"""Build a Stage 1 route cache for end-to-end (Non-Oracle) ProSys evaluation.

For a family, this generates top-k retrosynthesis routes from the trained
EditRetro checkpoint for every test product (from the Stage 2 gold split), then
aggregates the test-time-augmented beams into ranked unique reactant sets with
scores. The result is a route cache consumed by the Stage 2 Non-Oracle pipeline
(``stage2.v2.candidate_pool.load_route_records_from_cache``).

Generation uses ``fairseq-interactive`` (raw product SMILES in, augmentation +
SPE tokenization done internally by the vendored fairseq CLI, which must run
with cwd=stage1/). Aggregation reuses ``stage1/utils/get_ranked_topk.py``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

# Reuse the released aggregation logic.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'utils'))
from get_ranked_topk import canonicalize_smiles_clear_map, compute_rank, process_input  # noqa: E402


def dataset_name(family: str) -> str:
    return f'REAXYS_{family}_SINGLE_CATMERGE'


def resolve_checkpoint(repo_root: Path, family: str, override: str | None) -> Path:
    if override:
        return Path(override)
    finetune_root = repo_root / 'stage1' / 'results' / 'family_finetune' / dataset_name(family)
    candidates = sorted(finetune_root.glob('*/checkpoints/checkpoint_best.pt'))
    if not candidates:
        raise FileNotFoundError(f'no checkpoint_best.pt under {finetune_root}')
    return candidates[-1]


def load_test_reactions(gold_split_file: Path) -> list[dict]:
    """Ordered unique (reaction_id, product) test reactions from the gold split.

    Mirrors the deduplication used by ``load_route_records_from_split`` so the
    Non-Oracle slates line up with the Oracle ones.
    """
    reactions: list[dict] = []
    seen: set[tuple[str, str]] = set()
    with open(gold_split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            reaction_id, reactants, product = parts[0], parts[1], parts[2]
            key = (reaction_id, product)
            if key in seen:
                continue
            seen.add(key)
            reactions.append(
                {
                    'sample_index': len(reactions),
                    'reaction_id': reaction_id,
                    'product': product,
                    'gold_reactants': reactants,
                }
            )
    return reactions


def run_interactive(
    *,
    repo_root: Path,
    databin: Path,
    checkpoint: Path,
    input_file: Path,
    output_file: Path,
    aug: int,
    topk: int,
    repos_beam: int,
    token_beam: int,
    mask_beam: int,
    device: str,
    batch_size: int,
    buffer_size: int,
    max_tokens: int,
) -> None:
    stage1_dir = repo_root / 'stage1'
    cmd = [
        'fairseq-interactive',
        '--user-dir', 'editretro',
        str(databin),
        '-s', 'src', '-t', 'tgt',
        '--input', str(input_file),
        '--task', 'translation_retro',
        '--path', str(checkpoint),
        '--iter-decode-max-iter', '10',
        '--iter-decode-eos-penalty', '0',
        '--beam', '1', '--remove-bpe',
        '--init-src',
        '--buffer-size', str(buffer_size),
        '--batch-size', str(batch_size),
        '--max-tokens', str(max_tokens),
        '--TOPK', str(topk),
        '--inference-with-augmentation', '--aug', str(aug),
        '--repos-beam', str(repos_beam),
        '--mask-beam', str(mask_beam),
        '--token-beam', str(token_beam),
        '--print-step',
    ]
    import os

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = device
    env.setdefault('OMP_NUM_THREADS', '8')
    with output_file.open('w', encoding='utf-8') as out:
        subprocess.run(cmd, cwd=str(stage1_dir), env=env, stdout=out, check=True)


def aggregate_routes(
    generation_file: Path,
    *,
    num_reactions: int,
    aug: int,
    beam_size: int,
    n_best: int,
    score_alpha: float,
    processes: int,
) -> list[list[tuple[str, float]]]:
    """Return, per reaction (in order), a ranked list of (reactants, score)."""
    predictions, raw_scores, _ = process_input(str(generation_file), False)

    expected = num_reactions * aug * beam_size
    if len(predictions) < expected:
        raise ValueError(
            f'generation has {len(predictions)} hypotheses, expected {expected} '
            f'({num_reactions} reactions x {aug} aug x {beam_size} beam)'
        )
    predictions = predictions[:expected]
    raw_scores = raw_scores[:expected]

    pool = multiprocessing.Pool(processes=processes)
    canon = pool.map(func=canonicalize_smiles_clear_map, iterable=predictions)
    pool.close()
    pool.join()

    grouped_pred: list[list[list]] = [[[] for _ in range(aug)] for _ in range(num_reactions)]
    grouped_score: list[list[list]] = [[[] for _ in range(aug)] for _ in range(num_reactions)]
    stride = aug * beam_size
    for i, (pred, score) in enumerate(zip(canon, raw_scores)):
        r = i // stride
        a = (i % stride) // beam_size
        grouped_pred[r][a].append(pred)
        grouped_score[r][a].append(score)

    ranked_per_reaction: list[list[tuple[str, float]]] = []
    for r in range(num_reactions):
        rank, _ = compute_rank(
            grouped_pred[r], grouped_score[r], alpha=score_alpha, beam_size=beam_size
        )
        ranked = sorted(rank.items(), key=lambda kv: kv[1], reverse=True)[:n_best]
        # key is (canonical_smiles, max_frag); keep non-empty predictions only
        ranked_per_reaction.append([(key[0], float(score)) for key, score in ranked if key[0]])
    return ranked_per_reaction


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a Stage 1 route cache for Non-Oracle evaluation.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--family', type=str, required=True, help='e.g. Beckmann')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gold_split', type=str, default=None)
    parser.add_argument('--databin', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--aug', type=int, default=10)
    parser.add_argument('--topk', type=int, default=10, help='beams per augmentation (=repos*mask*token)')
    parser.add_argument('--repos_beam', type=int, default=5)
    parser.add_argument('--token_beam', type=int, default=2)
    parser.add_argument('--mask_beam', type=int, default=1)
    parser.add_argument('--n_best', type=int, default=10, help='top-k unique routes to keep per product')
    parser.add_argument('--score_alpha', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=2000)
    parser.add_argument('--max_tokens', type=int, default=4000)
    parser.add_argument('--max_products', type=int, default=None, help='limit test products (smoke)')
    parser.add_argument('--processes', type=int, default=min(16, multiprocessing.cpu_count()))
    parser.add_argument('--skip_generation', action='store_true', help='reuse existing generation.txt')
    parser.add_argument(
        '--generation_file',
        type=str,
        default=None,
        help='optional path to an existing generation.txt to reuse with --skip_generation',
    )
    args = parser.parse_args()

    if args.repos_beam * args.mask_beam * args.token_beam != args.topk:
        raise ValueError('repos_beam * mask_beam * token_beam must equal topk')

    repo_root = Path(args.repo_root).resolve()
    ds = dataset_name(args.family)
    checkpoint = resolve_checkpoint(repo_root, args.family, args.checkpoint)
    gold_split = Path(args.gold_split) if args.gold_split else (
        repo_root / 'data' / f'reaction_processed_{args.family}_catmerge'
        / 'For_second_part_model' / 'Splitted_second_test_labels_processed.txt'
    )
    databin = Path(args.databin) if args.databin else (
        repo_root / 'data' / 'editretro' / 'datasets' / ds / 'aug10' / 'data-bin'
    )
    output_dir = Path(args.output) if args.output else (repo_root / 'outputs' / 'stage1_routes' / args.family)
    output_dir = output_dir.resolve()  # subprocess runs with cwd=stage1/, so paths must be absolute
    output_dir.mkdir(parents=True, exist_ok=True)

    reactions = load_test_reactions(gold_split)
    if args.max_products is not None:
        reactions = reactions[:args.max_products]
    if not reactions:
        raise ValueError(f'no test reactions found in {gold_split}')

    input_file = output_dir / 'input_products.txt'
    input_file.write_text('\n'.join(r['product'] for r in reactions) + '\n', encoding='utf-8')

    if args.generation_file:
        if not args.skip_generation:
            raise ValueError('--generation_file is only supported together with --skip_generation')
        generation_file = Path(args.generation_file).resolve()
    else:
        generation_file = output_dir / 'generation.txt'

    if args.skip_generation and not generation_file.exists():
        raise FileNotFoundError(f'generation file not found: {generation_file}')

    if not args.skip_generation:
        print(f'[route_cache] {args.family}: generating routes for {len(reactions)} products '
              f'(checkpoint={checkpoint.name})')
        run_interactive(
            repo_root=repo_root, databin=databin, checkpoint=checkpoint,
            input_file=input_file, output_file=generation_file,
            aug=args.aug, topk=args.topk, repos_beam=args.repos_beam,
            token_beam=args.token_beam, mask_beam=args.mask_beam, device=args.device,
            batch_size=args.batch_size, buffer_size=args.buffer_size, max_tokens=args.max_tokens,
        )

    ranked = aggregate_routes(
        generation_file, num_reactions=len(reactions), aug=args.aug,
        beam_size=args.topk, n_best=args.n_best, score_alpha=args.score_alpha,
        processes=args.processes,
    )

    for reaction, routes in zip(reactions, ranked):
        total = sum(score for _, score in routes) or 1.0
        reaction['routes'] = [
            {
                'reactants': smiles,
                'retro_rank': rank + 1,
                'retro_score': score,
                'retro_probability': score / total,
            }
            for rank, (smiles, score) in enumerate(routes)
        ]

    cache = {
        'family': args.family,
        'checkpoint': str(checkpoint),
        'generation_file': str(generation_file),
        'aug': args.aug,
        'topk': args.topk,
        'n_best': args.n_best,
        'num_reactions': len(reactions),
        'reactions': reactions,
    }
    cache_file = output_dir / 'route_cache.json'
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    with_routes = sum(1 for r in reactions if r['routes'])
    print(f'[route_cache] {args.family}: wrote {cache_file} '
          f'({with_routes}/{len(reactions)} reactions have >=1 route)')


if __name__ == '__main__':
    main()
