"""Summarize ProSys hit-rate results across all families.

Reads:
- Stage 2 V2 Oracle evaluation JSONs: ``outputs/stage2_v2/<family>/train/eval_oracle_test.json``
  (system / context / route top-k, pool coverage, temperature).
- Stage 1 family finetune logs: ``stage1/results/family_finetune/<dataset>/<ts>/train.log``
  (best validation loss / perplexity as a convergence proxy).

Prints a per-family table plus a macro-average row.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TOPKS = (1, 3, 5, 10)


def _family_from_stage1_dataset(name: str) -> str:
    return name.replace('REAXYS_', '').replace('_SINGLE_CATMERGE', '')


def load_stage2_oracle(output_root: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for eval_file in sorted(output_root.glob('*/train/eval_oracle_test.json')):
        family = eval_file.parents[1].name
        try:
            results[family] = json.loads(eval_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def load_stage1_convergence(results_root: Path) -> dict[str, dict]:
    """Parse the latest train.log per family for best validation loss/ppl."""
    convergence: dict[str, dict] = {}
    if not results_root.is_dir():
        return convergence

    valid_re = re.compile(r'\| valid \|.*?\bloss (\d+\.\d+).*?\bppl (\d+\.\d+).*?num_updates (\d+)')
    for dataset_dir in sorted(results_root.glob('REAXYS_*_SINGLE_CATMERGE')):
        run_dirs = sorted([p for p in dataset_dir.glob('*') if p.is_dir()])
        if not run_dirs:
            continue
        log_file = run_dirs[-1] / 'train.log'
        if not log_file.exists():
            continue
        best_loss = float('inf')
        best_ppl = None
        last_updates = 0
        for line in log_file.read_text(encoding='utf-8', errors='ignore').splitlines():
            match = valid_re.search(line)
            if not match:
                continue
            loss, ppl, updates = float(match.group(1)), float(match.group(2)), int(match.group(3))
            last_updates = updates
            if loss < best_loss:
                best_loss, best_ppl = loss, ppl
        if best_loss < float('inf'):
            convergence[_family_from_stage1_dataset(dataset_dir.name)] = {
                'best_val_loss': best_loss,
                'best_val_ppl': best_ppl,
                'last_updates': last_updates,
            }
    return convergence


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return '-'
    return f'{value * 100:.1f}' if pct else f'{value:.3f}'


def print_stage2_table(stage2: dict[str, dict]) -> None:
    header = ['family', 'N', 'cover'] + [f'sys@{k}' for k in TOPKS] + ['temp_MAE', 'temp±20']
    rows = []
    agg = {f'system_top{k}_all': [] for k in TOPKS}
    agg['pool_coverage'] = []
    agg['temp_mae'] = []

    for family, result in sorted(stage2.items()):
        m = result['metrics']
        temp = m.get('temperature', {})
        rows.append([
            family,
            str(m.get('num_slates', '-')),
            _fmt(m.get('pool_coverage'), pct=True),
            *[_fmt(m.get(f'system_top{k}_all'), pct=True) for k in TOPKS],
            (f"{temp['mae']:.1f}" if temp.get('mae') is not None else '-'),
            (_fmt(temp.get('within_20c'), pct=True) if temp.get('within_20c') is not None else '-'),
        ])
        for k in TOPKS:
            if m.get(f'system_top{k}_all') is not None:
                agg[f'system_top{k}_all'].append(m[f'system_top{k}_all'])
        if m.get('pool_coverage') is not None:
            agg['pool_coverage'].append(m['pool_coverage'])
        if temp.get('mae') is not None:
            agg['temp_mae'].append(temp['mae'])

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    if rows:
        rows.append([
            'MACRO-AVG', '', _fmt(mean(agg['pool_coverage']), pct=True),
            *[_fmt(mean(agg[f'system_top{k}_all']), pct=True) for k in TOPKS],
            (f"{mean(agg['temp_mae']):.1f}" if agg['temp_mae'] else '-'),
            '',
        ])

    widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
    print('=' * (sum(widths) + 3 * len(widths)))
    print('Stage 2 V2 Oracle hit rates (%)')
    print('=' * (sum(widths) + 3 * len(widths)))
    for r in [header] + rows:
        print('   '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(r)))


def print_stage1_table(stage1: dict[str, dict]) -> None:
    if not stage1:
        print('\nStage 1 finetune: no train logs found yet.')
        return
    print('\nStage 1 finetune convergence (best validation)')
    print('-' * 60)
    print('family'.ljust(34), 'best_val_loss'.ljust(14), 'best_ppl'.ljust(10), 'updates')
    for family, info in sorted(stage1.items()):
        print(
            family.ljust(34),
            f"{info['best_val_loss']:.4f}".ljust(14),
            (f"{info['best_val_ppl']:.2f}" if info['best_val_ppl'] is not None else '-').ljust(10),
            info['last_updates'],
        )


def load_stage2_nonoracle(output_root: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for eval_file in sorted(output_root.glob('*/non_oracle/eval_non_oracle_test.json')):
        family = eval_file.parents[1].name
        try:
            results[family] = json.loads(eval_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def print_nonoracle_table(nonoracle: dict[str, dict]) -> None:
    if not nonoracle:
        print('\nStage 2 V2 Non-Oracle: no eval results yet.')
        return
    header = ['family', 'N', 'rr@1', 'rr@10', 'cover'] + [f'sys@{k}' for k in TOPKS] + ['temp_MAE']
    rows = []
    agg = {f'system_top{k}_all': [] for k in TOPKS}
    agg.update({'pool_coverage': [], 'temp_mae': [], 'rr1': [], 'rr10': []})

    for family, result in sorted(nonoracle.items()):
        m = result['metrics']
        rr = result.get('stage1_route_recall', {})
        temp = m.get('temperature', {})
        rows.append([
            family,
            str(m.get('num_slates', '-')),
            _fmt(rr.get('route_recall_top1'), pct=True),
            _fmt(rr.get('route_recall_top10'), pct=True),
            _fmt(m.get('pool_coverage'), pct=True),
            *[_fmt(m.get(f'system_top{k}_all'), pct=True) for k in TOPKS],
            (f"{temp['mae']:.1f}" if temp.get('mae') is not None else '-'),
        ])
        for k in TOPKS:
            if m.get(f'system_top{k}_all') is not None:
                agg[f'system_top{k}_all'].append(m[f'system_top{k}_all'])
        if m.get('pool_coverage') is not None:
            agg['pool_coverage'].append(m['pool_coverage'])
        if temp.get('mae') is not None:
            agg['temp_mae'].append(temp['mae'])
        if rr.get('route_recall_top1') is not None:
            agg['rr1'].append(rr['route_recall_top1'])
        if rr.get('route_recall_top10') is not None:
            agg['rr10'].append(rr['route_recall_top10'])

    def mean(v):
        return sum(v) / len(v) if v else None

    rows.append([
        'MACRO-AVG', '',
        _fmt(mean(agg['rr1']), pct=True), _fmt(mean(agg['rr10']), pct=True),
        _fmt(mean(agg['pool_coverage']), pct=True),
        *[_fmt(mean(agg[f'system_top{k}_all']), pct=True) for k in TOPKS],
        (f"{mean(agg['temp_mae']):.1f}" if agg['temp_mae'] else '-'),
    ])

    widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
    print('\n' + '=' * (sum(widths) + 3 * len(widths)))
    print('Stage 2 V2 Non-Oracle hit rates (%)  [rr = Stage 1 route recall; sys = end-to-end route+context]')
    print('=' * (sum(widths) + 3 * len(widths)))
    for r in [header] + rows:
        print('   '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(r)))


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize ProSys hit-rate results.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--output_root', type=str, default='outputs/stage2_v2')
    parser.add_argument('--stage1_results', type=str, default='stage1/results/family_finetune')
    parser.add_argument('--json_out', type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    stage2 = load_stage2_oracle(repo_root / args.output_root)
    nonoracle = load_stage2_nonoracle(repo_root / args.output_root)
    stage1 = load_stage1_convergence(repo_root / args.stage1_results)

    print_stage2_table(stage2)
    print_nonoracle_table(nonoracle)
    print_stage1_table(stage1)

    if args.json_out:
        payload = {
            'stage2_oracle': stage2,
            'stage2_non_oracle': nonoracle,
            'stage1_convergence': stage1,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
