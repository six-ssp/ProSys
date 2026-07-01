"""Prepare EditRetro text and binarized data for a family-specific route dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare family-specific EditRetro dataset.')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--augmentation', type=int, default=10)
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--repo_root', type=str, default='.')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    preprocess_dir = repo_root / 'stage1' / 'preprocess'
    dataset_root = repo_root / 'data' / 'editretro' / 'datasets' / args.dataset
    raw_dir = dataset_root / 'raw'
    if not raw_dir.is_dir():
        raise FileNotFoundError(f'raw dataset directory not found: {raw_dir}')
    run_env = os.environ.copy()
    run_env['EDITRETRO_DATASETS_ROOT'] = str(repo_root / 'data' / 'editretro' / 'datasets')

    run(
        [
            'python',
            'preprocess_data.py',
            '-dataset', args.dataset,
            '-augmentation', str(args.augmentation),
            '-processes', str(args.processes),
            '-splits', 'train,val,test',
            '-spe',
        ],
        cwd=preprocess_dir,
        env=run_env,
    )

    aug_dir = dataset_root / f'aug{args.augmentation}'
    data_bin = aug_dir / 'data-bin'
    data_bin.mkdir(parents=True, exist_ok=True)

    fairseq_cmd = [
        'fairseq-preprocess',
        '--source-lang', 'src',
        '--target-lang', 'tgt',
        '--trainpref', str(aug_dir / 'train'),
        '--validpref', str(aug_dir / 'val'),
        '--destdir', str(data_bin),
        '--workers', '8',
        '--srcdict', str(preprocess_dir / 'dict.txt'),
        '--tgtdict', str(preprocess_dir / 'dict.txt'),
    ]
    if (aug_dir / 'test.src').exists():
        fairseq_cmd.extend(['--testpref', str(aug_dir / 'test')])

    run(fairseq_cmd, cwd=repo_root / 'stage1')

    print(f'prepared dataset: {args.dataset}')
    print(f'data-bin: {data_bin}')


if __name__ == '__main__':
    main()
