"""Batch runner for ProSys Stage 2 V2 family experiments."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

from stage2.v2.candidate_pool import (
    FNNCandidateGenerator,
    ProductMemoryLookup,
    build_candidate_pool_for_routes,
    load_route_records_from_split,
)
from stage2.v2.product_memory import build_product_memory_artifacts
from stage2.v2.training_table import write_candidate_training_table


def infer_family_name_from_dir(family_dir: Path) -> str:
    return family_dir.name.replace('reaction_processed_', '').replace('_catmerge', '')


def parse_csv_arg(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(',') if item.strip()]


def resolve_fnn_checkpoint(pattern: str | None, family: str) -> str | None:
    if not pattern:
        return None
    return pattern.format(family=family)


def memory_artifact_paths(memory_dir: Path) -> dict[str, Path]:
    return {
        'exact_memory': memory_dir / 'exact_product_memory.csv',
        'scaffold_memory': memory_dir / 'scaffold_product_memory.csv',
        'knn_index': memory_dir / 'product_knn_index.npz',
        'metadata': memory_dir / 'product_memory_metadata.json',
    }


def build_candidate_pool_file(
    family_dir: Path,
    memory_dir: Path,
    split_file: Path,
    output_file: Path,
    fnn_checkpoint: str | None,
    device: str | None,
) -> Path:
    family = infer_family_name_from_dir(family_dir)
    routes = load_route_records_from_split(split_file, family=family)
    lookup = ProductMemoryLookup(memory_dir)

    fnn_generator = None
    if fnn_checkpoint:
        fnn_generator = FNNCandidateGenerator(
            family_dir=family_dir,
            checkpoint_path=fnn_checkpoint,
            device=device,
        )

    frame = build_candidate_pool_for_routes(routes, lookup, fnn_generator=fnn_generator)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)
    return output_file


def ensure_product_memory(
    train_split: Path,
    memory_dir: Path,
    family: str,
    *,
    n_bits: int,
    radius: int,
    force: bool,
) -> None:
    artifacts = memory_artifact_paths(memory_dir)
    if not force and all(path.exists() for path in artifacts.values()):
        print(f'[stage2] reuse product memory for {family}: {memory_dir}')
        return

    print(f'[stage2] build product memory for {family}')
    build_product_memory_artifacts(
        train_file=train_split,
        output_dir=memory_dir,
        n_bits=n_bits,
        radius=radius,
        family=family,
    )


def ensure_training_table(
    candidate_pool_file: Path,
    gold_split_file: Path,
    output_file: Path,
    *,
    family: str,
    split_name: str,
    force: bool,
) -> None:
    if not force and output_file.exists():
        print(f'[stage2] reuse training table for {family}/{split_name}: {output_file}')
        return

    print(f'[stage2] build training table for {family}/{split_name}')
    write_candidate_training_table(
        candidate_pool_file=candidate_pool_file,
        gold_split_file=gold_split_file,
        output_file=output_file,
    )


def start_training_process(
    *,
    python_bin: str,
    family_dir: Path,
    family_output: Path,
    table_dir: Path,
    train_device: str,
    epochs: int,
    slates_per_batch: int,
    train_num_workers: int,
    learning_rate: float,
    weight_decay: float,
    max_train_slates: int | None,
    max_val_slates: int | None,
) -> tuple[subprocess.Popen[bytes], object]:
    output_dir = family_output / 'train'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'train.log'
    log_handle = log_file.open('ab')

    cmd = [
        python_bin,
        '-m',
        'stage2.train_stage2_v2',
        '--family_dir',
        str(family_dir),
        '--train_table',
        str(table_dir / 'train.csv'),
        '--val_table',
        str(table_dir / 'val.csv'),
        '--test_table',
        str(table_dir / 'test.csv'),
        '--output_dir',
        str(output_dir),
        '--epochs',
        str(epochs),
        '--slates_per_batch',
        str(slates_per_batch),
        '--num_workers',
        str(train_num_workers),
        '--learning_rate',
        str(learning_rate),
        '--weight_decay',
        str(weight_decay),
        '--device',
        train_device,
    ]
    if max_train_slates is not None:
        cmd.extend(['--max_train_slates', str(max_train_slates)])
    if max_val_slates is not None:
        cmd.extend(['--max_val_slates', str(max_val_slates)])

    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    header = f"$ {' '.join(cmd)}\n".encode('utf-8')
    log_handle.write(header)
    log_handle.flush()
    process = subprocess.Popen(
        cmd,
        cwd=str(family_dir.parents[1]),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle


def wait_for_any_training_job(
    running_jobs: list[dict[str, object]],
    *,
    block: bool,
) -> dict[str, object] | None:
    while running_jobs:
        for index, job in enumerate(running_jobs):
            process = job['process']
            return_code = process.poll()
            if return_code is None:
                continue

            log_handle = job['log_handle']
            log_handle.close()
            completed = running_jobs.pop(index)
            completed['return_code'] = return_code
            return completed

        if not block:
            return None
        time.sleep(2)

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ProSys Stage 2 V2 pipeline across families.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--output_root', type=str, default='outputs/stage2_v2')
    parser.add_argument('--fnn_checkpoint_pattern', type=str, default=None)
    parser.add_argument('--candidate_device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--slates_per_batch', type=int, default=8)
    parser.add_argument('--train_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--parallel_preprocess', type=int, default=4)
    parser.add_argument('--train_devices', type=str, default='cuda:0')
    parser.add_argument('--parallel_train', type=int, default=0)
    parser.add_argument('--python_bin', type=str, default=sys.executable)
    parser.add_argument('--product_fp_bits', type=int, default=2048)
    parser.add_argument('--product_fp_radius', type=int, default=2)
    parser.add_argument('--force_rebuild_memory', action='store_true')
    parser.add_argument('--force_rebuild_candidates', action='store_true')
    parser.add_argument('--force_rebuild_tables', action='store_true')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--max_train_slates', type=int, default=None)
    parser.add_argument('--max_val_slates', type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = repo_root / 'data'
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    family_dirs = sorted(data_root.glob('reaction_processed_*_catmerge'))
    if args.families != 'all':
        allow = set(parse_csv_arg(args.families))
        family_dirs = [path for path in family_dirs if infer_family_name_from_dir(path) in allow]

    if not family_dirs:
        raise ValueError('No family directories matched the requested filters.')

    preprocess_jobs: list[Future[Path]] = []
    family_outputs: dict[str, Path] = {}

    # Reuse built artifacts unless the caller explicitly forces a rebuild.
    with ProcessPoolExecutor(max_workers=args.parallel_preprocess) as pool:
        for family_dir in family_dirs:
            family = infer_family_name_from_dir(family_dir)
            family_output = output_root / family
            family_outputs[family] = family_output

            split_root = family_dir / 'For_second_part_model'
            train_split = split_root / 'Splitted_second_train_labels_processed.txt'
            val_split = split_root / 'Splitted_second_validate_labels_processed.txt'
            test_split = split_root / 'Splitted_second_test_labels_processed.txt'
            memory_dir = family_output / 'memory'

            ensure_product_memory(
                train_split=train_split,
                memory_dir=memory_dir,
                family=family,
                n_bits=args.product_fp_bits,
                radius=args.product_fp_radius,
                force=args.force_rebuild_memory,
            )

            fnn_checkpoint = resolve_fnn_checkpoint(args.fnn_checkpoint_pattern, family)
            for split_name, split_file in (
                ('train', train_split),
                ('val', val_split),
                ('test', test_split),
            ):
                candidate_file = family_output / 'candidate_pool' / f'{split_name}.csv'
                if not args.force_rebuild_candidates and candidate_file.exists():
                    print(f'[stage2] reuse candidate pool for {family}/{split_name}: {candidate_file}')
                    continue

                print(f'[stage2] build candidate pool for {family}/{split_name}')
                preprocess_jobs.append(
                    pool.submit(
                        build_candidate_pool_file,
                        family_dir,
                        memory_dir,
                        split_file,
                        candidate_file,
                        fnn_checkpoint,
                        args.candidate_device,
                    )
                )

        for job in preprocess_jobs:
            built_file = job.result()
            print(f'[stage2] wrote candidate pool: {built_file}')

    for family_dir in family_dirs:
        family = infer_family_name_from_dir(family_dir)
        family_output = family_outputs[family]
        split_root = family_dir / 'For_second_part_model'
        candidate_dir = family_output / 'candidate_pool'
        table_dir = family_output / 'training_tables'
        table_dir.mkdir(parents=True, exist_ok=True)

        for split_name, gold_name in (
            ('train', 'Splitted_second_train_labels_processed.txt'),
            ('val', 'Splitted_second_validate_labels_processed.txt'),
            ('test', 'Splitted_second_test_labels_processed.txt'),
        ):
            ensure_training_table(
                candidate_pool_file=candidate_dir / f'{split_name}.csv',
                gold_split_file=split_root / gold_name,
                output_file=table_dir / f'{split_name}.csv',
                family=family,
                split_name=split_name,
                force=args.force_rebuild_tables,
            )

    train_devices = parse_csv_arg(args.train_devices)
    if not train_devices:
        raise ValueError('At least one training device must be provided.')

    max_parallel_train = args.parallel_train or len(train_devices)
    available_devices = train_devices[:max_parallel_train]
    running_jobs: list[dict[str, object]] = []

    for family_dir in family_dirs:
        family = infer_family_name_from_dir(family_dir)
        family_output = family_outputs[family]
        best_checkpoint = family_output / 'train' / 'best_model.pt'

        if not args.force_retrain and best_checkpoint.exists():
            print(f'[stage2] reuse trained model for {family}: {best_checkpoint}')
            continue

        while not available_devices:
            completed = wait_for_any_training_job(running_jobs, block=True)
            if completed is None:
                continue

            finished_family = completed['family']
            finished_device = completed['device']
            return_code = int(completed['return_code'])
            available_devices.append(str(finished_device))
            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code=return_code,
                    cmd=completed['cmd'],
                )
            print(f'[stage2] finished training for {finished_family} on {finished_device}')

        train_device = available_devices.pop(0)
        table_dir = family_output / 'training_tables'
        process, log_handle = start_training_process(
            python_bin=args.python_bin,
            family_dir=family_dir,
            family_output=family_output,
            table_dir=table_dir,
            train_device=train_device,
            epochs=args.epochs,
            slates_per_batch=args.slates_per_batch,
            train_num_workers=args.train_num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_train_slates=args.max_train_slates,
            max_val_slates=args.max_val_slates,
        )
        cmd = process.args if isinstance(process.args, list) else [str(process.args)]
        running_jobs.append(
            {
                'family': family,
                'device': train_device,
                'process': process,
                'log_handle': log_handle,
                'cmd': cmd,
            }
        )
        print(f'[stage2] started training for {family} on {train_device}')

    while running_jobs:
        completed = wait_for_any_training_job(running_jobs, block=True)
        if completed is None:
            continue

        finished_family = completed['family']
        finished_device = completed['device']
        return_code = int(completed['return_code'])
        available_devices.append(str(finished_device))
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code=return_code,
                cmd=completed['cmd'],
            )
        print(f'[stage2] finished training for {finished_family} on {finished_device}')


if __name__ == '__main__':
    main()
