"""Runtime preflight checks for ProSys training and evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path


def module_version(name: str) -> str:
    module = importlib.import_module(name)
    return getattr(module, '__version__', 'unknown')


def safe_import(name: str) -> tuple[bool, str]:
    try:
        version = module_version(name)
        return True, version
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, str(exc)


def fairseq_extension_status(repo_root: Path) -> dict[str, str | bool]:
    fairseq_root = repo_root / 'stage1_retrosynthesis' / 'fairseq'
    if str(fairseq_root) not in sys.path:
        sys.path.insert(0, str(fairseq_root))

    result: dict[str, str | bool] = {}
    for extension_name in ['fairseq.libnat', 'fairseq.libnat_cuda']:
        try:
            importlib.import_module(extension_name)
            result[extension_name] = True
        except Exception as exc:  # pragma: no cover - diagnostic path
            result[extension_name] = str(exc)
    return result


def build_report(repo_root: Path) -> dict:
    report: dict[str, object] = {
        'python': {
            'executable': sys.executable,
            'version': sys.version,
        },
        'env': {
            'CONDA_PREFIX': os.environ.get('CONDA_PREFIX', ''),
            'CUDA_HOME': os.environ.get('CUDA_HOME', ''),
            'PATH_has_nvcc': shutil.which('nvcc') is not None,
        },
    }

    required_modules = [
        'numpy',
        'pandas',
        'rdkit',
        'rxnmapper',
        'textdistance',
        'selfies',
        'SmilesPE',
    ]
    report['modules'] = {name: safe_import(name) for name in required_modules}

    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        report['torch'] = {
            'version': torch.__version__,
            'file': torch.__file__,
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_built': bool(torch.backends.cuda.is_built()),
            'device_count': int(torch.cuda.device_count()),
            'torch_cuda': torch.version.cuda,
            'cpp_extension_cuda_home': CUDA_HOME,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        report['torch'] = {'error': str(exc)}

    report['fairseq_extensions'] = fairseq_extension_status(repo_root)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Check ProSys runtime readiness.')
    parser.add_argument('--repo_root', type=str, default='.', help='Repository root path')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON only')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report = build_report(repo_root)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
