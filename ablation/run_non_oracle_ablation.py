"""Dedicated entrypoint for ProSys Non-Oracle ablation experiments.

This keeps the `ablation/` directory aligned with the actual executable entry.
The heavy lifting is shared with the unified Stage-2/Stage-3 experiment runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.run_non_oracle_stage23_experiments import main as run_stage23_experiments


def main() -> None:
    argv = sys.argv[1:]
    if '--run_set' not in argv:
        argv = ['--run_set', 'ablation', *argv]
    run_stage23_experiments(argv)


if __name__ == '__main__':
    main()
