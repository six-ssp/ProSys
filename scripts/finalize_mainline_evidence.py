#!/usr/bin/env python3
"""Wait for the observed coordinator, then replay and export complete evidence."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "Experiment/mainline_evidence_completion_20260913"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator_pid", type=int, required=True)
    args = parser.parse_args()
    process = Path(f"/proc/{args.coordinator_pid}/cmdline")
    while process.exists():
        command = process.read_bytes()
        if b"run_mainline_evidence.py" not in command:
            break
        time.sleep(10)
    status = json.loads((STUDY / "status.json").read_text())
    if status["status"] != "complete" or len(status["completed"]) != 18:
        raise RuntimeError(f"Coordinator terminated without 18 completed jobs: {status}")
    for script in ("summarize_mainline_evidence.py", "export_evidence_examples.py", "benchmark_product_inference.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    (STUDY / "finalization.json").write_text(json.dumps({"complete": True,
        "scope": "retained-row replay, exact controls, subgroups, failures, reconstruction cost and cases"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
