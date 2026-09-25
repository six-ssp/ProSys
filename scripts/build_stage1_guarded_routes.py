#!/usr/bin/env python3
"""Reuse frozen route builder, changing only the interactive augmentation guard."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage1_retrosynthesis import build_route_cache as builder
from scripts.audit_stage1_base_augmented import sha, write


def main():
    if '--skip_generation' in sys.argv or '--generation_file' in sys.argv:
        raise ValueError('Guarded routes require fresh guarded generation')
    receipts = []
    source = ROOT / 'stage1_retrosynthesis/fairseq/fairseq_cli/interactive.py'

    def run(command, **kwargs):
        if len(command) < 2 or Path(command[1]).resolve() != source:
            raise ValueError('Unexpected frozen-builder subprocess')
        command = list(command)
        command[1] = str(ROOT / 'scripts/stage1_interactive_guarded.py')
        output = Path(kwargs['stdout'].name).parent / 'augmentation_guard.json'
        kwargs['env'] = dict(kwargs['env'], PROSYS_AUGMENTATION_GUARD_RECEIPT=str(output))
        result = subprocess.run(command, **kwargs)
        record = json.loads(output.read_text())
        if record.get('pass') is not True:
            raise ValueError('Guarded inference did not pass')
        receipts.append(output)
        return result

    builder.subprocess = SimpleNamespace(run=run)
    builder.main()
    if len(receipts) != 1:
        raise RuntimeError('Expected exactly one guarded generation')
    receipt = receipts[0]
    path = receipt.parent / 'route_cache.json'
    cache = json.loads(path.read_text())
    record = json.loads(receipt.read_text())
    if record['query_count'] != len(cache['reactions']):
        raise ValueError('Guard/query count mismatch')
    cache['augmentation_guard'] = {'protocol': record['protocol'],
        'receipt_sha256': sha(receipt), 'replaced_variant_count': record['replaced_variant_count'],
        'wrapper_sha256': sha(Path(__file__))}
    write(path, cache)


if __name__ == '__main__':
    main()
