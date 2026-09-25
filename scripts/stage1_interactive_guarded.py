#!/usr/bin/env python3
"""Run frozen EditRetro inference with product-only augmentation protection."""

import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
sys.path.insert(0, str(ROOT / 'scripts'))
from stage1_identity_guard import augment
from audit_stage1_base_augmented import sha, write


def main():
    source = ROOT / 'stage1_retrosynthesis/fairseq/fairseq_cli/interactive.py'
    output = Path(os.environ['PROSYS_AUGMENTATION_GUARD_RECEIPT'])
    if output.exists():
        raise FileExistsError('Do not overwrite a guarded-inference receipt')
    paths = [Path(__file__), ROOT / 'scripts/stage1_identity_guard.py',
             ROOT / 'scripts/audit_stage1_base_augmented.py', source,
             ROOT / 'stage1_retrosynthesis/preprocess/SPE_ChEMBL.txt']
    before = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    spec = importlib.util.spec_from_file_location('prosys_frozen_interactive', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []

    def guarded(smiles, aug=10):
        tokens, replaced = augment(smiles, aug, module.smi_tokenizer)
        events.append({'query_index': len(events), 'augmentation_count': aug,
                       'input_normalized': ''.join(tokens[0].split()) != smiles,
                       'replaced_slots': replaced})
        return tokens

    module.get_aug_input = guarded
    module.cli_main()
    if not events:
        raise RuntimeError('Guard was not used; inference augmentation is required')
    if {name: sha(ROOT / name) for name in before} != before:
        raise RuntimeError('Guarded inference sources changed during decoding')
    write(output, {'pass': True, 'protocol': 'product_identity_fallback_v1',
        'source_sha256': before, 'query_count': len(events),
        'augmented_input_count': sum(e['augmentation_count'] for e in events),
        'replaced_variant_count': sum(len(e['replaced_slots']) for e in events),
        'normalized_input_count': sum(e['input_normalized'] for e in events),
        'events': events, 'argv': sys.argv[1:],
        'scope': 'Pre-vocabulary product identity and exact tokenization only; fixed-vocabulary UNK is not repaired; no gold precursors or conditions are read by the guard.'})


if __name__ == '__main__':
    main()
