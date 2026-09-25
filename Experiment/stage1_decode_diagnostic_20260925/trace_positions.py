"""Diagnose invalid embedding indices without changing any model outputs."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
import stage1_interactive_guarded as guard


def main():
    target = Path(os.environ['PROSYS_POSITION_DIAGNOSTIC'])
    assert not target.exists()
    original = torch.nn.Embedding.forward

    def checked(module, indices):
        low, high = torch.aminmax(indices)
        low, high = int(low.item()), int(high.item())
        if low < 0 or high >= module.num_embeddings:
            bad = ((indices < 0) | (indices >= module.num_embeddings))
            record = dict(failure='embedding_index_out_of_range',
                class_name=type(module).__name__, num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim, padding_idx=module.padding_idx,
                index_shape=list(indices.shape), minimum=low, maximum=high,
                bad_rows=bad.any(dim=1).nonzero().flatten().cpu().tolist(),
                bad_per_row=bad.sum(dim=1).cpu().tolist(),
                stack=traceback.format_stack(), argv=sys.argv,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Diagnostic only; stop before invalid lookup, do not clip or replace output')
            frames = {}
            for info in inspect.stack():
                local = info.frame.f_locals
                if info.function == 'main' and 'batch' in local:
                    batch = local['batch']
                    frames['interactive_batch_augmented_ids'] = batch.ids.cpu().tolist()
                    frames['interactive_batch_source_lengths'] = batch.src_lengths.cpu().tolist()
                if info.function in ('generate', 'forward_decoder', 'extract_features'):
                    payload = {}
                    for key in ('step', 'sent_idxs', 'can_ins_word', 'can_ins_mask', 'max_lens'):
                        value = local.get(key)
                        if torch.is_tensor(value):
                            payload[key] = value.detach().cpu().tolist()
                        elif isinstance(value, (int, float, str)):
                            payload[key] = value
                    for key in ('output_tokens', 'prev_output_tokens'):
                        value = local.get(key)
                        if torch.is_tensor(value):
                            payload[key + '_shape'] = list(value.shape)
                            payload[key + '_nonpadding_lengths'] = value.ne(module.padding_idx).sum(dim=1).cpu().tolist()
                    frames[info.function] = payload
            record['frames'] = frames
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2) + '\n')
            raise RuntimeError('Position diagnostic captured before invalid CUDA lookup: ' + str(target))
        return original(module, indices)

    torch.nn.Embedding.forward = checked
    guard.main()


if __name__ == '__main__':
    main()
