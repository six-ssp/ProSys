"""Experimental insertion guard: invalidate overflow hypotheses, never truncate chemistry."""

from contextvars import ContextVar
from functools import wraps

import torch

DEAD = -32768


def apply_guarded(original, limit, callback, tokens, marks, scores, counts, pad, unk, eos):
    if limit < 2 or marks is None or scores is None:
        raise ValueError('Inference guard requires marks, scores and a valid decoder capacity')
    lengths = tokens.ne(pad).sum(1)
    effective = counts.masked_fill(tokens[:, 1:].eq(pad), 0)
    proposed = lengths + effective.sum(1)
    previous = marks[:, 0].eq(DEAD)
    overflow = proposed.gt(limit)
    dead = previous | overflow
    if not bool(dead.any()):
        # Preserve the original call and its in-place semantics on the normal path.
        return original(tokens, marks, scores, counts, pad, unk, eos)
    fresh = overflow & ~previous
    if bool(fresh.any()):
        callback(dict(capacity=int(limit), rows=fresh.nonzero().flatten().cpu().tolist(),
                      before_lengths=lengths[fresh].cpu().tolist(),
                      proposed_lengths=proposed[fresh].cpu().tolist()))
    tokens, marks, scores, counts = (x.clone() for x in (tokens, marks, scores, counts))
    first = tokens[dead, 0].clone()
    tokens[dead] = pad
    tokens[dead, 0] = first
    tokens[dead, 1] = eos
    marks[dead] = 0
    marks[dead, 0] = DEAD
    scores[dead] = 0
    scores[dead, :2] = -float('inf')
    counts[dead] = 0
    output = original(tokens, marks, scores, counts, pad, unk, eos)
    # Repeated padded scatter indices can overwrite the terminal score with zero.
    output[2][dead] = 0
    output[2][dead, :2] = -float('inf')
    assert bool(output[0].ne(pad).sum(1).le(limit).all())
    assert bool(output[1][dead, 0].eq(DEAD).all())
    return output


def install(module, events):
    """Install only in a separate inference process; do not edit vendor sources."""
    active = ContextVar('prosys_decoder_capacity')
    original_masks = module._apply_ins_masks

    def masks(*args):
        return apply_guarded(original_masks, active.get(), events.append, *args)

    module._apply_ins_masks = masks
    for name in ('forward_decoder', 'forward_decoder_mask'):
        original = getattr(module.EditRetroModel, name)

        def wrap(function):
            @wraps(function)
            def forward(self, *args, **kwargs):
                if self.training:
                    raise RuntimeError('This diagnostic guard must not affect training')
                token = active.set(int(self.decoder.max_positions()))
                try:
                    return function(self, *args, **kwargs)
                finally:
                    active.reset(token)
            return forward

        setattr(module.EditRetroModel, name, wrap(original))
    original_token = module.EditRetroModel.forward_decoder_token

    @wraps(original_token)
    def token_forward(self, decoder_out, *args, **kwargs):
        count = int(kwargs.get('token_beam', 1))
        all_terminal = not bool(decoder_out.output_tokens.eq(self.unk).any())
        has_dead = bool(decoder_out.output_marks[:, 0].eq(DEAD).any())
        result = original_token(self, decoder_out, *args, **kwargs)
        if count > 1 and all_terminal and has_dead:
            # The vendor's no-insertion branch skips beam expansion altogether.
            values = {}
            for name in ('output_tokens', 'output_marks', 'output_scores', 'attn'):
                value = getattr(result, name)
                values[name] = None if value is None else value.repeat_interleave(count, dim=0)
            if result.history is not None:
                values['history'] = [value.repeat_interleave(count, dim=0) for value in result.history]
            result = result._replace(**values)
        return result

    module.EditRetroModel.forward_decoder_token = token_forward
