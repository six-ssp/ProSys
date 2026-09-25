"""CPU checks against the actual frozen insertion implementation, not a mock."""

import importlib.util
import io
from collections import namedtuple
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fail_closed_masks import DEAD, apply_guarded, install

spec = importlib.util.spec_from_file_location('vendor_levenshtein', ROOT / 'stage1_retrosynthesis/editretro/models/levenshtein_utils.py')
vendor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vendor)


class GuardTests(unittest.TestCase):
    def inputs(self):
        tokens = torch.tensor([[0, 7, 2, 1], [0, 8, 9, 2]])
        return tokens, torch.zeros_like(tokens), torch.zeros_like(tokens, dtype=torch.float), torch.tensor([[1, 2, 99], [0, 1, 0]])

    def call(self, args, limit):
        events = []
        output = apply_guarded(vendor._apply_ins_masks, limit, events.append, *args, 1, 3, 2)
        return output, events

    def test_normal_exact_and_padding_counts_ignored(self):
        args = self.inputs()
        expected = vendor._apply_ins_masks(*(a.clone() for a in args), 1, 3, 2)
        actual, events = self.call(args, 6)
        self.assertEqual(events, [])
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))

    def test_overflow_is_empty_not_truncated(self):
        actual, events = self.call(self.inputs(), 5)
        self.assertEqual(events[0]['rows'], [0])
        self.assertEqual(events[0]['proposed_lengths'], [6])
        self.assertEqual(actual[0][0][actual[0][0].ne(1)].tolist(), [0, 2])
        self.assertEqual(actual[1][0, 0].item(), DEAD)
        self.assertTrue(bool(torch.isneginf(actual[2][0, :2]).all()))

    def test_healthy_neighbor_identical(self):
        args = self.inputs()
        expected = vendor._apply_ins_masks(*(a.clone() for a in args), 1, 3, 2)
        actual, _ = self.call(args, 5)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a[1], b[1, :a.shape[1]]))

    def test_dead_hypothesis_cannot_resurrect(self):
        first, _ = self.call(self.inputs(), 5)
        counts = torch.ones((2, first[0].size(1) - 1), dtype=torch.long)
        actual, events = self.call((*first, counts), 20)
        self.assertEqual(events, [])
        self.assertEqual(actual[0][0][actual[0][0].ne(1)].tolist(), [0, 2])
        self.assertEqual(actual[1][0, 0].item(), DEAD)

    def test_all_overflow_preserves_rows(self):
        args = self.inputs()
        actual, events = self.call(args, 4)
        self.assertEqual(actual[0].shape, (2, 2))
        self.assertEqual(events[0]['rows'], [0, 1])
        self.assertEqual(actual[0].tolist(), [[0, 2], [0, 2]])

    def test_random_normal_inputs_exact(self):
        generator = torch.Generator().manual_seed(19)
        for _ in range(30):
            args = list(self.inputs())
            args[3] = torch.randint(0, 20, args[3].shape, generator=generator)
            expected = vendor._apply_ins_masks(*(a.clone() for a in args), 1, 3, 2)
            actual, events = self.call(args, 1024)
            self.assertEqual(events, [])
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a, b))

    def test_vendor_terminal_token_branch_keeps_beam_slots(self):
        sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis'))
        from editretro.models.editretro_nat import EditRetroModel

        class Model:
            pad, unk, training = 1, 3, False
            forward_decoder_token = EditRetroModel.forward_decoder_token

            def forward_decoder(self, value, *args, **kwargs):
                return value

            forward_decoder_mask = forward_decoder

        module = SimpleNamespace(EditRetroModel=Model, _apply_ins_masks=vendor._apply_ins_masks)
        install(module, [])
        State = namedtuple('State', 'output_tokens output_marks output_scores attn num_ops history')
        tokens, marks, scores = self.call(self.inputs(), 4)[0]
        state = State(tokens, marks, scores, None, (0, 0, 0), [tokens.clone()])
        output = Model().forward_decoder_token(state, None, None, token_beam=2)
        self.assertEqual(output.output_tokens.shape, (4, 2))
        self.assertEqual(output.history[0].shape, (4, 2))
        self.assertTrue(bool(output.output_marks[:, 0].eq(DEAD).all()))
        self.assertTrue(bool(torch.isneginf(output.output_scores).all()))

    def test_empty_hypothesis_keeps_generation_slot_and_is_invalid(self):
        path = ROOT / 'stage1_retrosynthesis/utils/get_ranked_topk.py'
        spec = importlib.util.spec_from_file_location('vendor_rank', path)
        rank = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rank)
        with patch('builtins.open', return_value=io.StringIO('H-0\t-inf\t\nP-0\t-inf -inf\n')):
            predictions, scores, _ = rank.process_input('memory-only-fixture', False)
        self.assertEqual(predictions, [''])
        self.assertEqual(scores, [-float('inf')])
        canonical = [rank.canonicalize_smiles_clear_map(p) for p in predictions]
        ranked, invalid = rank.compute_rank([canonical], [scores], beam_size=1)
        self.assertEqual(ranked, {})
        self.assertEqual(invalid, [1])


if __name__ == '__main__':
    unittest.main()
