import sys
from pathlib import Path
import tempfile
import unittest

from scripts.preview_clean_validation import inspect_augmented, reaction_key

ROOT = Path(__file__).resolve().parents[1]


class ValidationPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'stage1_retrosynthesis/fairseq'))

    def make_fixture(self, root):
        from fairseq.data import Dictionary, indexed_dataset
        data = root / 'data-bin'
        data.mkdir()
        text = {'src': ['C C O\n', 'C C = O\n'], 'tgt': ['C C Br\n', 'C C O\n']}
        for side in ('src', 'tgt'):
            dictionary = Dictionary()
            for line in text[side]:
                for token in line.split():
                    dictionary.add_symbol(token)
            dictionary.save(str(data / f'dict.{side}.txt'))
            builder = indexed_dataset.make_builder(str(data / f'valid.src-tgt.{side}.bin'),
                                                   'mmap', vocab_size=len(dictionary))
            for line in text[side]:
                builder.add_item(dictionary.encode_line(line, add_if_not_exist=False))
            builder.finalize(str(data / f'valid.src-tgt.{side}.idx'))
            (root / f'val.{side}').write_text(''.join(text[side]))

    def test_indices_match_exact_reactions_and_binaries(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.make_fixture(root)
            result = inspect_augmented(root, {reaction_key('CCBr', 'CCO')})
            self.assertEqual(result['remove_indices'], [0])
            self.assertEqual(result['remaining_rows'], 1)
            self.assertTrue(result['text_equals_bin_all_rows'])

    def test_stale_binary_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.make_fixture(root)
            (root / 'val.src').write_text('C C\nC C = O\n')
            with self.assertRaisesRegex(ValueError, 'Text/bin disagreement'):
                inspect_augmented(root, set())


if __name__ == '__main__':
    unittest.main()
