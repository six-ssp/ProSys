import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_stage1_nonempty_inputs import build_one, check_databin, fairseq_data


class NonemptyInputTests(unittest.TestCase):
    def test_external_exclusions_are_paired_and_recorded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'REAXYS_fixture'
            self.fixture(source)
            plan = {'train': [0], 'val': [], 'test': []}
            row = build_one(source, root / 'copy', excluded_indices=plan)
            self.assertEqual(row['dataset'], 'REAXYS_fixture')
            self.assertEqual([r['retained_pairs'] for r in row['splits']], [1, 2, 2])
            self.assertEqual((root / 'copy/train.src').read_text(), 'N\n')
            self.assertEqual((root / 'copy/train.tgt').read_text(), 'C\n')
            import gzip
            with gzip.open(root / 'copy/train.lineage.json.gz', 'rt') as handle:
                lineage = json.load(handle)
            self.assertEqual(lineage['excluded_by_external_plan_indices'], [0])
            self.assertEqual(lineage['source_index_by_output_row'], [2])

    def test_out_of_bounds_external_exclusion_fails_closed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root / 'source')
            with self.assertRaisesRegex(ValueError, 'outside source'):
                build_one(root / 'source', root / 'copy',
                          excluded_indices={'train': [100], 'val': [], 'test': []})
            self.assertFalse((root / 'copy').exists())

    def fixture(self, source):
        Dictionary, indexed = fairseq_data()
        (source / 'data-bin').mkdir(parents=True)
        lines = {'src': ['C C O\n', 'C C\n', 'N\n'], 'tgt': ['C C Br\n', '\n', 'C\n']}
        for side in lines:
            dictionary = Dictionary()
            for line in lines[side]:
                for token in line.split():
                    dictionary.add_symbol(token)
            dictionary.save(str(source / 'data-bin' / f'dict.{side}.txt'))
            for split in ('train', 'val', 'test'):
                prefix = ('valid' if split == 'val' else split) + '.src-tgt.' + side
                builder = indexed.make_builder(str(source / 'data-bin' / (prefix + '.bin')),
                                               'mmap', vocab_size=len(dictionary))
                for line in lines[side]:
                    builder.add_item(dictionary.encode_line(line, add_if_not_exist=False))
                builder.finalize(str(source / 'data-bin' / (prefix + '.idx')))
                (source / f'{split}.{side}').write_text(''.join(lines[side]))

    def test_copy_preserves_retained_tensors_and_does_not_choose_validation_policy(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, destination = root / 'source', root / 'copy'
            self.fixture(source)
            before = (source / 'train.tgt').read_bytes()
            row = build_one(source, destination, expected_empty={s: [1] for s in ('train', 'val', 'test')})
            self.assertEqual([r['retained_pairs'] for r in row['splits']], [2, 2, 2])
            self.assertEqual((source / 'train.tgt').read_bytes(), before)
            self.assertEqual(check_databin(destination / 'data-bin'), {'train': 2, 'valid': 2})
            manifest = json.loads((destination / 'manifest.json').read_text())
            self.assertTrue(manifest['protocol_selection_pending'])
            self.assertFalse(manifest['ready_for_formal_three_seed_training'])
            with self.assertRaises(FileExistsError):
                build_one(source, destination)

    def test_binary_empty_guard_rejects_legacy_eos_only_targets(self):
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / 'source'
            self.fixture(source)
            with self.assertRaisesRegex(ValueError, 'EOS-only'):
                check_databin(source / 'data-bin')

    def test_text_bin_mismatch_cannot_be_promoted_as_a_clean_copy(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root / 'source')
            (root / 'source/train.src').write_text('N\nC C\nN\n')
            with self.assertRaisesRegex(ValueError, 'text/bin mismatch'):
                build_one(root / 'source', root / 'copy')
            self.assertFalse((root / 'copy').exists())


if __name__ == '__main__':
    unittest.main()
