import unittest
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from stage1_retrosynthesis.preprocess.pair_validation import filter_nonempty_pairs, reaction_side_fragments


class PairValidationTests(unittest.TestCase):
    def test_empty_augmented_targets_are_removed_with_their_sources(self):
        src, tgt, rejected = filter_nonempty_pairs(['CC O', 'CC', '  ', 'N'], ['CC Br', '', 'O', 'C'])
        self.assertEqual(src, ['CC O', 'N'])
        self.assertEqual(tgt, ['CC Br', 'C'])
        self.assertEqual(rejected, [1, 2])

    def test_misaligned_arrays_fail_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            filter_nonempty_pairs(['C', 'O'], ['CO'])

    def test_cross_dot_graph_is_not_two_products(self):
        self.assertEqual(reaction_side_fragments('C1.C1'), ['CC'])
        self.assertEqual(len(reaction_side_fragments('CC.O')), 2)

    def test_mapping_is_preserved(self):
        from rdkit import Chem
        parts = reaction_side_fragments('[CH3:1]1.[CH3:2]1.[Na+:3]')
        maps = sorted(a.GetAtomMapNum() for p in parts for a in Chem.MolFromSmiles(p).GetAtoms())
        self.assertEqual(maps, [1, 2, 3])
        self.assertEqual(len(parts), 2)

    def test_invalid_side_is_not_salvaged(self):
        with self.assertRaises(ValueError):
            reaction_side_fragments('CC.C1')

    def test_real_preprocessing_cli_filters_empty_and_accepts_cross_dot(self):
        root = Path(__file__).resolve().parents[1]
        preprocess = root / 'stage1_retrosynthesis/preprocess'
        product = '[CH3:1][CH2:2][CH2:3][CH2:4][CH:5]=[O:6]'
        reactants = [
            '[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][OH:6]',
            '[CH3:11][CH2:12][CH2:13][CH2:14][CH2:15][OH:16]',
            '[CH3:1]1.[CH2:2]1[CH2:3][CH2:4][CH2:5][OH:6]',
        ]
        with tempfile.TemporaryDirectory() as directory:
            datasets = Path(directory)
            dataset = datasets / 'REAXYS_PAIR_FILTER_SMOKE'
            raw = dataset / 'raw'
            raw.mkdir(parents=True)
            with (raw / 'raw_train.csv').open('w') as handle:
                writer = csv.writer(handle)
                writer.writerow(['id', 'reactants>reagents>production'])
                writer.writerows((i, r + '>>' + product) for i, r in enumerate(reactants))
            env = dict(os.environ, EDITRETRO_DATASETS_ROOT=str(datasets))
            subprocess.run([sys.executable, str(preprocess / 'preprocess_data.py'),
                            '-dataset', dataset.name, '-augmentation', '2', '-processes', '1',
                            '-splits', 'train', '-spe'], cwd=preprocess, env=env,
                           check=True, capture_output=True, text=True)
            output = dataset / 'aug2'
            report = json.loads((output / 'train.pair_filter.json').read_text())
            self.assertEqual(report['input_augmented_pairs'], 6)
            self.assertEqual(report['kept_augmented_pairs'], 4)
            self.assertEqual(report['empty_pair_indices_before_filter'], [2, 3])
            for side in ('src', 'tgt'):
                lines = (output / f'train.{side}').read_text().splitlines()
                self.assertEqual(len(lines), 4)
                self.assertTrue(all(line.strip() for line in lines))


if __name__ == '__main__':
    unittest.main()
