"""Synthetic aggregation tests; these are not completed expert runs."""

import unittest
from pathlib import Path
import tempfile
from scripts.summarize_stage1_multiseed import FIELDS, aggregate_rows, check_invalidation


class Stage1SummaryTests(unittest.TestCase):
    def test_invalidated_study_cannot_be_summarized(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            check_invalidation(study)
            for marker in ('SCIENTIFIC_INVALIDATION.json', 'PAUSED_FOR_SPLIT_AUDIT.json'):
                (study / marker).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'split-integrity invalidation'):
                    check_invalidation(study)
                (study / marker).unlink()

    def rows(self):
        return [dict(family=f, seed=s, n=10 if f == 'a' else 1000,
                     **{k: value for k in FIELDS})
                for f, s, value in [('a', 0, 0.1), ('a', 1, 0.3), ('b', 0, 0.9), ('b', 1, 0.7)]]

    def test_equal_family_not_query_weighted(self):
        _, macro, _, summary = aggregate_rows(self.rows(), ['a', 'b'], [0, 1])
        self.assertEqual(macro[FIELDS[0]].tolist(), [0.5, 0.5])
        self.assertEqual(summary.iloc[0][FIELDS[0] + '_std'], 0)

    def test_missing_or_duplicate_seed_withheld(self):
        for rows in (self.rows()[:-1], self.rows() + [self.rows()[0]]):
            with self.assertRaisesRegex(ValueError, 'Incomplete or duplicate'):
                aggregate_rows(rows, ['a', 'b'], [0, 1])

    def test_denominator_changes_rejected(self):
        rows = self.rows()
        rows[0]['n'] = 9
        with self.assertRaisesRegex(ValueError, 'support differs'):
            aggregate_rows(rows, ['a', 'b'], [0, 1])

    def test_nonfinite_and_nonmonotonic_rejected(self):
        for value in (float('nan'), 0.99):
            rows = self.rows()
            rows[0][FIELDS[0]] = value
            with self.assertRaises(ValueError):
                aggregate_rows(rows, ['a', 'b'], [0, 1])

    def test_sample_standard_deviation(self):
        _, _, family, _ = aggregate_rows(self.rows(), ['a', 'b'], [0, 1])
        self.assertAlmostEqual(family.loc['a', FIELDS[0] + '_std'], 0.2 / 2**0.5)


if __name__ == '__main__':
    unittest.main()
