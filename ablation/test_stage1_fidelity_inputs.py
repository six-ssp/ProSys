import unittest

from scripts.prepare_stage1_fidelity_inputs import exclusion_plan


class FidelityPlanTests(unittest.TestCase):
    def report(self):
        return {'splits': {s: {'augmented_rows': 10, 'unmatched_augmented_pairs': 1}
                           for s in ('train', 'val', 'test')}}

    def rows(self):
        return [{'split': s, 'augmented_row': 3} for s in ('train', 'val', 'test')]

    def test_only_train_validation_are_filtered(self):
        self.assertEqual(exclusion_plan(self.report(), self.rows()),
                         {'train': [3], 'val': [3], 'test': []})

    def test_duplicate_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            exclusion_plan(self.report(), rows + rows[:1])

    def test_missing_details_rejected(self):
        with self.assertRaises(ValueError):
            exclusion_plan(self.report(), self.rows()[:2])

    def test_out_of_range_rejected(self):
        rows = self.rows()
        rows[0]['augmented_row'] = 10
        with self.assertRaises(ValueError):
            exclusion_plan(self.report(), rows)


if __name__ == '__main__':
    unittest.main()
