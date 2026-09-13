import unittest
import pandas as pd
from scripts.repeat_deterministic_pipeline import compare_predictions


class PipelineRepeatComparisonTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            {'sample_index': i, 'reaction_id': str(i), 'product': 'CCO', 'reactants': 'CCBr',
             'reagent_norm': 'O', 'solvent_norm': 'CCO', 'label': 1,
             'temperature_gold': 25., 'xgb_temperature_pred': 24., 'xgb_score': 0.5}
            for i in (0, 1)])

    def test_row_reordering_is_not_prediction_drift(self):
        a = self.fixture()
        report = compare_predictions(a, a.iloc[::-1].copy())
        self.assertEqual(report['temperature_support']['count'], 2)
        self.assertTrue(report['all_retained_candidate_columns_exactly_equal'])

    def test_temperature_difference_is_not_accepted(self):
        a, b = self.fixture(), self.fixture()
        b.loc[0, 'xgb_temperature_pred'] += 0.00001
        with self.assertRaises(AssertionError):
            compare_predictions(a, b)

    def test_duplicate_identities_are_rejected(self):
        a = self.fixture()
        a = pd.concat([a, a.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'duplicated candidate'):
            compare_predictions(a, a)


if __name__ == '__main__':
    unittest.main()
