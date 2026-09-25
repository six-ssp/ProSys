"""Pairing tests prevent incomplete or mismatched base/expert comparisons."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.collect_stage1_base_vs_tuned import append_macro_and_weighted, collect_rows


class PairedCollectorTests(unittest.TestCase):
    def test_missing_family_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                collect_rows(Path(temporary), Path(temporary), ['Beckmann'])

    def test_equal_count_different_queries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind, product in [('base', 'CO'), ('expert', 'CCO')]:
                folder = root / kind / 'Beckmann'
                folder.mkdir(parents=True)
                (folder / 'route_cache.json').write_text(json.dumps({'family': 'Beckmann', 'reactions': [
                    {'sample_index': 0, 'reaction_id': 'r0', 'product': product, 'gold_reactants': 'C.O', 'routes': []}]}))
            with self.assertRaisesRegex(ValueError, 'query identities differ'):
                collect_rows(root / 'expert', root / 'base', ['Beckmann'])

    def test_identical_queries_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'Beckmann'
            folder.mkdir()
            (folder / 'route_cache.json').write_text(json.dumps({'family': 'Beckmann', 'reactions': [
                {'sample_index': 0, 'reaction_id': 'r0', 'product': 'CO', 'gold_reactants': 'C.O', 'routes': []}]}))
            metrics = {'n': 1, **{'route_recall_top' + str(k): 0.0 for k in (1, 3, 5, 10)}}
            with patch('scripts.collect_stage1_base_vs_tuned.stage1_route_recall', return_value=metrics), \
                 patch('scripts.stage1_route_admission.verify_guard') as guard:
                rows = collect_rows(root, root, ['Beckmann'])
            self.assertEqual(guard.call_count, 2)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['test_products'], 1)

    def test_macro_count_is_total_but_rate_remains_equal_family_mean(self):
        rows = append_macro_and_weighted([
            dict(family='a', display_family='a', test_products=10, recall=.2),
            dict(family='b', display_family='b', test_products=90, recall=.8)], sample_key='test_products')
        self.assertEqual(rows[-2]['test_products'], 100)
        self.assertAlmostEqual(rows[-2]['recall'], .5)
        self.assertEqual(rows[-1]['test_products'], 100)
        self.assertAlmostEqual(rows[-1]['recall'], .74)

    def test_empty_or_duplicate_family_grid_is_rejected(self):
        for families in ([], ['Beckmann', 'Beckmann']):
            with self.assertRaisesRegex(ValueError, 'nonempty unique'):
                collect_rows(Path('.'), Path('.'), families)


if __name__ == '__main__':
    unittest.main()
