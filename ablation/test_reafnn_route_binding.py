"""No model fitting: reject stale no-KNN routes, references and retained evidence."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_reafnn_only_ablation as runner
from scripts.run_current_mainline_matched_ablations import _load_mainline_reference


class RouteBindingTests(unittest.TestCase):
    def test_validation_and_test_require_same_expert_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = root / 'a.pt', root / 'b.pt'
            a.write_bytes(b'a')
            b.write_bytes(b'b')
            with patch.object(runner, 'family_inputs', return_value={'checkpoint/test': a, 'checkpoint/val': b}):
                with self.assertRaisesRegex(ValueError, 'different expert weights'):
                    runner.input_bindings(['Beckmann'], root, root)

    def test_changed_retained_candidate_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {'routes/test': 'synthetic-hash'}
            for name in ('candidate_audit.json', 'test_candidates.csv.gz'):
                (root / name).write_bytes(b'fixture')
            result = {'bound_family_inputs': expected, 'retained_evidence_sha256': {
                name: runner.sha(root / name) for name in ('candidate_audit.json', 'test_candidates.csv.gz')}}
            runner.require_cached_evidence(result, root, expected)
            (root / 'test_candidates.csv.gz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                runner.require_cached_evidence(result, root, expected)

    def test_legacy_result_without_input_binding_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'unbound inputs'):
            runner.require_cached_evidence({}, Path('/not-used'), {'routes/test': 'new'})

    def test_historical_reference_is_rejected_for_new_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'seed_0/Beckmann'
            folder.mkdir(parents=True)
            splits = {s: 'hash-' + s for s in ('train', 'val', 'test')}
            expected = {'routes/test': 'test-routes', 'routes/val': 'val-routes',
                        **{'split/' + s: value for s, value in splits.items()}}
            provenance = {'family': 'Beckmann', 'seed': 0, 'route_test': 'test-routes',
                          'route_validation': 'val-routes', 'splits': splits}
            path = folder / 'provenance.json'
            path.write_text(json.dumps(provenance))
            (folder / 'result.json').write_text(json.dumps({'family': 'Beckmann', 'seed': 0}))
            self.assertEqual(len(runner.reference_bindings(root, ['Beckmann'], [0], {'Beckmann': expected})), 2)
            expected['routes/test'] = 'new-50k-routes'
            with self.assertRaisesRegex(ValueError, 'different routes/splits'):
                runner.reference_bindings(root, ['Beckmann'], [0], {'Beckmann': expected})

    def test_missing_route_metric_is_not_replaced_by_historical_constant(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'metrics.csv'
            row = {metric + '_' + stat: 0.2 for metric in
                   ('cover', 'sys1', 'sys3', 'sys5', 'sys10', 'mrr', 'ndcg10') for stat in ('mean', 'std')}
            with path.open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            self.assertIsNone(_load_mainline_reference(path)['route_at_10_mean'])
            row.update(route_at_10_mean=0.4, route_at_10_std=0.03)
            with path.open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            self.assertEqual(_load_mainline_reference(path)['route_at_10_mean'], 0.4)

    def test_macro_reference_is_recomputed_from_paired_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = ('route_at_10', 'candidate_recall', 'sys_at_1', 'sys_at_3',
                       'sys_at_5', 'sys_at_10', 'mrr', 'ndcg_at_10')
            for seed in (0, 1):
                folder = root / f'seed_{seed}/Beckmann'
                folder.mkdir(parents=True)
                (folder / 'result.json').write_text(json.dumps({'seed': seed}))
            def row(result, *, arm, seed):
                return {'seed': seed, **{metric: 0.2 + seed * 0.2 for metric in metrics}}
            with patch.object(runner.reporting, '_metric_row', side_effect=row), \
                    patch.object(runner.reporting, '_load_mainline_reference', return_value={
                        'arm': 'full_mainline', 'sys_at_10_mean': 0.3, 'route_at_10_mean': None}):
                result = runner.checked_reference_metrics(root / 'macro.csv', root, ['Beckmann'], [0, 1])
                self.assertAlmostEqual(result['route_at_10_mean'], 0.3)
                self.assertAlmostEqual(result['sys_at_10_std'], 0.1 * 2 ** 0.5)
            with patch.object(runner.reporting, '_metric_row', side_effect=row), \
                    patch.object(runner.reporting, '_load_mainline_reference', return_value={
                        'arm': 'full_mainline', 'sys_at_10_mean': 0.9}):
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    runner.checked_reference_metrics(root / 'macro.csv', root, ['Beckmann'], [0, 1])


if __name__ == '__main__':
    unittest.main()
