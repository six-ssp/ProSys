import json
from pathlib import Path
import tempfile
import unittest

from scripts.stage1_route_admission import PROTOCOL, require_paired_checkpoints, verify_guard
from scripts.audit_stage1_base_augmented import sha


class GuardEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / 'scripts/build_stage1_guarded_routes.py'
        source.parent.mkdir()
        source.write_text('synthetic source binding, not inference')
        self.source = source
        self.guard = {'pass': True, 'protocol': PROTOCOL, 'query_count': 1,
            'augmented_input_count': 10, 'replaced_variant_count': 1,
            'normalized_input_count': 0,
            'events': [{'query_index': 0, 'augmentation_count': 10,
                        'replaced_slots': [2], 'input_normalized': False}],
            'source_sha256': {str(source.relative_to(self.root)): sha(source)}}
        self.cache = {'reactions': [{}], 'aug': 10, 'augmentation_guard': {
            'protocol': PROTOCOL, 'replaced_variant_count': 1,
            'wrapper_sha256': sha(source)}}
        self.path = self.root / 'route_cache.json'
        self.persist()

    def persist(self):
        path = self.root / 'augmentation_guard.json'
        path.write_text(json.dumps(self.guard))
        self.cache['augmentation_guard']['receipt_sha256'] = sha(path)
        self.path.write_text(json.dumps(self.cache))

    def test_complete_synthetic_receipt(self):
        self.assertEqual(verify_guard(self.path, self.root)['query_count'], 1)

    def test_query_order_or_slots_rejected(self):
        self.guard['events'][0]['query_index'] = 1
        self.persist()
        with self.assertRaises(ValueError):
            verify_guard(self.path, self.root)

    def test_changed_source_rejected(self):
        self.source.write_text('changed')
        with self.assertRaises(ValueError):
            verify_guard(self.path, self.root)

    def test_wrong_count_rejected(self):
        self.guard['query_count'] = 2
        self.persist()
        with self.assertRaises(ValueError):
            verify_guard(self.path, self.root)

    def test_aggregate_replacements_rejected(self):
        self.guard['replaced_variant_count'] = 0
        self.persist()
        with self.assertRaises(ValueError):
            verify_guard(self.path, self.root)

    def test_mixed_validation_checkpoint_rejected(self):
        paths = [self.root / 'test.json', self.root / 'val.json']
        for index, path in enumerate(paths):
            path.write_text(json.dumps({'family': 'a', 'checkpoint': str(self.root / str(index)),
                                        'aug': 10, 'topk': 10, 'n_best': 10}))
        with self.assertRaisesRegex(ValueError, 'same family expert'):
            require_paired_checkpoints(*paths)


if __name__ == '__main__':
    unittest.main()
