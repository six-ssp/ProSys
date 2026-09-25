"""Process identity and completion guards without starting real training."""

from pathlib import Path
import tempfile
import unittest

from scripts.continue_stage1_50k_experts import process_identity, waiting_state


class HandoffTests(unittest.TestCase):
    def test_live_same_process_waits(self):
        self.assertEqual(waiting_state(False, False, (42, '100'), (42, '100')), 'waiting_for_base')

    def test_completion_goes_to_verification_not_automatic_acceptance(self):
        self.assertEqual(waiting_state(True, False, None, (42, '100')), 'ready_for_verification')

    def test_failure_overrides_stale_completion(self):
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            waiting_state(True, True, None, (42, '100'))

    def test_missing_process_without_completion_stops(self):
        with self.assertRaisesRegex(RuntimeError, 'gone'):
            waiting_state(False, False, None, (42, '100'))

    def test_reused_pid_is_not_a_live_base(self):
        with self.assertRaisesRegex(RuntimeError, 'gone'):
            waiting_state(False, False, (42, '101'), (42, '100'))

    def test_proc_start_ticks_and_zombie(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(process_identity(42, root))
            (root / '42').mkdir()
            # After the parenthesized comm field: state (3) through starttime (22).
            path = root / '42/stat'
            fields = ['S'] + ['0'] * 18 + ['12345']
            path.write_text('42 (a name with ) parentheses) ' + ' '.join(fields))
            self.assertEqual(process_identity(42, root), (42, '12345'))
            fields[0] = 'Z'
            path.write_text('42 (python) ' + ' '.join(fields))
            self.assertIsNone(process_identity(42, root))


if __name__ == '__main__':
    unittest.main()
