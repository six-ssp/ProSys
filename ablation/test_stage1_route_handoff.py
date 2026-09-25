import unittest

from scripts.prepare_stage1_downstream_routes import require_queue_state


class RouteHandoffTests(unittest.TestCase):
    def test_waits_only_for_same_live_queue(self):
        self.assertFalse(require_queue_state(False, False, (1, 'ticks'), (1, 'ticks')))

    def test_missing_queue_is_not_wait(self):
        with self.assertRaises(RuntimeError):
            require_queue_state(False, False, None, (1, 'ticks'))

    def test_reused_pid_is_not_same_queue(self):
        with self.assertRaises(RuntimeError):
            require_queue_state(False, False, (1, 'new'), (1, 'old'))

    def test_completed_job_can_be_checked_after_queue_exit(self):
        self.assertTrue(require_queue_state(True, False, None, (1, 'ticks')))

    def test_failed_queue_is_not_silently_accepted(self):
        with self.assertRaises(RuntimeError):
            require_queue_state(True, True, (1, 'ticks'), (1, 'ticks'))


if __name__ == '__main__':
    unittest.main()
