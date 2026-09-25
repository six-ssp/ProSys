from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import continue_50k_downstream_studies as controller


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "job.log"

    def run_command(self, free=20, **kwargs):
        with patch.object(controller, "require_frozen"), \
                patch.object(controller.shutil, "disk_usage", return_value=SimpleNamespace(free=free * 1024**3)), \
                patch.object(controller.subprocess, "run") as runner:
            controller.run(["python", "model.py"], self.log, {}, {}, **kwargs)
            return runner

    def test_existing_log_is_not_silent_retry(self):
        self.log.touch()
        with self.assertRaises(FileExistsError):
            self.run_command()

    def test_training_requires_space(self):
        with self.assertRaisesRegex(RuntimeError, "8 GiB"):
            self.run_command(free=7)

    def test_compaction_can_run_below_training_threshold(self):
        runner = self.run_command(free=2, minimum_free_gib=1)
        runner.assert_called_once()
        self.assertTrue(runner.call_args.kwargs["check"])

    def test_changed_sources_stop_before_subprocess(self):
        with patch.object(controller, "sha", return_value="changed"), \
                patch.object(controller.subprocess, "run") as runner:
            with self.assertRaises(ValueError):
                controller.run(["python"], self.log, {}, {"some_source.py": "original"})
            runner.assert_not_called()

    def test_subprocess_failure_is_propagated(self):
        with patch.object(controller, "require_frozen"), \
                patch.object(controller.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)), \
                patch.object(controller.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["python"])):
            with self.assertRaises(subprocess.CalledProcessError):
                controller.run(["python"], self.log, {}, {})

    def test_waiting_requires_same_live_controller(self):
        expected = (123, "456")
        self.assertEqual(controller.waiting_state(False, False, expected, expected), "waiting_for_base")
        for observed in (None, (123, "789")):
            with self.assertRaises(RuntimeError):
                controller.waiting_state(False, False, observed, expected)
        with self.assertRaises(RuntimeError):
            controller.waiting_state(True, True, expected, expected)


if __name__ == "__main__":
    unittest.main()
