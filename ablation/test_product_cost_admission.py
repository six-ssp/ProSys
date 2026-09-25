import unittest
from unittest.mock import Mock, patch

from scripts.benchmark_product_inference import require_idle


class ProductCostAdmissionTests(unittest.TestCase):
    def test_live_training_queue_rejects_cost_measurement(self):
        process = Mock()
        process.read_bytes.return_value = b"python\0scripts/run_stage1_multiseed.py\0"
        with patch("scripts.benchmark_product_inference.Path.glob", return_value=[process]), \
             patch("scripts.benchmark_product_inference.subprocess.check_output") as query:
            with self.assertRaisesRegex(RuntimeError, "queue is active"):
                require_idle()
            query.assert_not_called()

    def test_unrelated_gpu_process_also_rejects_measurement(self):
        with patch("scripts.benchmark_product_inference.Path.glob", return_value=[]), \
             patch("scripts.benchmark_product_inference.subprocess.check_output", return_value="12345\n"):
            with self.assertRaisesRegex(RuntimeError, "GPU compute processes"):
                require_idle()

    def test_idle_device_passes_preflight(self):
        with patch("scripts.benchmark_product_inference.Path.glob", return_value=[]), \
             patch("scripts.benchmark_product_inference.subprocess.check_output", return_value=""):
            require_idle()


if __name__ == "__main__":
    unittest.main()
