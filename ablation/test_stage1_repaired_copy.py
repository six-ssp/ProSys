import unittest
from types import SimpleNamespace
import numpy as np
from scripts.verify_stage1_repaired_copy import check_lineage, contiguous_runs, compare_binary


def fake_binary(rows):
    sizes = np.asarray([len(row) for row in rows], dtype=np.int32)
    pointers = np.cumsum(np.r_[0, sizes[:-1]], dtype=np.int64) * 2
    class Binary:
        def __len__(self):
            return len(self.sizes)
    result = Binary()
    result.sizes = sizes
    result._index = SimpleNamespace(dtype=np.uint16, _pointers=pointers)
    result._bin_buffer = memoryview(np.concatenate(rows).astype(np.uint16).tobytes())
    return result


class RepairedCopyTests(unittest.TestCase):
    def test_exhaustive_partition(self):
        lineage = {'source_index_by_output_row': [0, 2], 'removed_empty_pair_indices': [],
                   'excluded_by_external_plan_indices': [1]}
        self.assertEqual(check_lineage(lineage, 3), ([0, 2], {1}))
        for changed in ([0, 1], [0, 0], [0], [2, 0], [-1, 2]):
            with self.assertRaises(ValueError):
                check_lineage(dict(lineage, source_index_by_output_row=changed), 3)

    def test_runs_cover_all_retained_rows(self):
        self.assertEqual(list(contiguous_runs([0, 1, 3, 4, 8])), [(0, 2), (3, 5), (8, 9)])

    def test_full_tensor_byte_comparison(self):
        old = fake_binary([[4, 5, 2], [6, 2], [7, 8, 2], [9, 2]])
        new = fake_binary([[4, 5, 2], [7, 8, 2], [9, 2]])
        compare_binary(old, new, [0, 2, 3])
        for changed in ([[4, 5, 2], [7, 10, 2], [9, 2]], [[4, 5, 2], [7, 8, 2]]):
            with self.assertRaises(ValueError):
                compare_binary(old, fake_binary(changed), [0, 2, 3])

    def test_pointer_corruption_is_rejected(self):
        old, new = fake_binary([[4, 2], [5, 2]]), fake_binary([[4, 2], [5, 2]])
        new._index._pointers[1] = 0
        with self.assertRaisesRegex(ValueError, 'pointer'):
            compare_binary(old, new, [0, 1])


if __name__ == '__main__':
    unittest.main()
