import unittest

from scripts.stage1_identity_guard import augment, identity


class IdentityGuardTests(unittest.TestCase):
    def test_equivalent_order_kept(self):
        self.assertEqual(identity('CCO'), identity('OCC'))
        tokens, slots = augment('CCO', 3, lambda x: x, lambda _: 'OCC')
        self.assertEqual(tokens, ['CCO', 'OCC', 'OCC'])
        self.assertEqual(slots, [])

    def test_stereo_change_replaced_without_extra_draw(self):
        calls = []
        def randomize(_):
            calls.append(1)
            return 'F/C=C\\F'
        tokens, slots = augment('F/C=C/F', 10, lambda x: x, randomize)
        self.assertEqual(tokens, ['F/C=C/F'] * 10)
        self.assertEqual(slots, list(range(1, 10)))
        self.assertEqual(len(calls), 9)

    def test_graph_change_replaced(self):
        tokens, slots = augment('CCO', 2, lambda x: x, lambda _: 'CCC')
        self.assertEqual(tokens, ['CCO', 'CCO'])
        self.assertEqual(slots, [1])

    def test_isotopes_preserved(self):
        self.assertNotEqual(identity('[13CH3]CO'), identity('CCO'))

    def test_bad_tokenizer_is_fatal(self):
        with self.assertRaises(ValueError):
            augment('CCO', 1, lambda _: 'CC')

    def test_cx_metadata_parsed_not_truncated(self):
        # The CX radical annotation must survive in ordinary SMILES.
        source = 'CC |^1:0|'
        tokenize = lambda text: text.replace('|', '')
        tokens, slots = augment(source, 1, tokenize)
        self.assertEqual(identity(tokens[0]), identity(source))
        self.assertNotEqual(identity(tokens[0]), identity('CC'))
        self.assertEqual(slots, [])

    def test_invalid_count_is_fatal(self):
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                augment('CCO', count, lambda x: x)


if __name__ == '__main__':
    unittest.main()
