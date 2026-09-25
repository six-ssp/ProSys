import unittest
from types import SimpleNamespace

from scripts.audit_50k_paper_auxiliary import condition_metrics, selected_weight, fusion_replay


class PaperAuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.queries = [dict(sample_index=i, reaction_id=str(i), product='CC', routes=[])
                        for i in range(2)]
        self.predictions = [dict(q, contexts=[]) for q in self.queries]
        self.gold = {(str(i), 'CC'): {('', 'O')} for i in range(2)}

    def test_condition_hit_does_not_require_route(self):
        self.predictions[0]['contexts'] = [dict(context_rank=1, reagent_norm='', solvent_norm='O')]
        result = condition_metrics(self.queries, self.predictions, self.gold)
        self.assertEqual(result['num_product_queries'], 2)
        self.assertEqual(result['condition_top1_all'], .5)

    def test_missing_prediction_is_not_dropped(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            condition_metrics(self.queries, self.predictions[:1], self.gold)

    def test_duplicate_context_rejected(self):
        self.predictions[0]['contexts'] = [dict(context_rank=i, reagent_norm='', solvent_norm='O') for i in (1, 2)]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            condition_metrics(self.queries, self.predictions, self.gold)

    def test_bad_rank_rejected(self):
        self.predictions[0]['contexts'] = [dict(context_rank=2, reagent_norm='', solvent_norm='O')]
        with self.assertRaisesRegex(ValueError, 'ranks'):
            condition_metrics(self.queries, self.predictions, self.gold)

    def test_missing_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reference'):
            condition_metrics(self.queries, self.predictions, {})

    def test_tie_uses_larger_knn_weight(self):
        self.assertEqual(selected_weight([0., .5, 1.], [.4, .6, .6]), 1.)

    def test_no_forced_neural_contribution(self):
        self.assertEqual(selected_weight([0., .5, 1.], [.1, .2, .8]), 1.)

    def test_invalid_grid_rejected(self):
        for grid, coverage in [([0., 1., 1.], [.2, .3, .4]), ([0., 1.], [.2, float('nan')]),
                               ([.2, .8], [.3, .4])]:
            with self.assertRaises(ValueError):
                selected_weight(grid, coverage)

    def fusion_fixture(self):
        route = SimpleNamespace(sample_index=0, reaction_id='0', product='CC', reactants='C')
        bucket = SimpleNamespace(route_keys={'C'}, exact_keys={('C', '', 'O')})
        builder = SimpleNamespace(
            _independent_post_fusion_state=lambda *a, **k: [],
            _select_independent_post_fusion_contexts=lambda state, knn_weight:
                [dict(reagent_norm='', solvent_norm='O')] if knn_weight else [])
        calibration = dict(protocol={'weight_grid': [0., .5, 1.]},
            selected_knn_weight=1., selected_reafnn_weight=0.,
            validation_stage2_candidate_coverage=.5,
            weight_candidates=[dict(knn_weight=w, validation_sample_identities=2,
                validation_exact_candidate_hits=int(w > 0), validation_route_records=1,
                validation_matchable_route_records=1,
                validation_stage2_candidate_coverage=.5 if w else 0.) for w in (0., .5, 1.)])
        return builder, [route], self.queries, {('0', 'CC'): bucket}, calibration

    def test_fusion_keeps_no_route_query_in_denominator(self):
        result = fusion_replay(*self.fusion_fixture())
        self.assertEqual(result['n_validation_queries'], 2)
        self.assertEqual(result['fused_coverage'], .5)
        self.assertEqual(result['selected_knn_weight'], 1.)

    def test_fusion_wrong_support_rejected(self):
        args = self.fusion_fixture()
        args[-1]['weight_candidates'][0]['validation_sample_identities'] = 1
        with self.assertRaisesRegex(ValueError, 'counts'):
            fusion_replay(*args)

    def test_fusion_wrong_selected_weight_rejected(self):
        args = self.fusion_fixture()
        args[-1]['selected_knn_weight'] = .5
        with self.assertRaisesRegex(ValueError, 'weight differs'):
            fusion_replay(*args)


if __name__ == '__main__':
    unittest.main()
