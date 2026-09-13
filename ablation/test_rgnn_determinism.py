import unittest
from pathlib import Path
import tempfile
import torch
from stage3_XGBoost.reaction_gnn_features import (
    ReactionGNNConfig, ReactionGNNFeatureEncoder, deterministic_graph_runtime,
    train_reaction_gnn_feature_model,
)
from scripts.run_stage23_mainline_non_oracle import _reaction_gnn_cache_name


class DeterminismTests(unittest.TestCase):
    def test_old_configuration_remains_legacy(self):
        self.assertFalse(ReactionGNNConfig(**{'random_state': 0}).deterministic)

    def test_runtime_is_restored_even_on_error(self):
        original = torch.are_deterministic_algorithms_enabled()
        with self.assertRaisesRegex(ValueError, 'probe'):
            with deterministic_graph_runtime(True):
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                raise ValueError('probe')
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), original)

    def test_graph_cache_separates_numerical_protocols_and_hyperparameters(self):
        legacy = _reaction_gnn_cache_name(ReactionGNNConfig())
        fixed = _reaction_gnn_cache_name(ReactionGNNConfig(deterministic=True))
        changed_lr = _reaction_gnn_cache_name(ReactionGNNConfig(learning_rate=0.002))
        self.assertEqual(len({legacy, fixed, changed_lr}), 3)

    def test_repeated_fit_and_inference_through_public_api(self):
        # Synthetic interface test, not a held-out chemistry-performance study.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'tiny.tsv'
            routes = [('CCO', 'CC=O'), ('CNC', 'C=N'), ('CCBr', 'CCO'), ('C=O', 'CO')]
            data.write_text(''.join(f'{i}\t{r}\t{p}\t50\tO\tCCO\t25\n'
                                    for i, (r, p) in enumerate(routes)))
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            config = ReactionGNNConfig(hidden_dim=8, embedding_dim=8,
                message_passing_steps=1, batch_size=2, max_epochs=2,
                device=device, random_state=7, deterministic=True)
            before = torch.are_deterministic_algorithms_enabled()
            states, embeddings = [], []
            for repeat in (0, 1):
                out = root / str(repeat)
                train_reaction_gnn_feature_model(data, data, out, config=config)
                states.append(torch.load(out / 'reaction_gnn.pt', map_location='cpu',
                                         weights_only=False)['model_state'])
                encoder = ReactionGNNFeatureEncoder(out, device=device)
                embeddings.append(encoder.encode_routes(routes))
                self.assertEqual(torch.are_deterministic_algorithms_enabled(), before)
            for name in states[0]:
                self.assertTrue(torch.equal(states[0][name], states[1][name]), name)
            self.assertTrue((embeddings[0] == embeddings[1]).all())


if __name__ == '__main__':
    unittest.main()
