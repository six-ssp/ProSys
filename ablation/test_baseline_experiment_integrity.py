import json
import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline import experiment_integrity as integrity
from baseline import run_multiseed_baselines as runner
from prosys_shared.cache_integrity import CacheMismatchError, save_manifest
from prosys_shared.mainline import split_file_for_family


class BaselineIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.binding = {"schema": 1, "config": {"seed": 0}, "inputs": {"route": "a"}}

    def test_exact_study_resume_requires_explicit_flag(self):
        integrity.admit_study(self.root, self.binding, resume=False)
        with self.assertRaises(CacheMismatchError):
            integrity.admit_study(self.root, self.binding, resume=False)
        integrity.admit_study(self.root, self.binding, resume=True)

    def test_changed_study_preserves_old_binding(self):
        integrity.admit_study(self.root, self.binding, resume=False)
        manifest = self.root / "experiment_binding.json"
        before = manifest.read_bytes()
        with self.assertRaises(CacheMismatchError):
            integrity.admit_study(self.root, {**self.binding, "inputs": {"route": "b"}}, resume=True)
        self.assertEqual(before, manifest.read_bytes())

    def test_legacy_study_is_not_silently_upgraded(self):
        (self.root / "complete.json").write_text('{}')
        with self.assertRaises(CacheMismatchError):
            integrity.admit_study(self.root, self.binding, resume=True)
        self.assertFalse((self.root / "experiment_binding.json").exists())

    def test_completion_marker_alone_is_insufficient(self):
        (self.root / "complete.json").write_text('{}')
        with self.assertRaises(CacheMismatchError):
            integrity.reuse_completed(self.root, self.binding)

    def test_all_completed_output_bytes_are_bound(self):
        (self.root / "complete.json").write_text('{}')
        metrics = self.root / "metrics.json"
        metrics.write_text('{"score": 0.2}')
        integrity.seal_completed(self.root, self.binding)
        self.assertTrue(integrity.reuse_completed(self.root, self.binding))
        metrics.write_text('{"score": 0.9}')
        with self.assertRaises(CacheMismatchError):
            integrity.reuse_completed(self.root, self.binding)

    def test_partial_run_not_silently_resumed(self):
        (self.root / "unfinished_model.pt").write_bytes(b'partial')
        with self.assertRaises(CacheMismatchError):
            integrity.reuse_completed(self.root, self.binding)

    def test_export_rejects_same_path_changed_route_bytes(self):
        route = self.root / "route.json"
        route.write_text('{"prediction": 1}')
        directory = self.root / "exports" / "sequential_fnn" / "F"
        directory.mkdir(parents=True)
        (directory / "train_routes.jsonl").write_text('{}\n')
        with patch.object(integrity, "family_inputs", return_value={"route": route}):
            binding = integrity.export_binding(self.root, self.root, self.root, "F", "sequential_fnn")
            save_manifest(directory / "content_manifest.json", binding, integrity.artifact_files(directory))
            integrity.verify_export(self.root / "exports", self.root, self.root, self.root, "F", "sequential_fnn")
            route.write_text('{"prediction": 2}')
            with self.assertRaises(CacheMismatchError):
                integrity.verify_export(self.root / "exports", self.root, self.root, self.root, "F", "sequential_fnn")

    def test_nb_uses_this_study_and_reuses_only_verified_output(self):
        def fake_run(command, *, repo_root):
            out = Path(command[command.index("--output-root") + 1])
            metadata = out / "product_naive_bayes" / "F" / "run_metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({"test_metrics": {"system_top10_all": 0.25}}))
        kwargs = dict(repo_root=self.root, output_root=self.root, route_root=self.root / "new_test",
                      validation_route_root=self.root / "new_val", families=["F"],
                      top_contexts=20, binding=self.binding)
        with patch.object(runner, "_run", side_effect=fake_run) as run:
            runner._run_deterministic_nb(**kwargs)
            runner._run_deterministic_nb(**kwargs)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--route-root") + 1], str(self.root / "new_test"))
            self.assertEqual(command[command.index("--validation-route-root") + 1], str(self.root / "new_val"))
        self.assertEqual(runner._load_deterministic_nb_rows(self.root, ["F"])[0]["sys10"], 0.25)

    def test_missing_current_nb_never_falls_back_to_historical_results(self):
        with self.assertRaises(FileNotFoundError):
            runner._load_deterministic_nb_rows(self.root, ["F"])

    def test_retention_compression_is_lossless(self):
        source = self.root / 'full.csv'
        source.write_bytes(b'sample_index,score\r\n0,0.12345678901234567\r\n' * 1000)
        destination = self.root / 'retained' / 'full.csv.gz'
        runner._copy_verified(source, destination, compressed=True)
        with gzip.open(destination, 'rb') as reader:
            self.assertEqual(reader.read(), source.read_bytes())
        self.assertLess(destination.stat().st_size, source.stat().st_size)

    def test_corrupted_metadata_copy_is_rejected(self):
        source = self.root / 'source.json'
        source.write_bytes(b'{"complete": true}')
        destination = self.root / 'copied.json'
        with patch.object(runner.shutil, 'copy2', side_effect=lambda s, d: Path(d).write_bytes(b'wrong')):
            with self.assertRaisesRegex(CacheMismatchError, 'Lossless retention'):
                runner._copy_verified(source, destination)

    def test_full_external_retention_requires_prediction_evidence(self):
        work, compact, inputs = self.root / 'work', self.root / 'compact', self.root / 'inputs'
        work.mkdir()
        for filename in ('run_config.json', 'summary.csv', 'summary.json'):
            (work / filename).write_text('{}')
        for method in runner.EXTERNAL_METHODS:
            source = work / method / 'F'
            source.mkdir(parents=True)
            for filename in ('run_metadata.json', 'fusion_selection.json', 'validation_predictions.jsonl',
                             'test_predictions.jsonl', 'validation_candidates.csv', 'test_candidates.csv',
                             'test_labeled_candidates.csv'):
                (source / filename).write_text('{}\n')
            manifest_dir = inputs / method / 'F'
            manifest_dir.mkdir(parents=True)
            for filename in ('val_manifest.jsonl', 'test_manifest.jsonl'):
                (manifest_dir / filename).write_text('{"sample_index": 0}\n')
        files = runner._copy_external_compact(work, compact, ['F'], inputs)
        self.assertIn('sequential_fnn/F/test_labeled_candidates.csv.gz', files)
        (work / 'reaction_gcnn/F/test_predictions.jsonl').unlink()
        with self.assertRaises(FileNotFoundError):
            runner._copy_external_compact(work, self.root / 'incomplete_copy', ['F'], inputs)

    def test_real_export_contract_binds_both_methods_and_checkpoint(self):
        from baseline.external_adapters.build_datasets import _build_indirect_fnn_and_gcnn
        checkpoint = self.root / 'base.pt'
        checkpoint.write_bytes(b'fixture only, not a trained model')
        test_root, val_root = self.root / 'test_routes', self.root / 'val_routes'
        for split, product in (('train', 'CCO'), ('val', 'CCN'), ('test', 'CCC')):
            path = split_file_for_family(self.root, 'F', split)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'{split}\tCC\t{product}\t80\tN\tO\t25\n')
            if split != 'train':
                directory = (test_root if split == 'test' else val_root) / 'F'
                directory.mkdir(parents=True)
                (directory / 'route_cache.json').write_text(json.dumps({
                    'family': 'F', 'checkpoint': str(checkpoint), 'reactions': [{
                        'sample_index': 0, 'reaction_id': split, 'product': product,
                        'gold_reactants': 'CC', 'routes': [{'reactants': 'CO', 'retro_rank': 1,
                                                         'retro_score': -0.2, 'retro_probability': 0.8}],
                    }],
                }))
        export_root = self.root / 'exports'
        _build_indirect_fnn_and_gcnn(self.root, test_root, val_root, export_root, 'F',
                                    ['sequential_fnn', 'reaction_gcnn'])
        for method in runner.EXTERNAL_METHODS:
            integrity.verify_export(export_root, self.root, test_root, val_root, 'F', method)
            records = (export_root / method / 'F' / 'test_stage1_routes.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(records[0])['reactants'], 'CO')
        checkpoint.write_bytes(b'changed checkpoint at same path')
        with self.assertRaises(CacheMismatchError):
            integrity.verify_export(export_root, self.root, test_root, val_root, 'F', 'sequential_fnn')


if __name__ == '__main__':
    unittest.main()
