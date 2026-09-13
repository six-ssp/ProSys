import tempfile
import unittest
from pathlib import Path

from prosys_shared.cache_integrity import (
    CacheMismatchError, fingerprint, require_manifest, save_manifest,
)


class CacheIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "data"
        self.model = self.root / "model"
        self.source.write_bytes(b"train")
        self.model.write_bytes(b"weights")
        self.config = {"seed": 0, "k": 64}
        self.binding = fingerprint(self.config, {"data": self.source})
        self.manifest = self.root / "manifest.json"
        self.outputs = {"model": self.model}
        save_manifest(self.manifest, self.binding, self.outputs)

    def test_identical(self):
        require_manifest(self.manifest, self.binding, self.outputs)

    def test_same_path_changed_data(self):
        self.source.write_bytes(b"other")
        with self.assertRaises(CacheMismatchError):
            require_manifest(self.manifest, fingerprint(self.config, {"data": self.source}), self.outputs)

    def test_changed_seed(self):
        with self.assertRaises(CacheMismatchError):
            require_manifest(self.manifest, fingerprint({"seed": 1, "k": 64}, {"data": self.source}), self.outputs)

    def test_tampered_checkpoint(self):
        self.model.write_bytes(b"changed")
        with self.assertRaises(CacheMismatchError):
            require_manifest(self.manifest, self.binding, self.outputs)

    def test_missing_checkpoint(self):
        self.model.unlink()
        with self.assertRaises(CacheMismatchError):
            require_manifest(self.manifest, self.binding, self.outputs)

    def test_untracked_artifact(self):
        with self.assertRaises(CacheMismatchError):
            require_manifest(self.manifest, self.binding, {**self.outputs, "extra": self.source})

    def test_legacy_missing_manifest(self):
        self.manifest.unlink()
        with self.assertRaises(CacheMismatchError):
            require_manifest(self.manifest, self.binding, self.outputs)


if __name__ == "__main__":
    unittest.main()
