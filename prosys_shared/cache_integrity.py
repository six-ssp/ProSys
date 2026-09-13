"""Fail-closed content manifests for reproducible artifacts, not path-only caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CacheMismatchError(RuntimeError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(config, inputs):
    # Labels, rather than absolute paths, permit relocation of identical data.
    content = {"schema": 1, "config": config,
               "inputs": {k: file_sha256(p) for k, p in sorted(inputs.items())}}
    return json.loads(json.dumps(content, sort_keys=True, allow_nan=False))


def save_manifest(path, binding, outputs):
    path = Path(path)
    record = {"binding": binding, "outputs": {k: file_sha256(p) for k, p in sorted(outputs.items())}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def require_manifest(path, binding, outputs):
    try:
        record = json.loads(Path(path).read_text())
        if record.get("binding") != binding:
            raise CacheMismatchError("Input, source or configuration fingerprint changed")
        actual = {k: file_sha256(p) for k, p in sorted(outputs.items())}
        if record.get("outputs") != actual:
            raise CacheMismatchError("Cached table/model contents or artifact membership changed")
    except (OSError, ValueError) as exc:
        raise CacheMismatchError(f"Missing or invalid cache evidence: {path}") from exc
    return record
