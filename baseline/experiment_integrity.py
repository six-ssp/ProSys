"""Bind baseline exports and repeat results to their actual inputs.

Content consistency is not a certificate that the Stage 1 model was trained
without leakage. Stage 1 scientific admission must be established separately.
"""

from __future__ import annotations

import json
from pathlib import Path

from prosys_shared.cache_integrity import CacheMismatchError, fingerprint, require_manifest, save_manifest
from prosys_shared.mainline import split_file_for_family


def artifact_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*")
            if p.is_file() and p.name not in {"content_manifest.json", "content_manifest.json.tmp"}}


def source_files(repo_root: Path) -> dict[str, Path]:
    inputs = {}
    for directory in ("baseline", "prosys_shared", "stage3_XGBoost"):
        for p in (repo_root / directory).rglob("*.py"):
            # Third-party source archives are not used by the PyTorch adapters.
            if directory == "baseline" and len(p.relative_to(repo_root / directory).parts) > 2:
                continue
            inputs["code/" + str(p.relative_to(repo_root))] = p
    verifier = repo_root / "scripts" / "run_verified_mainline.py"
    if verifier.is_file():
        inputs["code/scripts/run_verified_mainline.py"] = verifier
    return inputs


def family_inputs(repo_root: Path, route_root: Path, validation_root: Path | None,
                  family: str) -> dict[str, Path]:
    from scripts.run_verified_mainline import verify_route_manifest
    inputs = {"split/" + s: split_file_for_family(repo_root, family, s)
              for s in ("train", "val", "test")}
    for split, root in (("test", route_root), ("val", validation_root)):
        if root is None:
            continue
        cache_path = root / family / "route_cache.json"
        verify_route_manifest(cache_path, inputs["split/" + split], family)
        inputs["routes/" + split] = cache_path
        cache = json.loads(cache_path.read_text())
        checkpoint = Path(cache["checkpoint"])
        inputs["checkpoint/" + split] = checkpoint if checkpoint.is_absolute() else repo_root / checkpoint
    return inputs


def export_binding(repo_root: Path, route_root: Path, validation_root: Path | None,
                   family: str, method: str) -> dict:
    return fingerprint({"kind": "baseline_export", "family": family, "method": method},
                       {**source_files(repo_root), **family_inputs(repo_root, route_root, validation_root, family)})


def verify_export(input_root: Path, repo_root: Path, route_root: Path,
                  validation_root: Path, family: str, method: str) -> dict:
    directory = input_root / method / family
    binding = export_binding(repo_root, route_root, validation_root, family, method)
    require_manifest(directory / "content_manifest.json", binding, artifact_files(directory))
    return binding


def admit_study(output_root: Path, binding: dict, *, resume: bool) -> None:
    manifest = output_root / "experiment_binding.json"
    entries = [p for p in output_root.iterdir() if p.name != ".experiment.lock"]
    if entries:
        if not resume:
            raise CacheMismatchError("Existing baseline study requires --resume and identical inputs; otherwise use a new output root")
        try:
            stored = json.loads(manifest.read_text())
        except (OSError, ValueError) as exc:
            raise CacheMismatchError("Legacy/partial study has no input binding; use a new output root") from exc
        if stored != binding:
            raise CacheMismatchError("Baseline data, routes, code or settings changed; use a new output root")
        return
    manifest.write_text(json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n")


def reuse_completed(directory: Path, binding: dict) -> bool:
    if not directory.exists() or not any(directory.iterdir()):
        return False
    if not (directory / "complete.json").is_file():
        raise CacheMismatchError(f"Incomplete unverified baseline run: {directory}; preserve it and use a new output root")
    require_manifest(directory / "content_manifest.json", binding, artifact_files(directory))
    return True


def seal_completed(directory: Path, binding: dict) -> None:
    save_manifest(directory / "content_manifest.json", binding, artifact_files(directory))
