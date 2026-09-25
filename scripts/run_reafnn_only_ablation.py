#!/usr/bin/env python3
"""Run an isolated, resumable no-KNN ablation with compact row-level evidence."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from ablation.reafnn_only_experiment import KNN_EVIDENCE_COLUMNS, ReaFNNOnlyPoolBuilder
from baseline.experiment_integrity import family_inputs
from prosys_shared.mainline import parse_families_arg, split_file_for_family
from scripts import run_current_mainline_matched_ablations as reporting

ARM = "reafnn_only"
SOURCE_FILES = (
    "ablation/reafnn_only_experiment.py", "scripts/run_reafnn_only_ablation.py",
    "scripts/run_stage23_mainline_non_oracle.py", "stage2_ReaFNN/knn_condition_selector.py",
    "stage2_ReaFNN/reafnn_selector.py", "stage3_XGBoost/xgb_reranker.py",
    "prosys_shared/mainline.py", "prosys_shared/features.py",
    "baseline/experiment_integrity.py", "scripts/run_verified_mainline.py",
    "scripts/run_current_mainline_matched_ablations.py",
    "prosys_shared/cache_integrity.py",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def input_bindings(families, route_root, validation_root):
    bindings = {}
    for family in families:
        paths = family_inputs(ROOT, route_root, validation_root, family)
        hashes = {name: sha(path) for name, path in paths.items()}
        if hashes['checkpoint/test'] != hashes['checkpoint/val']:
            raise ValueError('Test and validation routes use different expert weights: ' + family)
        bindings[family] = hashes
    return bindings


def require_cached_evidence(result, destination, expected):
    if result.get('bound_family_inputs') != expected:
        raise ValueError('Cached ablation belongs to different or unbound inputs; use a new output root')
    hashes = result.get('retained_evidence_sha256', {})
    if set(hashes) != {'candidate_audit.json', 'test_candidates.csv.gz'}:
        raise ValueError('Cached ablation lacks retained evidence bindings')
    for name, digest in hashes.items():
        if sha(destination / name) != digest:
            raise ValueError('Cached ablation evidence changed: ' + name)


def reference_bindings(reference_root, families, seeds, bindings):
    paths = []
    for seed in seeds:
        for family in families:
            folder = reference_root / f'seed_{seed}' / family
            path = folder / 'provenance.json'
            provenance = json.loads(path.read_text())
            expected = bindings[family]
            if (provenance.get('family') != family or provenance.get('seed') != seed or
                    provenance.get('route_test') != expected['routes/test'] or
                    provenance.get('route_validation') != expected['routes/val'] or
                    provenance.get('splits') != {s: expected['split/' + s] for s in ('train', 'val', 'test')}):
                raise ValueError('Mainline comparison uses different routes/splits: ' + str(folder))
            result_path = folder / 'result.json'
            result = json.loads(result_path.read_text())
            if result.get('family') != family or result.get('seed') != seed:
                raise ValueError('Mainline reference family/seed mismatch')
            paths.extend((path, result_path))
    return paths


def checked_reference_metrics(reference, reference_root, families, seeds):
    stored = reporting._load_mainline_reference(reference)
    if stored is None:
        raise ValueError('Missing explicit mainline macro table')
    rows = [reporting._metric_row(json.loads(
        (reference_root / f'seed_{seed}' / family / 'result.json').read_text()),
        arm='full_mainline', seed=seed) for seed in seeds for family in families]
    expected = {'arm': 'full_mainline'}
    for metric, _ in reporting.METRIC_FIELDS:
        values = [statistics.fmean(row[metric] for row in rows if row['seed'] == seed) for seed in seeds]
        expected[metric + '_mean'] = statistics.fmean(values)
        expected[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
    for name, value in stored.items():
        if name != 'arm' and value is not None and not math.isclose(
                value, expected[name], rel_tol=0, abs_tol=1e-8):
            raise ValueError('Mainline macro table does not match paired per-job results: ' + name)
    return expected


def child(argv):
    from scripts import run_stage23_mainline_non_oracle as pipeline
    # Explicit dependency substitution is confined to this child process.
    # The maintained runner and its default configuration are not edited.
    pipeline.KNNContextPoolBuilder = ReaFNNOnlyPoolBuilder
    sys.argv = [str(Path(__file__))] + argv
    pipeline.main()


def retain_audit(result, destination):
    scored_file = Path(result["scored_test_file"])
    frame = pd.read_csv(scored_file, keep_default_na=False)
    candidate_file = Path(result["candidate_table"])
    training_dir = candidate_file.parent
    checks = {}
    for split in ("train", "val", "test"):
        path = training_dir / (split + ".csv")
        df = pd.read_csv(path, keep_default_na=False)
        for field in KNN_EVIDENCE_COLUMNS:
            if field in df:
                assert (pd.to_numeric(df[field], errors="raise") == 0).all(), (split, field)
        assert (pd.to_numeric(df["from_reafnn_novel"], errors="raise") == 0).all()
        assert df.groupby("sample_index").size().max() <= 200
        checks[split] = {"rows": len(df), "sha256": sha(path), "knn_fields_zero": True}
    keys = ["sample_index", "reaction_id", "product", "reactants", "reagent_norm", "solvent_norm"]
    identities = frame[keys].astype(str).sort_values(keys, kind="mergesort")
    candidate_hash = hashlib.sha256(identities.to_csv(index=False).encode()).hexdigest()
    wanted = keys + [c for c in frame if c in (
        "label", "route_match", "context_match", "retro_rank", "retro_probability",
        "stage2_initial_score", "xgb_score", "xgb_score_raw", "stage3_score_fused",
        "temperature_gold", "temperature_pred")]
    # Compact per-candidate evidence supports later case, subgroup and pairing
    # audits without retaining large feature tables or neural checkpoints.
    frame[wanted].to_csv(destination / "test_candidates.csv.gz", index=False,
                         compression={"method": "gzip", "compresslevel": 6})
    audit = {"candidate_identity_sha256": candidate_hash,
             "candidate_identity_columns": keys, "tables": checks,
             "scored_file_sha256": sha(scored_file),
             "retained_rows": len(frame), "no_novel_contexts": True}
    write_json(audit, destination / "candidate_audit.json")
    return audit


def main():
    if "--child" in sys.argv:
        child(sys.argv[sys.argv.index("--child") + 1:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--output_root", type=Path, default=ROOT / "Experiment/stage2_reafnn_only_multiseed_20260913")
    parser.add_argument("--scratch_root", type=Path, default=ROOT / "outputs/reafnn_only_scratch_20260913")
    parser.add_argument("--route_root", type=Path, default=ROOT / "outputs/stage1_routes")
    parser.add_argument("--post_fusion_validation_route_root", type=Path,
                        default=ROOT / "outputs/stage1_routes_validation")
    parser.add_argument("--mainline_reference", type=Path, required=True,
                        help="Explicit matching mainline macro CSV; no historical implicit comparison")
    parser.add_argument("--mainline_compact_root", type=Path, required=True,
                        help="Matching mainline per-family/seed records")
    parser.add_argument("--keep_scratch", action="store_true")
    args = parser.parse_args()
    families = parse_families_arg(args.families)
    seeds = reporting._parse_seeds(args.seeds)
    output, scratch = args.output_root.resolve(), args.scratch_root.resolve()
    route_root = args.route_root.resolve()
    validation_root = args.post_fusion_validation_route_root.resolve()
    reference = args.mainline_reference.resolve()
    reference_root = args.mainline_compact_root.resolve()
    bindings = input_bindings(families, route_root, validation_root)
    reference_files = [reference] + reference_bindings(reference_root, families, seeds, bindings)
    reference_hashes = {str(path): sha(path) for path in reference_files}
    reference_metrics = checked_reference_metrics(reference, reference_root, families, seeds)
    if scratch == ROOT or output == scratch or scratch in output.parents or output in scratch.parents:
        parser.error("Output and scratch must be separate, non-nested study directories")
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {"arm": ARM, "seeds": seeds, "families": families,
                "intervention": "remove KNN proposals and all KNN evidence; retrain XGB-LTR",
                "fixed_w": 0.0, "contexts_per_route": 20, "neural_prefilter": 64,
                "temperature": "skipped; cannot affect system ranking",
                "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
                "family_inputs": bindings, "mainline_reference_sha256": reference_hashes,
                "route_root": str(route_root), "validation_route_root": str(validation_root)}
    manifest = output / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise RuntimeError("Study manifest differs; use a new output directory")
    write_json(protocol, manifest)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(args.cpu_threads)
    env["TMPDIR"] = str(scratch)
    env["PYTHONUNBUFFERED"] = "1"
    rows = []
    for seed in seeds:
        for family in families:
            dest = output / "compact" / f"seed_{seed}" / family
            record = dest / "result.json"
            if record.exists():
                result = json.loads(record.read_text())
                require_cached_evidence(result, dest, bindings[family])
                rows.append(reporting._metric_row(result, arm=ARM, seed=seed))
                continue
            if shutil.disk_usage(scratch).free < 5 * 1024**3:
                raise RuntimeError("Less than 5 GiB free on scratch disk; refusing next family")
            dest.mkdir(parents=True, exist_ok=True)
            job = scratch / f"seed_{seed}" / family
            if job.exists():
                raise RuntimeError(f"Incomplete scratch job exists: {job}; inspect before retry")
            command = [sys.executable, str(Path(__file__).resolve()), "--child",
                "--repo_root", str(ROOT), "--families", family, "--seed", str(seed),
                "--output_root", str(job), "--route_root", str(route_root),
                "--reafnn_device", args.device, "--skip_temperature",
                "--reafnn_enable_independent_post_fusion", "--no-reafnn_enable_knn_wide_refinement",
                "--reafnn_independent_contexts", "64", "--reafnn_post_fusion_weights", "0.0",
                "--reafnn_post_fusion_validation_route_root", str(validation_root)]
            start = time.time()
            print(f"START seed={seed} family={family}", flush=True)
            write_json({"status": "running", "seed": seed, "family": family,
                        "command": command, "started_unix": start}, output / "status.json")
            with (dest / "run.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            raw = job / family / "knn_xgb/non_oracle/result.json"
            result = json.loads(raw.read_text())
            assert result["seed"] == seed and result["family"] == family
            p = result["model"]["stage2_protocol"]
            assert p["reafnn_post_fusion_calibration"]["selected_knn_weight"] == 0.0
            assert p["reafnn_post_fusion_calibration"]["protocol"]["knn_retrieval_enabled"] is False
            audit = retain_audit(result, dest)
            result["baseline"] = "reafnn_only_xgb_ltr"
            p["architecture"] = "reafnn_only"
            p["knn_retrieval_enabled"] = False
            p["reafnn_candidate_policy"] = "historical_reafnn_only_no_knn_evidence"
            result["candidate_audit"] = audit
            result["elapsed_seconds"] = time.time() - start
            result["input_sha256"] = {
                **{split: sha(split_file_for_family(ROOT, family, split)) for split in ("train", "val", "test")},
                "stage1_test": sha(route_root / family / "route_cache.json"),
                "stage1_validation": sha(validation_root / family / "route_cache.json"),
            }
            if input_bindings([family], route_root, validation_root)[family] != bindings[family]:
                raise ValueError('Ablation inputs changed during training: ' + family)
            result['bound_family_inputs'] = bindings[family]
            result['retained_evidence_sha256'] = {
                name: sha(dest / name) for name in ('candidate_audit.json', 'test_candidates.csv.gz')}
            for name in ("xgb_ranker_meta.json", "xgb_temperature_meta.json"):
                source = raw.parent / "model" / name
                if source.exists():
                    shutil.copy2(source, dest / name)
            compact = reporting._compact_result(result)
            write_json(compact, record)
            rows.append(reporting._metric_row(result, arm=ARM, seed=seed))
            reporting._write_csv(output / "per_family_seed_metrics.csv", rows)
            if not args.keep_scratch:
                shutil.rmtree(job)
            print(f"DONE seed={seed} family={family} sys10={100*result['metrics']['system_top10_all']:.2f}% elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    # Reuse the tested aggregation, with an explicit arm label in the report.
    if input_bindings(families, route_root, validation_root) != bindings or \
            {str(path): sha(path) for path in reference_files} != reference_hashes or \
            {name: sha(ROOT / name) for name in SOURCE_FILES} != protocol['source_sha256']:
        raise ValueError('Ablation inputs, code or mainline references changed; refusing final report')
    reporting._write_report(output_root=output, rows=rows, families=families, seeds=seeds,
        mainline_reference=reference_metrics,
        mainline_compact_root=reference_root)
    write_json({"status": "complete", "records": len(rows)}, output / "status.json")
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
