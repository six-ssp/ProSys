#!/usr/bin/env python3
"""Audit dot-splitting against RDKit connected components without changing data."""

import csv
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from rdkit import Chem, RDLogger
    from prosys_shared.features import canonicalize_reaction_side
    from prosys_shared.mainline import FAMILY_ORDER, split_file_for_family
    RDLogger.DisableLog("rdApp.error")
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True,
                        help='New audit directory; never overwrite historical evidence')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'canonical_fragment_audit.json').exists():
        raise FileExistsError(output / 'canonical_fragment_audit.json')
    mismatches, invalid, overlaps, counts = [], [], [], []
    for family in FAMILY_ORDER:
        sets = {}
        for split in ("train", "val", "test"):
            keys = set()
            n = 0
            with split_file_for_family(ROOT, family, split).open() as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 3:
                        continue
                    n += 1
                    sides = []
                    for name, value in (("reactants", fields[1]), ("product", fields[2])):
                        mol = Chem.MolFromSmiles(value)
                        if mol is None:
                            invalid.append({"family": family, "split": split, "reaction_id": fields[0], "side": name})
                            sides.append("")
                            continue
                        complete = ".".join(sorted(Chem.MolToSmiles(m, canonical=True)
                            for m in Chem.GetMolFrags(mol, asMols=True)))
                        legacy = canonicalize_reaction_side(value)
                        if complete != legacy:
                            mismatches.append({"family": family, "split": split, "reaction_id": fields[0],
                                "side": name, "original": value, "complete_components": complete,
                                "current_dot_split": legacy})
                        sides.append(complete)
                    if all(sides):
                        keys.add(tuple(sides))
            sets[split] = keys
            counts.append({"family": family, "split": split, "condition_rows": n, "complete_reaction_keys": len(keys)})
        overlaps.append({"family": family, "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]), "val_test": len(sets["val"] & sets["test"])})
    result = {"scope": "condition splits only; full-molecule parsing followed by connected-component canonicalization",
        "mismatches": mismatches, "invalid_whole_sides": invalid, "overlaps": overlaps, "counts": counts,
        "data_or_models_modified": False}
    (output / "canonical_fragment_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mismatched_sides": len(mismatches), "invalid_whole_sides": len(invalid), "overlaps": overlaps}, indent=2))


if __name__ == "__main__":
    main()
