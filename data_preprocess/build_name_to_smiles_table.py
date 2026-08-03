"""Build a local name-to-canonical-token table from legacy Reaxys label resources.

This script absorbs the still-useful parts of the old top-level ``preprocess_data/``
workflow into ``data_preprocess/``:
- read the archived ``*_emerge*.txt`` label exports
- apply a small set of durable manual corrections / alias merges
- write the repo-local ``name_to_smiles.tsv`` used by the current pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.preprocess import SENTINEL_VALUES, normalize_label

MANUAL_OVERRIDES = {
    'dmap': 'CN(C)c1ccncc1',
    '4-dimethylaminopyridine': 'CN(C)c1ccncc1',
    'thf': 'C1CCOC1',
    'dmf': 'CN(C)C=O',
    'dmf (n,n-dimethyl-formamide)': 'CN(C)C=O',
    'dcm': 'ClCCl',
    'acoh': 'CC(=O)O',
    'tmeda': 'CN(C)CCN(C)C',
    'tetramethylethylenediamine': 'CN(C)CCN(C)C',
    'potassium-t-butoxide': 'CC(C)(C)[O-].[K+]',
    'sodium tertiary butoxide': 'CC(C)(C)[O-].[Na+]',
    'aluminum chloride': '[Cl-].[Cl-].[Cl-].[Al+3]',
    'ferric chloride': '[Cl-].[Cl-].[Cl-].[Fe+3]',
    'petroleum ether': 'petroleum ether',
    'aq. buffer': 'aq. buffer',
    'various solvent(s)': 'various solvent(s)',
    'xantphos': 'CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21',
    'palladium on carbon': 'C.[Pd]',
    'pd/c': 'C.[Pd]',
    '10% pd/c': 'C.[Pd]',
    'copper(l) iodide': '[Cu+].[I-]',
    'copper(i) iodide': '[Cu+].[I-]',
    'beta?cyclodextrin': 'OCC1OC2OC3C(CO)OC(OC4C(CO)OC(OC5C(CO)OC(OC6C(CO)OC(OC7C(CO)OC(OC8C(CO)OC(OC1C(O)C2O)C(O)C8O)C(O)C7O)C(O)C6O)C(O)C5O)C(O)C4O)C(O)C3O',
    'beta-cyclodextrin': 'OCC1OC2OC3C(CO)OC(OC4C(CO)OC(OC5C(CO)OC(OC6C(CO)OC(OC7C(CO)OC(OC8C(CO)OC(OC1C(O)C2O)C(O)C8O)C(O)C7O)C(O)C6O)C(O)C5O)C(O)C4O)C(O)C3O',
    'trifluorormethanesulfonic acid': 'O=S(=O)(O)C(F)(F)F',
    'cetyltrimethylammonim bromide': 'CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]',
    'lithium 1,1,1,3,3,3-hexamethyldisilazide': 'C[Si](C)(C)[N-][Si](C)(C)C.[Li+]',
    "1,1,1,3',3',3'-hexafluoro-propanol": 'OC(C(F)(F)F)C(F)(F)F',
    'hexafluoroisopropanol': 'OC(C(F)(F)F)C(F)(F)F',
    'p-toluenesulfonyl chloride': 'Cc1ccc(S(=O)(=O)Cl)cc1',
    'dichloromethane-d2': 'ClCCl',
    'd(4)-methanol': 'CO',
    '[d3]acetonitrile': 'CC#N',
    '3percent tfa': 'O=C(O)C(F)(F)F',
    'α,α,α-trifluorotoluene': 'FC(F)(F)c1ccccc1',
}


def canonicalize_token(raw_value: str) -> tuple[str | None, int]:
    """
    Convert a raw mapped value to a canonical token.

    Returns ``(token, score)`` where:
    - score=2: canonical SMILES
    - score=1: normalized canonical label string
    - score=0: unusable / missing
    """
    value = str(raw_value).strip()
    if not value or value.lower() in {'none', 'nan'}:
        return None, 0

    mol = Chem.MolFromSmiles(value)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True), 2

    label = normalize_label(value)
    if label is None or label.lower() in SENTINEL_VALUES:
        return None, 0
    return label, 1


def normalized_key(raw_name: str) -> str | None:
    key = normalize_label(raw_name)
    if key is None or key.lower() in SENTINEL_VALUES:
        return None
    return key.lower()


def source_specs(repo_root: Path) -> list[tuple[Path, int]]:
    legacy_root = repo_root / 'data' / 'reaxys_output' / 'label_processed'
    return [
        (legacy_root / 'class_names_reagent_emerge.txt', 0),
        (legacy_root / 'class_names_solvent_emerge.txt', 0),
        (legacy_root / 'class_names_reagent_emerge_backup.txt', 1),
        (legacy_root / 'class_names_solvent_emerge_backup.txt', 1),
    ]


def build_mapping(repo_root: Path) -> tuple[dict[str, str], dict[str, int], list[str]]:
    mapping: dict[str, str] = {}
    metadata: dict[str, tuple[int, int]] = {}
    stats = {'smiles': 0, 'label': 0}
    conflicts: list[str] = []

    for source_path, source_priority in source_specs(repo_root):
        if not source_path.exists():
            continue
        for line_no, line in enumerate(source_path.read_text(encoding='utf-8', errors='ignore').splitlines(), start=1):
            if '\t' not in line:
                continue
            raw_name, raw_value = line.split('\t', 1)
            key = normalized_key(raw_name)
            if key is None:
                continue

            token, score = canonicalize_token(raw_value)
            if token is None:
                continue

            current = metadata.get(key)
            candidate_rank = (score, -source_priority)
            if current is None or candidate_rank > current:
                mapping[key] = token
                metadata[key] = candidate_rank
            elif mapping[key] != token and candidate_rank == current:
                conflicts.append(
                    f'{key}\t{mapping[key]}\t{token}\t{source_path.name}:{line_no}'
                )

    for raw_key, raw_value in MANUAL_OVERRIDES.items():
        key = normalized_key(raw_key)
        token, score = canonicalize_token(raw_value)
        if key is None or token is None:
            continue
        mapping[key] = token
        metadata[key] = (score, 1)

    final_stats = {'smiles': 0, 'label': 0}
    for token in mapping.values():
        if Chem.MolFromSmiles(token) is not None:
            final_stats['smiles'] += 1
        else:
            final_stats['label'] += 1
    return mapping, final_stats, conflicts


def write_mapping(mapping: dict[str, str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for key in sorted(mapping):
            handle.write(f'{key}\t{mapping[key]}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build local name_to_smiles.tsv from legacy label resources.')
    parser.add_argument('--repo_root', type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', type=str, default=str(Path(__file__).resolve().with_name('name_to_smiles.tsv')))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    mapping, stats, conflicts = build_mapping(repo_root)
    write_mapping(mapping, output_path)

    print(f'[name_to_smiles] wrote {len(mapping)} entries -> {output_path}')
    print(f'[name_to_smiles] canonical SMILES entries: {stats["smiles"]}')
    print(f'[name_to_smiles] canonical label entries: {stats["label"]}')
    if conflicts:
        print(f'[name_to_smiles] conflicts kept by priority: {len(conflicts)}')
    else:
        print('[name_to_smiles] conflicts kept by priority: 0')


if __name__ == '__main__':
    main()
