"""Audit Stage 1 and Stage 2 dataset splits for leakage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rdkit import Chem


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True) if mol else ''


def canonical_reaction_key(reactants: str, product: str) -> str:
    reactant_parts = sorted(
        canonical for part in str(reactants).split('.')
        if (canonical := canonical_smiles(part.strip()))
    )
    product_parts = sorted(
        canonical for part in str(product).split('.')
        if (canonical := canonical_smiles(part.strip()))
    )
    if not reactant_parts or not product_parts:
        return ''
    return '.'.join(reactant_parts) + '>>' + '.'.join(product_parts)


def audit_stage2(data_root: Path) -> list[str]:
    lines = ['Stage2 split audit', 'family\ttrain\tval\ttest\ttrain_val\ttrain_test\tval_test']
    for family_dir in sorted(data_root.glob('reaction_processed_*_catmerge')):
        second_dir = family_dir / 'For_second_part_model'
        split_keys = {}
        for split in ['train', 'validate', 'test']:
            keys = set()
            with (second_dir / f'Splitted_second_{split}_labels_processed.txt').open(encoding='utf-8') as handle:
                for line in handle:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 3:
                        key = canonical_reaction_key(parts[1], parts[2])
                        if key:
                            keys.add(key)
            split_keys[split] = keys
        lines.append(
            '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                family_dir.name,
                len(split_keys['train']),
                len(split_keys['validate']),
                len(split_keys['test']),
                len(split_keys['train'] & split_keys['validate']),
                len(split_keys['train'] & split_keys['test']),
                len(split_keys['validate'] & split_keys['test']),
            )
        )
    return lines


def audit_stage1(data_root: Path) -> list[str]:
    lines = [
        '',
        'Stage1 route audit',
        'family\traw_train\traw_val\traw_test\ttrain_val\ttrain_test\tval_test\tval_vs_stage2_val\ttest_vs_stage2_test\ttrainval_vs_stage2_test',
    ]
    dataset_root = data_root / 'editretro' / 'datasets'
    for family_dir in sorted(dataset_root.glob('REAXYS_*_SINGLE_CATMERGE')):
        family_name = family_dir.name.replace('REAXYS_', '').replace('_SINGLE_CATMERGE', '')
        stage2_train = set()
        stage2_val = set()
        stage2_test = set()
        for split_name, holder in [('train', stage2_train), ('validate', stage2_val), ('test', stage2_test)]:
            stage2_path = (
                data_root / f'reaction_processed_{family_name}_catmerge'
                / 'For_second_part_model'
                / f'Splitted_second_{split_name}_labels_processed.txt'
            )
            with stage2_path.open(encoding='utf-8') as handle:
                for line in handle:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 3:
                        key = canonical_reaction_key(parts[1], parts[2])
                        if key:
                            holder.add(key)

        split_keys = {}
        for split in ['train', 'val', 'test']:
            keys = set()
            with (family_dir / 'raw' / f'raw_{split}.csv').open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    reactants, product = row['reactants>reagents>production'].split('>>', 1)
                    key = canonical_reaction_key(reactants, product)
                    if key:
                        keys.add(key)
            split_keys[split] = keys

        lines.append(
            '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                family_name,
                len(split_keys['train']),
                len(split_keys['val']),
                len(split_keys['test']),
                len(split_keys['train'] & split_keys['val']),
                len(split_keys['train'] & split_keys['test']),
                len(split_keys['val'] & split_keys['test']),
                len(split_keys['val'] ^ stage2_val),
                len(split_keys['test'] ^ stage2_test),
                len((split_keys['train'] | split_keys['val']) & stage2_test),
            )
        )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit ProSys dataset split leakage.')
    parser.add_argument('--strict', action='store_true', help='Exit non-zero if any leakage is detected.')
    args = parser.parse_args()

    data_root = Path('data')
    stage2_lines = audit_stage2(data_root)
    stage1_lines = audit_stage1(data_root)

    for line in stage2_lines:
        print(line)
    for line in stage1_lines:
        print(line)

    if args.strict:
        leaks = []
        for line in stage2_lines[2:]:
            fields = line.split('\t')
            if any(int(value) != 0 for value in fields[-3:]):
                leaks.append(line)
        for line in stage1_lines[3:]:
            fields = line.split('\t')
            if any(int(value) != 0 for value in fields[-6:]):
                leaks.append(line)
        if leaks:
            raise SystemExit('split leakage detected:\n' + '\n'.join(leaks))


if __name__ == '__main__':
    main()
