"""Final paired-sequence checks after atom-mapped root-aligned augmentation."""


def reaction_side_fragments(smiles):
    """Preserve mapping/stereochemistry while splitting actual connected graphs."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('Invalid or empty reaction side')
    return [Chem.MolToSmiles(m, canonical=False, isomericSmiles=True)
            for m in Chem.GetMolFrags(mol, asMols=True)]


def filter_nonempty_pairs(sources, targets):
    if len(sources) != len(targets):
        raise ValueError('Augmented source/target counts differ')
    keep_source, keep_target, rejected = [], [], []
    for index, (source, target) in enumerate(zip(sources, targets)):
        if not source.strip() or not target.strip():
            rejected.append(index)
            continue
        keep_source.append(source)
        keep_target.append(target)
    return keep_source, keep_target, rejected
