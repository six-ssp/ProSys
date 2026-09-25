"""Product-only inference augmentation with explicit chemical identity checks."""

from rdkit import Chem


def identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError('Invalid or empty product SMILES')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return '.'.join(sorted(Chem.MolToSmiles(fragment, isomericSmiles=True)
                           for fragment in Chem.GetMolFrags(mol, asMols=True)))


def augment(smiles, aug, tokenize, randomize=None):
    if type(aug) is not int or aug < 1:
        raise ValueError('Augmentation count must be a positive integer')
    expected = identity(smiles)
    mol = Chem.MolFromSmiles(smiles)
    original_tokens = tokenize(smiles)
    if ''.join(original_tokens.split()) != smiles:
        # Parse CXSMILES metadata before normalization; truncating a suffix
        # could drop chemically meaningful annotations such as radicals.
        normalized = Chem.MolToSmiles(mol, isomericSmiles=True)
        if (identity(normalized) != expected or
                ''.join(tokenize(normalized).split()) != normalized):
            raise ValueError('Product cannot be faithfully normalized/tokenized')
        smiles = normalized
    randomize = randomize or (lambda m: Chem.MolToSmiles(m, doRandom=True))
    strings, replaced = [smiles], []
    for index in range(1, aug):
        candidate = randomize(mol)
        try:
            same = identity(candidate) == expected
        except ValueError:
            same = False
        if not same:
            replaced.append(index)
            candidate = smiles
        # One random draw per slot, including failures: no retries that shift
        # the random stream for later products or alter augmentation weights.
        strings.append(candidate)
    tokens = [tokenize(text) for text in strings]
    if any(''.join(token.split()) != text for token, text in zip(tokens, strings)):
        raise ValueError('Tokenizer did not reconstruct a guarded input exactly')
    return tokens, replaced
