"""Feature helpers for ProSys Stage 2 V2."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.warning')


def _smiles_to_mol(smiles: str) -> Chem.Mol | None:
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        return None
    return Chem.MolFromSmiles(canonical)


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ''
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_reaction_side(smiles: str) -> str:
    fragments = []
    for fragment in str(smiles).split('.'):
        fragment = fragment.strip()
        if not fragment:
            continue
        canonical = canonicalize_smiles(fragment)
        if canonical:
            fragments.append(canonical)
    return '.'.join(sorted(fragments))


def product_morgan_fp(product: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    canonical = canonicalize_smiles(product)
    if not canonical:
        raise ValueError(f'Invalid product SMILES: {product}')

    mol = Chem.MolFromSmiles(canonical)
    fp_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol=mol,
        radius=radius,
        nBits=n_bits,
        useFeatures=False,
        useChirality=True,
    )
    fp = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp_bit, fp)
    return fp


def reaction_morgan_fp(reactants: str, product: str, fpsize: int = 4096, radius: int = 2) -> np.ndarray:
    reactant_mol = _smiles_to_mol(reactants)
    product_mol = _smiles_to_mol(product)
    if reactant_mol is None or product_mol is None:
        raise ValueError(f'Invalid reaction SMILES: {reactants} >> {product}')

    reactant_bits = AllChem.GetMorganFingerprintAsBitVect(
        mol=reactant_mol,
        radius=radius,
        nBits=fpsize,
        useFeatures=False,
        useChirality=True,
    )
    product_bits = AllChem.GetMorganFingerprintAsBitVect(
        mol=product_mol,
        radius=radius,
        nBits=fpsize,
        useFeatures=False,
        useChirality=True,
    )

    reactant_fp = np.zeros((fpsize,), dtype=np.float32)
    product_fp = np.zeros((fpsize,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(reactant_bits, reactant_fp)
    DataStructs.ConvertToNumpyArray(product_bits, product_fp)

    delta_fp = product_fp - reactant_fp
    return np.concatenate((product_fp, delta_fp)).astype(np.float32)


def product_scaffold_smiles(product: str) -> str:
    canonical = canonicalize_smiles(product)
    if not canonical:
        return ''

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return ''

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return canonicalize_smiles(scaffold) if scaffold else ''


def molecule_graph_descriptors(smiles: str) -> np.ndarray:
    mol = _smiles_to_mol(smiles)
    if mol is None:
        return np.zeros((8,), dtype=np.float32)

    num_atoms = float(mol.GetNumAtoms())
    num_bonds = float(mol.GetNumBonds())
    num_rings = float(rdMolDescriptors.CalcNumRings(mol))
    aromatic_atoms = float(sum(int(atom.GetIsAromatic()) for atom in mol.GetAtoms()))
    hetero_atoms = float(sum(int(atom.GetAtomicNum() not in (1, 6)) for atom in mol.GetAtoms()))
    exact_mw = float(rdMolDescriptors.CalcExactMolWt(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    fraction_sp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))

    return np.asarray(
        [
            num_atoms,
            num_bonds,
            num_rings,
            aromatic_atoms,
            hetero_atoms,
            exact_mw,
            tpsa,
            fraction_sp3,
        ],
        dtype=np.float32,
    )


def reaction_graph_descriptors(reactants: str, product: str) -> np.ndarray:
    reactant_desc = molecule_graph_descriptors(reactants)
    product_desc = molecule_graph_descriptors(product)
    return np.concatenate(
        (
            reactant_desc,
            product_desc,
            product_desc - reactant_desc,
        )
    ).astype(np.float32)


def normalize_fp(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def tanimoto_similarity_from_bitvect(query_fp: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    query_fp = np.asarray(query_fp, dtype=np.uint8)
    candidate_matrix = np.asarray(candidate_matrix, dtype=np.uint8)
    if candidate_matrix.ndim != 2:
        raise ValueError('candidate_matrix must be 2D')
    if query_fp.shape[0] != candidate_matrix.shape[1]:
        raise ValueError('Fingerprint length mismatch')

    intersections = np.bitwise_and(candidate_matrix, query_fp).sum(axis=1, dtype=np.int32)
    unions = np.bitwise_or(candidate_matrix, query_fp).sum(axis=1, dtype=np.int32)
    similarities = np.zeros(candidate_matrix.shape[0], dtype=np.float32)
    valid = unions > 0
    similarities[valid] = intersections[valid] / unions[valid]
    return similarities


def count_condition_tokens(labels: str) -> int:
    tokens = [token.strip() for token in str(labels).split(';') if token.strip()]
    return len(tokens)


def count_reactant_components(reactants: str) -> int:
    normalized = canonicalize_reaction_side(reactants)
    if not normalized:
        return 0
    return len(normalized.split('.'))
