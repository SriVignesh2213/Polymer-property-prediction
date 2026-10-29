"""Polymer SMILES preprocessing preserving the notebook methodology."""

from __future__ import annotations

import re

from rdkit import Chem


def cap_wildcard_smiles(smiles: str, cap: str = "C") -> str:
    """Replace polymer wildcard connectors with a carbon cap and canonicalize."""
    if not isinstance(smiles, str) or not smiles:
        return "C"

    capped = smiles.replace("[*]", cap).replace("*", cap)
    mol = Chem.MolFromSmiles(capped)
    if mol is None:
        capped = re.sub(r"\[\d*\*\]", cap, smiles)
        capped = capped.replace("*", cap)
        mol = Chem.MolFromSmiles(capped)

    if mol is not None:
        return Chem.MolToSmiles(mol)
    return capped


def reduce_oligomer(smiles: str, max_k: int = 6) -> str:
    """Reduce repeated polymer repeat units to the minimal monomer."""
    if not isinstance(smiles, str) or "*" not in smiles:
        return smiles

    current = smiles
    for _ in range(5):
        reduced = False
        mol = Chem.MolFromSmiles(current)
        if mol is None:
            break

        dummies = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummies) != 2:
            break

        d0, d1 = dummies[0], dummies[1]
        try:
            path = list(Chem.rdmolops.GetShortestPath(mol, d0, d1))
        except Exception:
            break

        if len(path) < 3:
            break

        backbone = path[1:-1]
        n_bb = len(backbone)

        for k in range(max_k, 1, -1):
            if n_bb % k != 0:
                continue
            seg = n_bb // k
            if seg < 2:
                continue

            cuts = []
            valid = True
            for i in range(1, k):
                bond = mol.GetBondBetweenAtoms(backbone[seg * i - 1], backbone[seg * i])
                if bond is None:
                    valid = False
                    break
                cuts.append(bond.GetIdx())

            if not valid or len(cuts) != k - 1:
                continue

            try:
                frags = Chem.MolToSmiles(
                    Chem.FragmentOnBonds(mol, cuts, addDummies=True)
                ).split(".")
                if len(frags) != k:
                    continue

                clean_frags = []
                for fragment in frags:
                    clean = re.sub(r"\[\d*\*\]", "[*]", fragment)
                    fragment_mol = Chem.MolFromSmiles(clean)
                    if fragment_mol is None:
                        valid = False
                        break
                    clean_frags.append(Chem.MolToSmiles(fragment_mol))

                if not valid or len(clean_frags) != k:
                    continue
                if len(set(clean_frags)) == 1:
                    current = clean_frags[0]
                    reduced = True
                    break
            except Exception:
                continue

        if not reduced:
            break

    return current


def preprocess_polymer(smiles: str) -> str:
    """Canonicalize, reduce oligomer periodicity, and return a canonical SMILES."""
    if not smiles or not isinstance(smiles, str):
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    canon = Chem.MolToSmiles(mol)
    reduced = reduce_oligomer(canon)
    reduced_mol = Chem.MolFromSmiles(reduced)
    return Chem.MolToSmiles(reduced_mol) if reduced_mol is not None else reduced


def is_valid_smiles(smiles: str) -> bool:
    """Return whether RDKit can parse the SMILES."""
    return isinstance(smiles, str) and bool(smiles) and Chem.MolFromSmiles(smiles) is not None