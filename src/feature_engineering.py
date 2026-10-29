"""Feature engineering copied from the notebook's preprocessing pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, rdMolDescriptors, Fragments

from .polymer_preprocessing import cap_wildcard_smiles, preprocess_polymer

RDLogger.DisableLog("rdApp.*")
_BondType_SINGLE = Chem.BondType.SINGLE
_BondType_DOUBLE = Chem.BondType.DOUBLE
_BondType_TRIPLE = Chem.BondType.TRIPLE
_BondType_AROMATIC = Chem.BondType.AROMATIC

ALREADY_INTENSIVE = {
    "FractionCSP3",
    "MolLogP",
    "BalabanJ",
    "HallKierAlpha",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "qed",
    "SPS",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
}
POLYMER_PATTERN_SMARTS = {
    "imide": "C(=O)NC(=O)",
    "ester": "C(=O)O[#6]",
    "amide": "C(=O)N[#6]",
    "carbonate": "OC(=O)O",
    "urethane": "NC(=O)O",
    "urea": "NC(=O)N",
    "sulfone": "S(=O)(=O)",
    "ether": "[#6]O[#6]",
    "siloxane": "[Si]O[Si]",
    "phosphazene": "N=P",
    "cf2": "C(F)(F)",
    "thiophene": "s1cccc1",
    "pyridine": "n1ccccc1",
}
POLYMER_PATTERNS = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in POLYMER_PATTERN_SMARTS.items()
}


@dataclass
class FeatureExtractionResult:
    feature_vector: np.ndarray
    physical_vector: np.ndarray
    clean_smiles: str


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def extract_features(smiles: str) -> FeatureExtractionResult:
    """Extract the exact feature vector used in the notebook for one SMILES."""
    clean_smiles = preprocess_polymer(smiles) if isinstance(smiles, str) and smiles else ""

    row: list[float] = []

    # Backbone features on minimal monomer
    mol_clean = None if not clean_smiles else _mol(clean_smiles)
    bb_feats = [0.0] * 12
    n_heavy_raw = 1
    if mol_clean is not None:
        dummies = [atom.GetIdx() for atom in mol_clean.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummies) == 2:
            d1, d2 = dummies[0], dummies[1]
            try:
                path = list(Chem.rdmolops.GetShortestPath(mol_clean, d1, d2))
            except Exception:
                path = []
            backbone_atoms = set(path[1:-1]) if path else set()
            all_heavy = [atom.GetIdx() for atom in mol_clean.GetAtoms() if atom.GetAtomicNum() > 1]
            n_heavy_raw = max(1, len(all_heavy))
            n_bb = len(backbone_atoms)
            n_sc = max(0, n_heavy_raw - n_bb)
            bb_arom = sum(1 for idx in backbone_atoms if mol_clean.GetAtomWithIdx(idx).GetIsAromatic())
            bb_rot = 0
            for bond in mol_clean.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a1 in backbone_atoms and a2 in backbone_atoms:
                    if bond.GetBondType() == _BondType_SINGLE and not bond.IsInRing():
                        bb_rot += 1
            bb_arom_ratio = bb_arom / max(1, n_bb)
            bb_rot_ratio = bb_rot / max(1, n_bb)
            sc_ratio = n_sc / n_heavy_raw
            bb_len_ratio = n_bb / n_heavy_raw
            bb_hetero = sum(
                1 for idx in backbone_atoms if mol_clean.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6)
            )
            bb_hetero_ratio = bb_hetero / max(1, n_bb)
            steric_hindrance = n_sc / max(1, n_bb)
            s_canon = Chem.MolToSmiles(mol_clean)
            has_amide = 1.0 if ("NC(=O)" in s_canon or "C(=O)N" in s_canon) else 0.0
            has_ester = 1.0 if ("OC(=O)" in s_canon or "C(=O)O" in s_canon) else 0.0
            has_ether = 1.0 if ("COC" in s_canon or "cOc" in s_canon or "COc" in s_canon) else 0.0
            has_sulfone = 1.0 if ("S(=O)(=O)" in s_canon) else 0.0
            bb_feats = [
                float(n_bb) / n_heavy_raw,
                float(n_sc) / n_heavy_raw,
                bb_len_ratio,
                sc_ratio,
                bb_arom_ratio,
                bb_rot_ratio,
                bb_hetero_ratio,
                steric_hindrance,
                has_amide,
                has_ester,
                has_ether,
                has_sulfone,
            ]

    row.extend(bb_feats)

    # Whole-molecule capped features
    capped = cap_wildcard_smiles(clean_smiles, cap="C") if clean_smiles else "C"
    mol = _mol(capped)
    if mol is None:
        mol = _mol("C")
    n_heavy = max(1, mol.GetNumHeavyAtoms())

    for name, func in Descriptors.descList:
        try:
            value = func(mol)
            if not np.isfinite(value):
                value = 0.0
        except Exception:
            value = 0.0
        value = float(np.clip(value, -1e6, 1e6))
        if (
            name in ALREADY_INTENSIVE
            or name.startswith("BCUT2D")
            or name.startswith("PEOE")
            or name.startswith("SMR")
            or name.startswith("SlogP")
        ):
            row.append(value)
        else:
            row.append(value / n_heavy)

    maccs = list(MACCSkeys.GenMACCSKeys(mol).ToBitString())
    row.extend(float(bit) for bit in maccs)

    def morgan_counts(radius: int, bits: int) -> list[float]:
        hashed = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=bits)
        counts = hashed.GetNonzeroElements()
        return [counts.get(i, 0) / n_heavy for i in range(bits)]

    row.extend(morgan_counts(1, 64))
    row.extend(morgan_counts(2, 128))
    row.extend(morgan_counts(3, 128))

    fcfp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=64, useFeatures=True)
    fcfp_counts = fcfp.GetNonzeroElements()
    row.extend(fcfp_counts.get(i, 0) / n_heavy for i in range(64))

    try:
        ap_fp = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=64)
        ap_counts = ap_fp.GetNonzeroElements()
        row.extend(ap_counts.get(i, 0) / n_heavy for i in range(64))
    except Exception:
        row.extend([0.0] * 64)

    try:
        tt_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol, nBits=64)
        tt_counts = tt_fp.GetNonzeroElements()
        row.extend(tt_counts.get(i, 0) / n_heavy for i in range(64))
    except Exception:
        row.extend([0.0] * 64)

    for fragment_name in dir(Fragments):
        if fragment_name.startswith("fr_"):
            try:
                count = getattr(Fragments, fragment_name)(mol)
            except Exception:
                count = 0
            row.append(count / n_heavy)

    for pattern_name, pattern in POLYMER_PATTERNS.items():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            row.append(1.0)
        else:
            row.append(0.0)

    ring_info = mol.GetRingInfo()
    ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
    row.append(sum(1 for size in ring_sizes if size == 3) / n_heavy)
    row.append(sum(1 for size in ring_sizes if size == 4) / n_heavy)
    row.append(sum(1 for size in ring_sizes if size == 5) / n_heavy)
    row.append(sum(1 for size in ring_sizes if size == 6) / n_heavy)
    row.append(sum(1 for size in ring_sizes if size >= 7) / n_heavy)

    bonds = mol.GetBonds()
    n_bonds = max(1, len(bonds))
    row.append(sum(1 for bond in bonds if bond.GetBondType() == _BondType_SINGLE) / n_bonds)
    row.append(sum(1 for bond in bonds if bond.GetBondType() == _BondType_DOUBLE) / n_bonds)
    row.append(sum(1 for bond in bonds if bond.GetBondType() == _BondType_TRIPLE) / n_bonds)
    row.append(sum(1 for bond in bonds if bond.GetBondType() == _BondType_AROMATIC) / n_bonds)

    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    bertz = Descriptors.BertzCT(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    row.extend(
        [
            sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / n_heavy,
            sum(1 for atom in mol.GetAtoms() if atom.IsInRing()) / n_heavy,
            sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)) / n_heavy,
            sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (9, 17, 35, 53)) / n_heavy,
            n_rot / n_bonds,
            n_rot / (mw + 1e-5),
            Descriptors.NumHDonors(mol) / n_heavy,
            Descriptors.NumHAcceptors(mol) / n_heavy,
            mw / n_heavy,
            tpsa / (mw + 1e-5),
            float(Descriptors.FractionCSP3(mol)),
            bertz / n_heavy,
        ]
    )

    feature_vector = np.asarray(row, dtype=np.float32)

    phys_vector = np.asarray(
        [
            float(Descriptors.MolMR(mol)) / n_heavy,
            float(Descriptors.TPSA(mol)) / n_heavy,
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.FractionCSP3(mol)),
            float(Descriptors.LabuteASA(mol)) / n_heavy,
        ],
        dtype=np.float32,
    )

    return FeatureExtractionResult(feature_vector, phys_vector, clean_smiles)


def build_feature_matrix(smiles_values: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return `(features, physical_features, clean_smiles)` arrays."""
    feature_rows = []
    phys_rows = []
    clean_values = []
    for smiles in smiles_values:
        result = extract_features(smiles)
        feature_rows.append(result.feature_vector)
        phys_rows.append(result.physical_vector)
        clean_values.append(result.clean_smiles)
    return (
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(phys_rows, dtype=np.float32),
        clean_values,
    )


def feature_names() -> list[str]:
    """Return names in the same order as ``extract_features``."""
    names = [f"bb_{i}" for i in range(12)]
    names.extend(name for name, _ in Descriptors.descList)
    names.extend(f"maccs_{i}" for i in range(167))
    names.extend(f"morgan_r1_{i}" for i in range(64))
    names.extend(f"morgan_r2_{i}" for i in range(128))
    names.extend(f"morgan_r3_{i}" for i in range(128))
    names.extend(f"fcfp4_{i}" for i in range(64))
    names.extend(f"atompair_{i}" for i in range(64))
    names.extend(f"torsion_{i}" for i in range(64))
    fragment_names = [name for name in dir(Fragments) if name.startswith("fr_")]
    names.extend(fragment_names)
    names.extend(list(POLYMER_PATTERN_SMARTS.keys()))
    names.extend(["rings_3", "rings_4", "rings_5", "rings_6", "rings_7plus"])
    names.extend(["bonds_single", "bonds_double", "bonds_triple", "bonds_aromatic"])
    names.extend(
        [
            "aromatic_atoms_ratio",
            "ring_atoms_ratio",
            "hetero_atoms_ratio",
            "halogen_atoms_ratio",
            "rotatable_per_bond",
            "rotatable_per_mass",
            "h_donors_per_heavy",
            "h_acceptors_per_heavy",
            "mass_per_heavy",
            "tpsa_per_mass",
            "fraction_csp3",
            "bertz_per_heavy",
        ]
    )
    return names