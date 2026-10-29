"""Pipeline and validation page."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem

from src.polymer_preprocessing import preprocess_polymer
from src.model_pipeline import load_artifact

st.title("Pipeline & Validation")
st.caption("Documented pipeline stages with actual test outcomes when executed.")

st.subheader("Pipeline stages")
st.markdown(
    "- SMILES canonicalization\n"
    "- Oligomer-periodicity reduction\n"
    "- Wildcard capping\n"
    "- Intensive RDKit descriptors\n"
    "- Polymer-class indicators, ring, and bond statistics\n"
    "- Molecular fingerprints\n"
    "- Five-model ensemble stacking\n"
    "- NNLS blend weight optimization\n"
    "- Physical transfer and domain clipping\n"
    "- SHAP explainability"
)

if st.button("Run preprocessing validation"):
    test_pairs = [
        ("*CC(Cl)*", "*CC(Cl)*"),
        ("*CC(Cl)CC(Cl)*", "*CC(Cl)*"),
        ("*CC(Cl)CC(Cl)CC(Cl)*", "*CC(Cl)*"),
        ("*CC(C)CC(C)*", "*CC(C)*"),
    ]
    rows = []
    for raw, expected in test_pairs:
        output = preprocess_polymer(raw)
        expected_clean = preprocess_polymer(expected)
        rows.append({"input": raw, "output": output, "expected": expected_clean, "passed": output == expected_clean})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.metric("Passed", f"{sum(r['passed'] for r in rows)} / {len(rows)}")

    artifact = load_artifact()
    st.write("Model artifact available:", artifact is not None)
    if artifact is not None:
        st.json({"cv_scores": artifact.get("cv_scores", {}), "official_mean": artifact.get("official_mean")})