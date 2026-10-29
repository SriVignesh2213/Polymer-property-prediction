"""Polymer prediction studio."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.config import TARGET_LABELS, PHYSICAL_BOUNDS
from src.polymer_preprocessing import preprocess_polymer, is_valid_smiles
from src.feature_engineering import build_feature_matrix, extract_features
from src.model_pipeline import load_artifact, predict_smiles_list

st.title("Polymer Prediction Studio")
st.caption("Validate a repeat-unit SMILES, preview structure, and predict all seven properties.")

artifact = load_artifact()
if artifact is None:
    st.warning("Model artifacts are unavailable. Run `python training/train_models.py` before prediction.")

sample_inputs = ["*CC(Cl)*", "*CCCCCCCCCCCC(=O)N*", "*Nc1ccc(-c2ccc(-c3ccc(*)cc3)s2)cc1"]
smiles_input = st.text_area("Polymer repeat-unit SMILES", value=sample_inputs[0], height=120)

col1, col2, col3 = st.columns(3)
validate = col1.button("Validate SMILES")
predict = col2.button("Predict Properties")
reset = col3.button("Reset")
if reset:
    st.session_state.pop("prediction_frame", None)
    st.rerun()

if validate or smiles_input:
    if is_valid_smiles(smiles_input):
        st.success("Valid RDKit SMILES.")
        mol = Chem.MolFromSmiles(smiles_input)
        img = Draw.MolToImage(mol, size=(320, 240))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        st.image(buffer.getvalue(), width=320)
        clean = preprocess_polymer(smiles_input)
        st.caption(f"Clean SMILES: {clean}")
        result = extract_features(smiles_input)
        st.caption(f"Extracted {result.feature_vector.shape[0]} invariant features.")
    else:
        st.error("Invalid SMILES: RDKit could not parse the structure.")

if predict:
    if artifact is None:
        st.error("Cannot predict: trained model artifacts are unavailable.")
    elif not is_valid_smiles(smiles_input):
        st.error("Cannot predict: invalid SMILES.")
    else:
        with st.spinner("Running polymer pipeline..."):
            rows, clean = predict_smiles_list([smiles_input], artifact)
        frame = pd.DataFrame(rows)
        display_columns = ["smiles", "clean_smiles"] + list(TARGET_LABELS.keys())
        st.session_state["prediction_frame"] = frame
        st.success("Prediction complete.")
        preview = frame[display_columns].T.reset_index()
        preview.columns = ["property", "value"]
        preview["name"] = preview["property"].map(TARGET_LABELS)
        st.dataframe(frame[display_columns], use_container_width=True)
        csv = frame[display_columns].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="amalgam_prediction.csv", mime="text/csv")

frame = st.session_state.get("prediction_frame")
if frame is not None:
    st.subheader("Latest prediction")
    st.dataframe(frame, use_container_width=True)