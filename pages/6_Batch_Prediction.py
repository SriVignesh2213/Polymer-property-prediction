"""Batch prediction page."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import TARGET_LABELS
from src.data_loader import load_csv
from src.polymer_preprocessing import is_valid_smiles
from src.model_pipeline import load_artifact, predict_smiles_list

st.title("Batch Prediction")
st.caption("Upload a CSV with a `smiles` column and predict all seven properties.")

artifact = load_artifact()
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    batch_df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Invalid CSV: {exc}")
    st.stop()

if "smiles" not in batch_df.columns:
    st.error("CSV must contain a `smiles` column.")
    st.stop()

st.subheader("Preview")
st.dataframe(batch_df.head(20), use_container_width=True)

valid_mask = batch_df["smiles"].astype(str).map(is_valid_smiles)
st.metric("Valid SMILES", f"{int(valid_mask.sum())} / {len(batch_df)}")
if (~valid_mask).any():
    st.warning("Invalid rows will be reported and retained without predictions.")

if artifact is None:
    st.error("Model artifacts unavailable; prediction cannot run.")
    st.stop()

if st.button("Run batch prediction"):
    frame = batch_df.copy()
    for target in TARGET_LABELS:
        frame[target] = pd.NA
    results_by_row = {}
    valid_indices = batch_df.index[valid_mask].tolist()
    progress = st.progress(0)
    for i, idx in enumerate(valid_indices):
        row_results, _ = predict_smiles_list([str(batch_df.loc[idx, "smiles"])], artifact)
        for target in TARGET_LABELS:
            frame.loc[idx, target] = row_results[0][target]
        progress.progress((i + 1) / max(1, len(valid_indices)))
    frame["prediction_status"] = valid_mask.map({True: "predicted", False: "invalid SMILES"})
    st.session_state["batch_results"] = frame
    st.success("Batch prediction complete.")

results = st.session_state.get("batch_results")
if isinstance(results, pd.DataFrame):
    st.subheader("Results")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download amalgam_predictions.csv",
        results.to_csv(index=False).encode("utf-8"),
        file_name="amalgam_predictions.csv",
        mime="text/csv",
    )