"""Model performance and architecture."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import TARGET_LABELS
from src.model_pipeline import load_artifact
from src.evaluation import mean_r2

st.title("Model Performance")
st.caption("Cross-validation metrics and cascaded transfer architecture.")

artifact = load_artifact()
learner_df = pd.DataFrame(
    {
        "Learner": ["LightGBM", "CatBoost", "XGBoost", "ExtraTrees", "Ridge"],
        "Role": ["Gradient boosting", "Oblivious trees", "Regularized boosting", "Variance reduction", "Linear group contribution"],
    }
)
st.subheader("Five-learner ensemble")
st.dataframe(learner_df, use_container_width=True)
st.write("NNLS blends out-of-fold predictions and constrains weights to non-negative values, minimizing residual error after normalization.")

if artifact is None:
    st.info("No model artifacts available. Cross-validation metrics cannot be displayed.")
else:
    scores = artifact.get("cv_scores", {})
    score_df = pd.DataFrame({"property": list(scores.keys()), "cv_r2": list(scores.values())})
    fig = px.bar(score_df, x="property", y="cv_r2", template="plotly_dark", title="Per-property 10-fold CV R2")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Official mean CV R2", f"{artifact.get('official_mean', mean_r2(scores)):.4f}")

    st.subheader("Cascaded architecture")
    stages = pd.DataFrame(
        {
            "Level": ["Level 1", "Level 2", "Level 3"],
            "Targets": ["tg, egc", "egb, nc, eea, ei", "eps"],
            "Transfer": ["Base features", "pred_tg, pred_egc, pred_eea", "pred_egc, pred_tg, pred_nc and nc squared"],
        }
    )
    st.dataframe(stages, use_container_width=True)

    st.note = st.markdown("Metrics shown are cross-validation results, not training scores or unseen test performance.")