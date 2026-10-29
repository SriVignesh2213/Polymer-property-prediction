"""Optional SHAP helpers for the explainability page."""

from __future__ import annotations

import numpy as np
import shap

from .model_pipeline import load_artifact


def shap_importance_for_target(artifact, target: str, X):
    if not artifact or (target not in ("tg", "egc")):
        return None
    model = artifact["models"][target]["folds"][0]["lgb"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[0]
    if hasattr(values, "values"):
        values = values.values
    return np.mean(np.abs(np.asarray(values)), axis=0)