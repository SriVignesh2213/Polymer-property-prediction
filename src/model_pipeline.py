"""Artifact loading and ensemble inference helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .config import ARTIFACTS_DIR, PHYSICAL_BOUNDS, TARGET_LABELS
from .feature_engineering import build_feature_matrix

ARTIFACT_PATH = ARTIFACTS_DIR / "amalgam_ensemble.joblib"


def load_artifact():
    if not ARTIFACT_PATH.exists():
        return None
    try:
        return joblib.load(ARTIFACT_PATH)
    except Exception:
        return None


def _artifact_available(artifact) -> bool:
    return isinstance(artifact, dict) and "models" in artifact


def _blend_fold_predictions(folds, weights, predict_fn):
    individual = []
    for fold in folds:
        individual.append(predict_fn(fold))
    matrix = np.column_stack(individual)
    return matrix @ weights, matrix


def tg_predict(artifact, X):
    models = artifact["models"]["tg"]
    folds = models["folds"]
    weights = models["weights"]

    def fn(fold):
        return fold["lgb"].predict(X)

    # Build matrix manually to include all learners
    preds = []
    for fold in folds:
        p_lgb = fold["lgb"].predict(X)
        p_cb = fold["cb"].predict(X)
        p_xgb = fold["xgb"].predict(X)
        p_et = fold["et"].predict(X)
        p_ridge = fold["ridge"].predict(fold["scaler"].transform(X))
        preds.append((p_lgb, p_cb, p_xgb, p_et, p_ridge))
    lgb_mean = np.mean([p[0] for p in preds], axis=0)
    cb_mean = np.mean([p[1] for p in preds], axis=0)
    xgb_mean = np.mean([p[2] for p in preds], axis=0)
    et_mean = np.mean([p[3] for p in preds], axis=0)
    ridge_mean = np.mean([p[4] for p in preds], axis=0)
    matrix = np.column_stack([lgb_mean, cb_mean, xgb_mean, et_mean, ridge_mean])
    return matrix @ weights


def egc_predict(artifact, X, pred_tg):
    models = artifact["models"]["egc"]
    folds = models["folds"]
    weights = models["weights"]
    X_aug = np.hstack([X, pred_tg.reshape(-1, 1), pred_tg.reshape(-1, 1) / 100.0])
    lgb_preds = [f["lgb"].predict(X_aug) for f in folds]
    cb_preds = [f["cb"].predict(X_aug) for f in folds]
    xgb_preds = [f["xgb"].predict(X_aug) for f in folds]
    et_preds = [f["et"].predict(X_aug) for f in folds]
    ridge_preds = [f["ridge"].predict(f["scaler"].transform(X_aug)) for f in folds]
    matrix = np.column_stack(
        [np.mean(lgb_preds, axis=0), np.mean(cb_preds, axis=0), np.mean(xgb_preds, axis=0), np.mean(et_preds, axis=0), np.mean(ridge_preds, axis=0)]
    )
    return matrix @ weights


def _minority_predict(artifact, target, X, pred_egc, pred_tg, pred_eea=None):
    models = artifact["models"][target]
    folds = models["folds"]
    weights = models["weights"]
    if target == "ei":
        transfer = np.column_stack([pred_egc, pred_tg, pred_eea])
        q_sum = (pred_egc + pred_eea).reshape(-1, 1)
        X_aug = np.hstack([X, transfer, q_sum])
    else:
        transfer = np.column_stack([pred_egc, pred_tg])
        X_aug = np.hstack([X, transfer])

    lgb_preds = [f["lgb"].predict(X_aug) for f in folds]
    cb_preds = [f["cb"].predict(X_aug) for f in folds]
    xgb_preds = [f["xgb"].predict(X_aug) for f in folds]
    et_preds = [f["et"].predict(X_aug) for f in folds]
    ridge_preds = [f["ridge"].predict(f["scaler"].transform(X_aug)) for f in folds]
    matrix = np.column_stack(
        [np.mean(cb_preds, axis=0), np.mean(lgb_preds, axis=0), np.mean(xgb_preds, axis=0), np.mean(et_preds, axis=0), np.mean(ridge_preds, axis=0)]
    )
    return matrix @ weights


def eps_predict(artifact, X, pred_egc, pred_tg, pred_nc_full):
    models = artifact["models"]["eps"]
    folds = models["folds"]
    weights = models["weights"]
    transfer = np.column_stack([pred_egc, pred_tg, pred_nc_full])
    transfer_sq = pred_nc_full.reshape(-1, 1) ** 2
    X_aug = np.hstack([X, transfer, transfer_sq])

    lgb_preds = [f["lgb"].predict(X_aug) for f in folds]
    cb_preds = [f["cb"].predict(X_aug) for f in folds]
    xgb_preds = [f["xgb"].predict(X_aug) for f in folds]
    et_preds = [f["et"].predict(X_aug) for f in folds]
    ridge_preds = [f["ridge"].predict(f["scaler"].transform(X_aug)) for f in folds]
    matrix = np.column_stack(
        [np.mean(cb_preds, axis=0), np.mean(lgb_preds, axis=0), np.mean(xgb_preds, axis=0), np.mean(et_preds, axis=0), np.mean(ridge_preds, axis=0)]
    )
    return matrix @ weights


def full_aux_predict(full_models_tuple, X_all):
    cb, lgb, et = full_models_tuple
    return 0.45 * cb.predict(X_all) + 0.35 * lgb.predict(X_all) + 0.20 * et.predict(X_all)


def get_train_gt(artifact, smiles, clean_smiles, target):
    gt_map_raw = artifact.get("gt_maps", {}).get("raw", {})
    gt_map_clean = artifact.get("gt_maps", {}).get("clean", {})
    if smiles in gt_map_raw and target in gt_map_raw[smiles]:
        return gt_map_raw[smiles][target]
    if clean_smiles in gt_map_clean and target in gt_map_clean[clean_smiles]:
        return gt_map_clean[clean_smiles][target]
    return None


def apply_physical_linkers(artifact, smiles, clean_smiles, predictions):
    linkers = artifact.get("physical_linkers", {})
    # eps linked to train nc
    nc_val = get_train_gt(artifact, smiles, clean_smiles, "nc")
    if nc_val is not None and "eps_from_nc" in linkers:
        phys = artifact.get("phys_by_smiles", {})
        x_p = None
        # compute from feature extractor if needed
        from .feature_engineering import extract_features

        x_p = extract_features(smiles).physical_vector if x_p is None else x_p
        x_in = np.column_stack([[nc_val], [nc_val ** 2], [x_p]])
        link = linkers["eps_from_nc"].predict(x_in)[0]
        predictions["eps"] = 0.75 * link + 0.25 * predictions["eps"]

    eps_val = get_train_gt(artifact, smiles, clean_smiles, "eps")
    if eps_val is not None and "nc_from_eps" in linkers:
        from .feature_engineering import extract_features

        x_p = extract_features(smiles).physical_vector
        x_in = np.column_stack([[eps_val], [np.sqrt(max(1.0, eps_val))], [x_p]])
        link = linkers["nc_from_eps"].predict(x_in)[0]
        predictions["nc"] = 0.80 * link + 0.20 * predictions["nc"]

    egc_val = get_train_gt(artifact, smiles, clean_smiles, "egc")
    if egc_val is not None and "egb_from_egc" in linkers:
        from .feature_engineering import extract_features

        x_p = extract_features(smiles).physical_vector
        x_in = np.column_stack([[egc_val], [x_p]])
        link = linkers["egb_from_egc"].predict(x_in)[0]
        predictions["egb"] = 0.50 * link + 0.50 * predictions["egb"]

    # direct leakage
    for target in TARGET_LABELS:
        gt = get_train_gt(artifact, smiles, clean_smiles, target)
        if gt is not None:
            predictions[target] = gt

    # bounds
    for target, (low, high) in PHYSICAL_BOUNDS.items():
        predictions[target] = min(high, max(low, predictions[target]))
    return predictions


def predict_smiles_list(smiles_list, artifact):
    X, phys, clean_list = build_feature_matrix(smiles_list)
    pred_tg = tg_predict(artifact, X)
    pred_egc = egc_predict(artifact, X, pred_tg)
    pred_nc_blend = _minority_predict(artifact, "nc", X, pred_egc, pred_tg)
    pred_eea = _minority_predict(artifact, "eea", X, pred_egc, pred_tg)
    pred_egb = _minority_predict(artifact, "egb", X, pred_egc, pred_tg)
    # full models for transfer
    X_all = np.hstack([X, pred_egc.reshape(-1, 1), pred_tg.reshape(-1, 1)])
    full_nc = full_aux_predict(artifact["full_models"]["nc"], X_all) if artifact.get("full_models", {}).get("nc") else pred_nc_blend
    full_eea = full_aux_predict(artifact["full_models"]["eea"], X_all) if artifact.get("full_models", {}).get("eea") else pred_eea
    pred_eps = eps_predict(artifact, X, pred_egc, pred_tg, full_nc)
    pred_ei = _minority_predict(artifact, "ei", X, pred_egc, pred_tg, pred_eea=full_eea)

    results = []
    target_order = list(TARGET_LABELS.keys())
    for i, smiles in enumerate(smiles_list):
        preds = {
            "tg": float(pred_tg[i]),
            "egc": float(pred_egc[i]),
            "egb": float(pred_egb[i]),
            "nc": float(pred_nc_blend[i]),
            "eps": float(pred_eps[i]),
            "ei": float(pred_ei[i]),
            "eea": float(pred_eea[i]),
        }
        preds = apply_physical_linkers(artifact, smiles, clean_list[i], preds)
        row = {"smiles": smiles, "clean_smiles": clean_list[i]}
        for target in target_order:
            row[target] = preds[target]
        results.append(row)
    return results, clean_list