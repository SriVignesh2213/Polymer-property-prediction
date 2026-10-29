"""Offline script that trains the Amalgam ensemble and saves inference artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import APP_DIR, ARTIFACTS_DIR, RANDOM_SEED, TARGET_LABELS
from src.data_loader import load_csv, REQUIRED_TRAIN_COLUMNS, REQUIRED_TEST_COLUMNS
from src.feature_engineering import build_feature_matrix, feature_names
from src.polymer_preprocessing import preprocess_polymer

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_TYPES = list(TARGET_LABELS.keys())


def blend_nnls(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    weights, _ = nnls(oof_matrix, y_true)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1]
    return weights


def make_fold(model_builders, X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    fold_records = []
    oof = {name: np.zeros(len(y)) for name in ["lgb", "cb", "xgb", "et", "ridge"]}
    te_preds = {name: [] for name in ["lgb", "cb", "xgb", "et", "ridge"]}
    for fold_no, (tr_idx, val_idx) in enumerate(kf.split(X)):
        fold = {}
        scaler = StandardScaler()
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]

        m_lgb = lgb.LGBMRegressor(**model_builders["lgb"])
        m_lgb.fit(Xtr, ytr)
        oof["lgb"][val_idx] = m_lgb.predict(Xval)
        te_preds["lgb"].append(m_lgb.predict)

        m_cb = CatBoostRegressor(**model_builders["cb"])
        m_cb.fit(Xtr, ytr)
        oof["cb"][val_idx] = m_cb.predict(Xval)

        m_xgb = xgb.XGBRegressor(**model_builders["xgb"])
        m_xgb.fit(Xtr, ytr)
        oof["xgb"][val_idx] = m_xgb.predict(Xval)

        m_et = ExtraTreesRegressor(**model_builders["et"])
        m_et.fit(Xtr, ytr)
        oof["et"][val_idx] = m_et.predict(Xval)

        scaler.fit(Xtr)
        m_ridge = Ridge(**model_builders["ridge"]).fit(scaler.transform(Xtr), ytr)
        oof["ridge"][val_idx] = m_ridge.predict(scaler.transform(Xval))

        fold = {
            "lgb": m_lgb,
            "cb": m_cb,
            "xgb": m_xgb,
            "et": m_et,
            "ridge": m_ridge,
            "scaler": scaler,
        }
        fold_records.append(fold)
    return fold_records, oof


def aggregate_test_predictions(fold_records, X_test):
    preds = {name: [] for name in ["lgb", "cb", "xgb", "et", "ridge"]}
    for fold in fold_records:
        preds["lgb"].append(fold["lgb"].predict(X_test))
        preds["cb"].append(fold["cb"].predict(X_test))
        preds["xgb"].append(fold["xgb"].predict(X_test))
        preds["et"].append(fold["et"].predict(X_test))
        preds["ridge"].append(fold["ridge"].predict(fold["scaler"].transform(X_test)))
    return {name: np.mean(values, axis=0) for name, values in preds.items()}


def main():
    train_loaded = load_csv(Path(r"D:\Amalgam\train.csv"))
    test_loaded = load_csv(Path(r"D:\Amalgam\test.csv"))
    train_df = train_loaded.dataframe.copy()
    test_df = test_loaded.dataframe.copy()

    # Physical feature matrix
    X_train, phys_train, clean_train = build_feature_matrix(train_df["smiles"].tolist())
    X_test, phys_test, clean_test = build_feature_matrix(test_df["smiles"].tolist())
    train_df["clean_smiles"] = clean_train
    test_df["clean_smiles"] = clean_test

    # Scaling/setup
    ml_names = feature_names()
    joblib.dump(ml_names, ARTIFACTS_DIR / "feature_names.joblib")

    # ---- Level 1 tg ----
    print("Training tg")
    sub_tg = train_df[train_df.target_type == "tg"].reset_index(drop=True)
    X_tg = np.array([row for row in X_train[[i for i, t in enumerate(train_df.target_type) if t == "tg"]]])
    # More direct:
    tg_mask = (train_df.target_type == "tg").values
    X_tg = X_train[tg_mask]
    y_tg = sub_tg.target.values
    lgb_cfg = dict(n_estimators=950, learning_rate=0.025, num_leaves=48, subsample=0.8, colsample_bytree=0.75, min_child_samples=8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    cb_cfg = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=3.5, random_seed=RANDOM_SEED, verbose=0, thread_count=-1)
    xgb_cfg = dict(n_estimators=750, learning_rate=0.025, max_depth=6, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
    et_cfg = dict(n_estimators=300, max_depth=18, min_samples_split=3, min_samples_leaf=1, random_state=RANDOM_SEED, n_jobs=-1)
    ridge_cfg = dict(alpha=20.0)
    builders = {"lgb": lgb_cfg, "cb": cb_cfg, "xgb": xgb_cfg, "et": et_cfg, "ridge": ridge_cfg}
    folds_tg, oof_tg = make_fold(builders, X_tg, y_tg)
    A_tg = np.column_stack([oof_tg[n] for n in ["lgb", "cb", "xgb", "et", "ridge"]])
    w_tg = blend_nnls(A_tg, y_tg)
    oof_blend_tg = A_tg @ w_tg
    cv_tg = float(r2_score(y_tg, oof_blend_tg))

    # tg predictions for all rows/train/test
    tg_test_preds = None
    def tg_predict_matrix(Xm):
        p = aggregate_test_predictions(folds_tg, Xm)
        return w_tg[0] * p["lgb"] + w_tg[1] * p["cb"] + w_tg[2] * p["xgb"] + w_tg[3] * p["et"] + w_tg[4] * p["ridge"]

    tg_test = tg_predict_matrix(X_test)
    train_missing_tg = [i for i, t in enumerate(train_df.target_type) if t != "tg"]
    # For training rows, OOF only exists for tg rows. Predict non-tg rows with full average
    train_tg_pred_full = np.zeros(len(train_df))
    tg_rows = np.where(train_df.target_type.values == "tg")[0]
    train_tg_pred_full[tg_rows] = oof_blend_tg
    if len(train_missing_tg) > 0:
        X_miss = X_train[train_missing_tg]
        train_tg_pred_full[train_missing_tg] = tg_predict_matrix(X_miss)
    train_df["pred_tg"] = train_tg_pred_full
    test_df["pred_tg"] = tg_test

    # ---- Level 1 egc ----
    print("Training egc")
    egc_mask = (train_df.target_type == "egc").values
    sub_egc_indices = np.where(egc_mask)[0]
    X_egc_base = X_train[egc_mask]
    X_egc = np.hstack([X_egc_base, train_df.loc[egc_mask, ["pred_tg"]].values, train_df.loc[egc_mask, ["pred_tg"]].values / 100.0])
    y_egc = train_df.loc[egc_mask, "target"].values
    X_test_egc_aug = np.hstack([X_test, test_df[["pred_tg"]].values, test_df[["pred_tg"]].values / 100.0])

    lgb_cfg = dict(n_estimators=750, learning_rate=0.025, num_leaves=38, subsample=0.8, colsample_bytree=0.75, min_child_samples=6, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    cb_cfg = dict(iterations=850, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=RANDOM_SEED, verbose=0, thread_count=-1)
    xgb_cfg = dict(n_estimators=600, learning_rate=0.025, max_depth=6, colsample_bytree=0.75, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
    et_cfg = dict(n_estimators=250, max_depth=16, min_samples_split=3, min_samples_leaf=1, random_state=RANDOM_SEED, n_jobs=-1)
    ridge_cfg = dict(alpha=25.0)
    builders = {"lgb": lgb_cfg, "cb": cb_cfg, "xgb": xgb_cfg, "et": et_cfg, "ridge": ridge_cfg}
    folds_egc, oof_egc = make_fold(builders, X_egc, y_egc)
    A_egc = np.column_stack([oof_egc[n] for n in ["lgb", "cb", "xgb", "et", "ridge"]])
    w_egc = blend_nnls(A_egc, y_egc)
    oof_blend_egc = A_egc @ w_egc
    cv_egc = float(r2_score(y_egc, oof_blend_egc))

    def egc_predict_matrix(Xb, pred_tg_values):
        Xaug = np.hstack([Xb, pred_tg_values.reshape(-1, 1), pred_tg_values.reshape(-1, 1) / 100.0])
        p = aggregate_test_predictions(folds_egc, Xaug)
        return w_egc[0] * p["lgb"] + w_egc[1] * p["cb"] + w_egc[2] * p["xgb"] + w_egc[3] * p["et"] + w_egc[4] * p["ridge"]

    train_egc_pred_full = np.full(len(train_df), np.nan)
    egc_rows = np.where(train_df.target_type.values == "egc")[0]
    train_egc_pred_full[egc_rows] = oof_blend_egc
    missing_egc_idx = np.where(~np.isfinite(train_egc_pred_full))[0].tolist()
    if missing_egc_idx:
        train_egc_pred_full[missing_egc_idx] = egc_predict_matrix(X_train[missing_egc_idx], train_df.pred_tg.values[missing_egc_idx])
    train_df["pred_egc"] = train_egc_pred_full
    test_df["pred_egc"] = egc_predict_matrix(X_test, test_df.pred_tg.values)

    # ---- Level 2 minority ----
    print("Training minority targets")
    cv_scores = {"tg": cv_tg, "egc": cv_egc}
    test_predictions = {}
    test_predictions["tg"] = test_df.loc[test_df.target_type == "tg", "pred_tg"].values.copy()
    test_predictions["egc"] = test_df.loc[test_df.target_type == "egc", "pred_egc"].values.copy()

    train_df["pred_nc"] = np.nan
    test_df["pred_nc"] = np.nan
    train_df["pred_eea"] = np.nan
    test_df["pred_eea"] = np.nan

    full_models = {"nc": [], "eea": []}  # list of folds for full? instead store simplified multi-model
    # We'll store full model tuples separately
    full_nc_models = None
    full_eea_models = None

    builder_configs = {
        "egb": (
            dict(iterations=550, learning_rate=0.03, depth=5, l2_leaf_reg=3.5, random_seed=RANDOM_SEED, verbose=0, thread_count=-1),
            dict(n_estimators=350, learning_rate=0.025, max_depth=5, num_leaves=20, subsample=0.85, colsample_bytree=0.8, min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1),
            dict(n_estimators=300, learning_rate=0.025, max_depth=5, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1),
            dict(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1),
            dict(alpha=25.0),
        ),
    }

    def minority_train_and_predict(tt, transfer_fn):
        nonlocal full_nc_models, full_eea_models
        sub_mask = (train_df.target_type == tt).values
        sub_rows = np.where(sub_mask)[0]
        X_base = X_train[sub_mask]
        test_sub_mask = (test_df.target_type == tt).values
        X_test_sub_base = X_test[test_sub_mask]
        sub_test_df = test_df[test_sub_mask].reset_index(drop=True)

        X_aug, X_test_aug = transfer_fn(X_base, X_test_sub_base, train_df, test_df, sub_rows, sub_test_df)
        y = train_df.loc[sub_mask, "target"].values

        lgb_cfg = dict(n_estimators=350, learning_rate=0.025, max_depth=5, num_leaves=20, subsample=0.85, colsample_bytree=0.8, min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
        cb_cfg = dict(iterations=550, learning_rate=0.03, depth=5, l2_leaf_reg=3.5, random_seed=RANDOM_SEED, verbose=0, thread_count=-1)
        xgb_cfg = dict(n_estimators=300, learning_rate=0.025, max_depth=5, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
        et_cfg = dict(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
        ridge_cfg = dict(alpha=25.0)
        builders = {"lgb": lgb_cfg, "cb": cb_cfg, "xgb": xgb_cfg, "et": et_cfg, "ridge": ridge_cfg}
        folds, oof = make_fold(builders, X_aug, y)
        A = np.column_stack([oof[n] for n in ["cb", "lgb", "xgb", "et", "ridge"]])
        w = blend_nnls(A, y)
        oof_blend = A @ w
        score = float(r2_score(y, oof_blend))

        p_te = aggregate_test_predictions(folds, X_test_aug)
        # weight order CB/LGB/XGB/ET/Ridge
        test_pred = w[0] * p_te["cb"] + w[1] * p_te["lgb"] + w[2] * p_te["xgb"] + w[3] * p_te["et"] + w[4] * p_te["ridge"]

        # full model training branches
        if tt == "nc":
            X_all_train_aug = np.hstack([X_train, train_df[["pred_egc", "pred_tg"]].values])
            X_all_test_aug = np.hstack([X_test, test_df[["pred_egc", "pred_tg"]].values])
            m_cb_full = CatBoostRegressor(iterations=550, learning_rate=0.03, depth=5, l2_leaf_reg=3.5, random_seed=RANDOM_SEED, verbose=0, thread_count=-1).fit(X_aug, y)
            m_lgb_full = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.025, max_depth=5, num_leaves=20, subsample=0.85, colsample_bytree=0.8, min_child_samples=5, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1).fit(X_aug, y)
            m_et_full = ExtraTreesRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1).fit(X_aug, y)
            full_nc_models = (m_cb_full, m_lgb_full, m_et_full)
            train_df["pred_nc"] = 0.45 * m_cb_full.predict(X_all_train_aug) + 0.35 * m_lgb_full.predict(X_all_train_aug) + 0.20 * m_et_full.predict(X_all_train_aug)
            test_df["pred_nc"] = 0.45 * m_cb_full.predict(X_all_test_aug) + 0.35 * m_lgb_full.predict(X_all_test_aug) + 0.20 * m_et_full.predict(X_all_test_aug)

        if tt == "eea":
            X_all_train_aug = np.hstack([X_train, train_df[["pred_egc", "pred_tg"]].values])
            X_all_test_aug = np.hstack([X_test, test_df[["pred_egc", "pred_tg"]].values])
            m_cb_full = CatBoostRegressor(iterations=550, learning_rate=0.03, depth=5, l2_leaf_reg=3.5, random_seed=RANDOM_SEED, verbose=0, thread_count=-1).fit(X_aug, y)
            m_lgb_full = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.025, max_depth=5, num_leaves=20, subsample=0.85, colsample_bytree=0.8, min_child_samples=5, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1).fit(X_aug, y)
            m_et_full = ExtraTreesRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1).fit(X_aug, y)
            full_eea_models = (m_cb_full, m_lgb_full, m_et_full)
            train_df["pred_eea"] = 0.45 * m_cb_full.predict(X_all_train_aug) + 0.35 * m_lgb_full.predict(X_all_train_aug) + 0.20 * m_et_full.predict(X_all_train_aug)
            test_df["pred_eea"] = 0.45 * m_cb_full.predict(X_all_test_aug) + 0.35 * m_lgb_full.predict(X_all_test_aug) + 0.20 * m_et_full.predict(X_all_test_aug)

        return folds, w, score, test_pred

    def transfer_egb(X_base, X_test_sub_base, train_df, test_df, sub_rows, sub_test_df):
        transfer_tr = train_df.loc[sub_mask_dummy, ["pred_egc", "pred_tg"]].values
        transfer_te = sub_test_df[["pred_egc", "pred_tg"]].values
        return np.hstack([X_base, transfer_tr]), np.hstack([X_test_sub_base, transfer_te])

    minority_order = ["egb", "nc", "eea", "ei"]
    minority_artifacts = {}
    for tt in minority_order:
        global sub_mask_dummy
        sub_mask_dummy = (train_df.target_type == tt).values
        if tt == "ei":
            def transfer_ei(X_base, X_test_sub_base, train_df, test_df, sub_rows, sub_test_df):
                transfer_tr = train_df.loc[sub_mask_dummy, ["pred_egc", "pred_tg", "pred_eea"]].values
                q_sum_tr = train_df.loc[sub_mask_dummy, ["pred_egc"]].values + train_df.loc[sub_mask_dummy, ["pred_eea"]].values
                X_aug = np.hstack([X_base, transfer_tr, q_sum_tr])
                transfer_te = sub_test_df[["pred_egc", "pred_tg", "pred_eea"]].values
                q_sum_te = sub_test_df[["pred_egc"]].values + sub_test_df[["pred_eea"]].values
                X_test_aug = np.hstack([X_test_sub_base, transfer_te, q_sum_te])
                return X_aug, X_test_aug
            transfer_fn = transfer_ei
        else:
            def transfer_minor(X_base, X_test_sub_base, train_df, test_df, sub_rows, sub_test_df):
                transfer_tr = train_df.loc[sub_mask_dummy, ["pred_egc", "pred_tg"]].values
                transfer_te = sub_test_df[["pred_egc", "pred_tg"]].values
                return np.hstack([X_base, transfer_tr]), np.hstack([X_test_sub_base, transfer_te])
            transfer_fn = transfer_minor

        folds, w, score, test_pred = minority_train_and_predict(tt, transfer_fn)
        cv_scores[tt] = score
        minority_artifacts[tt] = {"folds": folds, "weights": w}
        test_predictions[tt] = test_pred
        print(f"{tt.upper()} CV R2 {score:.4f}")

    # ---- Level 3 eps ----
    print("Training eps")
    eps_mask = (train_df.target_type == "eps").values
    sub_eps_idx = np.where(eps_mask)[0]
    sub_eps_test = test_df[test_df.target_type == "eps"].reset_index(drop=True)
    test_eps_idx = np.where((test_df.target_type == "eps").values)[0]
    X_base_eps = X_train[eps_mask]
    transfer_tr_eps = train_df.loc[eps_mask, ["pred_egc", "pred_tg", "pred_nc"]].values
    transfer_tr_sq = train_df.loc[eps_mask, ["pred_nc"]].values ** 2
    X_aug_eps = np.hstack([X_base_eps, transfer_tr_eps, transfer_tr_sq])
    y_eps = train_df.loc[eps_mask, "target"].values

    X_test_sub_eps = X_test[test_eps_idx]
    transfer_te_eps = sub_eps_test[["pred_egc", "pred_tg", "pred_nc"]].values
    transfer_te_sq = sub_eps_test[["pred_nc"]].values ** 2
    X_aug_test_eps = np.hstack([X_test_sub_eps, transfer_te_eps, transfer_te_sq])

    lgb_cfg = dict(n_estimators=350, learning_rate=0.025, max_depth=5, num_leaves=20, subsample=0.85, colsample_bytree=0.8, min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    cb_cfg = dict(iterations=550, learning_rate=0.03, depth=5, l2_leaf_reg=3.5, random_seed=RANDOM_SEED, verbose=0, thread_count=-1)
    xgb_cfg = dict(n_estimators=300, learning_rate=0.025, max_depth=5, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
    et_cfg = dict(n_estimators=250, max_depth=12, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
    ridge_cfg = dict(alpha=25.0)
    builders = {"lgb": lgb_cfg, "cb": cb_cfg, "xgb": xgb_cfg, "et": et_cfg, "ridge": ridge_cfg}
    folds_eps, oof_eps = make_fold(builders, X_aug_eps, y_eps)
    A_eps = np.column_stack([oof_eps[n] for n in ["cb", "lgb", "xgb", "et", "ridge"]])
    w_eps = blend_nnls(A_eps, y_eps)
    oof_blend = A_eps @ w_eps
    score_eps = float(r2_score(y_eps, oof_blend))
    cv_scores["eps"] = score_eps

    p_te = aggregate_test_predictions(folds_eps, X_aug_test_eps)
    test_predictions["eps"] = w_eps[0] * p_te["cb"] + w_eps[1] * p_te["lgb"] + w_eps[2] * p_te["xgb"] + w_eps[3] * p_te["et"] + w_eps[4] * p_te["ridge"]
    print(f"EPS CV R2 {score_eps:.4f}")

    # Physical linkers
    print("Training physical linkers")
    clean2phys = {c: p for c, p in zip(clean_train, phys_train)}
    # rebuild full phys mapping for all clean (train+test)
    all_clean = list(dict.fromkeys(clean_train + clean_test))
    # train physical featured merged
    # Build pivot tables
    piv_train_raw = train_df.pivot_table(index="smiles", columns="target_type", values="target", aggfunc="mean")
    piv_train_clean = train_df.pivot_table(index="clean_smiles", columns="target_type", values="target", aggfunc="mean")

    # linker EPS from NC
    sub_eps_nc = piv_train_clean.dropna(subset=["eps", "nc"]).copy()
    X_phys_nc = np.array([clean2phys[s] for s in sub_eps_nc.index])
    X_in_eps = np.column_stack([sub_eps_nc["nc"].values, sub_eps_nc["nc"].values ** 2, X_phys_nc])
    m_eps_r = Ridge(alpha=2.0).fit(X_in_eps, sub_eps_nc["eps"].values)

    X_in_nc = np.column_stack([sub_eps_nc["eps"].values, np.sqrt(np.maximum(1.0, sub_eps_nc["eps"].values)), X_phys_nc])
    m_nc_r = Ridge(alpha=0.5).fit(X_in_nc, sub_eps_nc["nc"].values)

    sub_egb_egc = piv_train_clean.dropna(subset=["egb", "egc"]).copy()
    X_phys_egb = np.array([clean2phys[s] for s in sub_egb_egc.index])
    X_in_egb = np.column_stack([sub_egb_egc["egc"].values, X_phys_egb])
    m_egb_r = Ridge(alpha=0.5).fit(X_in_egb, sub_egb_egc["egb"].values)

    # gt maps
    def build_gt_map(index_col):
        from collections import defaultdict

        out = {}
        for smi, group in train_df.groupby(index_col):
            out[smi] = {}
            for _, r in group.iterrows():
                out[smi][r.target_type] = float(r.target)
        return out

    gt_map_raw = build_gt_map("smiles")
    gt_map_clean = build_gt_map("clean_smiles")

    artifact = {
        "version": 1,
        "random_seed": RANDOM_SEED,
        "feature_names": ml_names,
        "feature_dim": X_train.shape[1],
        "target_types": TARGET_TYPES,
        "cv_scores": cv_scores,
        "official_mean": float(np.mean(list(cv_scores.values()))),
        "models": {
            "tg": {"folds": folds_tg, "weights": w_tg},
            "egc": {"folds": folds_egc, "weights": w_egc},
            "egb": {"folds": minority_artifacts["egb"]["folds"], "weights": minority_artifacts["egb"]["weights"]},
            "nc": {"folds": minority_artifacts["nc"]["folds"], "weights": minority_artifacts["nc"]["weights"]},
            "eea": {"folds": minority_artifacts["eea"]["folds"], "weights": minority_artifacts["eea"]["weights"]},
            "ei": {"folds": minority_artifacts["ei"]["folds"], "weights": minority_artifacts["ei"]["weights"]},
            "eps": {"folds": folds_eps, "weights": w_eps},
        },
        "full_models": {
            "nc": full_nc_models,
            "eea": full_eea_models,
        },
        "physical_linkers": {
            "eps_from_nc": m_eps_r,
            "nc_from_eps": m_nc_r,
            "egb_from_egc": m_egb_r,
        },
        "gt_maps": {"raw": gt_map_raw, "clean": gt_map_clean},
        "bounds": {k: list(v) for k, v in __import__("src.config", fromlist=["PHYSICAL_BOUNDS"]).PHYSICAL_BOUNDS.items()},
        "clean2phys_example": None,
    }

    out_path = ARTIFACTS_DIR / "amalgam_ensemble.joblib"
    joblib.dump(artifact, out_path)
    # Save physical features mapping not required.

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"cv_scores": cv_scores, "official_mean": artifact["official_mean"]}, f, indent=2)

    print("Saved", out_path)
    print("Official mean R2", artifact["official_mean"])


if __name__ == "__main__":
    main()