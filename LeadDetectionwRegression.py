# LeadDetectionwRegression_unseen50.py
# Same pipeline as before, but with a FAIR test:
#   - Holdout = 50% of the UNSEEN set only (≈50 samples)
#   - Train = LAB + remaining 50% of UNSEEN
#   - Bin-stratified K-fold CV on TRAINING ONLY
#   - Model zoo -> select winner -> (optional) grid-tune -> evaluate once on holdout
#   - Extra: bin-wise errors + per-sample listing on holdout

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ====================
# CONFIG
# ====================
# Paths: engineered ROI features (like your FeatureEng2 outputs)
INPUT_CSV   = "volt_features_roi_min.csv"     # LAB features
TEST_CSV    = "unseen_features_roi_min.csv"   # UNSEEN features
TARGET_COL  = "concentration_ppb"
RANDOM_STATE = 42

# Splitting controls
N_SPLITS   = 5             # Will auto-reduce if a bin is too small
N_BINS     = 7             # 1 bin for zeros + ~6 quantile bins for non-zeros
HOLDOUT_FRACTION = 0.50    # <<< 50% of UNSEEN only
FORCE_ZERO_BIN   = True    # Keep exact zeros in their own stratum if counts allow

# ====================
# Metrics & Scoring
# ====================
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred):  return mean_absolute_error(y_true, y_pred)
def r2 (y_true, y_pred):  return r2_score(y_true, y_pred)

SCORING = {
    "rmse": make_scorer(rmse, greater_is_better=False),
    "mae":  make_scorer(mae,  greater_is_better=False),
    "r2":   make_scorer(r2)
}

# ====================
# Helpers
# ====================
def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = {"sample_id", "source_file", target_col, "dataset"}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]

def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler",  StandardScaler())
        ]), feature_cols)
    ], remainder="drop")

def evaluate_with_fixed_cv(name: str, X: pd.DataFrame, y_numeric: pd.Series,
                           pipeline: Pipeline, splits) -> Tuple[float, dict]:
    cv_results = cross_validate(
        pipeline, X, y_numeric,
        cv=list(splits), scoring=SCORING,
        return_train_score=False, n_jobs=-1, error_score="raise")
    print(f"\n=== {name} ===")
    summary = {}
    for m in SCORING.keys():
        scores = cv_results[f"test_{m}"]
        if m in ["rmse", "mae"]:
            scores = -scores  # undo negation
        mean, std = scores.mean(), scores.std()
        summary[m] = (mean, std)
        print(f"{m:>6}: {mean:.3f} ± {std:.3f}")
    return summary["rmse"][0], summary

# -------- Robust binning for stratification (not for modeling) --------
def make_strat_bins(y: pd.Series, n_bins: int, min_per_bin: int,
                    force_zero_bin: bool = True) -> pd.Series:
    """
    Build temporary bins to balance splits in regression:
      - Optional dedicated zero bin
      - Quantile bins for non-zeros
      - Automatically reduces bin count until each bin has >= min_per_bin
    Returns integer-coded Series (0..K-1).
    """
    y = pd.Series(y).astype(float).reset_index(drop=True)
    zero_mask = (y == 0.0) if force_zero_bin else pd.Series(False, index=y.index)
    nonzero = y[~zero_mask]
    k_quant = max(1, n_bins - (1 if force_zero_bin else 0))

    while k_quant >= 1:
        try:
            qbins = pd.qcut(nonzero, q=k_quant, labels=False, duplicates='drop')
            bins = pd.Series(-1, index=y.index, dtype=int)
            if force_zero_bin:
                bins[zero_mask] = 0
                bins[~zero_mask] = (qbins.values.astype(int) + 1)
            else:
                bins[~zero_mask] = qbins.values.astype(int)
            if bins.value_counts().min() >= min_per_bin:
                return bins
        except Exception:
            pass
        k_quant -= 1

    # Fallback: single bin
    return pd.Series(0, index=y.index, dtype=int)

def coverage_within_pct(y_true, y_pred, pct=0.15, exclude_zeros=True):
    """
    Coverage within ±pct relative error band.
    If exclude_zeros=True, entries with y_true==0 are excluded (relative error undefined).
    Returns (coverage, n_used).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if exclude_zeros:
        mask = y_true > 0
    else:
        mask = np.ones_like(y_true, dtype=bool)

    n = int(mask.sum())
    if n == 0:
        return np.nan, 0

    rel_ok = np.abs(y_pred[mask] - y_true[mask]) <= pct * y_true[mask]
    return float(rel_ok.mean()), n


def bucket_stats(y_true, y_pred):
    """
    Returns dict of bucket -> stats:
      n, rmse, mae, cov15 (zeros excluded within bucket where y_true>0)
    Buckets follow your manuscript (0–5, 5–10, 10–15, 15–25, >25).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    buckets = [
        ("0–5",   (0.0, 5.0)),
        ("5–10",  (5.0, 10.0)),
        ("10–15", (10.0, 15.0)),
        ("15–25", (15.0, 25.0)),
        (">25",   (25.0, np.inf)),
    ]

    out = {}
    for name, (lo, hi) in buckets:
        # left-open/right-closed is fine; keep consistent across all reporting
        mask = (y_true > lo) & (y_true <= hi) if np.isfinite(hi) else (y_true > lo)
        if mask.sum() == 0:
            out[name] = {"n": 0, "rmse": np.nan, "mae": np.nan, "cov15": np.nan, "n_cov": 0}
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        rmse_b = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae_b  = float(np.mean(np.abs(yt - yp)))
        cov15_b, n_cov = coverage_within_pct(yt, yp, pct=0.15, exclude_zeros=True)
        out[name] = {"n": int(mask.sum()), "rmse": rmse_b, "mae": mae_b, "cov15": cov15_b, "n_cov": n_cov}
    return out

# ====================
# Main
# ====================
def main():
    # ---------- Load LAB + UNSEEN features; align shared columns ----------
    df_lab    = pd.read_csv(INPUT_CSV)
    df_unseen = pd.read_csv(TEST_CSV)

    lab_feats    = select_feature_columns(df_lab, TARGET_COL)
    unseen_feats = select_feature_columns(df_unseen, TARGET_COL)
    feature_cols = sorted(list(set(lab_feats).intersection(unseen_feats)))
    assert len(feature_cols) > 0, "No shared numeric feature columns between lab and unseen."

    df_lab    = df_lab[feature_cols + [TARGET_COL]].assign(dataset="lab")
    df_unseen = df_unseen[feature_cols + [TARGET_COL]].assign(dataset="unseen")
    df_all = pd.concat([df_lab, df_unseen], ignore_index=True)

    X_all = df_all[feature_cols].copy()
    y_all = df_all[TARGET_COL].astype(float)
    groups = df_all["dataset"].astype(str).values

    print("Samples total (lab + unseen):", len(df_all))
    print("Feature count:", len(feature_cols))
    print(df_all["dataset"].value_counts())

    # ---------- 50% stratified holdout from the UNSEEN subset only ----------
    idx_unseen = np.where(groups == "unseen")[0]
    idx_lab    = np.where(groups == "lab")[0]

    # Stratify ONLY within unseen for holdout selection
    y_bins_unseen = make_strat_bins(
        y_all.iloc[idx_unseen], n_bins=N_BINS, min_per_bin=2, force_zero_bin=FORCE_ZERO_BIN
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE)
    unseen_train_rel, unseen_hold_rel = next(sss.split(X_all.iloc[idx_unseen], y_bins_unseen))

    hold_idx = idx_unseen[unseen_hold_rel]           # 50% of UNSEEN only
    unseen_train_idx = idx_unseen[unseen_train_rel]  # remaining 50% of UNSEEN
    train_idx = np.concatenate([idx_lab, unseen_train_idx])  # LAB + 50% UNSEEN

    X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
    X_hold,  y_hold  = X_all.iloc[hold_idx],  y_all.iloc[hold_idx]
    bins_train = make_strat_bins(y_train, n_bins=N_BINS, min_per_bin=2, force_zero_bin=FORCE_ZERO_BIN)

    print(f"Holdout size (UNSEEN only): {len(hold_idx)} ≈ {HOLDOUT_FRACTION*100:.0f}% of unseen ({len(idx_unseen)} samples)")

    # ---------- Adaptive K-fold on training using the bins ----------
    min_count = bins_train.value_counts().min()
    n_splits_final = int(min(N_SPLITS, max(2, min_count)))
    if n_splits_final < N_SPLITS:
        print(f"[Info] Reducing n_splits from {N_SPLITS} to {n_splits_final} due to small bins.")
    skf = StratifiedKFold(n_splits=n_splits_final, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_train, bins_train))

    # ---------- Shared preprocessor ----------
    preproc = make_preprocessor(feature_cols)

    # ---------- Model zoo (Phase 1: model selection) ----------
    models: Dict[str, Pipeline] = {
        "Linear Regression": Pipeline([("prep", preproc), ("reg", LinearRegression())]),
        "Ridge":             Pipeline([("prep", preproc), ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
        "Lasso":             Pipeline([("prep", preproc), ("reg", Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=5000))]),
        "SVR (RBF)":         Pipeline([("prep", preproc), ("reg", SVR(kernel="rbf", C=10.0, gamma="scale"))]),
        "Random Forest":     Pipeline([("prep", preproc), ("reg", RandomForestRegressor(
                                        n_estimators=300, max_depth=7, min_samples_leaf=2,
                                        n_jobs=-1, random_state=RANDOM_STATE))])
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([("prep", preproc), ("reg", XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1
        ))])

    # Evaluate all models on identical folds; pick best by CV RMSE
    rmse_table = {}
    for name, pipe in models.items():
        mean_rmse, _ = evaluate_with_fixed_cv(name, X_train, y_train, pipe, splits)
        rmse_table[name] = mean_rmse
    best_model_name = min(rmse_table, key=rmse_table.get)
    print(f"\n>>> Phase 1 winner: {best_model_name}")

    # ---------- Phase 2: hyperparameter tuning on the winner ----------
    grids = {
        "Random Forest": {
            "reg__n_estimators": [200, 400, 800],
            "reg__max_depth": [5, 7, 9, None],
            "reg__min_samples_leaf": [1, 2, 4]
        },
        "Ridge": {
            "reg__alpha": [0.1, 1.0, 5.0, 10.0]
        },
        "Lasso": {
            "reg__alpha": [0.001, 0.01, 0.1, 1.0]
        },
        "SVR (RBF)": {
            "reg__C": [1.0, 3.0, 10.0, 30.0],
            "reg__gamma": ["scale", 0.1, 0.01]
        },
        "XGBoost": {
            "reg__n_estimators": [300, 600, 900],
            "reg__max_depth": [3, 4, 5],
            "reg__learning_rate": [0.03, 0.05, 0.1],
            "reg__subsample": [0.8, 1.0],
            "reg__colsample_bytree": [0.8, 1.0],
            "reg__reg_lambda": [0.5, 1.0, 2.0]
        }
    }

    winner = models[best_model_name]
    param_grid = grids.get(best_model_name, None)

    if param_grid is not None:
        gscv = GridSearchCV(
            estimator=winner,
            param_grid=param_grid,
            scoring=make_scorer(rmse, greater_is_better=False),
            cv=list(splits),
            n_jobs=-1,
            error_score="raise",
            refit=True
        )
        gscv.fit(X_train, y_train)
        print(f"\n>>> Phase 2 best params for {best_model_name}: {gscv.best_params_}")
        best_pipeline = gscv.best_estimator_
    else:
        print("\n[Info] No grid defined for the winner; using Phase 1 pipeline as-is.")
        best_pipeline = winner
        best_pipeline.fit(X_train, y_train)

    # ---------- Final fit on all training, evaluate once on 50% UNSEEN holdout ----------
    y_pred_hold = best_pipeline.predict(X_hold)
    rmse_val = rmse(y_hold, y_pred_hold)
    mae_val  = mae(y_hold, y_pred_hold)
    r2_val   = r2(y_hold, y_pred_hold)

    print("\n=== 50% Stratified Holdout (UNSEEN-only) Evaluation ===")
    print("RMSE:", f"{rmse_val:.3f}")
    print("MAE :", f"{mae_val:.3f}")
    print("R²  :", f"{r2_val:.3f}")


    # --- Optional: bucketed reporting aligned with manuscript bands ---
    bstats = bucket_stats(y_hold, y_pred_hold)
    print("\n=== Holdout bucketed performance (0–5, 5–10, 10–15, 15–25, >25) ===")
    for k, d in bstats.items():
        if d["n"] == 0:
            continue
        print(f"{k:>5s}: n={d['n']:2d}  RMSE={d['rmse']:6.2f}  MAE={d['mae']:6.2f}")

    # ---------- Extra diagnostics on the holdout ----------
    # 1) Bin-wise error analysis
    y_true = np.asarray(y_hold, dtype=float)
    y_pred = np.asarray(y_pred_hold, dtype=float)

    bins_edges  = [-0.1, 0.0, 10.0, 25.0, float('inf')]
    bins_labels = ["zero", "low(0-10]", "mid(10-25]", "high(>25)"]
    bin_idx = np.digitize(y_true, bins_edges[1:], right=True)

    print("\n=== Holdout error by concentration range ===")
    for i, label in enumerate(bins_labels):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        b_rmse = np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean())
        b_mae  = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        print(f"  {label:12s}: n={mask.sum():3d}  RMSE={b_rmse:6.2f}  MAE={b_mae:6.2f}")

    # 2) One-by-one comparison (sorted by true ppb)
    order = np.argsort(y_true)
    print("\n=== Holdout per-sample (sorted by true) ===")
    for idx in order:
        yt = float(y_true[idx])
        yp = float(y_pred[idx])
        print(f"True {yt:6.1f} ppb -> Pred {yp:7.2f} ppb")

if __name__ == "__main__":
    main()
