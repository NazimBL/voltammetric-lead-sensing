# PCR_PLS_benchmark_unseen50.py
# Fair comparison with a larger, "clean" test: 50% of UNSEEN held out (~50 samples).
# Pipeline and diagnostics match your earlier scripts.

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr

# ====================
# CONFIG
# ====================
DATA_DIR = "../../data"
INPUT_CSV = f"{DATA_DIR}/raw_matrix_all.csv"
TARGET_COL = "concentration_ppb"
DATASET_COL = "dataset"
RANDOM_STATE = 42

# For the final delta line, you can set your Lasso RMSE here once you re-run Lasso with the SAME holdout policy.
BASELINE_LASSO_RMSE = None  # e.g., 3.873 after you re-run Lasso with unseen-50% holdout

# Splitting controls (CHANGED: 50% of UNSEEN)
N_SPLITS = 5
HOLDOUT_FRACTION = 0.50
N_BINS = 7
FORCE_ZERO_BIN = True

# Model grids
PCR_N_COMPONENTS = [2, 3, 5, 8, 12, 15]
RIDGE_ALPHAS = [0.0, 0.1, 1.0, 5.0]  # 0.0 ≈ LinearRegression
PLS_N_COMPONENTS = [2, 3, 5, 8, 12, 15]

# ====================
# Metrics & Scoring
# ====================
def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred) -> float:
    return r2_score(y_true, y_pred)

SCORING = {
    "rmse": make_scorer(rmse, greater_is_better=False),
    "mae":  make_scorer(mae,  greater_is_better=False),
    "r2":   make_scorer(r2),
}



# ====================
# Helpers
# ====================
def make_strat_bins(y: pd.Series, n_bins: int, min_per_bin: int, force_zero_bin: bool = True) -> pd.Series:
    y = pd.Series(y).astype(float).reset_index(drop=True)
    zero_mask = (y == 0.0) if force_zero_bin else pd.Series(False, index=y.index)
    nonzero = y[~zero_mask]
    k_quant = max(1, n_bins - (1 if force_zero_bin else 0))
    while k_quant >= 1:
        try:
            qbins = pd.qcut(nonzero, q=k_quant, labels=False, duplicates="drop")
            bins = pd.Series(-1, index=y.index, dtype=int)
            if force_zero_bin:
                bins[zero_mask] = 0
                bins[~zero_mask] = qbins.values.astype(int) + 1
            else:
                bins[~zero_mask] = qbins.values.astype(int)
            if bins.value_counts().min() >= min_per_bin:
                return bins
        except Exception:
            pass
        k_quant -= 1
    return pd.Series(0, index=y.index, dtype=int)

def evaluate_with_fixed_cv(name: str, X: pd.DataFrame, y: pd.Series,
                           pipeline: Pipeline, splits) -> Tuple[float, Dict[str, Tuple[float, float]]]:
    cv_results = cross_validate(
        pipeline, X, y, cv=list(splits), scoring=SCORING,
        return_train_score=False, n_jobs=-1, error_score="raise"
    )
    print(f"\n=== {name} ===")
    summary: Dict[str, Tuple[float, float]] = {}
    for m in SCORING.keys():
        scores = cv_results[f"test_{m}"]
        if m in ["rmse", "mae"]:
            scores = -scores
        mean_val, std_val = scores.mean(), scores.std()
        summary[m] = (mean_val, std_val)
        print(f"{m:>6}: {mean_val:.3f} ± {std_val:.3f}")
    return summary["rmse"][0], summary


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
def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c.startswith("V_")]
    assert feature_cols, "No potential columns found (expected columns starting with 'V_')."

    X_all = df[feature_cols].copy()
    y_all = df[TARGET_COL].astype(float)
    groups = df[DATASET_COL].astype(str).values

    print("Samples total:", len(df))
    print("Potential points (features):", len(feature_cols))
    print(df[DATASET_COL].value_counts())

    # --- Holdout: 50% from UNSEEN only ---
    idx_unseen = np.where(groups == "unseen")[0]
    idx_lab    = np.where(groups == "lab")[0]

    y_bins_unseen = make_strat_bins(y_all.iloc[idx_unseen], n_bins=N_BINS, min_per_bin=2, force_zero_bin=FORCE_ZERO_BIN)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE)
    unseen_train_rel, unseen_hold_rel = next(sss.split(X_all.iloc[idx_unseen], y_bins_unseen))

    hold_idx = idx_unseen[unseen_hold_rel]            # 50% of UNSEEN only
    unseen_train_idx = idx_unseen[unseen_train_rel]   # remaining 50% of UNSEEN
    train_idx = np.concatenate([idx_lab, unseen_train_idx])  # LAB + 50% UNSEEN

    X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
    X_hold,  y_hold  = X_all.iloc[hold_idx],  y_all.iloc[hold_idx]

    print(f"Holdout size (UNSEEN only): {len(hold_idx)} ≈ {HOLDOUT_FRACTION*100:.0f}% of unseen ({len(idx_unseen)} samples)")

    # --- Build CV splits on TRAINING ONLY ---
    bins_train = make_strat_bins(y_train, n_bins=N_BINS, min_per_bin=2, force_zero_bin=FORCE_ZERO_BIN)
    min_count = bins_train.value_counts().min()
    n_splits_final = int(min(N_SPLITS, max(2, min_count)))
    if n_splits_final < N_SPLITS:
        print(f"[Info] Reducing n_splits from {N_SPLITS} to {n_splits_final} due to small bins.")
    skf = StratifiedKFold(n_splits=n_splits_final, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_train, bins_train))

    # --- Step 1: PCA correlations (train only; no leakage) ---
    pre_center_only = ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                                        ("center", StandardScaler(with_mean=True, with_std=False))]), feature_cols)],
        remainder="drop"
    )
    X_train_centered = pre_center_only.fit_transform(X_train)
    pca_diag = PCA(n_components=3, random_state=RANDOM_STATE)
    scores_diag = pca_diag.fit_transform(X_train_centered)

    print("\n--- Pearson correlation (train only) ---")
    for k in range(scores_diag.shape[1]):
        r, p = pearsonr(scores_diag[:, k], y_train)
        print(f"PC{k+1} vs y: r = {r:.3f}, R² = {r**2:.3f}, p = {p:.2e}")

    # --- Shared preprocessor ---
    preproc = ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                                        ("center", StandardScaler(with_mean=True, with_std=False))]), feature_cols)],
        remainder="drop"
    )

    # --- Models ---
    models: Dict[str, Pipeline] = {
        "PCR (Linear)": Pipeline([("prep", preproc),
                                  ("pca", PCA(n_components=5, svd_solver="auto", random_state=RANDOM_STATE)),
                                  ("reg", LinearRegression())]),
        "PCR (Ridge)" : Pipeline([("prep", preproc),
                                  ("pca", PCA(n_components=5, svd_solver="auto", random_state=RANDOM_STATE)),
                                  ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
        "PLS"         : Pipeline([("prep", preproc),
                                  ("pls", PLSRegression(n_components=5))])
    }

    # Phase 1: quick compare
    rmse_table: Dict[str, float] = {}
    for name, pipe in models.items():
        mean_rmse, _ = evaluate_with_fixed_cv(name, X_train, y_train, pipe, splits)
        rmse_table[name] = mean_rmse
    best_model_name = min(rmse_table, key=rmse_table.get)
    print(f"\n>>> Phase 1 winner: {best_model_name}")

    # Phase 2: tune n_components (and alpha for Ridge)
    grids = {
        "PCR (Linear)": {"pca__n_components": PCR_N_COMPONENTS},
        "PCR (Ridge)" : {"pca__n_components": PCR_N_COMPONENTS, "reg__alpha": RIDGE_ALPHAS},
        "PLS"         : {"pls__n_components": PLS_N_COMPONENTS}
    }
    winner = models[best_model_name]
    param_grid = grids[best_model_name]

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

    # --- Holdout evaluation ---
    y_pred_hold = best_pipeline.predict(X_hold)
    rmse_val = rmse(y_hold, y_pred_hold)
    mae_val  = mae(y_hold, y_pred_hold)
    r2_val   = r2(y_hold, y_pred_hold)

    print("\n=== 50% Holdout Evaluation (UNSEEN-only) — {0} ===".format(best_model_name))
    print("RMSE:", f"{rmse_val:.3f}")
    print("MAE :", f"{mae_val:.3f}")
    print("R²  :", f"{r2_val:.3f}")
    if BASELINE_LASSO_RMSE is not None:
        delta = rmse_val - BASELINE_LASSO_RMSE
        print(f"Compared to Lasso baseline RMSE {BASELINE_LASSO_RMSE:.3f}: ΔRMSE = {delta:+.3f} (negative = better)")

    # --- Optional: bucketed reporting aligned with manuscript bands ---
    bstats = bucket_stats(y_hold, y_pred_hold)
    print("\n=== Holdout bucketed performance (0–5, 5–10, 10–15, 15–25, >25) ===")
    for k, d in bstats.items():
        if d["n"] == 0:
            continue
        print(f"{k:>5s}: n={d['n']:2d}  RMSE={d['rmse']:6.2f}  MAE={d['mae']:6.2f}")

    # ---------- Extra diagnostics on the holdout ----------
    y_true = np.asarray(y_hold, dtype=float)
    y_pred = np.asarray(y_pred_hold, dtype=float)

    # Bin-wise error analysis
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

    # Per-sample listing
    order = np.argsort(y_true)
    print("\n=== Holdout per-sample (sorted by true) ===")
    for idx in order:
        yt = float(y_true[idx])
        yp = float(y_pred[idx])
        print(f"True {yt:6.1f} ppb -> Pred {yp:7.2f} ppb")


if __name__ == "__main__":
    main()
