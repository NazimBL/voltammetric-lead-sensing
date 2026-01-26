# PCR_benchmark.py
# Compare PCA→Regression (PCR) against your Lasso baseline using the SAME split logic:
# - 20% holdout drawn ONLY from the UNSEEN subset
# - Bin-stratified K-fold CV on the remaining training pool
# - Step 1: Print Pearson r / R² for PC1..PC3 vs y (train only; no leakage)
# - Step 2–3: Tune PCR (PCA inside pipeline) with identical CV
# - Step 4: Evaluate once on the 20% UNSEEN holdout and compare RMSE to Lasso baseline
# - Extra: Bin-wise errors on holdout + per-sample listing

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
from scipy.stats import pearsonr

# ====================
# CONFIG
# ====================
INPUT_CSV = "raw_matrix_all.csv"   # produced by your prep script
TARGET_COL = "concentration_ppb"
DATASET_COL = "dataset"
RANDOM_STATE = 42

# Baseline Lasso holdout RMSE (from your previous run) to compare against
BASELINE_LASSO_RMSE = 3.873

# Splitting controls
N_SPLITS = 5
HOLDOUT_FRACTION = 0.20
N_BINS = 7
FORCE_ZERO_BIN = True

# PCR grid
PCR_N_COMPONENTS = [2, 3, 5, 8, 12]
USE_RIDGE = True
RIDGE_ALPHAS = [0.0, 0.1, 1.0, 5.0]  # 0.0 ≈ LinearRegression

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
    """Temporary bins for stratified splitting in regression."""
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

# ====================
# Main
# ====================
def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c.startswith("V_")]
    assert len(feature_cols) > 0, "No potential columns found (expected columns starting with 'V_')."

    X_all = df[feature_cols].copy()
    y_all = df[TARGET_COL].astype(float)
    groups = df[DATASET_COL].astype(str).values

    print("Samples total:", len(df))
    print("Potential points (features):", len(feature_cols))
    print(df[DATASET_COL].value_counts())

    # --- Build 20% holdout ONLY from the UNSEEN subset ---
    idx_unseen = np.where(groups == "unseen")[0]
    idx_lab = np.where(groups == "lab")[0]

    y_bins_unseen = make_strat_bins(y_all.iloc[idx_unseen], n_bins=N_BINS, min_per_bin=2, force_zero_bin=FORCE_ZERO_BIN)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE)
    unseen_train_rel, unseen_hold_rel = next(sss.split(X_all.iloc[idx_unseen], y_bins_unseen))

    hold_idx = idx_unseen[unseen_hold_rel]            # 20% of UNSEEN only
    unseen_train_idx = idx_unseen[unseen_train_rel]   # remaining 80% of UNSEEN
    train_idx = np.concatenate([idx_lab, unseen_train_idx])  # LAB + 80% UNSEEN

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

    # --- Step 1: PCA on TRAINING ONLY for correlation diagnostics (no leakage) ---
    pre_center_only = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("center", StandardScaler(with_mean=True, with_std=False))
            ]), feature_cols)
        ],
        remainder="drop"
    )
    X_train_centered = pre_center_only.fit_transform(X_train)
    pca_diag = PCA(n_components=3, random_state=RANDOM_STATE)
    scores_diag = pca_diag.fit_transform(X_train_centered)

    print("\n--- Pearson correlation (train only) ---")
    for k in range(scores_diag.shape[1]):
        r, p = pearsonr(scores_diag[:, k], y_train)
        print(f"PC{k+1} vs y: r = {r:.3f}, R² = {r**2:.3f}, p = {p:.2e}")

    # --- Step 2: PCR pipelines (inside CV) ---
    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("center", StandardScaler(with_mean=True, with_std=False))
            ]), feature_cols)
        ],
        remainder="drop"
    )

    models: Dict[str, Pipeline] = {}
    models["PCR (Linear)"] = Pipeline([
        ("prep", preproc),
        ("pca", PCA(n_components=5, svd_solver="auto", random_state=RANDOM_STATE)),
        ("reg", LinearRegression())
    ])
    if USE_RIDGE:
        models["PCR (Ridge)"] = Pipeline([
            ("prep", preproc),
            ("pca", PCA(n_components=5, svd_solver="auto", random_state=RANDOM_STATE)),
            ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ])

    # Phase 1: quick compare with defaults
    rmse_table: Dict[str, float] = {}
    for name, pipe in models.items():
        mean_rmse, _ = evaluate_with_fixed_cv(name, X_train, y_train, pipe, splits)
        rmse_table[name] = mean_rmse
    best_model_name = min(rmse_table, key=rmse_table.get)
    print(f"\n>>> Phase 1 winner: {best_model_name}")

    # Phase 2: tune n_components (+ alpha for Ridge)
    grids = {
        "PCR (Linear)": {
            "pca__n_components": PCR_N_COMPONENTS
        },
        "PCR (Ridge)": {
            "pca__n_components": PCR_N_COMPONENTS,
            "reg__alpha": RIDGE_ALPHAS
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

    # --- Step 3 & 4: Final fit on training & Holdout evaluation ---
    y_pred_hold = best_pipeline.predict(X_hold)
    rmse_val = rmse(y_hold, y_pred_hold)
    mae_val  = mae(y_hold, y_pred_hold)
    r2_val   = r2(y_hold, y_pred_hold)

    print("\n=== 20% Holdout Evaluation (UNSEEN-only) — PCR ===")
    print("RMSE:", f"{rmse_val:.3f}")
    print("MAE :", f"{mae_val:.3f}")
    print("R²  :", f"{r2_val:.3f}")
    if BASELINE_LASSO_RMSE is not None:
        delta = rmse_val - BASELINE_LASSO_RMSE
        print(f"Compared to Lasso baseline RMSE {BASELINE_LASSO_RMSE:.3f}: ΔRMSE = {delta:+.3f} (negative = better)")

    # ---------- Extra diagnostics on the holdout ----------
    # 1) Bin-wise error analysis to see how error varies across ranges
    y_true = np.asarray(y_hold, dtype=float)
    y_pred = np.asarray(y_pred_hold, dtype=float)

    bins_edges = [-0.1, 0.0, 10.0, 25.0, float('inf')]
    bins_labels = ["zero", "low(0-10]", "mid(10-25]", "high(>25)"]
    bin_idx = np.digitize(y_true, bins_edges[1:], right=True)

    print("\n=== Holdout error by concentration range ===")
    for i, label in enumerate(bins_labels):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        b_rmse = np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean())
        b_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        print(f"  {label:12s}: n={mask.sum():3d}  RMSE={b_rmse:6.2f}  MAE={b_mae:6.2f}")

    # 2) One-by-one comparison (sorted by true ppb for readability)
    order = np.argsort(y_true)
    print("\n=== Holdout per-sample (sorted by true) ===")
    for idx in order:
        yt = float(y_true[idx])
        yp = float(y_pred[idx])
        print(f"True {yt:6.1f} ppb -> Pred {yp:7.2f} ppb")


if __name__ == "__main__":
    main()
