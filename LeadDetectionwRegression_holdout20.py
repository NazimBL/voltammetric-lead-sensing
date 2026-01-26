import pandas as pd
import numpy as np
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
# Paths: lab features (discrete ppb) + unseen features (mixed/continuous ppb)
INPUT_CSV = "volt_features_roi_min.csv"
TEST_CSV = "unseen_features_roi_min.csv"
TARGET_COL = "concentration_ppb"
RANDOM_STATE = 42

# Splitting controls
N_SPLITS = 5            # Will auto-reduce if any bin can't support K-fold
N_BINS = 7              # 1 bin for 0 ppb + ~6 quantile bins for non-zeros (auto-shrinks if needed)
HOLDOUT_FRACTION = 0.20 # 20% stratified holdout for final, unbiased check
FORCE_ZERO_BIN = True   # Keep exact zeros in their own stratum if counts allow

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
    "mae": make_scorer(mae, greater_is_better=False),
    "r2": make_scorer(r2),
}

# ====================
# Helpers
# ====================
def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = {"sample_id", "source_file", target_col}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]

def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )

def evaluate_with_fixed_cv(
    name: str,
    X: pd.DataFrame,
    y_numeric: pd.Series,
    pipeline: Pipeline,
    splits,
):
    """Run cross_validate on fixed splits (to reuse the same folds for all models)."""
    cv_results = cross_validate(
        pipeline,
        X,
        y_numeric,
        cv=list(splits),
        scoring=SCORING,
        return_train_score=False,
        n_jobs=-1,
        error_score="raise",
    )
    print(f"\n=== {name} ===")
    summary = {}
    for m in SCORING.keys():
        scores = cv_results[f"test_{m}"]
        # Undo negation for error metrics
        if m in ["rmse", "mae"]:
            scores = -scores
        mean_val, std_val = scores.mean(), scores.std()
        summary[m] = (mean_val, std_val)
        print(f"{m:>6}: {mean_val:.3f} ± {std_val:.3f}")
    # Return a single scalar to rank models (use RMSE)
    return summary["rmse"][0], summary

# -------- New: robust binning for stratification (not for modeling) --------
def make_strat_bins(
    y: pd.Series,
    n_bins: int,
    min_per_bin: int,
    force_zero_bin: bool = True,
    random_state: int = 42,
) -> pd.Series:
    """
    Build temporary bins to balance splits in regression:
      - Optional dedicated zero bin
      - Quantile bins for non-zeros
      - Automatically reduces bin count until each bin has >= min_per_bin
    Returns integer-coded Series (0..K-1).
    """
    y = pd.Series(y).astype(float).reset_index(drop=True)
    if force_zero_bin:
        zero_mask = y == 0.0
    else:
        zero_mask = pd.Series(False, index=y.index)

    nonzero = y[~zero_mask]
    # Start with desired bins; subtract 1 if we reserve a zero bin
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

            # Check minimum count constraint
            if bins.value_counts().min() >= min_per_bin:
                return bins
        except Exception:
            # If qcut fails due to ties, try fewer bins
            pass
        k_quant -= 1

    # Fallback: single bin
    return pd.Series(0, index=y.index, dtype=int)

# ====================
# Main
# ====================
def main() -> None:
    # ---------- Load lab + unseen, align features ----------
    df_lab = pd.read_csv(INPUT_CSV)
    df_unseen = pd.read_csv(TEST_CSV)

    lab_feats = select_feature_columns(df_lab, TARGET_COL)
    unseen_feats = select_feature_columns(df_unseen, TARGET_COL)
    feature_cols = sorted(list(set(lab_feats).intersection(unseen_feats)))
    assert len(feature_cols) > 0, "No shared numeric feature columns between lab and unseen."

    df_all = pd.concat(
        [
            df_lab[feature_cols + [TARGET_COL]].assign(dataset="lab"),
            df_unseen[feature_cols + [TARGET_COL]].assign(dataset="unseen"),
        ],
        ignore_index=True,
    )

    X_all = df_all[feature_cols].copy()
    y_all = df_all[TARGET_COL].astype(float)

    print("Samples total (lab + unseen):", len(df_all))
    print("Feature count:", len(feature_cols))
    print(df_all["dataset"].value_counts())

    # ---------- 20% stratified holdout by binned ranges ----------
    # Build bins that ensure variety (zeros + non-zero quantiles)
    min_needed = 2  # must allow at least 1 sample in train & 1 in holdout
    y_bins = make_strat_bins(
        y_all,
        n_bins=N_BINS,
        min_per_bin=min_needed,
        force_zero_bin=FORCE_ZERO_BIN,
        random_state=RANDOM_STATE,
    )

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE
    )
    train_idx, hold_idx = next(sss.split(X_all, y_bins))
    X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
    X_hold, y_hold = X_all.iloc[hold_idx], y_all.iloc[hold_idx]
    bins_train = y_bins.iloc[train_idx]

    print(f"Holdout size: {len(hold_idx)} ({HOLDOUT_FRACTION*100:.0f}%)")

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
        "Ridge": Pipeline([("prep", preproc), ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
        "Lasso": Pipeline([("prep", preproc), ("reg", Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=5000))]),
        "SVR (RBF)": Pipeline([("prep", preproc), ("reg", SVR(kernel="rbf", C=10.0, gamma="scale"))]),
        "Random Forest": Pipeline(
            [("prep", preproc), ("reg", RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))]
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline(
            [
                ("prep", preproc),
                (
                    "reg",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    # Evaluate all models on identical folds; pick best by CV RMSE
    rmse_table: Dict[str, float] = {}
    for name, pipe in models.items():
        mean_rmse, _ = evaluate_with_fixed_cv(name, X_train, y_train, pipe, splits)
        rmse_table[name] = mean_rmse
    best_model_name = min(rmse_table, key=rmse_table.get)
    print(f"\n>>> Phase 1 winner: {best_model_name}")

    # ---------- Phase 2: hyperparameter tuning on the winner ----------
    grids: Dict[str, Dict[str, list]] = {
        "Random Forest": {
            "reg__n_estimators": [200, 400, 800],
            "reg__max_depth": [5, 7, 9, None],
            "reg__min_samples_leaf": [1, 2, 4],
        },
        "Ridge": {
            "reg__alpha": [0.1, 1.0, 5.0, 10.0],
        },
        "Lasso": {
            "reg__alpha": [0.001, 0.01, 0.1, 1.0],
        },
        "SVR (RBF)": {
            "reg__C": [1.0, 3.0, 10.0, 30.0],
            "reg__gamma": ["scale", 0.1, 0.01],
        },
        "XGBoost": {
            "reg__n_estimators": [300, 600, 900],
            "reg__max_depth": [3, 4, 5],
            "reg__learning_rate": [0.03, 0.05, 0.1],
            "reg__subsample": [0.8, 1.0],
            "reg__colsample_bytree": [0.8, 1.0],
            "reg__reg_lambda": [0.5, 1.0, 2.0],
        },
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
            refit=True,
        )
        gscv.fit(X_train, y_train)
        print(f"\n>>> Phase 2 best params for {best_model_name}: {gscv.best_params_}")
        best_pipeline = gscv.best_estimator_
    else:
        print("[Info] No grid defined for the winner; using Phase 1 pipeline as-is.")
        best_pipeline = winner
        best_pipeline.fit(X_train, y_train)

    # ---------- Final fit on all training, evaluate once on 20% holdout ----------
    final_model = best_pipeline

    from matplotlib import pyplot as plt
    y_pred_hold = final_model.predict(X_hold)
    rmse_val = rmse(y_hold, y_pred_hold)
    mae_val = mae(y_hold, y_pred_hold)
    r2_val = r2(y_hold, y_pred_hold)

    print("\n=== 20% Stratified Holdout Evaluation ===")
    print("RMSE:", f"{rmse_val:.3f}")
    print("MAE :", f"{mae_val:.3f}")
    print("R²  :", f"{r2_val:.3f}")

    # Quick plot
    y_true = np.asarray(y_hold)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred_hold, s=30, alpha=0.7, edgecolor="none")
    lims = [min(float(y_true.min()), float(y_pred_hold.min())), max(float(y_true.max()), float(y_pred_hold.max()))]
    plt.plot(lims, lims, linewidth=1.5)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True concentration (ppb)")
    plt.ylabel("Predicted concentration (ppb)")
    plt.title(f"Predicted vs True (20% holdout)\nRMSE={rmse_val:.2f}, MAE={mae_val:.2f}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("pred_vs_true_holdout20.png", dpi=150)

    # ---------- Extra diagnostics on the holdout ----------
    # 1) Bin-wise error analysis to see how error varies across ranges


    # Define bins: zeros, low (0<ppb<=10), mid (10<ppb<=25), high (>25)
    bins_edges = [-0.1, 0.0, 10.0, 25.0, float('inf')]
    bins_labels = ["zero", "low(0-10]", "mid(10-25]", "high(>25)"]
    bin_idx = np.digitize(y_true, bins_edges[1:], right=True)

    print("\n=== Holdout error by concentration range ===")
    for i, label in enumerate(bins_labels):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        b_rmse = np.sqrt(((y_true[mask] - y_pred_hold[mask]) ** 2).mean())
        b_mae = np.mean(np.abs(y_true[mask] - y_pred_hold[mask]))
        print(f"  {label:12s}: n={mask.sum():3d}  RMSE={b_rmse:6.2f}  MAE={b_mae:6.2f}")

    # 2) One-by-one comparison (sorted by true ppb for readability)
    order = np.argsort(y_true)
    print("\n=== Holdout per-sample (sorted by true) ===")
    for idx in order:
        yt = float(y_true[idx])
        yp = float(y_pred_hold[idx])
        print(f"True {yt:6.1f} ppb -> Pred {yp:7.2f} ppb")


if __name__ == "__main__":
    main()