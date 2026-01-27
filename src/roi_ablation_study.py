#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Ablation Study for Lead Detection Feature Engineering
==========================================

Tests three ROI configurations on the feature engineering + Lasso regression pipeline:
  - ROI_A (Original): -0.3078 to -0.0629 V (current, tightest)
  - ROI_B (Expand Left): -0.4507 to -0.0629 V (wider, captures left shoulder)
  - ROI_C (Broader): -0.3385 to -0.0526 V (moderate expansion both sides)

For each ROI:
  1. Extract features from raw lab + unseen data
  2. Split data (train/holdout) using same stratification as model_feature_engineering_robust.py
  3. Refit Lasso (alpha=0.01) on training features
  4. Evaluate on holdout (outliers removed)
  5. Collect MAE, RMSE, R²

Then generates:
  - Comparison table with metrics
  - Visualization showing 5-6 representative samples (0, 10, 25, >25 ppb) with 3 ROI windows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================
# CONFIG
# ====================
DATA_DIR = "../data"
RAW_LAB_CSV = f"{DATA_DIR}/merged_voltammetry_baseline.csv"
RAW_UNSEEN_CSV = f"{DATA_DIR}/unseen_baseline.csv"
PLOT_DIR = "../plots"

TARGET_COL = "concentration_ppb"
RANDOM_STATE = 42

# Data split (same as model_feature_engineering_robust.py)
N_SPLITS = 5
N_BINS = 7
HOLDOUT_FRACTION = 0.50
FORCE_ZERO_BIN = True
OUTLIER_CONCENTRATION = 16.9

# Lasso hyperparameters (from Phase 2 tuning in model_feature_engineering_robust.py)
LASSO_ALPHA = 0.01
LASSO_MAX_ITER = 5000

# Three ROI configurations to test
ROI_CONFIGS = {
    "ROI_A (Original)": {"low": -0.3078, "high": -0.0629},
    "ROI_B (Expand Left)": {"low": -0.4507, "high": -0.0629},
    "ROI_C (Broader)": {"low": -0.3385, "high": -0.0526},
}

# Min points in ROI
MIN_POINTS_IN_ROI = 5
EPS = 1e-12
SMOOTH_BOXCAR = True
SMOOTH_WINDOW = 5

# ====================
# FEATURE EXTRACTION (same as feature_engineering.py)
# ====================

def boxcar_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window < 3 or window % 2 == 0:
        return y
    k = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, k, mode="same")

def trapezoid_area(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    return float(np.trapezoid(y, x))

def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

def compute_fwhm(E: np.ndarray, I: np.ndarray) -> float:
    if I.size < 3:
        return np.nan
    pk = int(np.argmax(I))
    Imax = float(I[pk])
    if not np.isfinite(Imax) or Imax <= 0:
        return np.nan
    half = Imax / 2.0
    
    left_cross = np.nan
    if pk > 0:
        below = np.where(I[:pk] < half)[0]
        if below.size > 0:
            i2 = below[-1]
            i1 = i2 + 1
            x0, y0 = E[i2], I[i2]
            x1, y1 = E[i1], I[i1]
            left_cross = x0 if y1 == y0 else x0 + (half - y0) * (x1 - x0) / (y1 - y0)
    
    right_cross = np.nan
    if pk < I.size - 1:
        below = np.where(I[pk:] < half)[0]
        if below.size > 0:
            j2 = pk + below[0]
            j1 = j2 - 1
            x0, y0 = E[j1], I[j1]
            x1, y1 = E[j2], I[j2]
            right_cross = x1 if y1 == y0 else x0 + (half - y0) * (x1 - x0) / (y1 - y0)
    
    if not np.isfinite(left_cross) or not np.isfinite(right_cross):
        return np.nan
    return float(abs(right_cross - left_cross))

def extract_features_for_row(Ew: np.ndarray, Iw_raw: np.ndarray) -> Dict[str, float]:
    n = Ew.size
    if n < MIN_POINTS_IN_ROI:
        return {
            "Ip_corr": np.nan, "Ep": np.nan, "Area_peak": np.nan, "FWHM": np.nan,
            "Peak_to_bg": np.nan, "Slope_rise": np.nan, "Slope_fall": np.nan,
            "Symmetry": np.nan, "dIdE_max": np.nan, "d2IdE2_max": np.nan,
            "Mean": np.nan, "Var": np.nan, "Skew": np.nan, "Kurtosis": np.nan, "Energy": np.nan
        }
    
    Iw = boxcar_smooth(Iw_raw, SMOOTH_WINDOW) if SMOOTH_BOXCAR else Iw_raw
    
    pk = int(np.argmax(Iw))
    Ip_corr = float(Iw[pk])
    Ep = float(Ew[pk])
    Area_peak = trapezoid_area(Ew, np.clip(Iw, 0, None))
    FWHM = compute_fwhm(Ew, Iw)
    
    k_bg = max(5, int(0.10 * n))
    bg_mean = float(np.mean(Iw[:k_bg]))
    Peak_to_bg = Ip_corr / (bg_mean if abs(bg_mean) > EPS else np.nan)
    
    Slope_rise = linear_slope(Ew[:pk + 1], Iw[:pk + 1]) if pk >= 1 else np.nan
    has_fall = (pk < n - 1)
    Slope_fall = linear_slope(Ew[pk:], Iw[pk:]) if has_fall else np.nan
    Symmetry = (Slope_rise / Slope_fall) if (has_fall and abs(Slope_fall) > EPS) else np.nan
    
    if n > 1:
        dIdE = np.gradient(Iw, Ew)
        dIdE_max = float(np.nanmax(dIdE))
    else:
        dIdE_max = np.nan
    
    if n > 2:
        d2IdE2 = np.gradient(dIdE, Ew)
        d2IdE2_max = float(np.nanmax(d2IdE2))
    else:
        d2IdE2_max = np.nan
    
    Mean = float(np.mean(Iw))
    Var = float(np.var(Iw))
    centered = Iw - Mean
    m2 = np.mean(centered ** 2) if n > 0 else np.nan
    m3 = np.mean(centered ** 3) if n > 0 else np.nan
    m4 = np.mean(centered ** 4) if n > 0 else np.nan
    Skew = float(m3 / ((m2 ** 1.5) + EPS)) if m2 > 0 else np.nan
    Kurtosis = float(m4 / ((m2 ** 2) + EPS)) if m2 > 0 else np.nan
    Energy = float(np.sum(Iw ** 2))
    
    return {
        "Ip_corr": Ip_corr, "Ep": Ep, "Area_peak": Area_peak, "FWHM": FWHM,
        "Peak_to_bg": Peak_to_bg, "Slope_rise": Slope_rise, "Slope_fall": Slope_fall,
        "Symmetry": Symmetry, "dIdE_max": dIdE_max, "d2IdE2_max": d2IdE2_max,
        "Mean": Mean, "Var": Var, "Skew": Skew, "Kurtosis": Kurtosis, "Energy": Energy
    }

def extract_features_with_roi(df_raw: pd.DataFrame, roi_low: float, roi_high: float) -> pd.DataFrame:
    """
    Extract features from raw voltammetry data using specified ROI bounds.
    Returns DataFrame with meta + 15 features + 4 validity flags.
    """
    pot_cols = [c for c in df_raw.columns if c.startswith("E_")]
    pot_vals = np.array([float(c.split("_", 1)[1]) for c in pot_cols])
    order = np.argsort(pot_vals)
    pot_cols = [pot_cols[i] for i in order]
    pot_vals = pot_vals[order]
    
    roi_mask = (pot_vals >= roi_low) & (pot_vals <= roi_high)
    if roi_mask.sum() < MIN_POINTS_IN_ROI:
        raise ValueError(f"ROI [{roi_low}, {roi_high}] has {roi_mask.sum()} points, need >= {MIN_POINTS_IN_ROI}")
    
    meta_cols = [c for c in ["sample_id", "concentration_ppb", "source_file"] if c in df_raw.columns]
    peak_shape_features = ["FWHM", "Slope_fall", "Symmetry", "Peak_to_bg"]
    
    out_rows = []
    for _, row in df_raw.iterrows():
        I_all = row[pot_cols].to_numpy(dtype=float)
        Ew = pot_vals[roi_mask]
        Iw = I_all[roi_mask]
        
        feats = extract_features_for_row(Ew, Iw)
        
        # Add validity flags + NaN->0 imputation
        for fname in peak_shape_features:
            val = feats.get(fname, np.nan)
            is_valid = 1 if (pd.notna(val) and np.isfinite(val)) else 0
            feats[f"{fname}_valid"] = is_valid
            if not is_valid:
                feats[fname] = 0.0
        
        out = {k: row[k] for k in meta_cols}
        out.update(feats)
        out_rows.append(out)
    
    return pd.DataFrame(out_rows)

# ====================
# HELPERS (from model_feature_engineering_robust.py)
# ====================

def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = {"sample_id", "source_file", target_col, "dataset"}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]

def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler())
        ]), feature_cols)
    ], remainder="drop")

def make_strat_bins(y: pd.Series, n_bins: int, min_per_bin: int, force_zero_bin: bool = True) -> pd.Series:
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
    
    return pd.Series(0, index=y.index, dtype=int)

def remove_outliers(y_true, y_pred, outlier_conc=OUTLIER_CONCENTRATION, tolerance=0.01):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    outlier_mask = np.abs(y_true - outlier_conc) <= (outlier_conc * tolerance)
    keep_mask = ~outlier_mask
    return y_true[keep_mask], y_pred[keep_mask], keep_mask

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# ====================
# MAIN ABLATION STUDY
# ====================

def run_roi_ablation():
    print("="*80)
    print("ROI ABLATION STUDY: Feature Engineering + Lasso Regression")
    print("="*80)
    
    # Load raw data
    print("\n[1/4] Loading raw voltammetry data...")
    df_lab = pd.read_csv(RAW_LAB_CSV)
    df_unseen = pd.read_csv(RAW_UNSEEN_CSV)
    print(f"  Lab samples: {len(df_lab)}")
    print(f"  Unseen samples: {len(df_unseen)}")
    
    results = {}
    raw_data_for_viz = {}  # Store raw data for visualization
    
    # Test each ROI
    for roi_name, roi_bounds in ROI_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Testing: {roi_name} (Low: {roi_bounds['low']}, High: {roi_bounds['high']})")
        print(f"{'='*80}")
        
        roi_low = roi_bounds['low']
        roi_high = roi_bounds['high']
        
        # Extract features with this ROI
        print(f"  Extracting features with ROI [{roi_low}, {roi_high}]...")
        df_lab_feat = extract_features_with_roi(df_lab, roi_low, roi_high)
        df_unseen_feat = extract_features_with_roi(df_unseen, roi_low, roi_high)
        
        # Align features
        lab_feats = select_feature_columns(df_lab_feat, TARGET_COL)
        unseen_feats = select_feature_columns(df_unseen_feat, TARGET_COL)
        feature_cols = sorted(list(set(lab_feats).intersection(unseen_feats)))
        
        df_lab_feat = df_lab_feat[feature_cols + [TARGET_COL]].assign(dataset="lab")
        df_unseen_feat = df_unseen_feat[feature_cols + [TARGET_COL]].assign(dataset="unseen")
        df_all = pd.concat([df_lab_feat, df_unseen_feat], ignore_index=True)
        
        X_all = df_all[feature_cols].copy()
        y_all = df_all[TARGET_COL].astype(float)
        groups = df_all["dataset"].astype(str).values
        
        # Stratified holdout
        idx_unseen = np.where(groups == "unseen")[0]
        idx_lab = np.where(groups == "lab")[0]
        
        y_bins_unseen = make_strat_bins(
            y_all.iloc[idx_unseen], n_bins=N_BINS, min_per_bin=2, force_zero_bin=FORCE_ZERO_BIN
        )
        sss = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE)
        unseen_train_rel, unseen_hold_rel = next(sss.split(X_all.iloc[idx_unseen], y_bins_unseen))
        
        hold_idx = idx_unseen[unseen_hold_rel]
        unseen_train_idx = idx_unseen[unseen_train_rel]
        train_idx = np.concatenate([idx_lab, unseen_train_idx])
        
        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_hold, y_hold = X_all.iloc[hold_idx], y_all.iloc[hold_idx]
        
        print(f"  Training samples: {len(X_train)}, Holdout samples: {len(X_hold)}")
        
        # Build and fit Lasso pipeline
        preproc = make_preprocessor(feature_cols)
        lasso_pipe = Pipeline([
            ("prep", preproc),
            ("reg", Lasso(alpha=LASSO_ALPHA, random_state=RANDOM_STATE, max_iter=LASSO_MAX_ITER))
        ])
        
        lasso_pipe.fit(X_train, y_train)
        
        # Evaluate on holdout (with outlier removal)
        y_pred_hold = lasso_pipe.predict(X_hold)
        y_hold_clean, y_pred_clean, _ = remove_outliers(y_hold, y_pred_hold, OUTLIER_CONCENTRATION)
        
        mae_val = mae(y_hold_clean, y_pred_clean)
        rmse_val = rmse(y_hold_clean, y_pred_clean)
        r2_val = r2(y_hold_clean, y_pred_clean)
        
        print(f"  MAE:  {mae_val:.4f} ppb")
        print(f"  RMSE: {rmse_val:.4f} ppb")
        print(f"  R²:   {r2_val:.4f}")
        
        results[roi_name] = {
            "MAE": mae_val,
            "RMSE": rmse_val,
            "R2": r2_val,
            "n_samples": len(y_hold_clean),
            "pipeline": lasso_pipe,
            "feature_cols": feature_cols,
            "roi_low": roi_low,
            "roi_high": roi_high
        }
        
        # Store raw data for visualization (only once)
        if "A" in roi_name:
            raw_data_for_viz = {"df_lab": df_lab, "df_unseen": df_unseen}
    
    # Print results table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'ROI Config':<30} {'MAE (ppb)':<15} {'RMSE (ppb)':<15} {'R²':<12} {'N':<5}")
    print("-" * 80)
    for roi_name, res in results.items():
        print(f"{roi_name:<30} {res['MAE']:<15.4f} {res['RMSE']:<15.4f} {res['R2']:<12.4f} {res['n_samples']:<5}")
    
    # Generate visualization
    print(f"\n[2/4] Generating visualization...")
    plot_roi_comparison(raw_data_for_viz, results, ROI_CONFIGS)
    
    print("\nROI Ablation Study Complete!")
    return results

def plot_roi_comparison(raw_data, results, roi_configs):
    """
    Plot 5-6 representative samples with 3 ROI windows overlaid on a single plot.
    Samples: 0 ppb, 10 ppb, 25 ppb, and 2-3 samples >25 ppb
    All curves overlaid on one figure, voltage restricted to -0.5 V onward.
    """
    df_lab = raw_data["df_lab"]
    
    # Select representative samples
    target_concs = [0, 10, 25, 35, 50]  # Target concentrations
    selected_samples = []
    sample_concs = []
    
    for target in target_concs:
        if target == 0:
            candidates = df_lab[df_lab[TARGET_COL] == 0].index.tolist()
        else:
            # Find closest to target
            candidates = df_lab[(df_lab[TARGET_COL] >= target - 3) & 
                               (df_lab[TARGET_COL] <= target + 3)].index.tolist()
        
        if candidates:
            idx = int(candidates[0])
            selected_samples.append(idx)
            sample_concs.append(df_lab.iloc[idx][TARGET_COL])
    
    selected_samples = selected_samples[:6]  # Take up to 6
    sample_concs = sample_concs[:6]
    
    # Parse voltage columns
    pot_cols = [c for c in df_lab.columns if c.startswith("E_")]
    pot_vals = np.array([float(c.split("_", 1)[1]) for c in pot_cols])
    order = np.argsort(pot_vals)
    pot_cols = [pot_cols[i] for i in order]
    pot_vals = pot_vals[order]
    
    # Create single figure with all voltammograms overlaid
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for samples (gradient from light to dark)
    sample_colors = plt.cm.viridis(np.linspace(0, 1, len(selected_samples)))
    
    # Define colors for ROIs
    roi_colors = {
        "ROI_A (Original)": "#FF6B6B",
        "ROI_B (Expand Left)": "#4ECDC4",
        "ROI_C (Broader)": "#45B7D1"
    }
    
    # Plot each sample as overlaid curves
    for idx_plot, (sample_idx, conc) in enumerate(zip(selected_samples, sample_concs)):
        sample = df_lab.iloc[sample_idx]
        
        # Get current values
        I_vals = sample[pot_cols].to_numpy(dtype=float)
        
        # Plot voltammogram
        ax.plot(pot_vals, I_vals, linewidth=2.5, label=f'{conc:.0f} ppb', 
               color=sample_colors[idx_plot], zorder=2)
    
    # Overlay ROI windows (only once, behind all curves)
    for roi_name, roi_bounds in roi_configs.items():
        roi_low = roi_bounds['low']
        roi_high = roi_bounds['high']
        ax.axvspan(roi_low, roi_high, alpha=0.15, color=roi_colors[roi_name], 
                  label=f"{roi_name}", zorder=1)
    
    ax.set_xlabel('Potential (V)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Current (µA)', fontsize=13, fontweight='bold')
    ax.set_title('ROI Ablation Study: Voltammograms with ROI Windows', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Restrict voltage view to start from -0.5 V
    ax.set_xlim([-0.5, pot_vals.max() + 0.01])
    
    # Create custom legend with two columns (samples + ROIs)
    handles, labels = ax.get_legend_handles_labels()
    # Separate sample curves from ROI patches
    n_samples = len(selected_samples)
    sample_handles = handles[:n_samples]
    sample_labels = labels[:n_samples]
    roi_handles = handles[n_samples:]
    roi_labels = labels[n_samples:]
    
    # Create legend with two sections
    legend1 = ax.legend(sample_handles, sample_labels, loc='upper left', 
                       fontsize=11, title='Concentration', title_fontsize=12,
                       framealpha=0.95)
    ax.add_artist(legend1)
    ax.legend(roi_handles, roi_labels, loc='upper right', 
             fontsize=11, title='ROI Windows', title_fontsize=12,
             framealpha=0.95)
    
    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/roi_ablation_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR}/roi_ablation_comparison.png")
    plt.close()

if __name__ == "__main__":
    results = run_roi_ablation()
