#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed feature engineering for voltammetry-based Pb prediction.
Keeps the original 15 features and adds a new validity flag for Peak_to_bg.
Key fixes vs your current script:
- FWHM now returns NaN if half-maximum crossings are missing (no ROI-edge fallback).
- Add Peak_to_bg_valid flag; when denominator ~0, Peak_to_bg is set to 0.0 and _valid=0.
- Optional light boxcar smoothing before derivative-based features (configurable).
- Same column order so it remains drop-in compatible with regression.py.

Outputs a CSV with meta columns, 15 features, and validity flags.
"""

import argparse
from typing import Dict

import numpy as np
import pandas as pd

# =========================
# CONFIG — edit as needed
# =========================
DATA_DIR = "../data"
INPUT_CSV  = f"{DATA_DIR}/unseen_baseline.csv"  # or unseen_baseline.csv
OUTPUT_CSV = f"{DATA_DIR}/unseen_features_roi.csv"                  # matches regression.py default

ROI_LOW_V  = -0.3078   # volts (engineer-specified Pb peak window)
ROI_HIGH_V = -0.0629   # volts
MIN_POINTS_IN_ROI = 5
EPS = 1e-12

# Optional smoothing for stability of derivatives/slopes
SMOOTH_BOXCAR = True
SMOOTH_WINDOW = 5  # odd integer >= 3; small to avoid peak distortion
# Which features get validity flags + NaN->0 imputation
PEAK_SHAPE_FEATURES = ["FWHM", "Slope_fall", "Symmetry", "Peak_to_bg"]


# =========================
# Helpers
# =========================
def boxcar_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window < 3 or window % 2 == 0:
        return y
    # simple moving average with edge handling via 'same' convolution
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
    """Full Width at Half Maximum around the main peak.
    Returns NaN if either half-maximum crossing is missing.
    """
    if I.size < 3:
        return np.nan
    pk = int(np.argmax(I))
    Imax = float(I[pk])
    if not np.isfinite(Imax) or Imax <= 0:
        return np.nan
    half = Imax / 2.0

    # Left crossing (strict: must exist)
    left_cross = np.nan
    if pk > 0:
        below = np.where(I[:pk] < half)[0]
        if below.size > 0:
            i2 = below[-1]
            i1 = i2 + 1
            x0, y0 = E[i2], I[i2]
            x1, y1 = E[i1], I[i1]
            left_cross = x0 if y1 == y0 else x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    # Right crossing (strict: must exist)
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
    """Ew, Iw_raw are ROI-sliced arrays (ascending Ew)."""
    n = Ew.size
    if n < MIN_POINTS_IN_ROI:
        return {
            "Ip_corr": np.nan, "Ep": np.nan, "Area_peak": np.nan, "FWHM": np.nan, "Peak_to_bg": np.nan,
            "Slope_rise": np.nan, "Slope_fall": np.nan, "Symmetry": np.nan,
            "dIdE_max": np.nan, "d2IdE2_max": np.nan,
            "Mean": np.nan, "Var": np.nan, "Skew": np.nan, "Kurtosis": np.nan, "Energy": np.nan
        }

    # Optional light smoothing (helps derivatives/slopes; leaves amplitudes mostly intact)
    Iw = boxcar_smooth(Iw_raw, SMOOTH_WINDOW) if SMOOTH_BOXCAR else Iw_raw

    # Peak features
    pk = int(np.argmax(Iw))
    Ip_corr = float(Iw[pk])
    Ep = float(Ew[pk])
    Area_peak = trapezoid_area(Ew, np.clip(Iw, 0, None))
    FWHM = compute_fwhm(Ew, Iw)

    # Background = first 10% (>=5 pts) within ROI
    k_bg = max(5, int(0.10 * n))
    bg_mean = float(np.mean(Iw[:k_bg]))
    Peak_to_bg = Ip_corr / (bg_mean if abs(bg_mean) > EPS else np.nan)

    # Slopes & symmetry (guard edges)
    Slope_rise = linear_slope(Ew[:pk + 1], Iw[:pk + 1]) if pk >= 1 else np.nan
    has_fall = (pk < n - 1)
    Slope_fall = linear_slope(Ew[pk:], Iw[pk:]) if has_fall else np.nan
    Symmetry = (Slope_rise / Slope_fall) if (has_fall and abs(Slope_fall) > EPS) else np.nan

    # Derivatives
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

    # Global stats
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
        "Ip_corr": Ip_corr, "Ep": Ep, "Area_peak": Area_peak, "FWHM": FWHM, "Peak_to_bg": Peak_to_bg,
        "Slope_rise": Slope_rise, "Slope_fall": Slope_fall, "Symmetry": Symmetry,
        "dIdE_max": dIdE_max, "d2IdE2_max": d2IdE2_max,
        "Mean": Mean, "Var": Var, "Skew": Skew, "Kurtosis": Kurtosis, "Energy": Energy
    }


# =========================
# Main
# =========================
def main():
    global SMOOTH_BOXCAR, SMOOTH_WINDOW

    parser = argparse.ArgumentParser(description="Extract voltammetry ROI features.")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV (baseline-corrected wide format)")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV for features")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                        help="Boxcar smoothing window (odd int, >=3). Set <=0 to disable.")
    args = parser.parse_args()

    # Configure smoothing globals based on CLI
    SMOOTH_BOXCAR = (args.smooth is not None and args.smooth >= 3 and (args.smooth % 2 == 1))
    SMOOTH_WINDOW = args.smooth if SMOOTH_BOXCAR else 0

    df = pd.read_csv(args.input)

    # Identify potential columns and parse numeric voltages
    pot_cols = [c for c in df.columns if c.startswith("E_")]
    if not pot_cols:
        raise ValueError("No potential columns found (expected names like 'E_-0.4507').")

    pot_vals = np.array([float(c.split("_", 1)[1]) for c in pot_cols])
    order = np.argsort(pot_vals)
    pot_cols = [pot_cols[i] for i in order]
    pot_vals = pot_vals[order]

    # ROI mask
    roi_mask = (pot_vals >= ROI_LOW_V) & (pot_vals <= ROI_HIGH_V)
    if roi_mask.sum() < MIN_POINTS_IN_ROI:
        raise ValueError(
            f"Not enough points in ROI [{ROI_LOW_V}, {ROI_HIGH_V}] — found {roi_mask.sum()}, "
            f"need >= {MIN_POINTS_IN_ROI}."
        )

    meta_cols = [c for c in ["sample_id", "concentration_ppb", "source_file"] if c in df.columns]

    out_rows = []
    for _, row in df.iterrows():
        I_all = row[pot_cols].to_numpy(dtype=float)
        Ew = pot_vals[roi_mask]
        Iw = I_all[roi_mask]

        feats = extract_features_for_row(Ew, Iw)

        # ----- add validity flags + NaN->0 imputation for peak-shape features -----
        for fname in PEAK_SHAPE_FEATURES:
            val = feats.get(fname, np.nan)
            is_valid = 1 if (pd.notna(val) and np.isfinite(val)) else 0
            feats[f"{fname}_valid"] = is_valid
            if not is_valid:
                feats[fname] = 0.0  # impute 'no measurable/defined peak'

        out = {k: row[k] for k in meta_cols}
        out.update(feats)
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    # Order columns: meta, features, then validity flags
    feat_order = [
        "Ip_corr", "Ep", "Area_peak", "FWHM", "Peak_to_bg",
        "Slope_rise", "Slope_fall", "Symmetry",
        "dIdE_max", "d2IdE2_max",
        "Mean", "Var", "Skew", "Kurtosis", "Energy"
    ]
    valid_cols = [f"{f}_valid" for f in PEAK_SHAPE_FEATURES]

    out_cols = meta_cols + feat_order + valid_cols
    out_cols = [c for c in out_cols if c in out_df.columns]
    out_df = out_df[out_cols]

    out_df.to_csv(args.output, index=False)
    print(f"Saved features to: {args.output}")
    print(out_df.head())


if __name__ == "__main__":
    main()
