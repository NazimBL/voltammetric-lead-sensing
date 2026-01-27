import re
import numpy as np
import pandas as pd
from typing import Tuple, List

# ====================
# CONFIG
# ====================
DATA_DIR = "../data"
LAB_BASELINE_CSV = f"{DATA_DIR}/lab_baseline.csv"
UNSEEN_BASELINE_CSV = f"{DATA_DIR}/unseen_baseline.csv"
TARGET_COL = "concentration_ppb"
OUT_CSV = f"{DATA_DIR}/raw_matrix_all.csv"
ROUND_DECIMALS = 4   # round numeric part of potential names to this many decimals

def _normalize_potential_columns(df: pd.DataFrame, target_col: str, round_decimals: int) -> pd.DataFrame:
    """
    Rename potential columns to a canonical format V_<rounded> based on the numeric part of each header.
    Keeps TARGET_COL intact. Returns a new DataFrame with columns: [TARGET_COL] + sorted V_*.
    """
    mapping = {}
    for c in df.columns:
        if c == target_col:
            continue
        m = re.search(r'(-?\d+(?:\.\d+)?)', str(c))
        if not m:
            raise ValueError(f"Cannot parse a numeric potential from column name '{c}'.")
        val = round(float(m.group(1)), round_decimals)
        mapping[c] = f"V_{val:.{round_decimals}f}"
    out = df.rename(columns=mapping).copy()
    pot_cols = sorted([c for c in out.columns if c != target_col])
    out = out[[target_col] + pot_cols]
    return out

def main() -> None:
    # Load both baseline tables (rows=samples, columns=concentration + potentials)
    df_lab = pd.read_csv(LAB_BASELINE_CSV)
    df_un  = pd.read_csv(UNSEEN_BASELINE_CSV)

    # Sanity: target column must exist
    if TARGET_COL not in df_lab.columns or TARGET_COL not in df_un.columns:
        raise ValueError(f"Expected a '{TARGET_COL}' column in both CSVs.")

    # Normalize potential column names
    lab_n = _normalize_potential_columns(df_lab, TARGET_COL, ROUND_DECIMALS)
    un_n  = _normalize_potential_columns(df_un, TARGET_COL, ROUND_DECIMALS)

    # Align to the shared set of potential columns
    lab_pots = set(lab_n.columns) - {TARGET_COL}
    un_pots  = set(un_n.columns) - {TARGET_COL}
    both = sorted(lab_pots & un_pots)

    if len(both) == 0:
        raise ValueError("No overlapping potential columns after normalization. Check headers and ROUND_DECIMALS.")

    # Keep only shared potentials, in the same sorted order
    lab_keep = [TARGET_COL] + both
    un_keep  = [TARGET_COL] + both
    lab_final = lab_n[lab_keep].copy()
    un_final  = un_n[un_keep].copy()

    # Add dataset labels
    lab_final["dataset"] = "lab"
    un_final["dataset"]  = "unseen"

    # Combine
    df_all = pd.concat([lab_final, un_final], ignore_index=True)

    # Report
    print(f"LAB samples:   {len(lab_final):4d} | UNSEEN samples: {len(un_final):4d}")
    print(f"Shared potential points: {len(both)} (rounded to {ROUND_DECIMALS} decimals)")
    print(f"Saving {OUT_CSV} with shape {df_all.shape} (rows=samples, cols=target + potentials + dataset)")

    # Save
    df_all.to_csv(OUT_CSV, index=False)

if __name__ == "__main__":
    main()
