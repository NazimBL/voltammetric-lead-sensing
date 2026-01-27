import os
import re
from glob import glob
import pandas as pd
import numpy as np

# =========================
# CONFIG — edit these
# =========================
INPUT_FOLDER = "TW"   # ← set to the folder containing the 5 files
DATA_DIR = "../data"
OUTPUT_CSV   = f"{DATA_DIR}/merged_voltammetry_wide.csv"

# If True, enforce that all files share the exact same potential grid (recommended).
# If grids differ, the script will raise with a clear message.
STRICT_SAME_POTENTIAL_GRID = True

# =========================
# helpers
# =========================
def concentration_from_name(filename: str) -> int:
    """
    Extract integer ppb from filename like 'Ac TW 10ppb.xlsx'.
    """
    m = re.search(r"(\d+)\s*ppb", filename, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse concentration from filename: {filename}")
    return int(m.group(1))

def load_excel_no_header(path: str) -> pd.DataFrame:
    """
    Read one Excel file with NO header.
      - Col 0 = potential (V)
      - Col 1..N = replicate currents (each column is a sample)
    Returns a DataFrame with columns: ['E', 'S1', 'S2', ..., 'Sn']
    """
    df = pd.read_excel(path, header=None)
    n_cols = df.shape[1]
    df.columns = ["E"] + [f"S{i}" for i in range(1, n_cols)]
    # sort by potential just in case
    df = df.sort_values("E").reset_index(drop=True)
    return df

def potentials_to_feature_names(E: np.ndarray) -> list:
    """
    Turn potential values into safe column names like:
      E_-0.8999  (prefix keeps them valid identifiers and ordered)
    Rounded to 4 decimals to keep names consistent while avoiding FP noise.
    """
    return [f"E_{e:.4f}" for e in E]

# =========================
# main
# =========================
def main():
    # 1) discover files
    paths = sorted(glob(os.path.join(INPUT_FOLDER, "*.xlsx")))
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in: {INPUT_FOLDER}")

    rows = []            # will collect per-sample rows here
    ref_E = None         # potential grid to check consistency
    ref_E_names = None

    for path in paths:
        filename = os.path.basename(path)
        conc = concentration_from_name(filename)
        df = load_excel_no_header(path)

        # establish / check potential grid
        E = df["E"].values.astype(float)
        feat_names = potentials_to_feature_names(E)

        if ref_E is None:
            ref_E = E
            ref_E_names = feat_names
        else:
            if STRICT_SAME_POTENTIAL_GRID:
                # check exact length & approximate equality
                if len(E) != len(ref_E) or not np.allclose(E, ref_E, rtol=0, atol=1e-6):
                    raise ValueError(
                        f"Potential grid mismatch detected in '{filename}'.\n"
                        f"Reference grid length: {len(ref_E)}, this file: {len(E)}.\n"
                        f"Set STRICT_SAME_POTENTIAL_GRID=False if you want to allow differences and handle them."
                    )

        # 2) build rows: one per replicate column
        for col in df.columns[1:]:  # skip 'E'
            current = df[col].values.astype(float)

            # assemble one row: sample_id, concentration, then currents keyed by potential feature names
            sample_id = f"{conc}ppb_{col}"  # e.g., '5ppb_S3'
            row = {"sample_id": sample_id, "concentration_ppb": conc, "source_file": filename}

            # map currents to potential-feature columns
            # ensure consistent column order using ref_E_names if available
            feat_vals = dict(zip(feat_names, current))
            if ref_E_names is not None:
                for k in ref_E_names:
                    row[k] = feat_vals.get(k, np.nan)  # if STRICT is false, fill missing with NaN
            else:
                for k, v in feat_vals.items():
                    row[k] = v

            rows.append(row)

    # 3) build final wide dataframe
    wide = pd.DataFrame(rows)

    # 4) sort columns: id/label first, then potentials ascending
    meta_cols = ["sample_id", "concentration_ppb", "source_file"]
    pot_cols = [c for c in wide.columns if c.startswith("E_")]
    pot_cols = sorted(pot_cols, key=lambda s: float(s.split("_")[1]))  # sort by numeric potential
    wide = wide[meta_cols + pot_cols]

    # 5) sanity checks
    #    each sample should have same number of potential columns
    assert wide[pot_cols].shape[1] == len(pot_cols), "Unexpected potential columns shape."
    #    basic counts per class
    class_counts = wide["concentration_ppb"].value_counts().sort_index()
    print("Samples per concentration (ppb):")
    print(class_counts.to_string())

    # 6) save
    wide.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved wide-format dataset to: {OUTPUT_CSV}")
    print("\nPreview:")
    print(wide.head())

if __name__ == "__main__":
    main()
