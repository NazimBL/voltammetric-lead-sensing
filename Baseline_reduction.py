import pandas as pd
import numpy as np

# =====================
# CONFIG
# =====================
INPUT_CSV = "merged_voltammetry_wide.csv"        # your wide-format file

#INPUT_CSV = "unseen_samples.csv"
OUTPUT_CSV = "merged_voltammetry_baseline.csv"
BASELINE_VOLTAGE = -0.4507                       # engineer’s reference

# =====================
# MAIN
# =====================
def main():
    df = pd.read_csv(INPUT_CSV)

    # Identify potential columns (those starting with "E_")
    pot_cols = [c for c in df.columns if c.startswith("E_")]
    if not pot_cols:
        raise ValueError("No potential columns found (expected columns named like 'E_-0.4507').")

    # Parse numeric voltages from column names
    pot_values = np.array([float(c.split("_", 1)[1]) for c in pot_cols])

    # Pick the column closest to the target baseline voltage
    idx_baseline = int(np.argmin(np.abs(pot_values - BASELINE_VOLTAGE)))
    baseline_col = pot_cols[idx_baseline]
    baseline_actual = pot_values[idx_baseline]
    print(f"Using baseline column: {baseline_col} (actual V ≈ {baseline_actual:.4f})")

    # Subtract baseline value from ALL potential columns (entire curve) per row
    df[pot_cols] = df[pot_cols].sub(df[baseline_col], axis=0)

    # Sanity check: baseline column should now be ~0 for every row
    print("\nPost-correction baseline stats (should be ~0):")
    print(df[baseline_col].describe())

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved baseline-corrected dataset to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
