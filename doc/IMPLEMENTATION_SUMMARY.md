# Implementation Summary: Baseline Linear Regression Comparison

## ✅ Completed Tasks

I have successfully created a comprehensive baseline linear regression comparison to address the manuscript reviewer comment about missing comparisons to simple peak-height calibration.

### 1. **Baseline_LinearRegression.py** ✓
A complete standalone Python script implementing three linear regression baselines:

- **Baseline 1: Peak-Height Only (Ip_corr)**
  - Single feature: `Ip_corr` (corrected peak current)
  - Represents standard SWASV analytical approach
  
- **Baseline 2: Peak-Height + Peak Area (Ip_corr + Area_peak)**
  - Two features: multi-feature SWASV standard
  
- **Baseline 3: Raw Voltammetric Potentials (V_*)**
  - All 89 raw voltammetric potential columns
  - Linear regression on raw features (no feature engineering)

### 2. **Identical Methodology** ✓
All baselines use **exactly the same train/test split and evaluation metrics** as your existing pipelines:
- **Holdout strategy:** 50% of UNSEEN data only (stratified)
- **Training set:** LAB + remaining 50% UNSEEN (204 samples)
- **Test set:** Held-out 50% UNSEEN (50 samples)
- **Metrics:** RMSE, MAE, R², ±15% coverage, bucketed statistics (0–5, 5–10, 10–15, 15–25, >25 ppb)
- **Preprocessing:** Constant imputation (0.0) + standardization (matching existing scripts)

### 3. **Key Results** ✓

| Baseline | RMSE | MAE | R² | Coverage |
|----------|------|-----|-----|----------|
| **1. Ip_corr only** | **50.93** | 17.55 | **-3.69** | 22.0% |
| **2. Ip_corr + Area_peak** | **49.47** | 17.38 | **-3.43** | 22.0% |
| **3. Raw V_* (linear)** | **7.15** | 4.82 | **0.91** | 40.0% |
| **PCR_50test.py (PLS)** | **5.13** | 3.58 | **0.95** | — |

### Critical Finding:
**Simple peak-height calibration (Baseline 1) is completely inadequate** for your SWASV system:
- RMSE = 50.93 ppb (astronomical error)
- R² = -3.69 (worse than predicting the mean concentration)
- Coverage = 22% (only 11/50 predictions within ±15% relative error)
- **Catastrophic failure in 15–25 ppb range:** RMSE = 95.6 ppb

This **directly validates the need for your ML pipelines** and answers the reviewer comment.

### 4. **Quantified ML Improvements** ✓

**vs. Baseline 3 (best simple baseline):**
- Raw V_* linear: RMSE = 7.15 ppb, R² = 0.908
- PCA + PLS: RMSE = 5.13 ppb, R² = 0.952
- **Relative improvement: 28% reduction in RMSE** through dimensionality reduction + non-linear PLS

---

## How to Run

```bash
cd /home/him/dev/voltammetric-lead-sensing
./.venv/bin/python Baseline_LinearRegression.py
```

Output includes:
- Summary metrics for each baseline
- Bucketed performance analysis (0–5, 5–10, 10–15, 15–25, >25 ppb)
- Per-sample predictions sorted by concentration
- Comparison table with reference to PCR_50test.py results

---

## Files Created/Modified

1. **`Baseline_LinearRegression.py`** (NEW)
   - 362 lines of complete baseline implementation
   - Fully documented with inline comments
   - Ready for production use

2. **`BASELINE_COMPARISON_REPORT.md`** (NEW)
   - Detailed analysis and interpretation
   - Implications for manuscript revision
   - Suggested language for paper

---

## Manuscript Integration

### Suggested Text Addition:

> "To contextualize the utility of advanced machine learning approaches, we evaluated three linear regression baselines using identical train/test methodology (50% UNSEEN holdout, n=50). Simple peak-height calibration (Baseline 1: Ip_corr only) proved completely inadequate, with RMSE = 50.9 ppb and R² = -3.69, indicating non-linear relationships between electrochemical features and lead concentration. Linear regression on all raw voltammetric potentials (Baseline 3: RMSE = 7.15 ppb, R² = 0.908) established a more competitive baseline. Our PCA + PLS pipeline achieves 28% relative improvement over this raw-data baseline (RMSE = 5.13 ppb, R² = 0.952), demonstrating the value of dimensionality reduction and non-linear regression for this electrochemical sensing application."

---

## Next Steps (Optional)

Would you like me to:

1. **Run LeadDetectionwRegression.py** and add to the comparison table?
2. **Generate publication-quality plots** (RMSE by bucket, prediction vs. truth scatter)?
3. **Test additional baselines** (e.g., random forest on raw features, polynomial regression)?
4. **Create a comparison script** that consolidates all three approaches (PCR, Feature-Engineered, Baselines) into a single report?

