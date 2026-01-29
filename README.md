# Voltammetric Lead Sensing

A machine learning system for **rapid, real-time prediction of lead (Pb) levels in water** using voltammetric data. This project combines two complementary modeling pipelines—feature engineering and principal component regression (PCR)/partial least squares (PLS)—to deliver reliable environmental monitoring for water safety.

## Overview

Lead contamination in drinking water is a critical environmental health concern. This project leverages electrochemical voltammetry to capture the electrochemical signature of lead ions, enabling instant predictions of lead concentration without lengthy chemical analysis.

### Key Features
- **Two modeling pipelines** for robust lead level prediction
- **Robust outlier handling** to improve model reliability
- **Comprehensive feature engineering** from raw voltammetric signals
- **Multiple regression approaches** (Linear, PCR, PLS)
- **Automated data processing** from raw voltammetry to model-ready features

---

## Project Architecture

```
voltammetric-lead-sensing/
├── data/                          # Raw and processed voltammetric data
│   ├── *_baseline.csv            # Baseline-corrected voltammetry signals
│   ├── *_roi.csv                 # Region-of-interest (ROI) extracted features
│   └── *.csv                      # Various processing stages
├── src/                           # Core processing scripts
│   ├── dataprep.py               # Data loading and preparation
│   ├── baseline_reduction.py      # Baseline correction for PCA/PCR/PLS pipeline
│   ├── feature_engineering.py     # ROI feature extraction for Feature Engineering pipeline
│   ├── pca_preparation.py         # PCA setup for PCR approach
│   ├── visualization_pca.py       # PCA visualization tools
│   ├── roi_ablation_study.py      # ROI parameter sensitivity analysis
│   ├── analysis_uncertainty_decision.py  # Uncertainty quantification
│   └── model/                     # Trained models and prediction engines
│       ├── baseline_linear_regression.py    # Simple baseline model
│       ├── model_feature_engineering.py     # Feature Engineering pipeline
│       ├── model_feature_engineering_robust.py  # Outlier-resistant FE pipeline
│       ├── model_pcr_pls.py                # PCR & PLS models
│       └── model_pcr_pls_robust.py         # Outlier-resistant PCR/PLS
├── plots/                         # Generated visualizations
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

---

## Two Modeling Pipelines

### Pipeline 1: Feature Engineering

This pipeline extracts interpretable, domain-specific features from the voltammetric signal within the lead peak region of interest (ROI).

**Data Flow:**
```
Raw Voltammetry Data
        ↓
Baseline Correction (raw_matrix_all.csv → *_baseline.csv)
        ↓
feature_engineering.py
  • Extract ROI (typically -0.3078V to -0.0629V)
  • Compute 15 key features:
    - Peak features: Ip_corr (peak current), Ep (peak potential), FWHM (width)
    - Morphology: Area_peak, Peak_to_bg (peak-to-background), Symmetry
    - Slopes: Slope_rise, Slope_fall
    - Derivatives: dIdE_max (max 1st derivative), d2IdE2_max (max 2nd derivative)
    - Statistics: Mean, Variance, Skewness, Kurtosis, Energy
        ↓
Feature CSV (*_roi.csv)
        ↓
model_feature_engineering.py (or model_feature_engineering_robust.py)
        ↓
Lead Concentration Prediction
```

**Key Scripts:**
- [feature_engineering.py](src/feature_engineering.py) — Extracts 15 features from ROI
- [model_feature_engineering.py](src/model/model_feature_engineering.py) — Trains/predicts with features
- [model_feature_engineering_robust.py](src/model/model_feature_engineering_robust.py) — Handles outliers automatically

**Advantages:**
- Interpretable features tied to electrochemistry
- Fast inference once features are computed
- Fewer parameters to tune

---

### Pipeline 2: PCA/PCR/PLS

This pipeline operates directly on the full voltammetric spectrum, using dimensionality reduction to extract latent patterns.

**Data Flow:**
```
Raw Voltammetry Data
        ↓
Baseline Correction (raw_matrix_all.csv → *_baseline.csv)
        ↓
baseline_reduction.py
  • Applies baseline correction to raw signal
        ↓
pca_preparation.py
  • Standardizes features
  • Computes PCA decomposition
        ↓
Full Spectrum Data + PCA Components
        ↓
model_pcr_pls.py (or model_pcr_pls_robust.py)
  • PCR: Principal Component Regression
  • PLS: Partial Least Squares Regression
        ↓
Lead Concentration Prediction
```

**Key Scripts:**
- [baseline_reduction.py](src/baseline_reduction.py) — Baseline correction
- [pca_preparation.py](src/pca_preparation.py) — PCA setup and transformation
- [model_pcr_pls.py](src/model/model_pcr_pls.py) — PCR/PLS models
- [model_pcr_pls_robust.py](src/model/model_pcr_pls_robust.py) — Robust variants with outlier handling

**Advantages:**
- Captures multi-scale patterns in the full spectrum
- Automatic dimensionality reduction
- Complements feature engineering approach

---

## Robust Model Variants

Both pipelines include **robust versions** that automatically detect and handle outliers:

- **model_feature_engineering_robust.py** — Outlier-resistant feature-based prediction
- **model_pcr_pls_robust.py** — Outlier-resistant spectrum-based prediction

These models use robust regression techniques (e.g., iteratively reweighted least squares, isolation forests, or Z-score filtering) to reduce the impact of anomalous samples while preserving predictive power.

---

## Data Format

### Input: Raw Voltammetry (Wide Format)
```
sample_id, concentration_ppb, source_file, E_-0.4507, E_-0.4402, E_-0.4297, ...
sample_001, 10.5, experiment_1.txt, -0.025, -0.018, -0.012, ...
sample_002, 50.0, experiment_1.txt, -0.032, -0.028, -0.021, ...
```
- Columns prefixed with `E_` contain current measurements at specific potentials
- Potential values parsed from column names

### Output: Extracted Features
```
sample_id, concentration_ppb, source_file, Ip_corr, Ep, Area_peak, FWHM, Peak_to_bg, ..., FWHM_valid, Slope_fall_valid, ...
```
- Includes all 15 engineered features
- Validity flags indicate feature reliability

---

## Usage

### Installation

```bash
# Clone repository
git clone <repo-url>
cd voltammetric-lead-sensing

# Install dependencies
pip install -e .
```

### Feature Engineering Pipeline

```bash
# Extract features from baseline-corrected data
python src/feature_engineering.py \
    --input data/unseen_baseline.csv \
    --output data/unseen_features_roi.csv \
    --smooth 5

# Train feature-based model
python src/model/model_feature_engineering.py \
    --train data/volt_features_roi.csv \
    --test data/unseen_features_roi.csv

# Or use robust variant (automatic outlier handling)
python src/model/model_feature_engineering_robust.py \
    --train data/volt_features_roi.csv \
    --test data/unseen_features_roi.csv
```

### PCA/PCR/PLS Pipeline

```bash
# Prepare baseline-corrected data
python src/baseline_reduction.py \
    --input data/raw_matrix_all.csv \
    --output data/merged_voltammetry_baseline.csv

# Prepare PCA components
python src/pca_preparation.py \
    --input data/merged_voltammetry_baseline.csv \
    --output data/pca_model.pkl

# Train PCR/PLS models
python src/model/model_pcr_pls.py \
    --train data/merged_voltammetry_baseline.csv \
    --test data/unseen_baseline.csv

# Or use robust variant
python src/model/model_pcr_pls_robust.py \
    --train data/merged_voltammetry_baseline.csv \
    --test data/unseen_baseline.csv
```

### Analysis & Visualization

```bash
# ROI sensitivity analysis
python src/roi_ablation_study.py --roi-low -0.35 --roi-high -0.05

# Visualize PCA decomposition
python src/visualization_pca.py --input data/pca_model.pkl

# Uncertainty quantification
python src/analysis_uncertainty_decision.py --model <model_path>
```

---

## Configuration

Key parameters (edit in script headers or pass as arguments):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROI_LOW_V` | -0.3078 | Lower bound of Pb peak region (volts) |
| `ROI_HIGH_V` | -0.0629 | Upper bound of Pb peak region (volts) |
| `SMOOTH_WINDOW` | 5 | Boxcar smoothing window for derivative stability |
| `MIN_POINTS_IN_ROI` | 5 | Minimum data points required in ROI |

---

## Key Scientific Concepts

### Region of Interest (ROI)
The voltammetric peak for lead oxidation/reduction occurs in a characteristic potential window (-0.31 to -0.06 V). Features are extracted only from this window to reduce noise and focus on signal quality.

### Feature Definitions

| Feature | Meaning |
|---------|---------|
| **Ip_corr** | Peak current (amplitude of Pb oxidation/reduction wave) |
| **Ep** | Peak potential (position of Pb signal) |
| **FWHM** | Full Width at Half Maximum (peak sharpness) |
| **Area_peak** | Integrated area under the peak (total charge) |
| **Peak_to_bg** | Ratio of peak height to background current |
| **Slope_rise/fall** | Rate of current change approaching/leaving peak |
| **Symmetry** | Rise slope / Fall slope ratio |
| **dIdE_max** | Maximum 1st derivative (steepness) |
| **d2IdE2_max** | Maximum 2nd derivative (curvature) |
| **Mean, Var, Skew, Kurt** | Statistical moments of signal distribution |
| **Energy** | Sum of squared currents |

---

## Performance Notes

- **Feature Engineering**: Fast, interpretable, stable with small datasets
- **PCA/PCR/PLS**: Captures complex patterns, better with larger datasets
- **Robust Variants**: ~5-10% accuracy loss but much better stability with noisy/real-world data
- **Typical RMSE**: 2–5 ppb (depending on concentration range and data quality)

---

## File Organization

| Directory | Purpose |
|-----------|---------|
| `data/` | Raw CSV data, intermediate processing stages, feature matrices |
| `src/` | Data preparation, feature extraction, analysis scripts |
| `src/model/` | Regression models and prediction engines |
| `plots/` | Generated visualizations and diagnostic plots |

---

## References & Further Reading

- **Voltammetry**: Electrochemical technique for measuring ion concentration
- **PCR/PLS**: Dimensionality reduction regression methods for spectroscopic data
- **Robust Regression**: Techniques for handling outliers in real-world measurements



