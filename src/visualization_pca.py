# visualize_pca.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ====================
# CONFIG
# ====================
DATA_DIR = "../data"
PLOTS_DIR = "../plots"
INPUT_CSV = f"{DATA_DIR}/raw_matrix_all.csv"
TARGET_COL = "concentration_ppb"
DATASET_COL = "dataset"
N_COMPONENTS = 5  # enough to cover variance, we’ll only plot first 2

def main():
    # --- Load data
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c.startswith("V_")]
    X = df[feature_cols].to_numpy()
    y = df[TARGET_COL].to_numpy()
    groups = df[DATASET_COL].to_numpy()

    potentials = np.array([float(c.replace("V_", "")) for c in feature_cols])

    # --- PCA
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_scores = pca.fit_transform(X)  # scores (samples × PCs)
    loadings = pca.components_       # loadings (PC × potentials)
    explained = pca.explained_variance_ratio_

    # 1. Explained variance (scree plot)
    plt.figure(figsize=(6,4))
    plt.bar(range(1, len(explained)+1), explained*100, alpha=0.7, label="Individual")
    plt.plot(range(1, len(explained)+1), np.cumsum(explained)*100,
             marker="o", color="red", label="Cumulative")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/pca_explained_variance.png", dpi=150)

    # 2. Scores scatter (PC1 vs PC2, colored by concentration, shape by dataset)
    # 2. Scores scatter: color = concentration (continuous), marker = dataset


    norm = Normalize(vmin=y.min(), vmax=y.max())
    cmap = plt.cm.viridis
    markers = {"lab": "o", "unseen": "s"}  # shapes per dataset

    plt.figure(figsize=(6, 6))
    mappable = None
    for g, mk in markers.items():
        mask = (groups == g)
        sc = plt.scatter(
            X_scores[mask, 0], X_scores[mask, 1],
            c=y[mask], cmap=cmap, norm=norm,
            marker=mk, s=50, edgecolor="k", linewidths=0.5, alpha=0.9,
            label=g, zorder=2
        )
        mappable = sc  # keep a handle for the colorbar

    # make a single colorbar for concentration
    cbar = plt.colorbar(mappable)
    cbar.set_label("Concentration (ppb)")

    plt.xlabel("PC1 scores")
    plt.ylabel("PC2 scores")
    plt.title("PC1 vs PC2 scores (color = concentration; shape = dataset)")
    plt.legend(title="Dataset")
    plt.grid(True, linestyle="--", alpha=0.3, zorder=1)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/pca_scores_scatter.png", dpi=150)

    # --- PC1 vs concentration with R²
    pc1 = X_scores[:, 0]
    r, pval = pearsonr(pc1, y)  # correlation and p-value
    r2_val = r ** 2  # R² as squared correlation

    plt.figure(figsize=(6, 4))
    plt.scatter(pc1, y, c=y, cmap="viridis",
                edgecolor="k", s=50, alpha=0.8)
    plt.xlabel("PC1 scores")
    plt.ylabel("Concentration (ppb)")
    plt.title(f"PC1 vs Concentration (r = {r:.3f}, R² = {r2_val:.3f})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/pca_pc1_vs_concentration.png", dpi=150)

    # --- 3D scatter: PC1, PC2, PC3
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X_scores[:, 0], X_scores[:, 1], X_scores[:, 2],
                    c=y, cmap="viridis", s=40, alpha=0.8, edgecolor="k")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PC1 vs PC2 vs PC3 (colored by concentration)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Concentration (ppb)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/pca_3d_scatter.png", dpi=150)

    # 3. Loadings for PC1 and PC2
    plt.figure(figsize=(7,4))
    plt.plot(potentials, loadings[0], label="PC1")
    plt.plot(potentials, loadings[1], label="PC2")
    plt.xlabel("Potential")
    plt.ylabel("Loading value")
    plt.title("PCA Loadings (PC1 & PC2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/pca_loadings.png", dpi=150)

    print("Saved plots: pca_explained_variance.png, pca_scores_scatter.png, pca_loadings.png")

if __name__ == "__main__":
    main()
