# save as remove_pca_outliers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
eigenvec_path = "PCA/out.eigenvec"
eigenval_path = "PCA/out.eigenval"
metadata_path = "Metadata.MissRem.tsv"
keep_output = "keep_samples.txt"
plot_output = "pca_outlier_removal_comparison.png"
variance_cutoff = 0.85  # % cumulative variance to use
outlier_quantile = 0.95  # top 1% distance outliers per site

# --- Load PCA ---
header = ["FID", "IID"] + [f"PC{i}" for i in range(1, 21)]
pca = pd.read_csv(eigenvec_path, delim_whitespace=True, header=None)
pca.columns = header
pca = pca.drop(columns=["IID"]).rename(columns={"FID": "Sample_ID"})

# --- Load Eigenvalues and compute variance explained ---
eigenvals = np.loadtxt(eigenval_path)
pct_var = eigenvals / eigenvals.sum()
cum_var = np.cumsum(pct_var)
N = np.argmax(cum_var >= variance_cutoff) + 1
print(f"Using top {N} PCs to capture {cum_var[N-1]*100:.2f}% variance")

# --- Load metadata and merge ---
meta = pd.read_csv(metadata_path, sep="\t")
df = pd.merge(meta, pca, on="Sample_ID")

# --- Calculate Euclidean distance using top N PCs ---
pc_cols = [f"PC{i}" for i in range(1, N + 1)]
df["PC_dist"] = np.linalg.norm(df[pc_cols].values, axis=1)

# --- Flag top 1% outliers per site ---
def flag_outliers(site_df):
    cutoff = site_df["PC_dist"].quantile(outlier_quantile)
    site_df["is_outlier"] = site_df["PC_dist"] > cutoff
    return site_df

df = df.groupby("site", group_keys=False).apply(flag_outliers)

# --- Plot before and after ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
sns.scatterplot(data=df, x="PC1", y="PC2", hue="site", ax=axes[0], s=10, alpha=0.8)
axes[0].set_title("Before Outlier Removal")
sns.scatterplot(data=df[~df["is_outlier"]], x="PC1", y="PC2", hue="site", ax=axes[1], s=10, alpha=0.8)
axes[1].set_title("After Outlier Removal")
plt.suptitle("PCA Before and After site-wise Outlier Removal", fontsize=14)
plt.tight_layout()
plt.savefig(plot_output, dpi=300)
plt.close()

# --- Save filtered metadata ---
keep_df = df[~df["is_outlier"]]
keep_df.to_csv("MetadataOutlierRemoved.tsv", sep="\t", index=True, header=True)


















