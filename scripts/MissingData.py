import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import sys


# Load the data
md = pd.read_csv('Metadata.M.tsv', sep='\t')


# Extract the "site" column for row labels
row_labels = md["site"]

# Create a color palette for the site categories
unique_sites = row_labels.unique()
site_palette = sns.color_palette("tab10", len(unique_sites))  # Adjust palette as needed
site_colors = {site: site_palette[i] for i, site in enumerate(unique_sites)}

# Map site categories to colors
row_colors = row_labels.map(site_colors)

# Select relevant columns for the heatmap
df = md[['PW_AGE', 'PW_EDUCATION', 'MAT_HEIGHT', 'MAT_WEIGHT', 'BMI', 'ALCOHOL_FREQ', 'SMOK_YR', 'SNIFF_FREQ',
         'TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
         'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'CHRON_HTN',
         'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
         'ALCOHOL', 'SMOKE_HIST', 'SMOK_FREQ', 'SMOK_TYP', 'SNIFF_TOBA']]


# Replace -88 with -77 (if necessary)
df.replace(-88, -77, inplace=True)
df.replace(-99, -77, inplace=True)

# Create a mask for missing data (-77) and transform it into a binary form for visualization
missing_data = df.replace(-77, np.nan).isnull().astype(int)


# ============================================================
# IDENTIFY FEATURES WITH >90% MISSINGNESS IN ANY SITE
# ============================================================

# Add site labels back temporarily for site-specific missingness calculation
missing_by_site_input = missing_data.copy()
missing_by_site_input["site"] = row_labels.values

# Calculate missing fraction for every feature within each site
site_missing_fraction = (
    missing_by_site_input
    .groupby("site")
    .mean()
)

print("\n========================================")
print("MISSINGNESS BY SITE (%)")
print("========================================")
print((site_missing_fraction * 100).round(2))


# Features that fail:
# >90% missing in at least one site
failed_features = [
    feature
    for feature in df.columns
    if (site_missing_fraction[feature] > 0.90).any()
]

# Features that survive:
# <=90% missing in every site
surviving_features = [
    feature
    for feature in df.columns
    if feature not in failed_features
]


print("\n========================================")
print("FEATURES REMOVED")
print(">90% missing in at least one site")
print("========================================")

for feature in failed_features:
    bad_sites = site_missing_fraction.index[
        site_missing_fraction[feature] > 0.90
    ]
    details = [
        f"{site} ({site_missing_fraction.loc[site, feature] * 100:.2f}%)"
        for site in bad_sites
    ]
    print(f"{feature}: " + ", ".join(details))


print("\n========================================")
print("FEATURES THAT SURVIVE")
print("========================================")

for feature in surviving_features:
    print(feature)

print("\nInitial features:", len(df.columns))
print("Failed features:", len(failed_features))
print("Surviving features:", len(surviving_features))


# ============================================================
# SAVE SURVIVING FEATURES TO TXT
# ============================================================

with open("SurvivingFeatures.txt", "w") as f:
    for feature in surviving_features:
        f.write(feature + "\n")

print("\nSaved surviving feature list:")
print("SurvivingFeatures.txt")


# Plot the clustered heatmap with row colors for the site categories
g = sns.clustermap(
    missing_data,
    cmap="viridis",               # Choose a colormap
    row_cluster=False,            # Do not cluster rows
    col_cluster=True,             # Cluster columns
    figsize=(15, 10),
    cbar_kws=None,                # Disable the color bar
    row_colors=row_colors,        # Add row color annotations
    yticklabels=False             # Remove default row labels
)

# Remove the color bar manually after plotting
g.cax.remove()

# Create a legend for the site categories
for site, color in site_colors.items():
    g.ax_row_dendrogram.bar(0, 0, color=color, label=site, linewidth=0)

g.ax_row_dendrogram.legend(
    title="Site Categories",
    loc="upper left",          # Position legend in the upper left
    bbox_to_anchor=(0, 1),     # Adjust position to be at the top-left
    ncol=1
)

# plt.title("Missing Data Heatmap with Site Categories")
plt.show()
plt.savefig("MissingDataHeatmap.png", bbox_inches="tight")
plt.clf()



# ============================================================
# REMOVE SAMPLES MISSING ANY SURVIVING FEATURE
# ============================================================

md=md.dropna(subset=["GAGEBRTH","PTB"])

# Treat -77 as missing for the surviving features
complete_case_mask = (
    ~md[surviving_features].isin([-88, -77, -99]).any(axis=1)
    & md[surviving_features].notna().all(axis=1)
)
print("\n========================================")
print("COMPLETE-CASE FILTER")
print("========================================")
print("Samples before:", len(md))
print("Samples removed:", (~complete_case_mask).sum())
print("Samples retained:", complete_case_mask.sum())

# Keep only samples complete for all surviving features
df_complete = md.loc[complete_case_mask].copy()


df_complete.to_csv('Metadata.MissRem.tsv', index=False, sep="\t") 
#Need dummy column for plink PCA
df_complete["Dummy"]=0
df_complete[["Dummy","id"]].to_csv("IDs.txt", index=False,header=False,sep='\t') 


























