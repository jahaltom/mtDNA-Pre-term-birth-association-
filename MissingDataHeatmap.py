import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


CoM=["C","M"]
for i in CoM:
    
    # Load the data
    df = pd.read_csv("Metadata."+i+".tsv", sep='\t')
    
    
        # Extract the "site" column for row labels
    row_labels = df["site"]
    
    # Create a color palette for the site categories
    unique_sites = row_labels.unique()
    site_palette = sns.color_palette("tab10", len(unique_sites))  # Adjust palette as needed
    site_colors = {site: site_palette[i] for i, site in enumerate(unique_sites)}
    
    # Map site categories to colors
    row_colors = row_labels.map(site_colors)
    
    # Select relevant columns for the heatmap
    df = df[['PW_AGE', 'PW_EDUCATION', 'MAT_HEIGHT', 'MAT_WEIGHT',
                       'TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
                           'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK','CHRON_HTN',
                           'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
                           'ALCOHOL' , 'ALCOHOL_FREQ' ,'SMOKE_HIST','SMOK_FREQ', 'SMOK_TYP','SMOK_YR','SNIFF_TOBA','SNIFF_FREQ']]
                          
    
    
    
    # Replace -88 with -77 (if necessary)
    df.replace(-88, -77, inplace=True)
    df.replace(-99, -77, inplace=True)
    
    # Create a mask for missing data (-77) and transform it into a binary form for visualization
    missing_data = df.replace(-77, np.nan).isnull().astype(int)
    
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
    
    #plt.title("Missing Data Heatmap with Site Categories")
    plt.show()
    plt.savefig("MissingDataHeatmap."+i+".png", bbox_inches="tight")
    plt.clf()

