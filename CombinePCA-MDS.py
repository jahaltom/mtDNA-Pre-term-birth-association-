import pandas as pd
from functools import reduce





header=["FID","IID"]+[f"PC{i}" for i in range(1, 21)]
pca = pd.read_csv("PCA-MDS/out.eigenvec",sep='\s+',header=None)
pca.columns=header
pca = pca.drop(columns=['IID'])
mds = pd.read_csv("PCA-MDS/out.mds",sep='\s+')
mds = mds.drop(columns=['IID','SOL'])
df=pd.merge(pca,mds,on=["FID"]) 
df = df.rename(columns={"FID":"Sample_ID"})
md = pd.read_csv("Metadata.Weibull.tsv",sep='\t')  
dfFinal=pd.merge(md,df,on=["Sample_ID"])     
dfFinal.to_csv("Metadata.Final.tsv", index=False,sep="\t") 





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Read eigenvalues
eigenvalues = pd.read_csv("PCA-MDS/out.eigenval", header=None)
total_variance = eigenvalues.sum().values[0]
pc_percentage = (eigenvalues / total_variance) * 100

features = ["MainHap", "SubHap", "site"]

for f in features:
    plt.figure(figsize=(12, 8))  # Adjusted figure size to accommodate legend
    # Create a scatter plot for each category within the feature
    categories = dfFinal[f].unique()
    for category in categories:
        subset = dfFinal[dfFinal[f] == category]
        plt.scatter(subset['PC1'], subset['PC2'], label=category, alpha=0.8, s=10)  # s increased for visibility
    plt.title("PC1 vs PC2")
    plt.xlabel(f"PC1 ({round(pc_percentage.iloc[0], 1)[0]}% variance)")
    plt.ylabel(f"PC2 ({round(pc_percentage.iloc[1], 1)[0]}% variance)")
    # Adjust legend to be outside the plot
    plt.legend(title=f, markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the plot area to make room for the legend
    plt.savefig(f"{f}.PCA.png")
    plt.close()
    # Repeat for C1 vs C2
    plt.figure(figsize=(12, 8))
    for category in categories:
        subset = dfFinal[dfFinal[f] == category]
        plt.scatter(subset['C1'], subset['C2'], label=category, alpha=0.8, s=10)
    plt.title("C1 vs C2")
    plt.xlabel("C1")
    plt.ylabel("C2")
    plt.legend(title=f, markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the plot area to make room for the legend
    plt.savefig(f"{f}.MDS.png")
    plt.close()
