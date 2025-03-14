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

# Read in raw counts
md = pd.read_csv("Metadata.Final.tsv", sep='\t', quotechar='"')

# Read eigenvalues
eigenvalues = pd.read_csv("PCA-MDS/out.eigenval", header=None)
total_variance = eigenvalues.sum().values[0]
pc_percentage = (eigenvalues / total_variance) * 100

features = ["MainHap", "SubHap", "site"]

for f in features:
    # Create PC1 vs PC2 scatter plot with color by feature
    plt.figure()
    plt.scatter(md['PC1'], md['PC2'], c=md[f], cmap='viridis', alpha=0.8, s=1)
    plt.colorbar(label=f)
    plt.title("PC1 vs PC2")
    plt.xlabel(f"PC1 ({round(pc_percentage.iloc[0], 1)}% variance)")
    plt.ylabel(f"PC2 ({round(pc_percentage.iloc[1], 1)}% variance)")
    plt.savefig(f"{f}.PCA.png")
    plt.close()

    # Create C1 vs C2 scatter plot with color by feature
    plt.figure()
    plt.scatter(md['C1'], md['C2'], c=md[f], cmap='viridis', alpha=0.8, s=1)
    plt.colorbar(label=f)
    plt.title("C1 vs C2")
    plt.xlabel("C1")
    plt.ylabel("C2")
    plt.savefig(f"{f}.MDS.png")
    plt.close()
