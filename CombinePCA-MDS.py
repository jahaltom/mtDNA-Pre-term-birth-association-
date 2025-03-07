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

