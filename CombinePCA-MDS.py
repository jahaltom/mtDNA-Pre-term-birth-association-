import pandas as pd
from functools import reduce



sets=["M","C"]
for s in sets:   
    header=["FID","IID"]+[f"PC{i}" for i in range(1, 21)]
    pca = pd.read_csv("PCA-MDS/"+s+".eigenvec",sep='\s+',header=None)
    pca.columns=header
    pca = pca.drop(columns=['IID'])
    mds = pd.read_csv("PCA-MDS/"+s+".mds",sep='\s+')
    mds = mds.drop(columns=['IID','SOL'])
    df=pd.merge(pca,mds,on=["FID"]) 
    df = df.rename(columns={"FID":"Sample_ID"})
    md = pd.read_csv("Metadata."+s+".Weibull.tsv",sep='\t')  
    dfFinal=pd.merge(md,df,on=["Sample_ID"])     
    dfFinal.to_csv("Metadata."+s+".Final.tsv", index=False,sep="\t") 

