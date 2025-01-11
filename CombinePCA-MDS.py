import pandas as pd
from functools import reduce






sets=["M","C"]
for s in sets:   
    dfComb=[]   

    header=["FID","IID"]+[f"PC{i}"+"_" for i in range(1, 21)]
    pca = pd.read_csv("PCA-MDS/"+"_"+s+".eigenvec",sep='\s+',header=None)
    pca.columns=header
    pca = pca.drop(columns=['IID'])
    mds = pd.read_csv("PCA-MDS/"+"_"+s+".mds",sep='\s+')
    mds = mds.drop(columns=['IID','SOL'])
    mds.columns=mds.columns + "_" 
    mds = mds.rename(columns={"FID_" : "FID"})
    df=pd.merge(pca,mds,on=["FID"]) 
    dfComb.append(df)                
    dfComb = reduce(lambda left, right: pd.merge(left, right, on="FID",how="left"), dfComb)
    dfComb = dfComb.rename(columns={"FID":"Sample_ID"})
    md = pd.read_csv("Metadata."+s+".Weibull.tsv",sep='\t')  
    dfFinal=pd.merge(md,dfComb,on=["Sample_ID"])     
    sa=["GAPPS-Bangladesh","AMANHI-Pakistan","AMANHI-Bangladesh"]
    afr=["AMANHI-Pemba","GAPPS-Zambia"]  
    dfFinal.loc[dfFinal["site"].isin(sa), "population"] = "SouthAsian"
    dfFinal.loc[dfFinal["site"].isin(afr), "population"] = "African"
    dfFinal.to_csv("Metadata."+s+".Final.tsv", index=False,sep="\t") 
 
