import pandas as pd



pops=["All","SouthAsian","Africa"]
cm=["M","C"]

 
for j in cm:   
    for i in pops:
        ###Combine metadata and haplogroup data
        
        #Read in metadata
        df=pd.read_csv('MetadataFinal.'+j+'.tsv',sep='\t')  
        
        #Read in MDS clusters and combine
        mds=pd.read_csv(i +'_'+ j +'.mds',sep="\s+")
        mds = mds.add_suffix('.'+j+'_'+i)
        mds=mds.rename(columns={"FID."+j+'_'+i: 'SampleID'})
        mds=mds.drop(columns=['IID.'+j+'_'+i, 'SOL.'+j+'_'+i])
        df=pd.merge(df,mds,on=["SampleID"],how="left")
        
        
        #Read in PCs and combine
        pca=pd.read_csv(i +'_'+ j +'.eigenvec',sep="\s+",header=None)
        pca.drop(columns=pca.columns[0], axis=1, inplace=True)
        pca=pca.T.reset_index(drop=True).T
        pca = pca.add_suffix('.'+j+'_'+i)
        pca = pca.add_prefix('PC')
        pca=pca.rename(columns={"PC0."+j+'_'+i: 'SampleID'})
        df=pd.merge(df,pca,on=["SampleID"],how="left")
        
        df.to_csv("MetadataFinal."+j+".2.tsv", index=False, sep='\t')  
    
    
    
    
    
    





