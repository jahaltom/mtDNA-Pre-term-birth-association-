import pandas as pd
import numpy as np


###Combine metadata and haplogroup data

#Read in metadata
md=pd.read_csv('momi5.pheno',sep='\t')  
#Read in haplogroup data from Haplogrep3
haplo=pd.read_csv('haplogrep3OUT',sep='\t')  


#Remove _C and _M  (Mother, Child) and repeat IDs from IDs
haplo['ORIG_ID'] = haplo['SampleID'].str.split('_').str[0]


#Merge
md2=pd.merge(md,haplo,on=["ORIG_ID"])




#Grab Main haplogroup
md2["MainHap"] = md2["Haplogroup"].map(lambda s: next((hap for hap in ["L0","L1","L2","L3","L4","L5","HV"] if hap in s), "other"))
md2['MainHap'] = np.where(md2['MainHap'] == 'other', md2['Haplogroup'].astype(str).str[0:1],md2["MainHap"])

#Grab sub haplogroups
speHaps=["L0","L1","L2","L3","L4","L5","HV"]
md2['SubHap'] = np.where(md2['MainHap'].isin(speHaps), md2['Haplogroup'].astype(str).str[0:3], md2['Haplogroup'].astype(str).str[0:2])




md2=md2[['ORIG_ID', 'site_name', 'age', 'sex', 'ht', 'gday', 'ptb', 'bwt',
       'twin', 'livebirth', 'SampleID', 'Haplogroup', 'MainHap', 'SubHap','Rank', 'Quality',
       'Range', 'Not_Found_Polys', 'Found_Polys', 'Remaining_Polys',
       'AAC_In_Remainings', 'Input_Sample']]


    
#Break md2 up by child and mother
dfM = md2[md2['SampleID'].str.contains('_M')]
dfC = md2[md2['SampleID'].str.contains('_C')]


def makemetadata(d,string):
    #Subset South Asian and African from data set   
    afr=d[(d["site_name"]=="GAPPS-Zambia") | (d["site_name"]=="AMANHI-Tanzania") ].copy()
    afr["Population"]="African"
    sa =d[(d["site_name"]=="AMANHI-Bangladesh") | (d["site_name"]=="GAPPS-Bangladesh") | (d["site_name"]=="AMANHI-Pakistan") ].copy()
    sa["Population"]="South_Asian"
    
    
    def isAtLeast10(df):
        #Make a df that marks a main haplogroup that has at least 10 samples.
        x=(df["MainHap"].value_counts() >= 10).to_frame()
        x= x.rename(columns={'MainHap': 'IsAtLeast10MainHap'})
        x.index.name = "MainHap"
        x.reset_index(inplace=True)
        #Merge
        df=pd.merge(df,x,on=["MainHap"])
        #Make a df that marks a sub haplogroup that has at least 10 samples.
        x=(df["SubHap"].value_counts() >= 10).to_frame()
        x= x.rename(columns={'SubHap': 'IsAtLeast10SubHap'})
        x.index.name = "SubHap"
        x.reset_index(inplace=True)
        #Merge
        return pd.merge(df,x,on=["SubHap"])
    
    
    
    AFR=isAtLeast10(afr)
    SA=isAtLeast10(sa)
    
    dfFinal=pd.concat([AFR,SA])
    
    dfFinal.to_csv("MetadataFinal."+string+".tsv", index=False, sep='\t')  



makemetadata(dfM, "M")
makemetadata(dfC, "C") 







