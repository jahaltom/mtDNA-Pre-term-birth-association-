import pandas as pd
import numpy as np


###Combine metadata and haplogroup data

#Read in metadata
md=pd.read_csv('momi5.pheno',sep='\t')  
#Read in haplogroup data from Haplogrep3
haplo=pd.read_csv('haplogrep3OUT',sep='\t')  


#Remove _C and _M  (Mother, Child) and repeat IDs from IDs
haplo['ORIG_ID'] = haplo['SampleID'].str.split('_').str[0]


#Merge and keep 1st occurance of dup IDs. Assuming haplogroup info are the same for mother and child. 
md2=pd.merge(md,haplo,on=["ORIG_ID"])
md2=md2.drop_duplicates(subset=['ORIG_ID'], keep='first').reset_index()



#Grab Main haplogroup
md2["MainHap"] = md2["Haplogroup"].map(lambda s: next((hap for hap in ["L0","L1","L2","L3","L4","L5","HV"] if hap in s), "other"))
md2['MainHap'] = np.where(md2['MainHap'] == 'other', md2['Haplogroup'].astype(str).str[0:1],md2["MainHap"])

#Grab sub haplogroups
speHaps=["L0","L1","L2","L3","L4","L5","HV"]
md2['SubHap'] = np.where(md2['MainHap'].isin(speHaps), md2['Haplogroup'].astype(str).str[0:3], md2['Haplogroup'].astype(str).str[0:2])




md2=md2[['ORIG_ID', 'site_name', 'age', 'sex', 'ht', 'gday', 'ptb',
       'bwt', 'twin', 'livebirth', 'Haplogroup', 'SubHap', 'MainHap', 'Rank', 'Quality',
       'Range', 'Not_Found_Polys', 'Found_Polys', 'Remaining_Polys',
       'AAC_In_Remainings', 'Input_Sample']]



###Get Stats

def getStats(df,string):
    #Gather main haplogroup and ptb column 
    main=df[["MainHap","ptb"]].copy()
    #Add column with 1 to keep track of total births
    main["TotalBirths"]=1
    #Group by main haplogroup and sum ptb and TotalBirths. 
    main=main.groupby(['MainHap'],as_index = False).sum()
    #Since ptb: preterm status (1: >=259; 2: <259) do the follwing to get stats
    main["ptbTotal"]=main["ptb"]-main["TotalBirths"]
    main["non-ptbTotal"]=main["TotalBirths"]-main["ptbTotal"]
    #Clean 
    main=main[["MainHap","TotalBirths","non-ptbTotal","ptbTotal"]]  
    main.to_csv(string+"MainHaplogroupStats.tsv", index=False, sep='\t')   
    #Get list of main haplogroups with TotalBirths >=10
    mainL=main[main["TotalBirths"]>=10]["MainHap"].to_list()
    #Make copy of whole df
    df2m=df.copy()
    #Make all main haplogroups with TotalBirths <10 "other" and set the rest to their main haplogroups
    df2m["PASSMain"] = df2m["MainHap"].map(lambda s: next((hap for hap in mainL if hap in s), "other"))
    
    
    #Gather sub haplogroup and ptb column 
    sub=df[["SubHap","ptb"]].copy()
    #Add column with 1 to keep track of total births
    sub["TotalBirths"]=1
    #Group by sub haplogroup and sum ptb and TotalBirths. 
    sub=sub.groupby(['SubHap'],as_index = False).sum()
    #Since ptb: preterm status (1: >=259; 2: <259) do the follwing to get stats
    sub["ptbTotal"]=sub["ptb"]-sub["TotalBirths"]
    sub["non-ptbTotal"]=sub["TotalBirths"]-sub["ptbTotal"]
    #Clean 
    sub=sub[["SubHap","TotalBirths","non-ptbTotal","ptbTotal"]] 
    sub.to_csv(string+"SubHaplogroupStats.tsv", index=False, sep='\t')  
    #Get list of sub haplogroups with TotalBirths >=10
    subL=sub[sub["TotalBirths"]>=10]["SubHap"].to_list()
    #Make copy of whole df
    df2s=df.copy()
    #Make all sub haplogroups with TotalBirths <10 "other" and set the rest to their sub haplogroups
    df2s["PASSSub"] = df2s["SubHap"].map(lambda s: next((hap for hap in subL if hap in s), "other"))   
    df2s=df2s[["ORIG_ID","PASSSub"]]
    #Combine main and sub
    dfFinal=pd.merge(df2m,df2s,on=["ORIG_ID"])
    dfFinal["Population"]=string
    return dfFinal

    
    
#Subset South Asian and African from data set   
afr=md2[(md2["site_name"]=="GAPPS-Zambia") | (md2["site_name"]=="AMANHI-Tanzania") ]
sa =md2[(md2["site_name"]=="AMANHI-Bangladesh") | (md2["site_name"]=="GAPPS-Bangladesh") | (md2["site_name"]=="AMANHI-Pakistan") ]

saDF=getStats(sa,"SouthAsian")
afrDF=getStats(afr,"African")

#Combine all populations
df=pd.concat([saDF,afrDF])
df.to_csv("MetadataFinal.tsv", index=False, sep='\t')  










