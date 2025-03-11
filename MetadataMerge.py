import pandas as pd
import numpy as np
 
        

###########The subject ID in samples.tab  is BABY, PArticipant and ORIG_ID in momi_combined.data.tsv!!!!!!!!!!!!!!!!!!!!!!!!

#Read in metadata
md=pd.read_csv('samples.tab',sep='\t')  
md = md.rename(columns={'id': 'SampleID'})

haplo=pd.read_csv('/scr1/users/haltomj/PTB/haplogrep3OUT_22175',sep='\t')  
haplo=pd.merge(md,haplo,on=["SampleID"])

haplo=haplo.drop_duplicates(subset=['Subject_ID'])








#Read in other metadata 
md=pd.read_csv('/scr1/users/haltomj/PTB/MOMI_derived_data.tsv',sep='\t')  
md=md.drop_duplicates() 



# #Only live births
md=md[md["PREG_OUTCOME"]==2]



#Merge haplogroups with metadata
mdPAR = md.rename(columns={'PARTICIPANT_ID': 'Subject_ID'})
x=pd.merge(mdPAR,haplo,on=["Subject_ID"])


mdBABY = md.rename(columns={'BABY_ID': 'Subject_ID'})
x2=pd.merge(mdBABY,haplo,on=["Subject_ID"])

mdORIG = md.rename(columns={'ORIG_ID': 'Subject_ID'})
haplo['Subject_ID'] = haplo['Subject_ID'].str.replace('-M','')
haplo['Subject_ID'] = haplo['Subject_ID'].str.replace('-C','')
x3=pd.merge(mdORIG,haplo,on=["Subject_ID"])


df=pd.concat([x,x2,x3])
df=df.drop_duplicates(subset=['Subject_ID'])
#Only high quality haplogroup calls. 
df=df[df["Quality"]>=0.9]






#Grab Main haplogroup
df["MainHap"] = df["Haplogroup"].map(lambda s: next((hap for hap in ["L0","L1","L2","L3","L4","L5","HV"] if hap in s), "other"))
df['MainHap'] = np.where(df['MainHap'] == 'other', df['Haplogroup'].astype(str).str[0:1],df["MainHap"])

#Grab sub haplogroups
speHaps=["L0","L1","L2","L3","L4","L5","HV"]
df['SubHap'] = np.where(df['MainHap'].isin(speHaps), df['Haplogroup'].astype(str).str[0:3], df['Haplogroup'].astype(str).str[0:2])

df.loc[df['ALCOHOL'] == 1, 'ALCOHOL_FREQ'] = 0
df.loc[df['SMOKE_HIST'] == 1, 'SMOK_FREQ'] = 0
df.loc[df['SNIFF_TOBA'] == 1, 'SNIFF_FREQ'] = 0





# calulate BMI
df["BMI"] = df["MAT_WEIGHT"]/(df["MAT_HEIGHT"]/100)**2


#Sep M and C 

dfM=df[df["M/C"]=="M"]
dfM["Sample_ID"]=("0_"+dfM["SampleID"])
dfM.to_csv("Metadata.M.tsv", index=False, sep='\t')  


dfC=df[df["M/C"]=="C"]
dfC["Sample_ID"]=("0_"+dfC["SampleID"])
dfC.to_csv("Metadata.C.tsv", index=False, sep='\t')  
