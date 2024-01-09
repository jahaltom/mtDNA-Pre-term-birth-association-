import pandas as pd




hGroups=["H","M","L3","L2","L0","L1","U","D","R","L4","T","F","A","C","J","N","G","E","W"]

#Read in haplogrep3 output 
df=pd.read_csv("haplogrep3OUT",sep='\t')
#Get only high quality calls that are 1-16569.
df=df[(df["Quality"]>=0.90) & (df["Range"]=="1-16569")]


dfH=[]
for h in hGroups:
    #Get specific haplogroups
    dfH.append(df[df["Haplogroup"].str.startswith(h)].head(10))



dfH=pd.concat(dfH)
dfH.to_csv("Haplogroups.tsv",mode="w", header=True,index=False,sep="\t")