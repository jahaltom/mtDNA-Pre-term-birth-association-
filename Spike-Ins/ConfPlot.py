import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\haltomj\Documents")




hGroups=["H","M","L3","L2","L0","L1","U","D","R","L4","T","F","A","C","J","N","G","E","W"]



#Read in haplogrep3 output 
df=pd.read_csv("haplogrepCompOUT",sep='\t')



#Grab Main haplogroup
df["MainHapPred"] = df["Haplogroup"].map(lambda s: next((hap for hap in ["L0","L1","L2","L3","L4","L5","HV"] if hap in s), "other"))
df['MainHapPred'] = np.where(df['MainHapPred'] == 'other', df['Haplogroup'].astype(str).str[0:1],df["MainHapPred"])

#Grab sub haplogroups
speHaps=["L0","L1","L2","L3","L4"]
df['SubHapPred'] = np.where(df['MainHapPred'].isin(speHaps), df['Haplogroup'].astype(str).str[0:3], df['Haplogroup'].astype(str).str[0:2])




#Grab Main haplogroup
df["MainHapTrue"] = df["Haplogroup_Whole"].map(lambda s: next((hap for hap in ["L0","L1","L2","L3","L4","L5","HV"] if hap in s), "other"))
df['MainHapTrue'] = np.where(df['MainHapTrue'] == 'other', df['Haplogroup_Whole'].astype(str).str[0:1],df["MainHapTrue"])

#Grab sub haplogroups
speHaps=["L0","L1","L2","L3","L4"]
df['SubHapTrue'] = np.where(df['MainHapTrue'].isin(speHaps), df['Haplogroup_Whole'].astype(str).str[0:3], df['Haplogroup_Whole'].astype(str).str[0:2])




actual=np.array(df["MainHapTrue"].tolist())

predicted=np.array(df["MainHapPred"].tolist())

#To make counts based CM
#cm = confusion_matrix(predicted,actual,labels=["H","M","L0","L1","L2","L3","L4","U","D","R","T","F","A","C","J","N","G","E","W"])
#To make % pased CM
cm = confusion_matrix(predicted,actual,labels=["H","M","L0","L1","L2","L3","L4","U","D","R","T","F","A","C","J","N","G","E","W"],normalize="pred")
cm=cm.round(decimals=2, out=None)

sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=["H","M","L0","L1","L2","L3","L4","U","D","R","T","F","A","C","J","N","G","E","W"],
            yticklabels=["H","M","L0","L1","L2","L3","L4","U","D","R","T","F","A","C","J","N","G","E","W"])
sns.set(font_scale=0.5)
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.savefig("ConfPlot.png",format='png',dpi=450,bbox_inches='tight')

plt.show()




