import pandas as pd
import glob
#import numpy as np
import dask.dataframe as dd
#from functools import reduce
import sys
#import pyarrow

sys.setrecursionlimit(25000)



# List all .txt files in the current directory
txt_files = glob.glob("*.txt")

dataframes=[]

for t in txt_files:
    df = pd.read_csv(t,sep='\t',header=None)
    df = df[~df[4].astype(str).str.contains(",", na=False)]       
    df.columns = ['CHROM', 'POS', 'REF',"ALT",t.split('.')[0],"DP"]  
    df["Var"]=df['POS'].astype(str)+"_"+df['REF']+">"+df['ALT']
    df=df[["Var",t.split('.')[0]]]
    dataframes.append(df)


# Convert Pandas DataFrames to Dask DataFrames
dask_dataframes = [dd.from_pandas(df, npartitions=4) for df in dataframes]

# Function to merge DataFrames in batches
def merge_in_batches(dfs, batch_size=100):
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = dd.merge(merged_df, dfs[i], on='Var', how='outer')
    return merged_df

# If you have too many DataFrames (e.g., 1000+), process them in smaller chunks
batch_size = 100  # Adjust this value based on your memory capacity

# Split the list of DataFrames into batches
chunks = [dask_dataframes[i:i + batch_size] for i in range(0, len(dask_dataframes), batch_size)]

# Merge DataFrames in chunks
merged_dfs = []
for chunk in chunks:
    merged_dfs.append(merge_in_batches(chunk))

# Final merge across all chunks
final_merged_df = merged_dfs[0]
for df in merged_dfs[1:]:
    final_merged_df = dd.merge(final_merged_df, df, on='Var', how='outer')

# Compute the final result (this triggers the actual computation)
merged_df = final_merged_df.compute() 



# Extract the numeric part and sort by it
merged_df['POS'] = merged_df['Var'].str.extract('(\d+)').astype(int)
df_sorted = merged_df.sort_values('POS')
df_sorted.iloc[:, 1:df_sorted.shape[1]-1] = df_sorted.iloc[:, 1:df_sorted.shape[1]-1].apply(pd.to_numeric, errors='coerce')


df_sorted = df_sorted.fillna(0) 





# Save the DataFrame as a pickle file
# df_sorted.to_pickle("dataframe.pkl")



# df_sorted = pd.read_pickle("dataframe.pkl")




cOm=["C","M"]

for f in cOm:
    dfCM = pd.read_csv("/scr1/users/haltomj/PTB/heteroplasmy/Metadata."+f+".Final.tsv",sep='\t')
    
    #Subset df to only have C or M
    cols=["Var"]+dfCM["SampleID"].to_list()+["POS"]
    
    existing_columns = [col for col in df_sorted if col in cols]
    df_filt = df_sorted[existing_columns]
    
    per1=df_filt[df_filt.columns[1:df_filt.shape[1]-1]].shape[1]*.01
    # Condition: Remove rows where less than 1% columns are > 0  
    df_filt=df_filt[(df_filt[df_filt.columns[1:df_filt.shape[1]-1]] > 0).sum(axis=1) >= per1] 
    
    
    
    
    # Step 1: Reshape df_freq into long format
    df_long = pd.melt(df_filt, id_vars=['Var', 'POS'], 
                  value_vars=df_filt.columns[1:df_filt.shape[1]-1],
                  var_name='SampleID', value_name='Frequency')
    
    # Step 2: Merge with race metadata
    df_merged = df_long.merge(dfCM, on='SampleID')
    
    df_merged.to_csv('forPlotting.'+f+'.csv', index=None,sep="\t")
    
    
    
    
    
    
    
    ####NN
    
    dfCM=dfCM[["SampleID"]]
    #Drop the 'POS' column if it's no longer needed
    df = df_filt.drop(columns='POS')
    df = df.set_index('Var')
    df=df.T
    # Convert rownames (index) to a column
    df = df.reset_index()
    # Rename the new column if needed (optional)
    df.rename(columns={'index': 'SampleID'}, inplace=True)
    df=pd.merge(dfCM,df,on=["SampleID"])     
    
    
    df.to_csv('HetroplasmyNN.'+f+'.tsv', index=None,sep='\t')
    








