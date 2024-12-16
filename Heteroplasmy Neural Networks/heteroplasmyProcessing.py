import pandas as pd
import glob
import matplotlib.pyplot as plt

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


# Start with the first DataFrame
merged_df = dataframes[0]
# Iterate over the remaining DataFrames and merge them on the 'Var' column
for d in dataframes[1:]:
    merged_df = pd.merge(merged_df, d, on='Var', how='outer')


# Extract the numeric part and sort by it
merged_df['POS'] = merged_df['Var'].str.extract('(\d+)').astype(int)
df_sorted = merged_df.sort_values('POS')



df_sorted = df_sorted.fillna(0) 
  

 
df_sorted.to_csv('data.csv', index=None)


 

#df_sorted.iloc[:, 1:df_sorted.shape[1]-1] = df_sorted.iloc[:, 1:df_sorted.shape[1]-1].apply(pd.to_numeric, errors='coerce')



      # Condition: Remove rows where less than 10 columns are > 0  
df_sorted=df_sorted[(df_sorted[df_sorted.columns[1:df_sorted.shape[1]-1]] > 0).sum(axis=1) >= 100] 




# Step 1: Reshape df_freq into long format
df_long = pd.melt(df_sorted, id_vars=['Var', 'POS'], 
                  value_vars=df_sorted.columns[1:df_sorted.shape[1]-1],
                  var_name='SampleID', value_name='Frequency')



 
 
dfC = pd.read_csv("Metadata.C.Final.tsv",sep='\t')
dfM = pd.read_csv("Metadata.M.Final.tsv",sep='\t')
dfCM=pd.concat([dfC, dfM], axis=0)

# Step 2: Merge with race metadata
df_merged = df_long.merge(dfCM, on='SampleID')






##########################################Population
# Step 3: Map colors to races
race_to_color = {'SouthAsian': 'blue', 'African': 'green'}  # Define color mapping
df_merged['Color'] = df_merged['population'].map(race_to_color)

# Step 4: Plot the data
plt.figure(figsize=(20, 6))

# Scatter plot colored by race
for race in df_merged['population'].unique():
    subset = df_merged[df_merged['population'] == race]
    plt.scatter(subset['POS'], subset['Frequency'], 
                label=race, c=subset['Color'], alpha=0.7, s=50)

# Add labels, legend, and title
plt.xlabel('Position')
plt.ylabel('Frequency')
plt.title('Position vs Frequency Colored by population Metadata')
plt.legend(title='population')
plt.grid(True)

# Show the plot
plt.show()
plt.savefig("Heteroplasmy.Population.png", bbox_inches="tight")
plt.clf()      
##########################################PTB
# Step 3: Map colors to races
ptb = {0.0: 'blue', 1.0: 'red'}  # Define color mapping
df_merged['Color'] = df_merged['PTB'].map(ptb)

# Step 4: Plot the data
plt.figure(figsize=(20, 6))

# Scatter plot colored by race
for race in df_merged['PTB'].unique():
    subset = df_merged[df_merged['PTB'] == race]
    plt.scatter(subset['POS'], subset['Frequency'], 
                label=race, c=subset['Color'], alpha=0.7, s=50)

# Add labels, legend, and title
plt.xlabel('Position')
plt.ylabel('Frequency')
plt.title('Position vs Frequency Colored by PTB Metadata')
plt.legend(title='PTB')
plt.grid(True)

# Show the plot
plt.show()
plt.savefig("Heteroplasmy.PTB.png", bbox_inches="tight")
plt.clf()   









dfCM=dfCM[["PTB","GAGEBRTH","SampleID"]]


#Drop the 'POS' column if it's no longer needed
df = df_sorted.drop(columns='POS')
df = df.set_index('Var')
df=df.T
# Convert rownames (index) to a column
df = df.reset_index()

# Rename the new column if needed (optional)
df.rename(columns={'index': 'SampleID'}, inplace=True)

df=pd.merge(dfCM,df,on=["SampleID"])     



df.to_csv('HetroplasmyNN.tsv', index=None,sep='\t')



# Assuming `vaf_matrix` is your matrix of variant allele frequencies
scaler = MinMaxScaler()
normalized_vaf = scaler.fit_transform(vaf_matrix)











