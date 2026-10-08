import pandas as pd
import numpy as np
 
        

###########The subject ID in samples.tab  is BABY, PArticipant and ORIG_ID in momi_combined.data.tsv!!!!!!!!!!!!!!!!!!!!!!!!

#Read in metadata
df=pd.read_csv('momi_mothers_ultimate.csv')  






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
df["BMI"] = np.where(
    (df["MAT_WEIGHT"] < 0) | (df["MAT_HEIGHT"] < 0),
    -77,
    df["MAT_WEIGHT"] / (df["MAT_HEIGHT"] / 100) ** 2
)






# Function to categorize population based on site
def categorize_population(site):
    if 'Pemba' in site or 'Zambia' in site:
        return 'African'
    else:
        return 'South Asian'
# Apply function to create a new column
df['population'] = df['site'].apply(categorize_population)





mapping_dict = {
    **{k: "M_lineage" for k in ["M", "D", "G","Q","C","Z","E"]},
    **{k: "N_lineage" for k in ["N","O","S","I","W","Y","A","X"]},
    **{k: "R_lineage" for k in ["R", "J","T","H", "HV", "V", "P", "F", "B", "K", "U"]}
}

df["SuperHap"] = df["MainHap"].map(mapping_dict).fillna("Other")


print(df["SuperHap"].value_counts())

# Also check mapping coverage
print(df.groupby(["MainHap", "SuperHap"]).size().reset_index())







mapping_dict = {
    **{k: "M_lineage" for k in ["M", "D", "G","Q","C","Z","E"]},
    **{k: "NR_lineage" for k in ["N","O","S","I","W","Y","A","X","R", "J","T","H", "HV", "V", "P", "F", "B", "K", "U"]}
}

df["SuperHap2"] = df["MainHap"].map(mapping_dict).fillna("Other")


print(df["SuperHap2"].value_counts())

# Also check mapping coverage
print(df.groupby(["MainHap", "SuperHap2"]).size().reset_index())






import pandas as pd
import numpy as np

# Longest/prefix-specific matches should come before broad matches like M or R
phylo_map = {
    "M9": ["M9", "E"],
    "M8": ["M8", "C", "Z"],
    "M7": ["M7"],
    "M6": ["M6"],
    "M5": ["M5"],
    "M4": ["M4"],
    "M3": ["M3"],
    "M2": ["M2"],
    "M1": ["M1"],
    "G": ["G"],
    "Q": ["Q"],
    "D": ["D"],
    "N1": ["N1","I"],
    "N2": ["N2","W"],
    "N9": ["N9","Y"],
    "A": ["A"],
    "O": ["O"],
    "S": ["S"],
    "X": ["X"],
    "R":  ["R", "B", "F", "J", "T", "H", "V", "U", "K","P"],
   
}

def assign_phylohap(hap):
    if pd.isna(hap):
        return np.nan

    hap = str(hap).strip().upper()

    for phylohap, prefixes in phylo_map.items():
        if any(hap.startswith(prefix) for prefix in prefixes):
            return phylohap

    return "Other"

df["PhyloHap"] = df["Haplogroup"].apply(assign_phylohap)






df.to_csv("Metadata.M.tsv", index=False, sep='\t')  


 
