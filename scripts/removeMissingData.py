import pandas as pd
import sys


# Load metadata
md = pd.read_csv(sys.argv[1],sep='\t')
md=md.dropna(subset=["GAGEBRTH","PTB"])


catigoricalFeat= [item for item in sys.argv[2].split(',') if item != '']  #
contFeat= [item for item in sys.argv[3].split(',') if item != '']  #




#All columns we want analized
wantedCol = catigoricalFeat + contFeat

# Apply filtering only in the wantedCol columns. Remove missing data rows. 
md = md[~md[wantedCol].isin([-88, -77,-99]).any(axis=1)]



md.to_csv('Metadata.MissRem.tsv', index=False, sep="\t") 
md[["Sample_ID","Sample_ID"]].to_csv("IDs.txt", index=False,header=False) 
