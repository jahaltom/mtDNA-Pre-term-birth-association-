# mtDNA Pre-term birth association


### plink2VCF.sh: 
Takes in plink files for nDNA and makes vcf.

## Run Haplogrep3 to assign haplogroups to samples.

Use tree "rCRS PhyloTree 17.2" and  Kulczynski Distance function. Run this on merged_chrM_22175.vcf. Outputs haplogroups to haplogrep3OUT_22175. 

```
./haplogrep3 classify  --extend-report --tree phylotree-rcrs@17.2 --in merged_chrM_22175.vcf --out haplogrep3OUT_22175
```

## Metadata curration, filtering, and conversion. 
### MetadataMerge.py: 
Takes in Haplogrep3 output and metadata files (momi_combined_data.txt and samples.tab) and performs merge. Filters for only high quality haplogroup calls "Quality">=0.9 and only live births "PREG_OUTCOME"==2. Identifies main and sub haplogroups. 
Seperates mother and child in dataset, then filters to include only main haplogroups with at least 10 occurrences. Writes two tsvs (Metadata.C.tsv and Metadata.M.tsv). 

## Outlier removal with Weibull
### WeibullFiltering.py:
Takes in (Metadata.C.tsv and Metadata.M.tsv) and removes samples where GA "GAGEBRTH" is na. Fit the Weibull distribution to the data and defines cutoff thresholds for outlier detection (upper/lower GA in days ...1st percentile and 99th percentile). Filter the data on these threshholds. 
Outputs (Metadata.M.Weibull.tsv Metadata.C.Weibull.tsv).
Plots the original data, filtered data, and Weibull distribution. Includes lower_cutoff and upper_cutoff in plot (weibullFiltering.M.png weibullFiltering.C.png).
Also outputs C.txt and M.txt which are subset from (Metadata.M.Weibull.tsv Metadata.C.Weibull.tsv) and used for sample selection in plink2VCF.sh.

C: Lower Cutoff: 232.24832311358944, Upper Cutoff: 297.5082654877975

M: Lower Cutoff: 229.40491328561183, Upper Cutoff: 298.6448367835414


### Subset nDNA VCF: 
Subsets nDNA vcf. Selects for only snps, excludes chrs (x,y,and M), selects for samples from previous dataset (C.txt and M.txt). Outputs two vcfs (plink2.C.vcf and plink2.M.vcf) that will used below. 
```
bcftools view --types snps -t ^26,24,23 -S C.txt --force-samples /scr1/users/haltomj/PTB/plink2.vcf   >  plink2.C.vcf
bcftools view --types snps -t ^26,24,23 -S M.txt --force-samples /scr1/users/haltomj/PTB/plink2.vcf   >  plink2.M.vcf
```
## Dimensionality reduction via PCA and MDS.
Generates PCs and MDS clusters for (plink2.C.vcf and plink2.M.vcf). Does this across all data (All) and South Asian/African seperatly. Below is for the plink2.C.vcf data only. To do the plink2.M.vcf, just swap C and M. 
```
mkdir PCA-MDS
#Run plink PCA and MDS All
plink --vcf plink2.C.vcf --pca --double-id --out PCA-MDS/C
plink --vcf plink2.C.vcf --cluster --mds-plot 5 --double-id --out PCA-MDS/C
rm listC

plink --vcf plink2.M.vcf --pca --double-id --out PCA-MDS/M
plink --vcf plink2.M.vcf --cluster --mds-plot 5 --double-id --out PCA-MDS/M
rm listM
```

## Combine PCA/MDS results with metadata. 
### CombinePCA-MDS.py: 
Takes in eigenvec and mds files and adds this data to (Metadata.M.Weibull.tsv Metadata.C.Weibull.tsv). Outputs ("Metadata.M.Final.tsv" and "Metadata.C.Final.tsv"). 



## Plotting
### PCA-MDA_Plot.r:
Takes in Metadata.M.Final.tsv, Metadata.C.Final.tsv, and eigenval, and makes PCA/MDS plots. Lables Main/Sub haplogroup and site. Does this for All populations, african only, and south asian only. Also splits data by child/mother. 


## Check for mtDNA haplogroup association with nDNA PCA clusters.
To determine whether the nDNA PCA clusters correlated with the mtDNA haplogroups due to assortative mating;

### Pearson correlation:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')
# Clean the dataset

df['GAGEBRTH'] = pd.to_numeric(df['GAGEBRTH'], errors='coerce')  # Ensure GAGEBRTH is numeric
df=df[[  'DIABETES','PW_AGE', 'MAT_HEIGHT',"PC1", "PC2", "PC3", "PC4", "PC5","MainHap","PTB", "GAGEBRTH"]]

categorical_columns=["MainHap"]
continuous_columns=['DIABETES', 'PW_AGE', 'MAT_HEIGHT', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# Mixed Feature Correlation: One-Hot Encode categorical features
encoded_df = pd.get_dummies(df[categorical_columns], drop_first=True)
encoded_df = encoded_df.astype(int)
mixed_df = pd.concat([df[continuous_columns + ['GAGEBRTH','PTB']], encoded_df], axis=1)








# Compute and visualize correlation matrix  (Pearson correlation).
plt.figure(figsize=(20, 15))
corr_matrix = mixed_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Matrix for All Variables")
plt.tight_layout()
plt.show()
plt.savefig("PCACorr.png")
plt.close()
```
