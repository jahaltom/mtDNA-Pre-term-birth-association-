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
Takes in Haplogrep3 output and metadata files (MOMI_derived_data.tsv and samples.tab) and performs merge. Filters for only high quality haplogroup calls "Quality">=0.9 and only live births "PREG_OUTCOME"==2. Identifies main and sub haplogroups. This script also sets (ALCOHOL_FREQ, SMOK_FREQ, and SNIFF_FREQ) to 0 if (ALCOHOL,SMOKE_HIST, and SNIFF_TOBA) = never. Seperates mother and child in dataset and writes two tsvs (Metadata.C.tsv and Metadata.M.tsv). 

 
## MissingDataHeatmap.py
Takes in Metadata.C.tsv or Metadata.M.tsv and analyzes the dataset for missing data using features of interest. Outputs heatmap (MissingDataHeatmap.png). Missing data is in yellow. 
```
python Metadata.C.tsv
```
![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/plots/MissingDataHeatmap.M.png?raw=true)
![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/plots/MissingDataHeatmap.C.png?raw=true)

## Outlier removal with Weibull
### WeibullFiltering.py:
Takes in (Metadata.C.tsv and Metadata.M.tsv) and removes samples where GA "GAGEBRTH" and PTB is na. Fit the Weibull distribution to the data and defines cutoff thresholds for outlier detection (upper/lower GA in days ...1st percentile and 99th percentile). Filter the data on these threshholds. 
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
Generates PCs and MDS clusters for (plink2.C.vcf and plink2.M.vcf). 
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
Takes in Metadata.M.Final.tsv, Metadata.C.Final.tsv, and eigenval, and makes PCA/MDS plots. Lables Main/Sub haplogroup and site.  Also splits data by child/mother. 


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
### Discretize PCA components into clusters and Calculate Cohen's Kappa



#### Use this to inform your KMeans clustering. 
Pick the elbow in the plot (where an increase in the x-axis is no longer making a notable changein y-axis). 

Chose the highest Silhouett score. 

Both of these should agree. In this case it was n_clusters=4

```python 
#Elbow Method (for KMeans)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortions = []
K = range(1, 10)  # Try different numbers of clusters
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[['PC1', 'PC2']])
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, distortions, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()
plt.savefig("Elbow.png")
plt.close()

#Silhouette Score
from sklearn.metrics import silhouette_score

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df[['PC1', 'PC2']])
    score = silhouette_score(df[['PC1', 'PC2']], labels)
    print(f'Number of clusters: {k}, Silhouette Score: {score:.3f}')


```

#### Discretize PCA components into clusters ( KMeans)
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df['PC1_cluster'] = kmeans.fit_predict(df[['PC1']])
df['PC2_cluster'] = kmeans.fit_predict(df[['PC2']])


# Step 1: Convert 'MainHap' to numeric labels
label_encoder = LabelEncoder()
df['MainHap_numeric'] = label_encoder.fit_transform(df['MainHap'])



# Step 2: Calculate Cohen's Kappa for PCA1 and MainHap
kappa_pca1_hap = cohen_kappa_score(df['PC1_cluster'], df['MainHap_numeric'])
print(f"Weighted Cohen's Kappa for PCA1 and mtDNA Haplogroups: {kappa_pca1_hap:.3f}")

# You can do the same for PCA2 and MainHap
kappa_pca2_hap = cohen_kappa_score(df['PC2_cluster'], df['MainHap_numeric'])
print(f"Weighted Cohen's Kappa for PCA2 and mtDNA Haplogroups: {kappa_pca2_hap:.3f}")
```

Negative values (PC1 -0.001 and PC2 -0.007) suggest no agreement or random association.
Conclusion: No significant evidence of assortative mating effecting the mtDNA haplogroups.


