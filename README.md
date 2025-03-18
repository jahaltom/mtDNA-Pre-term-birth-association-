# mtDNA Pre-term birth association


## plink2VCF.sh: 
Takes in plink files for nDNA and makes vcf.

## Run Haplogrep3 to assign haplogroups to samples.

Use tree "rCRS PhyloTree 17.2" and  Kulczynski Distance function. Run this on merged_chrM_22175.vcf. Outputs haplogroups to haplogrep3OUT_22175. 

```
./haplogrep3 classify  --extend-report --tree phylotree-rcrs@17.2 --in merged_chrM_22175.vcf --out haplogrep3OUT_22175
```

## Metadata curration, filtering, and conversion. 
### MetadataMerge.py: 
- Takes in Haplogrep3 output and metadata files (MOMI_derived_data.tsv and samples.tab) and performs merge. 
- Filters for only high quality haplogroup calls "Quality">=0.9 and only live births "PREG_OUTCOME"==2. 
- Identifies main and sub haplogroups. 
- This script also sets (ALCOHOL_FREQ, SMOK_FREQ, and SNIFF_FREQ) to 0 if (ALCOHOL,SMOKE_HIST, and SNIFF_TOBA) = never.
- Calculates BMI. (df["BMI"] = df["MAT_WEIGHT"]/(df["MAT_HEIGHT"]/100)**2)
- Categorizes population based on site. 
- Seperates mother and child in dataset and writes two tsvs (Metadata.C.tsv and Metadata.M.tsv). 
```
python  MetadataMerge.py
```
 
## MissingDataHeatmap.py
Takes in Metadata.C.tsv or Metadata.M.tsv and analyzes the dataset for missing data using features of interest. Outputs heatmap (MissingDataHeatmap.png). Missing data is in yellow. 
```
python MissingDataHeatmap.py Metadata.C.tsv
```
![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/plots/MissingDataHeatmap.M.png?raw=true)
![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/plots/MissingDataHeatmap.C.png?raw=true)




## worfkflow.sh
Specify:
- Input file: (Metadata.M.tsv or Metadata.C.tsv)
- FeaturesPASS: All features that passed missing data analysis. 
- Categorical and continuous features to be used for EDA 

In the script alter 4 lines:
```
#Input file
file="Metadata.M.tsv"
# Define FeaturesPASS features
columnsAll=('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','TYP_HOUSE','HH_ELECTRICITY','FUEL_FOR_COOK','DRINKING_SOURCE','TOILET','WEALTH_INDEX','CHRON_HTN','DIABETES','TB','THYROID','EPILEPSY','BABY_SEX','MainHap','SMOKE_HIST','SMOK_FREQ')
# Define Categorical features
columnsCat=('TYP_HOUSE','HH_ELECTRICITY','FUEL_FOR_COOK','DRINKING_SOURCE','TOILET','WEALTH_INDEX','CHRON_HTN','DIABETES','TB','THYROID','EPILEPSY','BABY_SEX','MainHap','SMOKE_HIST','SMOK_FREQ')
# Define Continuous  features
columnsCont=('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT',"PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","C1","C2","C3","C4","C5")
```
Then run with 
```
bash workflow.sh
```


#### Outlier removal with Weibull (WeibullFiltering.py):
- Takes in (input file and columns in FeaturesPASS)  
- Removes samples where gestational age "GAGEBRTH" or  PTB (0 or 1) is na. Also removes samples with missing data in any of the input columns "FeaturesPASS". 
- Fits the Weibull distribution to the data for "GAGEBRTH".
   - Defines lower/upper cutoff thresholds, in days, for outlier detection (1st percentile and 99th percentile).
   - Filters the data on these threshholds (>= lower_cutoff) & <= upper_cutoff). 
- Additionaly, removes samples who are in a haplogroup with <25 samples.

- Reports Weibull parameters (Shape, Scale, and Location) and upper/lower cutoffs in days. 
- Outputs filtered metadata as (Metadata.Weibull.tsv). Also outputs IDs.txt which are only SampleIDs  from (Metadata.Weibull.tsv) which will be used for sample selection form the nDNA vcf. 
- Plots the original data, filtered data, and Weibull distribution. Includes lower_cutoff and upper_cutoff in plot (weibullFiltering.png).


#### Subset nDNA VCF: 
- Selects for only snps, excludes chrs (x,y,and M), selects for samples from previous dataset (IDs.txt). 
- Outputs (plink2.vcf) that will used for PCA/MDS. 

#### Dimensionality reduction via PCA and MDS.
- Runs plink PCA and MDS using plink2.vcf
- MDS dimension count = 5. 
- Outputs results into PCA-MDS


#### Combine PCA/MDS results with metadata and plot PCA/MDS (CombinePCA-MDS.py):    
- Takes in eigenvec and mds files and adds this data to (Metadata.Weibull.tsv). 
- Outputs (Metadata.Final.tsv). 
- Takes in eigenval(for PCA), and makes PCA/MDS plots.
- Lables Main/Sub haplogroup and site.



#### Exploratory Data Analysis (EDA)
- see EDA folder.






## Check for mtDNA haplogroup association with nDNA PCA clusters (assortative mating).



### Discretize PCA components into clusters (Kmeans) and calculate Cohen's Kappa



#### Use this to inform your KMeans clustering. 
- Pick the elbow in the plot (where an increase in the x-axis is no longer making a notable changein y-axis). 
- Chose the highest Silhouett score. 
- Both of these should agree. In this case it was n_clusters=4

```python 
#Elbow Method (for KMeans)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Metadata.Final.tsv", sep='\t')

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

#### Discretize PCA components into clusters ( KMeans), plot, and calculate Cohen's Kappa
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score
import seaborn as sns

# Clustering using both PC1 and PC2
kmeans = KMeans(n_clusters=4, random_state=42)
df['combined_cluster'] = kmeans.fit_predict(df[['PC1', 'PC2']])


# Visualize 
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='combined_cluster', data=df, palette='viridis', style=df['combined_cluster'], markers=True, s=100)
plt.title('KMeans Clustering on PCA Components')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend(title='Cluster')
plt.grid(True)  # Optional: adds a grid for easier visualization
plt.savefig("KmeansPlot.png")
plt.clf()



# Step 1: Convert 'MainHap' to numeric labels
label_encoder = LabelEncoder()
df['MainHap_numeric'] = label_encoder.fit_transform(df['MainHap'])

# Step 2: Calculate Cohen's Kappa for the combined PCA clusters and MainHap
kappa_combined = cohen_kappa_score(df['combined_cluster'], df['MainHap_numeric'])
print(f"Cohen's Kappa for combined PCA clusters and mtDNA Haplogroups: {kappa_combined:.3f}")

```

Calculate Cohen's Kappa for site and population.

```python
# Convert 'site' to numeric labels
label_encoder = LabelEncoder()
df['site'] = label_encoder.fit_transform(df['site'])

# Calculate Cohen's Kappa for site and MainHap
kappa_combined = cohen_kappa_score(df['site'], df['MainHap_numeric'])
print(f"Cohen's Kappa for site and mtDNA Haplogroups: {kappa_combined:.3f}")



# Convert 'population' to numeric labels
label_encoder = LabelEncoder()
df['population'] = label_encoder.fit_transform(df['population'])

# Calculate Cohen's Kappa for population and MainHap
kappa_combined = cohen_kappa_score(df['population'], df['MainHap_numeric'])
print(f"Cohen's Kappa for population and mtDNA Haplogroups: {kappa_combined:.3f}")

```

