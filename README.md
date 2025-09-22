# mtDNA Pre-term birth association


## Make plink input VCF
sbatch scripts/plink2VCF.sh

## Run Haplogrep3 to assign haplogroups to samples.

```
./haplogrep3 classify  --extend-report --tree phylotree-rcrs@17.2 --in merged_chrM_22175.vcf --out haplogrep3OUT_22175
```

## Metadata curration and filtering. 
This script merges Haplogrep3 output with metadata files (MOMI_derived_data.csv and samples.tab), filters for high-quality haplogroup calls (Quality ≥ 0.9) and live births (PREG_OUTCOME = 2), and assigns main/sub-haplogroups. It sets ALCOHOL_FREQ, SMOK_FREQ, and SNIFF_FREQ to 0 when ALCOHOL, SMOKE_HIST, and SNIFF_TOBA are "never," calculates BMI, and categorizes population by site. Finally, it splits the dataset into mother and child subsets and writes them to Metadata.M.tsv and Metadata.C.tsv.
```
python  scripts/MetadataMerge.py
```
Takes in Metadata.C.tsv or Metadata.M.tsv and analyzes the dataset for missing data using features of interest. Outputs heatmap (MissingDataHeatmap.png). Missing data is in yellow.
```
python scripts/MissingDataHeatmap.py Metadata.M.tsv
```
![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/plots/MissingDataHeatmap.M.png?raw=true)



#### workflow.sh
- Use the missing data plot to exclude Categorical/Continuous features from those in workflow.sh (columnsCat and columnsCont). (Edit with vim)
```
sbatch workflow.sh
```
- Removes samples where gestational age "GAGEBRTH" or  PTB (0 or 1) is na. Also removes samples with missing data in any of the input columns. Makes (IDs.txt) to be used for PCA. 
- Using nDNA VCF, selects for only snps, excludes chrs (x,y,and M), selects for samples from previous dataset (IDs.txt). 
- Runs plink PCA
- PCA outlieres removed
- 	Calculates site-wise Euclidean distance (Using top N PCs needed to reach ~85% variance). Flags the top 1% as outliers per site
- 	Saves a MetadataOutlierRemoved.tsv file
- 	Creates a before/after PCA plot





#### workflow.sh
- Looking at out.txt from above, place "Categorical variables to keep for future model" in columnCat below.
- Look for outliers in continuous features in plotsAll. Adjust if necessary.
- Update workflow.sh and run.
```
columnCat="('TYP_HOUSE','HH_ELECTRICITY','TOILET','WEALTH_INDEX','THYROID','CHRON_HTN','DIABETES','TB','FUEL_FOR_COOK','MainHap','DRINKING_SOURCE','BABY_SEX')"
columnCont="('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI')"

sed -i "s/CAT/$columnCat/g" workflow.sh
sed -i "s/CONT/$columnCont/g" workflow.sh

sbatch workflow.sh
```


#### Subset nDNA VCF: 
- Selects for only snps, excludes chrs (x,y,and M), selects for samples from previous dataset (IDs.txt). 
- Outputs (plink2.vcf) that will used for PCA. 

#### Dimensionality reduction via PCA.
- Runs plink PCA using plink2.vcf
- Outputs results into PCA

#### outlierPCA.py
- Loads PLINK .eigenvec and .eigenval
- Computes how many PCs are needed to reach ~85% variance
- Calculates site-wise Euclidean distance
- Flags the top 1% as outliers per site
- Saves a keep_samples.txt file
- Creates a before/after PCA plot

#### Combine PCA results with metadata and plot PCA (CombinePCA.py):    
- Takes in eigenvec and adds this data to (Metadata.Weibull.tsv). 
- Outputs (Metadata.Final.tsv). 
- Takes in eigenval(for PCA), and makes PCA plots.
- Lables Main/Sub haplogroup and site.



#### Exploratory Data Analysis (EDA)
- see EDA folder.

#### featureSelection.sh
- Remove weight and heigth if using BMI (columnCont).
- Look at PCA plots and make sure they look good.
- See out.txt for;
	- Categorical variables with exactly two classes.  Will be used as binary variables for Feature selection. Place in columnBin
	- Categorical variables for Feature selection. Place in columnCat


```
columnCat="('DRINKING_SOURCE','FUEL_FOR_COOK','MainHap','TOILET','WEALTH_INDEX')"
columnCont="('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI','PC1','PC2','PC3','PC4','PC5')"
columnBin="('BABY_SEX','CHRON_HTN','DIABETES','HH_ELECTRICITY','TB','THYROID','TYP_HOUSE')"

sed -i "s/CAT/$columnCat/g" featureSelection.sh
sed -i "s/CONT/$columnCont/g" featureSelection.sh
sed -i "s/BIN/$columnBin/g" featureSelection.sh
sbatch featureSelection.sh
```



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

