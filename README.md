# mtDNA Pre-term birth association




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
- Use the missing data plot to exclude Categorical/Continuous features from those in workflow.sh (columnCat and columnCont).
```
#Excluded: 'SNIFF_TOBA','PASSIVE_SMOK','ALCOHOL','SMOK_TYP'
columnCat="('TYP_HOUSE','HH_ELECTRICITY','FUEL_FOR_COOK','DRINKING_SOURCE','TOILET','WEALTH_INDEX','CHRON_HTN','DIABETES','TB','THYROID','EPILEPSY','BABY_SEX','MainHap','SMOKE_HIST','SMOK_FREQ','population','site')"

#Excluded:  'SNIFF_FREQ','ALCOHOL_FREQ','SMOK_YR'
columnCont="('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI')"

sed -i "s/CAT/$columnCat/g" workflow.sh
sed -i "s/CONT/$columnCont/g" workflow.sh

sbatch workflow.sh
```
- Removes samples where gestational age "GAGEBRTH" or  PTB (0 or 1) is na. Also removes samples with missing data in any of the input columns. Makes (IDs.txt) to be used for PCA. 
- Using nDNA plink files:  

| Step                      | Filter              | Meaning                            |
| ------------------------- | ------------------- | ---------------------------------- |
| `--bfile nDNA_raw`        | input               | your starting `.bed/.bim/.fam`     |
| `--keep IDs.txt  `        | input               | subset samples by IDs.txt          |
| `--chr 1-22`              | autosomes only      | drops chr M, X, Y                  |
| `--snps-only just-acgt`   | variant type        | drop indels and non-ACGT calls     |
| `--biallelic-only strict` | allele structure    | keeps only clean biallelic SNPs    |
| `--geno 0.05`             | variant missingness | removes SNPs with > 5 % missing    |
| `--mind 0.05`             | sample missingness  | removes samples with > 5 % missing |
| `--maf 0.01`              | allele frequency    | keeps MAF ≥ 1 %                    |
| `--hwe 1e-6 midp`         | Hardy–Weinberg      | removes SNPs failing HWE ≤ 1e-6    |
| `--make-bed`              | output              | writes new binary dataset          |
| `--out nDNA_final`        | prefix              | output name for filtered data      |

- Runs plink PCA
- Makes qc.log and QC summery stats file nDNA_stats
- PCA outlieres removed
	- 	Calculates site-wise Euclidean distance (Using top N PCs needed to reach ~85% variance). Flags the top 5% as outliers per site
	- 	Creates a before/after PCA plot
- Fits a Weibull distribution to GAGEBRTH, defines 1st and 99th percentile cutoffs, and filters samples outside this range. It summarizes pre-term vs. normal birth counts per categorical class, removes low-count classes (<25 total), and drops categorical variables entirely if only one class would remain. It reports which categorical variables are retained and PTB #s(in CategoricalVariablesToKeepTable.tsv), flags binary variables for feature selection, and outputs Weibull parameters, cutoffs, and plots. It finds haplogroups that appear in at least 2 sites, have ≥ 20 total samples and ≥ 4 PTB cases, keeps those unchanged, and relabels all other haplogroups as Other_population.Finally, it saves the filtered metadata (Metadata.Weibull.tsv), writes IDs2.txt for downstream nDNA selection, and plots both filtered/unfiltered distributions and all continuous feature relationships with PTB and GA (outputs to plotsAll). Makes a report (out.txt) which will indicate which variables to keep/exclude in future modeling. 



#### workflow2.sh
- Looking at out.txt from above, place "Categorical variables to keep for future model" in columnCat below.
- Look for outliers in continuous features in plotsAll. Adjust if necessary.
- Update workflow2.sh and run.
```
columnCat="('TYP_HOUSE','HH_ELECTRICITY','TOILET','WEALTH_INDEX','THYROID','CHRON_HTN','DIABETES','TB','FUEL_FOR_COOK','MainHap','DRINKING_SOURCE','BABY_SEX','population','site')"
columnCont="('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI')"

sed -i "s/CAT/$columnCat/g" workflow2.sh
sed -i "s/CONT/$columnCont/g" workflow2.sh

sbatch workflow2.sh
```


- Subset nDNA plink files by selecting for samples from previous dataset (IDs2.txt). 
- Runs plink PCA (Outputs results into PCA2)
- Combine PCA results with metadata and plot PCA
##### Launches Exploratory Data Analysis (EDA)!


#### Feature Selection 
##### featureSelection.sh
- Remove weight and heigth if using BMI (columnCont).
- Look at PCA plots and make sure they look good.
- See out.txt for;
	- Categorical variables with exactly two classes.  Will be used as binary variables for Feature selection. Place in columnBin
	- Categorical variables for Feature selection. Place in columnCat


```
columnCat="('FUEL_FOR_COOK','site')"
columnCont="('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI','TOILET','WEALTH_INDEX','DRINKING_SOURCE')"
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

