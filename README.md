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
- Takes in Haplogrep3 output and metadata files (MOMI_derived_data.csv and samples.tab) and performs merge. 
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
python MissingDataHeatmap.py Metadata.M.tsv
```
![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/plots/MissingDataHeatmap.M.png?raw=true)



#### Use the missing data plot to exclude Categorical/Continuous features from those below (columnsCat and columnsCont). Then run WeibullFiltering.py.

```
#Input file
file="Metadata.M.tsv"
# Define Categorical features
#Excluded: 'SNIFF_TOBA','PASSIVE_SMOK','ALCOHOL','SMOK_TYP'
columnsCat=('TYP_HOUSE','HH_ELECTRICITY','FUEL_FOR_COOK','DRINKING_SOURCE','TOILET','WEALTH_INDEX','CHRON_HTN','DIABETES','TB','THYROID','EPILEPSY','BABY_SEX','MainHap','SMOKE_HIST','SMOK_FREQ') 
# Define Continuous  features
#Excluded:  'SNIFF_FREQ','ALCOHOL_FREQ','SMOK_YR'
columnsCont=('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI')

# Convert the array to a comma-separated string
columnCat_string=$( echo "${columnsCat[*]}")
columnCont_string=$( echo "${columnsCont[*]}")

# Call the Python script with the column string as an argument
python WeibullFiltering.py $file "$columnCat_string" "$columnCont_string"
```
#### Outlier removal with Weibull (WeibullFiltering.py):
- Takes in (input file and Categorical/Continuous features)  
- Removes samples where gestational age "GAGEBRTH" or  PTB (0 or 1) is na. Also removes samples with missing data in any of the input columns. 
- Fits the Weibull distribution to the data for "GAGEBRTH".
   - Defines lower/upper cutoff thresholds, in days, for outlier detection (1st percentile and 99th percentile).
   - Filters the data on these threshholds (>= lower_cutoff) & <= upper_cutoff). 
- Additionaly, removes samples who are in a haplogroup with <25 samples.

- Reports Weibull parameters (Shape, Scale, and Location) and upper/lower cutoffs in days. 
- Outputs filtered metadata as (Metadata.Weibull.tsv). Also outputs IDs.txt which are only SampleIDs  from (Metadata.Weibull.tsv) which will be used for sample selection form the nDNA vcf. 
- Plots the original data, filtered data, and Weibull distribution. Includes lower_cutoff and upper_cutoff in plot (weibullFiltering.png).

- For each categorical variable class, determine the number of pre-term births and normal births (PTB=1 normal=0) and the % of PTB=1.  (Categorical_counts.csv). All continuous features are ploted against PTB and GAGEBRTH (in plotsAll). 


#### Look at Categorical_counts.csv (below) for features to exclude. Look for outliers in continuous features in plotsAll. Then run WeibullFiltering.py once more. 

```
             Column Value  PTB_0_Count  PTB_1_Count  PTB_1_Percentage
0         TYP_HOUSE     1         3961          307          7.193065
1         TYP_HOUSE     2         3290          240          6.798867
2    HH_ELECTRICITY     1         4309          381          8.123667
3    HH_ELECTRICITY     0         2942          166          5.341055
4     FUEL_FOR_COOK     1         1258          154         10.906516
5     FUEL_FOR_COOK     4         2825          259          8.398184
6     FUEL_FOR_COOK     3         2614          105          3.861714
7     FUEL_FOR_COOK     5          358           22          5.789474
8     FUEL_FOR_COOK     2          196            7          3.448276
9   DRINKING_SOURCE     1         4059          260          6.019912
10  DRINKING_SOURCE     2         2771          259          8.547855
11  DRINKING_SOURCE     4          420           28          6.250000
12  DRINKING_SOURCE     3            1            0          0.000000
13           TOILET     1         3765          360          8.727273
14           TOILET     2         2280          110          4.602510
15           TOILET     3         1193           76          5.988968
16           TOILET     4           13            1          7.142857
17     WEALTH_INDEX     5         1583          120          7.046389
18     WEALTH_INDEX     1         1364           96          6.575342
19     WEALTH_INDEX     3         1439          114          7.340631
20     WEALTH_INDEX     2         1412           83          5.551839
21     WEALTH_INDEX     4         1453          134          8.443604
22        CHRON_HTN     0         7071          523          6.887016
23        CHRON_HTN     1          180           24         11.764706
24         DIABETES     0         7219          538          6.935671
25         DIABETES     1           32            9         21.951220
26               TB     0         7213          545          7.025006
27               TB     1           38            2          5.000000
28          THYROID     0         7229          542          6.974649
29          THYROID     1           22            5         18.518519
30         EPILEPSY     0         7239          547          7.025430
31         EPILEPSY     1           12            0          0.000000
32         BABY_SEX     1         3675          289          7.290616
33         BABY_SEX     2         3576          258          6.729264
34          MainHap     M         2692          288          9.664430
35          MainHap     U          503           55          9.856631
36          MainHap     R          305           20          6.153846
37          MainHap     N           37            3          7.500000
38          MainHap     D          112           13         10.400000
39          MainHap     T           87           10         10.309278
40          MainHap     F           70            5          6.666667
41          MainHap     J           43            4          8.510638
42          MainHap     H           74            9         10.843373
43          MainHap     W          109           11          9.166667
44          MainHap     G           56            1          1.754386
45          MainHap     Z           23            2          8.000000
46          MainHap     K           32            7         17.948718
47          MainHap     A           57            3          5.000000
48          MainHap    L1          203            5          2.403846
49          MainHap    L2          768           39          4.832714
50          MainHap    L3         1465           53          3.491436
51          MainHap    L0          451           11          2.380952
52          MainHap    L4          115            6          4.958678
53          MainHap     E           49            2          3.921569
54       SMOKE_HIST     1         7241          547          7.023626
55       SMOKE_HIST     4            8            0          0.000000
56       SMOKE_HIST     2            2            0          0.000000
57        SMOK_FREQ     0         7241          547          7.023626
58        SMOK_FREQ     1           10            0          0.000000
```
#### Remove all unwanted features and re-run WeibullFiltering.py. Here I get rid of 
```python
import pandas as pd

df = pd.read_csv('Metadata.Weibull.tsv',sep='\t')
df=df[df["DRINKING_SOURCE"]!=3]
dfto_csv('Metadata.Weibull.tsv', index=False, sep="\t")
```


#### workflow begins

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

