


#### MetadataMerge.py: 
- Takes in Haplogrep3 output and metadata files (MOMI_derived_data.csv and samples.tab) and performs merge. 
- Filters for only high quality haplogroup calls "Quality">=0.9 and only live births "PREG_OUTCOME"==2. 
- Identifies main and sub haplogroups. 
- This script also sets (ALCOHOL_FREQ, SMOK_FREQ, and SNIFF_FREQ) to 0 if (ALCOHOL,SMOKE_HIST, and SNIFF_TOBA) = never.
- Calculates BMI. (df["BMI"] = df["MAT_WEIGHT"]/(df["MAT_HEIGHT"]/100)**2)
- Categorizes population based on site. 
- Seperates mother and child in dataset and writes two tsvs (Metadata.C.tsv and Metadata.M.tsv).


#### MissingDataHeatmap.py
Takes in Metadata.C.tsv or Metadata.M.tsv and analyzes the dataset for missing data using features of interest. Outputs heatmap (MissingDataHeatmap.png). Missing data is in yellow. 

#### removeMissingData.py
- Removes samples where gestational age "GAGEBRTH" or  PTB (0 or 1) is na. Also removes samples with missing data in any of the input columns. Makes IDs.txt to be used for PCA. 


#### outlierPCA.py
- Loads PLINK .eigenvec and .eigenval
- Computes how many PCs are needed to reach ~85% variance
- Calculates site-wise Euclidean distance
- Flags the top 5% as outliers per site
- Saves a MetadataOutlierRemoved.tsv file
- Creates a before/after PCA plot



#### WeibullFiltering.py:
- Takes in ( Categorical/Continuous features)  
- Fits the Weibull distribution to the data for "GAGEBRTH".
   - Defines lower/upper cutoff thresholds, in days, for outlier detection (1st percentile and 99th percentile).
   - Filters the data on these threshholds (>= lower_cutoff) & <= upper_cutoff). 
- For each categorical variable class, determine the number of pre-term births and normal births (PTB=1 normal=0) and the % of PTB=1. Remove rows(samples) corresponding to a class from a categorical variable that total counts (PTB=1 normal=0) < 25. If only 1 class would remain after the prior filtering, don't exclude any samples and simply exclude the categorical variable from any future model.Reports categorical variables to keep/exclude for future models (Those kept are in CategoricalVariablesToKeepTable.tsv). Also reports those classes removed due to low counts.
- Also reports categorical variables with exactly two classes (binary). These will be used as binary variables for Feature selection. Outputs Categorical variables for Feature selection.
- Reports Weibull parameters (Shape, Scale, and Location) and upper/lower cutoffs in days.
- It finds haplogroups that appear in at least 2 sites, have ≥ 20 total samples and ≥ 4 PTB cases, keeps those unchanged, and relabels all other haplogroups as Other_<population>.
- Outputs filtered metadata as (Metadata.Weibull.tsv). Also outputs (IDs2.txt) which are only Sample_IDs  from (Metadata.Weibull.tsv) which will be used for sample selection form the nDNA plink data. 
- Plots the original data, filtered data, and Weibull distribution. Includes lower_cutoff and upper_cutoff in plot (weibullFiltering.png).
- All continuous features are ploted against PTB and GAGEBRTH (in plotsAll). 


#### CombinePCA.py:    
- Takes in eigenvec and adds this data to (Metadata.Weibull.tsv). 
- Outputs (Metadata.Final.tsv). 
- Takes in eigenval(for PCA), and makes PCA plots.
- Lables Main/Sub haplogroup and site.
