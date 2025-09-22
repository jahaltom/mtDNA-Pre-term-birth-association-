#### outlierPCA.py
- Loads PLINK .eigenvec and .eigenval
- Computes how many PCs are needed to reach ~85% variance
- Calculates site-wise Euclidean distance
- Flags the top 1% as outliers per site
- Saves a MetadataOutlierRemoved.tsv file
- Creates a before/after PCA plot



#### WeibullFiltering.py:
- Takes in ( Categorical/Continuous features)  
- Fits the Weibull distribution to the data for "GAGEBRTH".
   - Defines lower/upper cutoff thresholds, in days, for outlier detection (1st percentile and 99th percentile).
   - Filters the data on these threshholds (>= lower_cutoff) & <= upper_cutoff). 
- Additionaly, removes samples who are in a haplogroup with <25 samples.
- For each categorical variable class, determine the number of pre-term births and normal births (PTB=1 normal=0) and the % of PTB=1. Remove rows(samples) corresponding to a class from a categorical variable that total counts (PTB=1 normal=0) < 25. If only 1 class would remain after the prior filtering, don't exclude any samples and simply exclude the categorical variable from any future model.Reports categorical variables to keep/exclude for future models (Those kept are in CategoricalVariablesToKeepTable.tsv). Also reports those classes removed due to low counts.
- Also reports categorical variables with exactly two classes (binary). These will be used as binary variables for Feature selection. Outputs Categorical variables for Feature selection.

- Reports Weibull parameters (Shape, Scale, and Location) and upper/lower cutoffs in days. 
- Outputs filtered metadata as (Metadata.Weibull.tsv). Also outputs (IDs.txt) which are only SampleIDs  from (Metadata.Weibull.tsv) which will be used for sample selection form the nDNA vcf. 
- Plots the original data, filtered data, and Weibull distribution. Includes lower_cutoff and upper_cutoff in plot (weibullFiltering.png).
- All continuous features are ploted against PTB and GAGEBRTH (in plotsAll). 
