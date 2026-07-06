# mtDNA Pre-term birth association

```
snakemake -j 22 -s Prepare_1KGP --latency-wait 60 --cluster "sbatch -t 02:00:00 -c 24 -N 1"
```


## Run Haplogrep3 to assign haplogroups to samples.

```
./haplogrep3 classify  --extend-report --tree phylotree-rcrs@17.2 --in merged_chrM_22175.vcf --out haplogrep3OUT_22175
```

## Metadata curration and filtering. 
This script merges Haplogrep3 output with metadata files (MOMI_derived_data.csv and samples.tab), filters for high-quality haplogroup calls (Quality ≥ 0.9) and live births (PREG_OUTCOME = 2), and assigns main/sub-haplogroups. It sets ALCOHOL_FREQ, SMOK_FREQ, and SNIFF_FREQ to 0 when ALCOHOL, SMOKE_HIST, and SNIFF_TOBA are "never," calculates BMI, and categorizes population by site. Makes SuperHap, SuperHap2, and PhyloHap (for south asian only) classification based on mtDNA phylogeny https://forensicgenomics.github.io/mitoLeaf/. All other haplogroups classified as "Other". Use MainHap and SubHap for African. Finally, it splits the dataset into mother and child subsets and writes them to Metadata.M.tsv and Metadata.C.tsv.
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
- Specify target (MainHap,SubHap,SuperHap). Target must also be in columnCat.
```
#Excluded: 'SNIFF_TOBA','PASSIVE_SMOK','ALCOHOL','SMOK_TYP'
columnCat="('TYP_HOUSE','HH_ELECTRICITY','FUEL_FOR_COOK','DRINKING_SOURCE','TOILET','WEALTH_INDEX','CHRON_HTN','DIABETES','TB','THYROID','EPILEPSY','BABY_SEX','MainHap','SMOKE_HIST','SMOK_FREQ','population','site')"

#Excluded:  'SNIFF_FREQ','ALCOHOL_FREQ','SMOK_YR'
columnCont="('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI')"

target="MainHap"

sed -i "s/CAT/$columnCat/g" workflow.sh
sed -i "s/CONT/$columnCont/g" workflow.sh
sed -i "s/TARGET/$target/g" workflow.sh

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
- Fits a Weibull distribution to GAGEBRTH, defines 1st and 99th percentile cutoffs, and filters samples outside this range. It summarizes pre-term vs. normal birth counts per categorical class, removes low-count classes (<20 total) or where there is not at least 1 term and 1 pre-term birth, and drops categorical variables entirely if only one class would remain. It reports which categorical variables are retained for workflow2.sh and PTB #s(in CategoricalVariablesToKeepTable.tsv), flags binary variables and multi-class categorical for feature selection, and outputs Weibull parameters, cutoffs, and plots. For the multi-site analysis: Haplogroup must be present in ALL sites,in EACH site: >= 5 total, >= 1 PTB, >= 1 term births, and overall across all sites: >= 20 total, >= 4 PTB.
For the single-site analysis:overall: >= 20 total and >= 4 PTB. Finally, it saves the filtered metadata (Metadata.Weibull.tsv), writes IDs2.txt for downstream nDNA selection,a FinalCategoricalClassSummary.tsv is also produced, and plots both filtered/unfiltered distributions and all continuous and categorical features are ploted against PTB and GAGEBRTH (in plotsAll) labeled by site. Makes a report (out.txt) which will indicate which variables to keep/exclude in future modeling. 



#### workflow2.sh
- Looking at out.txt from above, place "Categorical variables to keep for workflow2.sh" in columnCat below.
- Look for outliers in categorical/continuous features in plotsAll. 
- Carry same continuous features down from input into workflow.sh.
- Update workflow2.sh and run.
- Specify target (MainHap,SubHap, PhyloHap)
```
columnCat="('TYP_HOUSE','HH_ELECTRICITY','TOILET','WEALTH_INDEX','THYROID','CHRON_HTN','DIABETES','TB','FUEL_FOR_COOK','MainHap','DRINKING_SOURCE','BABY_SEX','population','site')"
columnCont="('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI')"
target="MainHap"

sed -i "s/CAT/$columnCat/g" workflow2.sh
sed -i "s/CONT/$columnCont/g" workflow2.sh
sed -i "s/TARGET/$target/g" workflow2.sh

sbatch workflow2.sh
```


- Subset nDNA plink files by selecting for samples from previous dataset (IDs2.txt). 
- Runs plink PCA (Outputs results into PCA2)
- Combine PCA results with metadata and plot PCA
##### Launches Exploratory Data Analysis (EDA)!


#### Feature Selection 
##### featureSelection.sh
- Look at PCA plots and make sure they look good.
- Carry same continuous features down from input into workflow.sh.
- Do not include Haplogroup as a predictor in feature-selection scripts.
- Do not include site if nDNA PCs are included, especially for joint multi-site runs.
- See out.txt for;
	- Binary categorical variables for feature selection. Place in columnBin
	- Multi-class categorical variables for feature selection. Place in columnCat


```
columnCat="('FUEL_FOR_COOK','site','MainHap')"
columnCont="('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI','TOILET','WEALTH_INDEX','DRINKING_SOURCE')"
columnBin="('BABY_SEX','CHRON_HTN','DIABETES','HH_ELECTRICITY','TB','THYROID','TYP_HOUSE')"

sed -i "s/CAT/$columnCat/g" featureSelection.sh
sed -i "s/CONT/$columnCont/g" featureSelection.sh
sed -i "s/BIN/$columnBin/g" featureSelection.sh
sbatch featureSelection.sh
```
Run ConsensusFeatureTable.py once all MLs finished. 

#### Final Model

##### finalModel.sh
- Specify target (MainHap,SubHap,PhyloHap).

```

columnCat="('MainHap')"
columnCont="('PW_AGE','MAT_HEIGHT')"
target="MainHap"
covs='"PW_AGE + MAT_HEIGHT + site"'  ### Fixed "site" or random effect "(1 | site)"
ref='"M"'

sed -i "s/CAT/$columnCat/g" finalModel.sh
sed -i "s/CONT/$columnCont/g" finalModel.sh
sed -i "s/TARGET/$target/g" finalModel.sh
sed -i "s/COVARIATES/$covs/g" finalModel.sh
sed -i "s/REF/$ref/g" finalModel.sh

sbatch finalModel.sh
```





### Site and nDNA PC associated

This script tests whether nDNA principal components are strongly associated with study site. It performs ANOVA with R² estimation for individual PCs, MANOVA across all PCs, and PERMANOVA to quantify the proportion of overall ancestry structure explained by site. These results help determine whether study site can be used as a proxy for ancestry in downstream association models.

```
python site_pc_structure_tests.py \
  --input Metadata.Final.tsv \
  --sep $'\t' \
  --site-col site \
  --pc-prefix PC \
  --n-pcs 5 \
  --permutations 999 \
  --out-prefix nDNA_PC_site

```
Output Files:

1.  *_anova_r2.csv – Per-PC ANOVA results showing the strength of association between study site and each nDNA principal component, including R² (variance explained by site).
- PC – principal component tested (e.g., PC1–PC5)
- F – ANOVA F-statistic
- p_value – statistical significance of site effect
- R2 – proportion of variance in the PC explained by study site
- adj_R2 – adjusted R² accounting for model complexity

2. *_manova.txt – MANOVA results testing whether study site explains overall variation across all included principal components simultaneously.
- Wilks’ lambda
- Pillai’s trace
- Hotelling–Lawley trace
- Roy’s greatest root
  
4. *_permanova.csv – PERMANOVA results quantifying the proportion of multivariate ancestry structure explained by study site (R²) and its statistical significance.
- permanova_F – PERMANOVA F-statistic
- permanova_R2 – fraction of total multivariate variance explained by site
- permanova_p – permutation-based significance value
- df_between / df_within – model degrees of freedom
- n_permutations – number of permutations performed
  
5. *_site_pc_summary.csv – Summary statistics (mean, standard deviation, sample count) for each principal component stratified by study site.

