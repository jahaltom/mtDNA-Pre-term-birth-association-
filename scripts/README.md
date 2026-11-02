


#### MetadataMerge.py: 
- Takes in momi_mothers_ultimate.csv. 
- Identifies main and sub haplogroups.
- This script also sets (ALCOHOL_FREQ, SMOK_FREQ, and SNIFF_FREQ) to 0 if (ALCOHOL,SMOKE_HIST, and SNIFF_TOBA) = never.
- Calculates BMI. (df["BMI"] = df["MAT_WEIGHT"]/(df["MAT_HEIGHT"]/100)**2)
- Categorizes population based on site.
- Makes SuperHap, SuperHap2, and PhyloHap (for south asian only) classification based on mtDNA phylogeny https://forensicgenomics.github.io/mitoLeaf/. All other haplogroups classified as "Other". Use MainHap and SubHap for African.
- Writes Metadata.M.tsv.


#### MissingData.py
Takes in Metadata.M.tsv and assesses missingness in selected metadata features by study site. Uses features.json to specify selected categorical and continuous features. Features with >90% missingness in any site are excluded and the surviving feature list is saved to SurvivingFeatures_Categorical.txt and SurvivingFeatures_Continuous.txt. A missingness heatmap is generated (MissingDataHeatmap.png). Samples missing GAGEBRTH_NEW, PTB, or any surviving feature are removed. The filtered metadata are saved as Metadata.MissRem.tsv, and IDs.txt is created for downstream PCA/PLINK analysis. 



#### DataFiltering.py:
- Initial run: reads candidate categorical and continuous features from SurvivingFeatures_Categorical.txt and SurvivingFeatures_Continuous.txt. Final-model runs: use user-specified feature lists supplied by finalModel.sh.
- Fits a Weibull distribution to GAGEBRTH_NEW and reports the fitted parameters and 1st/99th percentile cutoffs for diagnostic purposes only. Weibull percentiles are not used to exclude samples. Samples are restricted only to biologically plausible gestational ages of 140–300 days.   
- Evaluates support for non-target categorical variables. Classes with <20 observations or without at least one PTB and one term birth are excluded. Categorical variables are excluded from downstream feature selection if fewer than two supported classes remain. Retained categorical variables are recorded for next step in workflow (EDA) and are classified as binary or multiclass for feature selection and written to JSON/summary files CategoricalVariables.json.
- Applies separate support criteria to the target haplogroup variable. For multi-site analyses, a haplogroup must be present in every site and have ≥5 samples, ≥1 PTB, and ≥1 term birth within each site, plus ≥20 samples and ≥4 PTBs overall. For single-site analyses, haplogroups require ≥20 samples and ≥4 PTBs overall.
- Saves the resulting metadata as Metadata.Filtered.tsv and writes IDs2.txt for downstream nDNA subsetting. CategoricalVariablesToKeepTable.tsv and FinalCategoricalClassSummary.tsv contain categorical-variable and PTB-count summaries before and after haplogroup filtering (respectfully).
- Plots the original data, filtered data, and Weibull distribution (weibullFiltering.png). Generates diagnostic plots of gestational age, PTB status, continuous features, and categorical features, stratified by site, in plotsAll/. A summary of retained/excluded variables and target haplogroups is written to out.txt.



#### CombinePCA.py:    
- Takes in eigenvec and adds this data to (Metadata.Filtered.tsv). 
- Outputs (Metadata.Final.tsv). 
- Takes in eigenval(for PCA), and makes PCA plots.
- Lables Main/Sub haplogroup and site.
