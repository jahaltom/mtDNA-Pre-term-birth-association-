# mtDNA Pre-term birth association

## Run Haplogrep3 to assign haplogroups to samples.

```
./haplogrep3 classify  --extend-report --tree phylotree-rcrs@17.2 --in merged_chrM_22175.vcf --out haplogrep3OUT_22175
```




## 1KGP





## Metadata curration and filtering. 
Takes in (momi_mothers_ultimate.csv), and assigns main/sub-haplogroups. It sets ALCOHOL_FREQ, SMOK_FREQ, and SNIFF_FREQ to 0 when ALCOHOL, SMOKE_HIST, and SNIFF_TOBA are "never," calculates BMI, and categorizes population by site. Makes SuperHap, SuperHap2, and PhyloHap (for south asian only) classification based on mtDNA phylogeny https://forensicgenomics.github.io/mitoLeaf/. All other haplogroups classified as "Other". Use MainHap and SubHap for African. Writes all to Metadata.M.tsv.
```
python  scripts/MetadataMerge.py
```
Takes in Metadata.M.tsv and assesses missingness in selected metadata features by study site. Uses features.json to specify selected categorical and continuous features. Features with >90% missingness in any site are excluded and the surviving feature list is saved to SurvivingFeatures_Categorical.txt and SurvivingFeatures_Continuous.txt. A missingness heatmap is generated (MissingDataHeatmap.png). Samples missing GAGEBRTH_NEW, PTB, or any surviving feature are removed. The filtered metadata are saved as Metadata.MissRem.tsv, and IDs.txt is created for downstream PCA/PLINK analysis. 

```
python scripts/MissingData.py
```
![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/plots/MissingDataHeatmap.M.png?raw=true)



#### workflow.sh
- Specify target (MainHap,SubHap,SuperHap) and run. 
```
target="MainHap"
sed -i "s/TARGET/$target/g" workflow.sh
sbatch workflow.sh
```

- Using nDNA plink files:  

| Step                      | Filter              | Meaning                            |
| ------------------------- | ------------------- | ---------------------------------- |
| `--bfile nDNA_raw`        | input               | your starting `.bed/.bim/.fam`     |
| `--keep IDs.txt  `        | input               | subset samples by IDs.txt          |
| `--chr 1-22`              | autosomes only      | drops chr M, X, Y                  |
| `--snps-only just-acgt`   | variant type        | drop indels and non-ACGT calls     |
| `--max-alleles` | allele structure    | keeps only clean biallelic SNPs    |
| `--geno 0.05`             | variant missingness | removes SNPs with > 5 % missing    |
| `--mind 0.05`             | sample missingness  | removes samples with > 5 % missing |
| `--maf 0.01`              | allele frequency    | keeps MAF ≥ 1 %                    |
| `--hwe 1e-6 midp`         | Hardy–Weinberg      | removes SNPs failing HWE ≤ 1e-6    |
| `--make-bed`              | output              | writes new binary dataset          |
| `--out nDNA_final`        | prefix              | output name for filtered data      |

- Makes qc.log and QC summery stats file nDNA_stats
- Subsets Metadata.MissRem.tsv to match plink samples above and saves ot Metadata.cleaned.tsv., 


- Initial run: reads candidate categorical and continuous features from SurvivingFeatures_Categorical.txt and SurvivingFeatures_Continuous.txt. Final-model runs: use user-specified feature lists supplied by finalModel.sh.
- Fits a Weibull distribution to GAGEBRTH_NEW and reports the fitted parameters and 1st/99th percentile cutoffs for diagnostic purposes only. Weibull percentiles are not used to exclude samples. Samples are restricted only to biologically plausible gestational ages of 140–300 days.   
- Evaluates support for non-target categorical variables. Classes with <20 observations or without at least one PTB and one term birth are excluded. Categorical variables are excluded from downstream feature selection if fewer than two supported classes remain. Retained categorical variables are classified as binary or multiclass and written to JSON/summary files.
- Applies separate support criteria to the target haplogroup variable. For multi-site analyses, a haplogroup must be present in every site and have ≥5 samples, ≥1 PTB, and ≥1 term birth within each site, plus ≥20 samples and ≥4 PTBs overall. For single-site analyses, haplogroups require ≥20 samples and ≥4 PTBs overall.
- Saves the resulting metadata as Metadata.Weibull.tsv and writes IDs2.txt for downstream nDNA subsetting. CategoricalVariablesToKeepTable.tsv and FinalCategoricalClassSummary.tsv contain categorical-variable and PTB-count summaries.
- Generates diagnostic plots of gestational age, PTB status, continuous features, and categorical features, stratified by site, in plotsAll/. A summary of retained/excluded variables and target haplogroups is written to out.txt.



- Subset nDNA plink files by selecting for samples from previous dataset (IDs2.txt). 
- Runs plink PCA (Outputs results into PCA)
- Combine PCA results with metadata and plot PCA
- Launches Exploratory Data Analysis (see Exploratory Data Analysis directory)


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
##### This is the final model that will associate mtDNA haplogroup with PTB/GA. (finalModel.sh)
- Specify target (MainHap,SubHap,PhyloHap).
- Choose categorical and continuous covariates found to be important in feature selection. Include some known covariates that effect PTB (BABY_SEX, MAT_HEIGHT, PW_AGE, nDNA PCs). Don't include nDNA PCs in continuous (columnCont), only place in covs (covariates) section.
- Choose a reference haplogroup.
- Construct an informative ID.
- The dataset is filtered from the very beginning (Metadata.M.tsv)  to remove samples with missing data in any of the covs, then plink filtering, PCA outlier removal, WeibullFiltering, and a Final PCA. Skipping EDA and feature selection. Doing it this way preserves significantly more samples.
- The alt model you run here (PTB/GA ~ Haplogroup + BABY_SEX + MAT_HEIGHT + BMI + PW_AGE + nDNA PCs) will also get run with the base model (PTB/GA ~ Haplogroup + BABY_SEX + MAT_HEIGHT + PW_AGE + nDNA PCs) on the same dataset. 
- Once complete OR_ShiftCheck.py and ES_ShiftCheck.py can be run to check for a >=10% shift in odds-ratio/Effect-size from the base for significant haplogeoups (padj< 0.05). If such a shift were to occur, it would mean the cov in the alt model is confounding with haplogroup and should be included in the final model. 
- See Final_Model tab for more info.

```
columnCat="('PhyloHap','BABY_SEX')"
columnCont="('PW_AGE','MAT_HEIGHT','BMI')"
target="PhyloHap"
covs='"BABY_SEX + PW_AGE + MAT_HEIGHT + BMI + PC1 + PC2 + PC3"'
ref='"M3"'
ID="PCsM3"

sed "s/CAT/$columnCat/g" finalModel.sh > finalModel.$ID.sh
sed -i "s/CONT/$columnCont/g" finalModel.$ID.sh
sed -i "s/TARGET/$target/g" finalModel.$ID.sh
sed -i "s/COVARIATES/$covs/g" finalModel.$ID.sh
sed -i "s/REF/$ref/g" finalModel.$ID.sh
sed -i "s/IDENT/$ID/g" finalModel.$ID.sh

sbatch finalModel.$ID.sh
```





