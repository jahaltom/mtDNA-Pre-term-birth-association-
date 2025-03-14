#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 24:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL



conda activate ML

#Input file
file="Metadata.M.tsv"
# Define Categorical features
columnsCat=('TYP_HOUSE','HH_ELECTRICITY','FUEL_FOR_COOK','DRINKING_SOURCE','TOILET','WEALTH_INDEX','PASSIVE_SMOK','CHRON_HTN','DIABETES','TB','THYROID','EPILEPSY','BABY_SEX','MainHap','ALCOHOL','SMOKE_HIST','SMOK_FREQ','SMOK_TYP','SNIFF_TOBA')
# Define Continuous  features
columnsCont=('PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI','ALCOHOL_FREQ','SMOK_YR','SNIFF_FREQ')






# Convert the array to a comma-separated string

columnCat_string=$( echo "${columnsCat[*]}")
columnCont_string=$( echo "${columnsCont[*]}")






# Call the Python script with the column string as an argument
python WeibullFiltering.py $file "$columnCat_string" "$columnCont_string" 




bcftools view --types snps -t ^26,24,23 -S IDs.txt --force-samples /scr1/users/haltomj/PTB/plink2.vcf   >  plink2.vcf

conda activate plink

mkdir PCA-MDS
#Run plink PCA and MDS All
plink --vcf plink2.vcf --pca --double-id --out PCA-MDS/out
plink --vcf plink2.vcf --cluster --mds-plot 5 --double-id --out PCA-MDS/out

conda activate ML

python  CombinePCA-MDS.py






##EDA!

cp Metadata.Final.tsv "Exploratory Data Analysis"
cd "Exploratory Data Analysis"
python PearsonCorrelationAll.py "$columnCat_string" "$columnCont_string" 

mv Metadata.Final.tsv Continuous/
cd Continuous
python ContinuousEDA.py "$columnCont_string"

mv Metadata.Final.tsv ../Categorical/
cd ../Categorical
python CategoricalEDA.py "$columnCat_string"
rm Metadata.Final.tsv















