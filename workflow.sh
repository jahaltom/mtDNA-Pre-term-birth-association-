#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 24:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL

source /home/haltomj/miniconda3/etc/profile.d/conda.sh


conda activate ML


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








#Removes samples where gestational age "GAGEBRTH" or  PTB (0 or 1) is na. Also removes samples with missing data in any of the input columns.
python scripts/removeMissingData.py $file "$columnCat_string" "$columnCont_string"





conda activate plink



plink \
  --bfile /scr1/users/haltomj/PTB/plink2 \
  --keep IDs.txt \
  --chr 1-22 \
  --snps-only just-acgt \
  --biallelic-only strict \
  --geno 0.05 \
  --mind 0.05 \
  --maf 0.01 \
  --hwe 1e-6 midp \
  --threads 8 \
  --make-bed --out nDNA_final > qc.log 2>&1



mkdir PCA
# PCA for ancestry covariates
plink --bfile nDNA_final --pca 10 --out PCA/out

# QC summary stats
plink --bfile nDNA_final --missing --freq --out nDNA_stats




conda activate ML
python scripts/outlierPCA.py











# Call the Python script with the column string as an argument
python scripts/WeibullFiltering.py "$columnCat_string" "$columnCont_string" > out.txt
