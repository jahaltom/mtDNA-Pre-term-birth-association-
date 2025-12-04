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
columnsCat=('MainHap')
# Define Continuous  features
columnsCont=CONT

# Convert the array to a comma-separated string
columnCat_string=$( echo "${columnsCat[*]}")
columnCont_string=$( echo "${columnsCont[*]}")






cp $file Final_Model
cd Final_Model

#Removes samples where gestational age "GAGEBRTH" or  PTB (0 or 1) is na. Also removes samples with missing data in any of the input columns.
python ../scripts/removeMissingData.py $file "$columnCat_string" "$columnCont_string"





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
python ../scripts/outlierPCA.py


# Call the Python script with the column string as an argument
python ../scripts/WeibullFiltering.py "$columnCat_string" "$columnCont_string" > out.txt







conda activate plink



plink --bfile nDNA_final --keep IDs2.txt --make-bed --out nDNA_final2

mkdir PCA2
#Run plink PCA
plink --bfile nDNA_final2 --pca 10 --out PCA2/cleaned




conda activate ML
python  ../scripts/CombinePCA.py





