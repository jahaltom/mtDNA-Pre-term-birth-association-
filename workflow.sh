#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 24:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL

source /home/haltomj/miniconda3/etc/profile.d/conda.sh


conda activate ML


# Define Categorical features
columnsCat=CAT
# Define Continuous  features
columnsCont=CONT

# Convert the array to a comma-separated string
columnCat_string=$( echo "${columnsCat[*]}")
columnCont_string=$( echo "${columnsCont[*]}")




bcftools view --types snps -t ^26,24,23 -S IDs.txt --force-samples /scr1/users/haltomj/PTB/plink2.vcf   >  plink2.vcf

conda activate plink

mkdir PCA
#Run plink PCA
plink --vcf plink2.vcf --pca --double-id --out PCA/out

conda activate ML
python outlierPCA.py

conda activate plink
# Recompute PCA using filtered samples
plink --vcf plink2.vcf --keep keep_samples.txt --pca --double-id --out PCA/cleaned



conda activate ML
python  CombinePCA.py






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

















