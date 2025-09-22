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




bcftools view -S IDs2.txt --force-samples plink2.vcf   >  plink2.2.vcf

conda activate plink

mkdir PCA2
#Run plink PCA
plink --vcf plink2.2.vcf --pca --double-id --out PCA2/out





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

















