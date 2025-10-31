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



conda activate plink


plink --bfile nDNA_final --keep IDs2.txt --out nDNA_final2

mkdir PCA2
#Run plink PCA
plink --bfile nDNA_final2 --pca 10 --out PCA2/cleaned




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

















