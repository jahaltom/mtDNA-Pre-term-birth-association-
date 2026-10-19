#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 24:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL

source /home/haltomj/miniconda3/etc/profile.d/conda.sh


conda activate ML



#Target
target=TARGET



conda activate plink



plink2 --bfile nDNA_final --keep IDs2.txt --make-bed --out nDNA_final2

mkdir PCA2
#Run plink PCA
plink2 --bfile nDNA_final2 --pca 10 --out PCA2/cleaned



conda activate ML
python  scripts/CombinePCA.py




conda activate ML

##EDA!

cp Metadata.Final.tsv "Exploratory Data Analysis"
cd "Exploratory Data Analysis"
python PearsonCorrelationAll.py 

mv Metadata.Final.tsv Continuous/
cd Continuous
python ContinuousEDA.py "$target"

mv Metadata.Final.tsv ../Categorical/
cd ../Categorical
python CategoricalEDA.py
rm Metadata.Final.tsv

















