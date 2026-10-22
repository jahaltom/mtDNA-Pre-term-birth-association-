#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 24:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL

source /home/haltomj/miniconda3/etc/profile.d/conda.sh

target=TARGET

conda activate plink


# ============================================================
# GENOTYPE QC
# ============================================================

plink2  \
  --bfile /scr1/users/haltomj/PTB/Data/qc/maternal_ptb_globalPCA \
  --keep IDs.txt \
  --chr 1-22 \
  --snps-only just-acgt \
  --max-alleles 2 \
  --geno 0.05 \
  --mind 0.05 \
  --maf 0.01 \
  --hwe 1e-6 midp \
  --threads 8 \
  --make-bed \
  --out nDNA_cleaned > qc.log 2>&1



# ============================================================
# QC SUMMARY STATS
# ============================================================

plink2  \
  --bfile nDNA_cleaned \
  --missing \
  --freq \
  --out nDNA_stats


# ============================================================
# PYTHON STEPS
# ============================================================

conda activate ML

python scripts/WeibullFiltering.py "Inital" "$target" > out.txt



conda activate plink

plink2 --bfile nDNA_cleaned --keep IDs2.txt --make-bed --out nDNA_final

mkdir PCA
#Run plink PCA
plink2 --bfile nDNA_final --pca 10 --out PCA/out



conda activate ML
python  scripts/CombinePCA.py






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

