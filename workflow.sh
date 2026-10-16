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
  --out nDNA_final > qc.log 2>&1


# ============================================================
# PCA
# ============================================================

mkdir -p PCA

plink2  \
  --bfile nDNA_final \
  --pca 10 \
  --out PCA/out


# ============================================================
# QC SUMMARY STATS
# ============================================================

plink2  \
  --bfile nDNA_final \
  --missing \
  --freq \
  --out nDNA_stats


# ============================================================
# PYTHON STEPS
# ============================================================

conda activate ML

python scripts/outlierPCA.py

python scripts/WeibullFiltering.py "$target" > out.txt
