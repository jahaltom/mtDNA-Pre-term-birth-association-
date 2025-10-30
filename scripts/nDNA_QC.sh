#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 8:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL



conda activate plink

#| Step                      | Filter              | Meaning                            |
#| ------------------------- | ------------------- | ---------------------------------- |
#| `--bfile nDNA_raw`        | input               | your starting `.bed/.bim/.fam`     |
#| `--chr 1-22`              | autosomes only      | drops chr M, X, Y                  |
#| `--snps-only just-acgt`   | variant type        | drop indels and non-ACGT calls     |
#| `--biallelic-only strict` | allele structure    | keeps only clean biallelic SNPs    |
#| `--geno 0.05`             | variant missingness | removes SNPs with > 5 % missing    |
#| `--mind 0.05`             | sample missingness  | removes samples with > 5 % missing |
#| `--maf 0.01`              | allele frequency    | keeps MAF ≥ 1 %                    |
#| `--hwe 1e-6 midp`         | Hardy–Weinberg      | removes SNPs failing HWE ≤ 1e-6    |
#| `--make-bed`              | output              | writes new binary dataset          |
#| `--out nDNA_final`        | prefix              | output name for filtered data      |



plink \
  --bfile plink2 \
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




# PCA for ancestry covariates
plink --bfile nDNA_final --pca 10 --out nDNA_pca

# QC summary stats
plink --bfile nDNA_final --missing --freq --out nDNA_stats

