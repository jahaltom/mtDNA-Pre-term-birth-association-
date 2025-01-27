#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 120:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=500G


conda activate plink

#Take in plink files and makes VCF
plink --bfile plink2 --recode vcf --out plink2



