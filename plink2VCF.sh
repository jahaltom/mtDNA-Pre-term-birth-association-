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

#bgzip plink2.vcf


#bcftools index plink2.vcf

#Makes ntDNA vcf for PCA.
bcftools view --types snps -t ^26,24,23 -S child.txt --force-samples plink2.vcf   >  plink2.C.vcf
bcftools view --types snps -t ^26,24,23 -S mom.txt --force-samples plink2.vcf   >  plink2.M.vcf
