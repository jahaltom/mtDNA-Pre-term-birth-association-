#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 5:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL







head -1 samples.tab > header



#Run plink PCA and MDS South Asian
cat samples.tab | grep 'POPULATION'   > samples.tail.tab
cat header samples.tail.tab > samples.tab
rm samples.tail.tab header









conda activate plink 


python MetadataMerge.py


python WeibullFiltering.py



#Makes ntDNA vcf for PCA.
bcftools view --types snps -t ^26,24,23 -S C.txt --force-samples /scr1/users/haltomj/PTB/plink2.vcf   >  plink2.C.vcf
bcftools view --types snps -t ^26,24,23 -S M.txt --force-samples /scr1/users/haltomj/PTB/plink2.vcf   >  plink2.M.vcf

    
    
    
mkdir PCA-MDS
#Run plink PCA and MDS All
plink --vcf plink2.C.vcf --pca --double-id --out PCA-MDS/C
plink --vcf plink2.C.vcf --cluster --mds-plot 5 --double-id --out PCA-MDS/C
rm listC

plink --vcf plink2.M.vcf --pca --double-id --out PCA-MDS/M
plink --vcf plink2.M.vcf --cluster --mds-plot 5 --double-id --out PCA-MDS/M
rm listM
    
    


python CombinePCA-MDS.py




module load R-/4.3.2

Rscript  PCA-MDA_Plot.r
