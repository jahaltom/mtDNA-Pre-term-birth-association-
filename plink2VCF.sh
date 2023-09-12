#Take in plink files and makes VCF
plink --bfile momi5.clean --recode vcf --out clean

bgzip clean.vcf


bcftools index clean.vcf.gz

#Extract mtDNA SNPs
bcftools view --types snps  --regions 26 -S samplesC.txt clean.vcf.gz >  outmtDNA_C.vcf
bcftools view --types snps  --regions 26 -S samplesM.txt clean.vcf.gz >  outmtDNA_M.vcf

#Makes ntDNA vcf for PCA.
bcftools view --types snps -t ^26,25,23 -S samplesC.txt clean.vcf.gz   >  outntDNA_C.vcf
bcftools view --types snps -t ^26,25,23 -S samplesM.txt clean.vcf.gz   >  outntDNA_M.vcf

