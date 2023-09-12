#Take in plink files and make VCF
plink --bfile momi5.clean --recode vcf --out yay

bgzip yay.vcf


bcftools index yay.vcf.gz

#Extract mtDNA SNPs
bcftools view yay.vcf.gz --regions 26 | bcftools view --types snps | bcftools view -S samplesC.txt >  outmtDNA_C.vcf
bcftools view yay.vcf.gz --regions 26 | bcftools view --types snps | bcftools view -S samplesM.txt >  outmtDNA_M.vcf

#Makes ntDNA vcf for PCA.
bcftools view --types snps -t ^26,25,23 yay.vcf.gz  | bcftools view -S samplesC.txt >  outntDNA_C.vcf
bcftools view --types snps -t ^26,25,23 yay.vcf.gz  | bcftools view -S samplesM.txt >  outntDNA_M.vcf
