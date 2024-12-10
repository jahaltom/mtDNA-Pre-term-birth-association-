# mtDNA Pre-term birtha association



## Run Haplogrep3 to assign haplogroups to samples.


Install Haplogrep3
```
wget https://github.com/genepi/haplogrep3/releases/download/v3.2.1/haplogrep3-3.2.1-linux.zip
unzip haplogrep3-3.2.1-linux.zip

```

Use tree "rCRS PhyloTree 17.2" and  Kulczynski Distance function. Run this on outmtDNA.vcf. Outputs haplogroups to haplogrep3OUT. 

```
./haplogrep3 classify  --extend-report --tree phylotree-rcrs@17.2 --in outmtDNA.vcf --out haplogrep3OUT
```

## Metadata curration
MetadataMerge.py: 

plink2VCF.sh

## PCA and MDS 

```
#Run plink PCA and MDS South Asian
cat Metadata.C.tsv | grep 'GAPPS-Bangladesh\|AMANHI-Pakistan\|AMANHI-Bangladesh'  | awk -F'\t' '{print $NF}'  > listC
#Extract nt DNA SNPs for each sample in list
bcftools view -S listC plink2.C.vcf > plink2.C.SouthAsian.vcf
plink --vcf plink2.C.SouthAsian.vcf --pca --double-id --out PCA-MDS/SouthAsian_C
plink --vcf plink2.C.SouthAsian.vcf --cluster --mds-plot 5 --double-id --out PCA-MDS/SouthAsian_C
rm plink2.C.SouthAsian.vcf


#Run plink PCA and MDS African
cat Metadata.C.tsv | grep 'AMANHI-Pemba\|GAPPS-Zambia'  | awk -F'\t' '{print $NF}'  > listC
#Extract nt DNA SNPs for each sample in list
bcftools view -S listC plink2.C.vcf > plink2.C.African.vcf
plink --vcf plink2.C.African.vcf --pca --double-id --out PCA-MDS/African_C
plink --vcf plink2.C.African.vcf --cluster --mds-plot 5 --double-id --out PCA-MDS/African_C
rm plink2.C.African.vcf

#Run plink PCA and MDS All
plink --vcf plink2.C.vcf --pca --double-id --out PCA-MDS/All_C
plink --vcf plink2.C.vcf --cluster --mds-plot 5 --double-id --out PCA-MDS/All_C
rm listC
```

