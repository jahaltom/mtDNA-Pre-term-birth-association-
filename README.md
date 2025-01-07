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
### MetadataMerge.py: 
Takes in Haplogrep3 output and metadata files (momi_combined_data.txt and samples.tab) and performs merge. Filters for only high quality haplogroup calls "Quality">=0.9 and only live births "PREG_OUTCOME"==2. Identifies main and sub haplogroups. 
Seperates mother and child in dataset, then filters to include only main haplogroups with at least 10 occurrences. Writes two tsvs (Metadata.C.tsv and Metadata.M.tsv). Also outputs child.txt and mom.txt which are subset from (Metadata.C.tsv and Metadata.M.tsv) and used for sample selection  plink2VCF.sh.


### plink2VCF.sh: 
Takes in plink files and makes vcfs. Selects for only snps, excludes chrs (x,y,and M), selects for samples from previous dataset (child.txt and mom.txt). Outputs two vcfs (plink2.C.vcf and plink2.M.vcf) that will used below. 
                                                                                                                        


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

