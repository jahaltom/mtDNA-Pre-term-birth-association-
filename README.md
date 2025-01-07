# mtDNA Pre-term birth association



## Run Haplogrep3 to assign haplogroups to samples.

Use tree "rCRS PhyloTree 17.2" and  Kulczynski Distance function. Run this on merged_chrM_22175.vcf. Outputs haplogroups to haplogrep3OUT_22175. 

```
./haplogrep3 classify  --extend-report --tree phylotree-rcrs@17.2 --in merged_chrM_22175.vcf --out haplogrep3OUT_22175
```

## Metadata curration
### MetadataMerge.py: 
Takes in Haplogrep3 output and metadata files (momi_combined_data.txt and samples.tab) and performs merge. Filters for only high quality haplogroup calls "Quality">=0.9 and only live births "PREG_OUTCOME"==2. Identifies main and sub haplogroups. 
Seperates mother and child in dataset, then filters to include only main haplogroups with at least 10 occurrences. Writes two tsvs (Metadata.C.tsv and Metadata.M.tsv). Also outputs child.txt and mom.txt which are subset from (Metadata.C.tsv and Metadata.M.tsv) and used for sample selection  plink2VCF.sh.


### plink2VCF.sh: 
Takes in plink files and makes vcfs. Selects for only snps, excludes chrs (x,y,and M), selects for samples from previous dataset (child.txt and mom.txt). Outputs two vcfs (plink2.C.vcf and plink2.M.vcf) that will used below. 
                                                                                                                        


## PCA and MDS 
Generates PCs and MDS clusters for (plink2.C.vcf and plink2.M.vcf). Does this across all data (All) and South Asian/African seperatly. Below is for the plink2.C.vcf data only. To do the plink2.M.vcf, just swap C and M. 
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
### WeibullFiltering.py
