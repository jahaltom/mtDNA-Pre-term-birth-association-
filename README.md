# mtDNA Pre-term birtha association

momi5.pheno: Phenotype file for each pregnancy. Each line is ether a child, mother, or child mother pair. 

## Genrate mother and child specific ntDNA and mtDNA VCFs from plink files.

### plink2VCF.sh
Requires: 
* momi5.clean.bed  momi5.clean.bim  momi5.clean.fam
* samplesC.txt: 1909 children  IDs
* samplesM.txt: 2176 mother IDs

```
conda activate plink

bash plink2VCF.sh
```
Output VCFs will only contaion SNPs. 
* outmtDNA.vcf: Only mtDNA (chr26)
* outntDNA_C.vcf and outntDNA_M.vcf: Only autosomes (chr 1-22)
  


## Run Haplogrep3 to assign haplogroups to samples.
(Doing this on the online Haplogrep3 server gives errors.) Do I need the --chip param? https://haplogrep.readthedocs.io/en/latest/parameters/#parameters

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
Combines Haplogrep3 output and momi5.pheno metadata.  Main and sub haplogroups are reported. The result is divided into 2 datasets (mother and child). For each population separately (African,South Asian), samples associated with a main and/or sub haplogroup <10 are marked in the "IsAtLeast10MainHap" and "IsAtLeast10SubHap" columns as False.
 
Outputs MetadataFinal.M.tsv for mother and MetadataFinal.C.tsv for child.


## PCA and MDS 

Starting with the mother dataset, remove any samples with a main and/or sub haplogroup <10. These are marked "False". Then garther PCA/MDS components for each populaton separately and then together. 

```
#Run plink PCA and MDS African
cat MetadataFinal.M.tsv | grep -v "False" | grep "African" | awk -F'\t' '{print $11}'  > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_M.vcf > outntDNA_M.Africa.vcf
plink --vcf outntDNA_M.Africa.vcf --pca --double-id --out Africa_M
plink --vcf outntDNA_M.Africa.vcf --cluster --mds-plot 5 --double-id --out Africa_M

#Run plink PCA and MDS South Asian
cat MetadataFinal.M.tsv | grep -v "False" | grep "South_Asian" | awk -F'\t' '{print $11}'  > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_M.vcf > outntDNA_M.SouthAsian.vcf
plink --vcf outntDNA_M.SouthAsian.vcf --pca --double-id --out SouthAsian_M
plink --vcf outntDNA_M.SouthAsian.vcf --cluster --mds-plot 5 --double-id --out SouthAsian_M



#######################
#Run plink PCA and MDS all populations
cat MetadataFinal.M.tsv | grep -v "False" | grep -v "SampleID" | awk -F'\t' '{print $11}' > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_M.vcf > outntDNA.All.M.vcf
plink --vcf outntDNA.All.M.vcf --pca --double-id --out All_M
plink --vcf outntDNA.All.M.vcf --cluster --mds-plot 5 --double-id --out All_M
```


Do the same for the children 


```
#Run plink PCA and MDS African
cat MetadataFinal.C.tsv | grep -v "False" | grep "African" | awk -F'\t' '{print $11}'  > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_C.vcf > outntDNA_C.Africa.vcf
plink --vcf outntDNA_C.Africa.vcf --pca --double-id --out Africa_C
plink --vcf outntDNA_C.Africa.vcf --cluster --mds-plot 5 --double-id --out Africa_C

#Run plink PCA and MDS South Asian
cat MetadataFinal.C.tsv | grep -v "False" | grep "South_Asian" | awk -F'\t' '{print $11}'  > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_C.vcf > outntDNA_C.SouthAsian.vcf
plink --vcf outntDNA_C.SouthAsian.vcf --pca --double-id --out SouthAsian_C
plink --vcf outntDNA_C.SouthAsian.vcf --cluster --mds-plot 5 --double-id --out SouthAsian_C



#######################
#Run plink PCA and MDS all populations
cat MetadataFinal.C.tsv | grep -v "False" | grep -v "SampleID" | awk -F'\t' '{print $11}' > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_C.vcf > outntDNA.All.C.vcf
plink --vcf outntDNA.All.C.vcf --pca --double-id --out All_C
plink --vcf outntDNA.All.C.vcf --cluster --mds-plot 5 --double-id --out All_C
```

## Combinbe MDS/PCA data with mother and child metadata.
### Combine.py:
Takes MDS/PCS files generated above and adds it into the metadata for mother and child. Outputs MetadataFinal.C.2.tsv and MetadataFinal.M.2.tsv.

## Multiple logistic linear regression
