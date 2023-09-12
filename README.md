### mtDNA Pre-term birtha association

momi5.pheno: metadata file

## Genrate mother and child specific ntDNA and mtDNA VCFs from plink files.

# plink2VCF.sh
Requires: 
momi5.clean.bed  momi5.clean.bim  momi5.clean.fam
samplesC.txt: 1909 children  IDs
samplesM.txt: 2176 mother IDs

```
conda activate plink

bash plink2VCF.sh
```
Output VCFs will only contaion SNPs. 
outmtDNA_C.vcf: Only mtDNA (chr26)
outntDNA_C.vcf: Only autosomes (Chrs 1-22)
Same for outmtDNA_M.vcf and outntDNA_M.vcf






