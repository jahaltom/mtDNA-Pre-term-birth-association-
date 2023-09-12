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
  


## Run Haplogrep3. 
Install Haplogrep3
```
wget https://github.com/genepi/haplogrep3/releases/download/v3.2.1/haplogrep3-3.2.1-linux.zip
unzip haplogrep3-3.2.1-linux.zip
./haplogrep3
```

Use tree "rCRS PhyloTree 17.2" and  Kulczynski Distance function. Run this on outmtDNA.vcf. Outputs haplogroups to haplogrep3OUT. 

```
/scr1/users/haltomj/tools/haplogrep3/haplogrep3 classify --tree phylotree-rcrs@17.2 --in outmtDNA.vcf --out haplogrep3OUT
```


