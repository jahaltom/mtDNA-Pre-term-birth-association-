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
MetadataMergeAndStats.py: Makes files in Stuff folder which contains files for african and south asian (at main and sub haplotype level) stats files that contain totalbirths, total ptb, total non-ptb. 
Also makes MetadataFinal.tsv which is african and south asian combined containg all haplogroup info. Samples with a main or sub haplogroup <10 are marked other.
 
