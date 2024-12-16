
Prepare mtDNA reference NC_012920.1
```
sed -i 's/NC_012920.1 Homo sapiens mitochondrion, complete genome/chrM/g' NC_012920.1.fasta
/scr1/users/haltomj/tools/samtools-1.21/bin/samtools  faidx NC_012920.1.fasta
gatk CreateSequenceDictionary -R NC_012920.1.fasta
```
