
### Prepare mtDNA reference NC_012920.1
```
sed -i 's/NC_012920.1 Homo sapiens mitochondrion, complete genome/chrM/g' NC_012920.1.fasta
/scr1/users/haltomj/tools/samtools-1.21/bin/samtools  faidx NC_012920.1.fasta
gatk CreateSequenceDictionary -R NC_012920.1.fasta
```

#### USe GNU parallel to process bam files with GATK. Implements Mutect2 in mitocondrial mode and filters. allele frequency
```
        parallel --jobs 70 \
        """
          #Reheader bam to only have chrM. 
          cat {} | grep -E '^@HD|^@PG|^@RG|^@CO|^@SQ.*SN:chrM|^[^@]' |   /scr1/users/haltomj/tools/samtools-1.21/bin/samtools view -bo {.}.reheader.bam
          /scr1/users/haltomj/tools/samtools-1.21/bin/samtools sort {.}.reheader.bam -o {.}.reheader.sorted.bam
          /scr1/users/haltomj/tools/samtools-1.21/bin/samtools index {.}.reheader.sorted.bam
          
          
          gatk Mutect2 \
           -R /scr1/users/haltomj/PTB/heteroplasmy/NC_012920.1.fasta \
           -I {.}.reheader.sorted.bam \
           --mitochondria-mode \
           -O /scr1/users/haltomj/PTB/heteroplasmy/vcf/{.}.raw.vcf
          
          gatk FilterMutectCalls \
           -R /scr1/users/haltomj/PTB/heteroplasmy/NC_012920.1.fasta \
           -V /scr1/users/haltomj/PTB/heteroplasmy/vcf/{.}.raw.vcf \
           -O /scr1/users/haltomj/PTB/heteroplasmy/vcf/{.}.filtered.vcf \
           --max-events-in-region 2 \
           --min-allele-fraction 0.01 \
           --unique-alt-read-count 3 \
           --min-median-mapping-quality 30

        bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%AF]\t[%DP]\n' /scr1/users/haltomj/PTB/heteroplasmy/vcf/{.}.filtered.vcf > /scr1/users/haltomj/PTB/heteroplasmy/out/{.}.heteroplasmy.txt

        """ ::: *chrM.bam
```
