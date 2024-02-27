haplogrep3OUT: Download 17,830 mtDNA complete genomes in fasta format from NCBI and run through Haplogrep3.


GetSpikes.py: Extract the 19 haplogroups of interest ["H","M","L3","L2","L0","L1","U","D","R","L4","T","F","A","C","J","N","G","E","W"].Get only high quality calls (>=0.90) that are 1-16569 in length. Makes Haplogroups.tsv. Takes in haplogrep3OUT. (n=11,477)


#Run list of 11,477 accession numbers (Haplogroups.tsv) through efetch. Generates single fasta with 11,477 seqs.  
```
efetch -db nucleotide -id $(tr '\n' , < list | sed 's/,$/\n/') -format fasta > seqs.fa
```

#Takes fasta from NCBI that has 11,477 seqs and makes a fasta for each seq. 
```
cat seqs.fa | awk '{
        if (substr($0, 1, 1)==">") {filename=(substr($1,2) ".fasta")}
        print $0 >> filename
        close(filename)
}'
```


#For each fasta, combines with NC_012920.1.fasta (Revised Cambridge Reference Sequence (rCRS)), and performs MEGA alignment. Outputs meg and summary file. file.mao needed to MEGA specs. 
```
cat fastaFiles | while read i;do
	cat NC_012920.1.fasta fastas/$i > fastas2/$i 
	../../tools/MEGA/megacc -a file.mao -d fastas2/$i
done
```

#"FAILED"
SpikeinFasta.py:  Takes in list of meg prefixes (one for each spike-in), meg, and list of positions of interest. Outputs fasta for each spike-in that has positions 
of interest marked and everything else Ns. 
#Combinbe fastas and run through Haplogrep3. 





SpikeinVCF.py: Takes in list of meg prefixes (one for each spike-in), meg, list of positions of interest, and a vcf that contain the positions of interest. Outputs VCF for each spike-in that has corresponding ALT and GT positions marked;
* GT=1/1: ALT "*" for deltion, or if ALT diff from REF.
* GT=0/0: if ALT match REF (ALT will be "."). 



#run VCFs through Haplogrep3

#Only 11,381/11,477 were able to have haplogroup called on vcf because some had NTs like Y.

haplogrepCompOUT: Haploghrep3 results from whole mtDNA (Haplogroups.tsv) combined with haplogrep results from positons of interest. 


ConfPlot.py: Takes in haplogrepCompOUT and makes confusion matrix (ConfPlot.png). Switch between % based and count based CM by switching the #s in ConfPlot.py. Below is set to the % based.  

```
#To make count based CM
#cm = confusion_matrix(predicted,actual,labels=["H","M","L0","L1","L2","L3","L4","U","D","R","T","F","A","C","J","N","G","E","W"])
#To make % based CM
cm = confusion_matrix(predicted,actual,labels=["H","M","L0","L1","L2","L3","L4","U","D","R","T","F","A","C","J","N","G","E","W"],normalize="pred")
cm=cm.round(decimals=2, out=None)
```

![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/Spike-Ins/ConfPlot.png?raw=true)


![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/Spike-Ins/ConfPlot%25.png?raw=true)

