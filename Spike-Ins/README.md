haplogrep3OUT: Download 17,830 mtDNA complete genomes in fasta format from NCBI and run through haplogrep3


GetSpikes.py: Get n=10 for each of the 19 haplogroups of interest ["H","M","L3","L2","L0","L1","U","D","R","L4","T","F","A","C","J","N","G","E","W"].Get only high quality calls that are 1-16569. Makes Haplogroups.tsv. Takes in haplogrep3OUT.


#Put list of accession numbers into this site (Haplogroups.tsv). Generates single fasta with 190 seqs.  
https://www.ncbi.nlm.nih.gov/sites/batchentrez


#Takes fasta from NCBI that has 190 seqs and makes a fasta for each seq. 
```
cat seqs.fa | awk '{
        if (substr($0, 1, 1)==">") {filename=(substr($1,2) ".fasta")}
        print $0 >> filename
        close(filename)
}'
```


#For each fasta, combines with NC_012920.1.fasta (Revised Cambridge Reference Sequence (rCRS)), and performs MEGA alignment. Outputs meg and summary file. file.mao needed to MEGA specs. 
```
cat fastas | while read i;do
	cat NC_012920.1.fasta test/$i > Fastas/$i 
	../tools/MEGA/megacc -a file.mao -d Fastas/$i
done
```


#SpikeinFasta.py: Takes in list of meg prefixes (one for each spike-in), meg, and list of positions of interest. Outputs fasta for each spike-in that has positions 
of interest marked and everything else Ns. 



#Combinbe fastas and run through haplogrep3. Failed







#SpikeinVCF.py: Takes in list of meg prefixes (one for each spike-in), meg, list of positions of interest, and a vcf that contain the positions of interest. Outputs VCF for each spike-in that has corresponding ALT positions marked;
* 1/1: "*" for deltion, or if ALT diff from REF.
* 0/0: if ALT match REF, ALT is ".". 



#run through haplogrep3


haplogrepCompOUT: Haploghrep3 results from whole mtDNA (Haplogroups.tsv) and positons of interest. 


ConfPlot.py: Takes in haplogrepCompOUT and makes confusion matrix (ConfPlot.png). 
