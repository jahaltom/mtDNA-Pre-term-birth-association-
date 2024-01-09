import pandas as pd



#Loop through all meg files. 
with open('list') as infile: 
    for file in infile:           
        #read in orig VCF
        vcfO=pd.read_csv("vcf",sep='\t')
        vcfO=vcfO.drop(columns=['ALT'])
        #To store position and alt from spike in seq
        vcfA=[]       
        #Create a list that has NC_012920.1 bp number and include -. - will be skiped over in bp number (e.g. 1,2,-,3,4)
        indx=[]
        length=0 
        #Get NC_012920.1 seq only
        with open(file.replace("\n", "")+'.meg') as infile: 
            copy = False
            for line in infile:
                if line.startswith("#NC_012920.1"):
                    copy = True
                    continue
                elif line.startswith("#" + file.split("-")[0]):
                    copy = False
                    continue
                elif copy:
                    for c in line:
                        if c != "-" and c != "\n":
                            length=length+1
                            indx.append(length) 
                        elif c == "-":
                            indx.append("-")
     
        ###Other seq   
        #Will be seq of Ns and -. NTs of interst will be marked. 
        seqN=[]
        #Will be other seq and all its NTs
        seq=[]
        #Get other seq only
        with open(file.replace("\n", "")+'.meg') as infile: 
            copy = False
            for line in infile:
                if line.startswith("#"+file.split("-")[0]):
                    copy = True
                    continue
                elif copy:
                    for c in line:
                        if  c != "\n" and c != "-":
                            seqN.append("N")
                            seq.append(c)
                        elif c == "-":
                            seqN.append(c)
                            seq.append(c)  
        #Using positons, marke NTs of interst. 
        #All should be same length
        if len(indx) == len(seqN) == len(seq):  
            with open('positions') as infile: 
                for line in infile:                      
                    vcfA.append([int(line.replace("\n", "")),seq[indx.index(int(line.replace("\n", "")))]])
        #Raname GT column to match spike-in name
        vcfO = vcfO.rename(columns={'30-001098_30-001098_C': file.split("-")[0]})
        #Make spike in df   
        vcfA=pd.DataFrame(vcfA,columns=['POS','ALT'])                      
        #Merge orig vcf with spike in vcf
        vcf=pd.merge(vcfO,vcfA,on=["POS"])
        vcf=vcf[['#CHROM', 'POS', 'ID', 'REF','ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT',
               file.split("-")[0]]]      
        #Change GT by compairing REF to ALT
        vcf.loc[ vcf.REF ==vcf.ALT, file.split("-")[0]] = "0/0"
        vcf.loc[ vcf.REF !=vcf.ALT, file.split("-")[0]] = "1/1"
        # Make ALT . where 0/0
        vcf.loc[ vcf.REF == vcf.ALT, 'ALT'] = "."
        
        vcf.loc[ vcf.ALT == "-", 'ALT'] = "*"
        
        #Write 
        with open(file.split("-")[0]+".vcf", 'w') as fp:
            fp.write("##fileformat=VCFv4.2")
            fp.write('\n')
            vcf.to_csv(fp, sep='\t', index=False)
            
            
        
        
        
    
    
    








