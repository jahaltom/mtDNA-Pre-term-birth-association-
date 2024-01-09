

#Loop through all meg files. 
with open('list') as infile: 
    for file in infile:      
        
        
        
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
                    seqN[indx.index(int(line.replace("\n", "")))]=seq[indx.index(int(line.replace("\n", "")))]
                    
        
        
        
        #Remove - and make fasta!
        mystr=""
        for c in seqN:
            if c != "-":
                mystr+=c
        
        fasta=open(file.split("-")[0]+".fasta", "w")
        fasta.write(">"+file.split("-")[0])
        fasta.write('\n')
        fasta.write(mystr)
        fasta.write('\n')
        


