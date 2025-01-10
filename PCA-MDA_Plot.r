

library("ggplot2")

sets=c("M","C")
pops=c("All","African","SouthAsian")

features=c("MainHap","SubHap","site")

for (s in sets){
    for (p in pops){

        #Read in raw counts
        md = read.table(paste("Metadata.",s,".Final.tsv",sep=""),header=TRUE,sep = '\t',quote = "")
            
        
        if (p != "All"){
            md=md[md$population==p,]}
                                      
        eigenvalues = read.table(paste(""PCA-MDS/",p,"_",s,".eigenval",sep=""))
        total_variance = sum(eigenvalues)
        pc_percentage = (eigenvalues[1] / total_variance) * 100
        
        
        
        for (f in features){
        
            # Create PC1 vs PC2 scatter plot with color by region
            ggplot(md, aes(x = md[[paste("PC1_",p,sep="")]], y = md[[paste("PC2_",p,sep="")]], color = md[[f]])) +
              geom_point(size = 1, alpha = 0.8) +  # Scatter plot points
              labs(
                title = "PC1 vs PC2",
                x = paste0("PC1 (", round(pc_percentage[1],1)[1,], "% variance)"),
                y = paste0("PC2 (", round(pc_percentage[1],1)[2,], "% variance)"),
                color = f  # Legend title
              ) 
            
            ggsave(paste(p,s,f,"PCA.png",sep="."))
            
            
            # Create PC1 vs PC2 scatter plot with color by region
            ggplot(md, aes(x = md[[paste("C1_",p,sep="")]], y = md[[paste("C2_",p,sep="")]], color = md[[f]])) +
              geom_point(size = 1, alpha = 0.8) +  # Scatter plot points
              labs(
                title = "C1 vs C2",
                x = paste0("C1"),
                y = paste0("C2"),
                color = f  # Legend title
              ) 
            
            ggsave(paste(p,s,f,"MDS.png",sep="."))
    
        
        }}}















