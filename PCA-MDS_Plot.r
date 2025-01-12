

library("ggplot2")

sets=c("M","C")


features=c("MainHap","SubHap","site")

for (s in sets){


    #Read in raw counts
    md = read.table(paste("Metadata.",s,".Final.tsv",sep=""),header=TRUE,sep = '\t',quote = "")
        
    
    
                                  
    eigenvalues = read.table(paste("PCA-MDS/",s,".eigenval",sep=""))
    total_variance = sum(eigenvalues)
    pc_percentage = (eigenvalues[1] / total_variance) * 100
    
    
    
    for (f in features){
    
        # Create PC1 vs PC2 scatter plot with color by region
        ggplot(md, aes(x = md[[paste("PC1",sep="")]], y = md[[paste("PC2",sep="")]], color = md[[f]])) +
          geom_point(size = 1, alpha = 0.8) +  # Scatter plot points
          labs(
            title = "PC1 vs PC2",
            x = paste0("PC1 (", round(pc_percentage[1],1)[1,], "% variance)"),
            y = paste0("PC2 (", round(pc_percentage[1],1)[2,], "% variance)"),
            color = f  # Legend title
          ) 
        
        ggsave(paste(s,f,"PCA.png",sep="."))
        
        
        # Create PC1 vs PC2 scatter plot with color by region
        ggplot(md, aes(x = md[[paste("C1",sep="")]], y = md[[paste("C2",sep="")]], color = md[[f]])) +
          geom_point(size = 1, alpha = 0.8) +  # Scatter plot points
          labs(
            title = "C1 vs C2",
            x = paste0("C1"),
            y = paste0("C2"),
            color = f  # Legend title
          ) 
        
        ggsave(paste(s,f,"MDS.png",sep="."))
    
        
        }}















