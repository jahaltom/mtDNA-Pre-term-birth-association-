library(ISLR)
library(ggplot2)
library(dplyr)


stats <- function(model) {
        # Get coefficients
        coefficients <- coef(model)
        # Compute odds ratios
        odds_ratios <- exp(coefficients)
        # Compute confidence intervals
        conf_int <- confint(model)
        # Exponentiate confidence intervals
        odds_ratios_conf <- exp(conf_int)  
        results <- cbind(odds_ratio = odds_ratios, lower_CI = odds_ratios_conf[, 1], upper_CI = odds_ratios_conf[, 2])
        options(scipen = 999)
        print(round(results,3))
        options(scipen = 0)
        }






df=read.table("Metadata.M.Final.tsv",header=TRUE,sep = '\t',quote="")
df$PTB <- factor(df$PTB)
print(c("PTB levels ",levels(df$PTB)))

print(c("Population GA","Mother"))
glm.fit=glm(GAGEBRTH~population , data=df  ) 
summary (glm.fit )
print(c("Population PTB(binary)","Mother"))
glm.fit=glm(PTB~population , family="binomial", data=df  )
summary (glm.fit )
stats(glm.fit)

print(c("Site GA","Mother"))
glm.fit=glm(GAGEBRTH~site , data=df  ) 
summary (glm.fit )
print(c("Site PTB(binary)","Mother"))
glm.fit=glm(PTB~site , family="binomial", data=df  )
summary (glm.fit )
stats(glm.fit)
        
df=read.table("Metadata.C.Final.tsv",header=TRUE,sep = '\t',quote="")
df$PTB <- factor(df$PTB)
print(c("PTB levels ",levels(df$PTB)))

print(c("Population GA","Child"))
glm.fit=glm(GAGEBRTH~population , data=df  ) 
summary (glm.fit )
print(c("Population PTB(binary)","Child"))
glm.fit=glm(PTB~population , family="binomial", data=df  )
summary (glm.fit )
stats(glm.fit)
print(c("Site GA","Child"))
glm.fit=glm(GAGEBRTH~site , data=df  ) 
summary (glm.fit )
print(c("Site PTB(binary)","Child"))
glm.fit=glm(PTB~site , family="binomial", data=df  )
summary (glm.fit )
stats(glm.fit)

model <- function(pop,ref,type,CoM) {
    
    df=read.table(paste("Metadata.",CoM,".Final.tsv",sep=""),header=TRUE,sep = '\t',quote="")
    df$PTB <- factor(df$PTB)
    print(c("PTB levels ",levels(df$PTB)))
        
    if (type == 1) {

    } else {
      df=df[df$population==pop,]
      df <- df %>%
        group_by(MainHap) %>%
        filter(n() >= 10) %>%
        ungroup()  

        
        }
      
      
     

    #Set reference haplogroups
    df$MainHap= relevel(factor(df$MainHap), ref=ref)
    
    
    
    
    pca=as.formula(paste("GAGEBRTH ~ MainHap + PC1_", pop,"+ PC2_", pop, "+ PC3_", pop, "+ PC4_", pop, "+ PC5_", pop,sep=""))
    mds=as.formula(paste("GAGEBRTH ~ MainHap + C1_", pop,"+ C2_", pop, "+ C3_", pop, "+ C4_", pop, "+ C5_", pop,sep=""))
    
        
    pcaBi=as.formula(paste("PTB ~ MainHap + PC1_", pop,"+ PC2_", pop, "+ PC3_", pop, "+ PC4_", pop, "+ PC5_", pop,sep=""))
    mdsBi=as.formula(paste("PTB ~ MainHap + C1_", pop,"+ C2_", pop, "+ C3_", pop, "+ C4_", pop, "+ C5_", pop,sep=""))
    
    
    
    
    x=paste("Population:",pop, "Reference:", ref,"Child or Mother:", CoM,sep=" ")
        
    print(x)
    glm.fit=glm( pca , data=df  )
    print(summary (glm.fit ))
    
    print(x)
    glm.fit=glm( pcaBi , family="binomial", data=df  )
    print(summary (glm.fit ))
    stats(glm.fit)

    print(x)
    glm.fit=glm( mds , data=df  )
    print(summary (glm.fit ))
    
    print(x)
    glm.fit=glm( mdsBi , family="binomial", data=df  )
    print(summary (glm.fit ))
    stats(glm.fit)
   
}   


    
model("All","M",1,"C")   
model("All","L3",1,"C")  
model("SouthAsian","M",2,"C")  
model("African","L3",2,"C")  
    
model("All","M",1,"M")   
model("All","L3",1,"M")  
model("SouthAsian","M",2,"M")  
model("African","L3",2,"M")  
    
    
    
    
    
    
    
    
    
    
    # popsSA=c("AMANHI-Bangladesh","AMANHI-Pakistan","GAPPS-Bangladesh")
    # popsAf=c("AMANHI-Pemba","GAPPS-Zambia")
    
    

    
    
    
    
    # #Haplogroup and ga
    # png(paste("GAMainbHapJitterPTB.png",sep="_"),width=10,height=15,units="in",res=300)    
    # print(ggplot(data=df, aes(x=MainHap, y=PTB)) +
    # geom_violin(aes(fill=MainHap, color=MainHap)) +
    # geom_crossbar(stat="summary", fun.y=mean, fun.ymax=mean, fun.ymin=mean, fatten=2, width=.5) +
    # geom_point(color="black", size=1, position = position_jitter(w=0.05)) +
    # theme_minimal())    
    # dev.off()  
    
    
    
    # png(paste("GASubbHapJitterPTB.png",sep="_"),width=30,height=15,units="in",res=300)    
    # print(ggplot(data=df, aes(x=SubHap, y=PTB)) +
    # geom_violin(aes(fill=SubHap, color=SubHap)) +
    # geom_crossbar(stat="summary", fun.y=mean, fun.ymax=mean, fun.ymin=mean, fatten=2, width=.5) +
    # geom_point(color="black", size=1, position = position_jitter(w=0.05)) +
    # theme_minimal())    
    # dev.off()   
    
    
    # png(paste("GAMainbHapJitterWealthIndex.png",sep="_"),width=10,height=15,units="in",res=300)
    # print(ggplot(data=df, aes(x=WEALTH_INDEX, y=PTB)) +
    # geom_violin(aes(fill=WEALTH_INDEX, color=WEALTH_INDEX)) +
    # geom_crossbar(stat="summary", fun.y=mean, fun.ymax=mean, fun.ymin=mean, fatten=2, width=.5) +
    # geom_point(color="black", size=1, position = position_jitter(w=0.05)) +
    # theme_minimal())    
    # dev.off()   

  
              










