library(ISLR)
library(ggplot2)

#Read in mother dataset of south asian and african 
df=read.table("MetadataFinal.C.2.tsv",header=TRUE,sep = '\t',quote="")
#Remove any samples with a main and/or sub haplogroup <10.
df=df[!grepl("False", df$IsAtLeast10MainHap),]
df=df[!grepl("False", df$IsAtLeast10SubHap),]


#Set reference haplogroups
df$MainHap= relevel(factor(df$MainHap), ref="H")
df$SubHap= relevel(factor(df$SubHap), ref="H2")

#Fit models predicting gestational age using maim/sub haplogroups, PCA/MDS comps, and sex. 

glm.fit=glm(gday~MainHap + sex + PC1.C_All  + PC2.C_All  + PC3.C_All  + PC4.C_All  + PC5.C_All , data=df  )
summary (glm.fit )



glm.fit=glm(gday~SubHap + sex  + PC1.C_All  + PC2.C_All  + PC3.C_All  + PC4.C_All  + PC5.C_All , data=df  )
summary (glm.fit )




glm.fit=glm(gday~MainHap + sex + C1.C_All  + C2.C_All  + C3.C_All  + C4.C_All  + C5.C_All , data=df  )
summary (glm.fit )



glm.fit=glm(gday~SubHap + sex  + C1.C_All  + C2.C_All  + C3.C_All  + C4.C_All  + C5.C_All  , data=df  )
summary (glm.fit )




#Subset to south asian


dfSA=df[grepl("South_Asian", df$Population),]

#Fit models predicting gestational age using maim/sub haplogroups, PCA/MDS comps, and sex. 

glm.fit=glm(gday~MainHap + sex + PC1.C_SouthAsian  + PC2.C_SouthAsian  + PC3.C_SouthAsian  + PC4.C_SouthAsian  + PC5.C_SouthAsian , data=dfSA  )
summary (glm.fit )


glm.fit=glm(gday~SubHap + sex  + PC1.C_SouthAsian  + PC2.C_SouthAsian  + PC3.C_SouthAsian  + PC4.C_SouthAsian  + PC5.C_SouthAsian , data=dfSA  )
summary (glm.fit )




glm.fit=glm(gday~MainHap + sex + C1.C_SouthAsian  + C2.C_SouthAsian  + C3.C_SouthAsian  + C4.C_SouthAsian  + C5.C_SouthAsian , data=dfSA  )
summary (glm.fit )



glm.fit=glm(gday~SubHap + sex  + C1.C_SouthAsian  + C2.C_SouthAsian  + C3.C_SouthAsian  + C4.C_SouthAsian  + C5.C_SouthAsian  , data=dfSA  )
summary (glm.fit )





#Subset to African


dfAFR=df[grepl("African", df$Population),]

#Fit models predicting gestational age using maim/sub haplogroups, PCA/MDS comps, and sex. 

glm.fit=glm(gday~MainHap + sex + PC1.C_Africa  + PC2.C_Africa  + PC3.C_Africa  + PC4.C_Africa  + PC5.C_Africa , data=dfAFR  )
summary (glm.fit )


glm.fit=glm(gday~SubHap + sex  + PC1.C_Africa  + PC2.C_Africa  + PC3.C_Africa  + PC4.C_Africa  + PC5.C_Africa , data=dfAFR  )
summary (glm.fit )



glm.fit=glm(gday~MainHap + sex + C1.C_Africa  + C2.C_Africa  + C3.C_Africa  + C4.C_Africa  + C5.C_Africa , data=dfAFR  )
summary (glm.fit )



glm.fit=glm(gday~SubHap + sex  + C1.C_Africa  + C2.C_Africa  + C3.C_Africa  + C4.C_Africa  + C5.C_Africa  , data=dfAFR  )
summary (glm.fit )


#Make plots for south asian data 

png(paste("MainHaplogroupVSHeight.C.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= ht)) + geom_boxplot()
dev.off() 

png(paste("MainHaplogroupVSBirthWeight.C.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= bwt)) + geom_boxplot()
dev.off() 

png(paste("MainHaplogroupVSSex.C.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= sex)) + geom_boxplot()
dev.off() 

png(paste("MainHaplogroupVSAge.C.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= age)) + geom_boxplot()
dev.off() 

png(paste("MainHaplogroupVSGday.C.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= gday)) + geom_boxplot()
dev.off() 



