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



        
CoM="M"


df=read.table(paste("Metadata.",CoM,".Final.tsv",sep=""),header=TRUE,sep = '\t',quote="")

print(paste("What it is" , CoM,sep=""))



df$PTB <- factor(df$PTB)
print(c("PTB levels ",levels(df$PTB)))



df$tab=paste(df$MainHap,df$PTB,sep="_")
table(df$tab)


table(df$MainHap)
df <- df %>%
        group_by(MainHap) %>%
        filter(n() >= 25) %>%
        ungroup()
table(df$MainHap)





# Initialize Population column if it does not exist
if (!"Population" %in% colnames(df)) {
  df$Population <- NA  # Create Population column with NA values
}

# Assign "African" to Population if site is "AMANHI-Pemba" or "GAPPS-Zambia"
df$Population <- ifelse(df$site %in% c("AMANHI-Pemba", "GAPPS-Zambia"), "African", "South Asian")
                        
                        
table(df$Population)
print(c("Population - GA Dataset: ",CoM))
glm.fit=glm(GAGEBRTH~Population , data=df  ) 
summary (glm.fit )
print(c("Population - PTB Dataset: ",CoM))
glm.fit=glm(PTB~Population , family="binomial", data=df  )
summary (glm.fit )
stats(glm.fit)

table(df$site)
df$site= relevel(factor(df$site), ref="AMANHI-Pemba")
print(c("Site - GA Dataset: ",CoM))
glm.fit=glm(GAGEBRTH~site , data=df  ) 
summary (glm.fit )
print(c("Site - PTB Dataset: )",CoM))
glm.fit=glm(PTB~site , family="binomial", data=df  )
summary (glm.fit )
stats(glm.fit)




ref="M" 
#Set reference haplogroups
df$MainHap= relevel(factor(df$MainHap), ref=ref)
print(paste("Hapologroup and PCA - GA, Ref=", ref ," Dataset: ",CoM,sep=""))
glm.fit=glm( GAGEBRTH ~ MainHap + PC1 + PC2 + PC3 + PC4 + PC5 + DIABETES + PW_AGE + MAT_HEIGHT , data=df  )
print(summary (glm.fit ))
print(paste("Hapologroup and PCA - PTB, Ref=", ref ," Dataset: ",CoM,sep=""))
glm.fit=glm( PTB ~ MainHap + PC1 + PC2 + PC3 + PC4 + PC5 + DIABETES + PW_AGE + MAT_HEIGHT , family="binomial", data=df  )
print(summary (glm.fit ))
stats(glm.fit)

# print(paste("Hapologroup and MDS - GA, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( GAGEBRTH ~ MainHap + C1 + C2 + C3 + C4 + C5  , data=df  )
# print(summary (glm.fit ))
# print(paste("Hapologroup and MDS - PTB, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( PTB ~ MainHap + C1 + C2 + C3 + C4 + C5  , family="binomial", data=df  )
# print(summary (glm.fit ))
# stats(glm.fit)
   

ref="HV" 
#Set reference haplogroups
df$MainHap= relevel(factor(df$MainHap), ref=ref)
print(paste("Hapologroup and PCA - GA, Ref=", ref ," Dataset: ",CoM,sep=""))
glm.fit=glm( GAGEBRTH ~ MainHap + PC1 + PC2 + PC3 + PC4 + PC5 + DIABETES + PW_AGE + MAT_HEIGHT , data=df  )
print(summary (glm.fit ))
print(paste("Hapologroup and PCA - PTB, Ref=", ref ," Dataset: ",CoM,sep=""))
glm.fit=glm( PTB ~ MainHap + PC1 + PC2 + PC3 + PC4 + PC5 + DIABETES + PW_AGE + MAT_HEIGHT  , family="binomial", data=df  )
print(summary (glm.fit ))
stats(glm.fit)

# print(paste("Hapologroup and MDS - GA, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( GAGEBRTH ~ MainHap + C1 + C2 + C3 + C4 + C5  , data=df  )
# print(summary (glm.fit ))
# print(paste("Hapologroup and MDS - PTB, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( PTB ~ MainHap + C1 + C2 + C3 + C4 + C5  , family="binomial", data=df  )
# print(summary (glm.fit ))
# stats(glm.fit)
    
    
ref="L0" 
#Set reference haplogroups
df$MainHap= relevel(factor(df$MainHap), ref=ref)
print(paste("Hapologroup and PCA - GA, Ref=", ref ," Dataset: ",CoM,sep=""))
glm.fit=glm( GAGEBRTH ~ MainHap + PC1 + PC2 + PC3 + PC4 + PC5 + DIABETES + PW_AGE + MAT_HEIGHT , data=df  )
print(summary (glm.fit ))
print(paste("Hapologroup and PCA - PTB, Ref=", ref ," Dataset: ",CoM,sep=""))
glm.fit=glm( PTB ~ MainHap + PC1 + PC2 + PC3 + PC4 + PC5 + DIABETES + PW_AGE + MAT_HEIGHT  , family="binomial", data=df  )
print(summary (glm.fit ))
stats(glm.fit)

# print(paste("Hapologroup and MDS - GA, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( GAGEBRTH ~ MainHap + C1 + C2 + C3 + C4 + C5  , data=df  )
# print(summary (glm.fit ))
# print(paste("Hapologroup and MDS - PTB, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( PTB ~ MainHap + C1 + C2 + C3 + C4 + C5  , family="binomial", data=df  )
# print(summary (glm.fit ))
# stats(glm.fit)
    
    
    
    



# table(df$SubHap)
# df <- df %>%
#         group_by(SubHap) %>%
#         filter(n() >= 25) %>%
#         ungroup()

# table(df$SubHap)

# ref="M3" 
# #Set reference haplogroups
# df$SubHap= relevel(factor(df$SubHap), ref=ref)
# print(paste("Hapologroup and PCA - GA, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( GAGEBRTH ~ SubHap + PC1 + PC2 + PC3 + PC4 + PC5  , data=df  )
# print(summary (glm.fit ))
# print(paste("Hapologroup and PCA - PTB, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( PTB ~ SubHap + PC1 + PC2 + PC3 + PC4 + PC5  , family="binomial", data=df  )
# print(summary (glm.fit ))
# stats(glm.fit)
# print(paste("Hapologroup and MDS - GA, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( GAGEBRTH ~ SubHap + C1 + C2 + C3 + C4 + C5  , data=df  )
# print(summary (glm.fit ))
# print(paste("Hapologroup and MDS - PTB, Ref=", ref ," Dataset: ",CoM,sep=""))
# glm.fit=glm( PTB ~ SubHap + C1 + C2 + C3 + C4 + C5  , family="binomial", data=df  )
# print(summary (glm.fit ))
# stats(glm.fit)

