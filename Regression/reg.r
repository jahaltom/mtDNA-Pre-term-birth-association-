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

print("What it is")
table(df$MainHap)


df$PTB <- factor(df$PTB)
print(c("PTB levels ",levels(df$PTB)))


# Initialize Population column if it does not exist
if (!"Population" %in% colnames(df)) {
  df$Population <- NA  # Create Population column with NA values
}

# Assign "African" to Population if site is "AMANHI-Pemba" or "GAPPS-Zambia"
df$Population <- ifelse(df$site %in% c("AMANHI-Pemba", "GAPPS-Zambia"), "African", "South Asian")



print(c("Population GA","Child"))
glm.fit=glm(GAGEBRTH~Population , data=df  ) 
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







  


        

    #Set reference haplogroups
    df$MainHap= relevel(factor(df$MainHap), ref="M")
    
    
    

    glm.fit=glm( GAGEBRTH ~ MainHap + PC1 + PC2 + PC3 + PC4 + PC5  , data=df  )
    print(summary (glm.fit ))
    

    glm.fit=glm( PTB ~ MainHap + PC1 + PC2 + PC3 + PC4 + PC5  , family="binomial", data=df  )
    print(summary (glm.fit ))
    stats(glm.fit)
   


    glm.fit=glm( GAGEBRTH ~ MainHap + C1 + C2 + C3 + C4 + C5  , data=df  )
    print(summary (glm.fit ))
    

    glm.fit=glm( PTB ~ MainHap + C1 + C2 + C3 + C4 + C5  , family="binomial", data=df  )
    print(summary (glm.fit ))
    stats(glm.fit)
   


    
    
    
    
    
    
    








