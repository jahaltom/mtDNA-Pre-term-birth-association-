
```r
df=read.table("Metadata.M.Final.tsv",header=TRUE,sep = '\t',quote="")

print("All Haplogroup breakdown")
table(df$MainHap)

print("African Haplogroup breakdown")
table(df[df$population=="African",]$MainHap)

print("South Asian Haplogroup breakdown")
table(df[df$population=="SouthAsian",]$MainHap)
```
```
[1] "All Haplogroup breakdown"

   A    B    C    D    E    F    G    H   HV    I    J    K   L0   L1   L2   L3
  68   13   20  146   59   84   62  103   20   10   60   42  672  380 1110 2049
  L4   L5    M    N    R    T    U    W    X    Z
 143   10 3355   42  359  106  628  131   24   28
[1] "African Haplogroup breakdown"

   B    E    I   L0   L1   L2   L3   L4   L5    M    R    T    U
   9   59    1  667  376 1100 2043  139   10   52    1   24   22
[1] "South Asian Haplogroup breakdown"

   A    B    C    D    F    G    H   HV    I    J    K   L0   L1   L2   L3   L4
  68    4   20  146   84   62  103   20    9   60   42    5    4   10    6    4
   M    N    R    T    U    W    X    Z
3303   42  358   82  606  131   24   28
```



```r
print("Population GA")
glm.fit=glm(GAGEBRTH_IN~population , data=df  ) 
summary (glm.fit )
print("Population PTB(binary)")
glm.fit=glm(PTB~population , family="binomial", data=df  )
summary (glm.fit )
```

```
[1] "Population GA"

Call:
glm(formula = GAGEBRTH_IN ~ population, data = df)

Coefficients:
                     Estimate Std. Error t value Pr(>|t|)
(Intercept)          276.1902     0.2371 1164.93   <2e-16 ***
populationSouthAsian  -4.4289     0.3091  -14.33   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 162.5607)

    Null deviance: 1175369  on 7026  degrees of freedom
Residual deviance: 1141989  on 7025  degrees of freedom
  (2697 observations deleted due to missingness)
AIC: 55721

Number of Fisher Scoring iterations: 2

[1] "Population PTB(binary)"

Call:
glm(formula = PTB ~ population, family = "binomial", data = df)

Coefficients:
                     Estimate Std. Error z value Pr(>|z|)
(Intercept)          -2.92483    0.06843 -42.742   <2e-16 ***
populationSouthAsian  0.73376    0.08246   8.898   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 5270.2  on 9637  degrees of freedom
Residual deviance: 5185.3  on 9636  degrees of freedom
  (86 observations deleted due to missingness)
AIC: 5189.3

Number of Fisher Scoring iterations: 5
```

```r
print("Site GA")
glm.fit=glm(GAGEBRTH_IN~site , data=df  ) 
summary (glm.fit )
print("Site PTB(binary)")
glm.fit=glm(PTB~site , family="binomial", data=df  )
summary (glm.fit )
```
