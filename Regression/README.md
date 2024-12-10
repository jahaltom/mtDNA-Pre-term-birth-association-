# Linear and logistic regression

# Extract sample size for each haplogroup across populations
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


# Use regression models to see population association with gestational age and pre-term birth (PTB)
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
# Use regression models to see site specific association with gestational age and pre-term birth (PTB)
```r
print("Site GA")
glm.fit=glm(GAGEBRTH_IN~site , data=df  ) 
summary (glm.fit )
print("Site PTB(binary)")
glm.fit=glm(PTB~site , family="binomial", data=df  )
summary (glm.fit )
```


```
[1] "Site GA"

Call:
glm(formula = GAGEBRTH_IN ~ site, data = df)

Coefficients:
                     Estimate Std. Error t value Pr(>|t|)
(Intercept)          273.9250     0.4698 583.022  < 2e-16 ***
siteAMANHI-Pakistan   -3.1554     0.6473  -4.874 1.12e-06 ***
siteAMANHI-Pemba       1.7385     0.5576   3.118  0.00183 **
siteGAPPS-Bangladesh  -2.4640     0.5323  -4.629 3.74e-06 ***
siteGAPPS-Zambia       3.1270     0.6068   5.153 2.63e-07 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 161.807)

    Null deviance: 1175369  on 7026  degrees of freedom
Residual deviance: 1136209  on 7022  degrees of freedom
  (2697 observations deleted due to missingness)
AIC: 55691

Number of Fisher Scoring iterations: 2

[1] "Site PTB(binary)"

Call:
glm(formula = PTB ~ site, family = "binomial", data = df)

Coefficients:
                     Estimate Std. Error z value Pr(>|z|)
(Intercept)           -2.8631     0.1266 -22.623  < 2e-16 ***
siteAMANHI-Pakistan    0.8049     0.1519   5.300 1.16e-07 ***
siteAMANHI-Pemba      -0.1938     0.1518  -1.277    0.202
siteGAPPS-Bangladesh   0.8285     0.1407   5.888 3.90e-09 ***
siteGAPPS-Zambia       0.2623     0.1736   1.511    0.131
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 5270.2  on 9637  degrees of freedom
Residual deviance: 5132.7  on 9633  degrees of freedom
  (86 observations deleted due to missingness)
AIC: 5142.7

Number of Fisher Scoring iterations: 5


```
