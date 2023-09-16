# mtDNA Pre-term birtha association

momi5.pheno: Phenotype file for each pregnancy. Each line is ether a child, mother, or child mother pair. 

## Genrate mother and child specific ntDNA and mtDNA VCFs from plink files.

### plink2VCF.sh
Requires: 
* momi5.clean.bed  momi5.clean.bim  momi5.clean.fam
* samplesC.txt: 1909 children  IDs
* samplesM.txt: 2176 mother IDs

```
conda activate plink

bash plink2VCF.sh
```
Output VCFs will only contaion SNPs. 
* outmtDNA.vcf (both mothers and children): Only mtDNA (chr26)
* outntDNA_C.vcf for child and outntDNA_M.vcf for mother: Only autosomes (chr 1-22)
  


## Run Haplogrep3 to assign haplogroups to samples.
(Doing this on the online Haplogrep3 server gives errors.) Do I need the --chip param? https://haplogrep.readthedocs.io/en/latest/parameters/#parameters

Install Haplogrep3
```
wget https://github.com/genepi/haplogrep3/releases/download/v3.2.1/haplogrep3-3.2.1-linux.zip
unzip haplogrep3-3.2.1-linux.zip

```

Use tree "rCRS PhyloTree 17.2" and  Kulczynski Distance function. Run this on outmtDNA.vcf. Outputs haplogroups to haplogrep3OUT. 

```
./haplogrep3 classify  --extend-report --tree phylotree-rcrs@17.2 --in outmtDNA.vcf --out haplogrep3OUT
```

## Metadata curration
### MetadataMerge.py: 
Combines Haplogrep3 output and momi5.pheno metadata.  Main and sub haplogroups are reported. The result is divided into 2 datasets (mother and child). For each population separately (African,South Asian), samples associated with a main and/or sub haplogroup <10 are marked in the "IsAtLeast10MainHap" and "IsAtLeast10SubHap" columns as False.
 
Outputs MetadataFinal.M.tsv for mother and MetadataFinal.C.tsv for child.


## PCA and MDS 

Starting with the mother dataset, remove any samples with a main and/or sub haplogroup <10. These are marked "False". Then garther PCA/MDS components for each populaton separately and then together. 

```
#Run plink PCA and MDS African
cat MetadataFinal.M.tsv | grep -v "False" | grep "African" | awk -F'\t' '{print $11}'  > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_M.vcf > outntDNA_M.Africa.vcf
plink --vcf outntDNA_M.Africa.vcf --pca --double-id --out Africa_M
plink --vcf outntDNA_M.Africa.vcf --cluster --mds-plot 5 --double-id --out Africa_M

#Run plink PCA and MDS South Asian
cat MetadataFinal.M.tsv | grep -v "False" | grep "South_Asian" | awk -F'\t' '{print $11}'  > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_M.vcf > outntDNA_M.SouthAsian.vcf
plink --vcf outntDNA_M.SouthAsian.vcf --pca --double-id --out SouthAsian_M
plink --vcf outntDNA_M.SouthAsian.vcf --cluster --mds-plot 5 --double-id --out SouthAsian_M



#######################
#Run plink PCA and MDS all populations
cat MetadataFinal.M.tsv | grep -v "False" | grep -v "SampleID" | awk -F'\t' '{print $11}' > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_M.vcf > outntDNA.All.M.vcf
plink --vcf outntDNA.All.M.vcf --pca --double-id --out All_M
plink --vcf outntDNA.All.M.vcf --cluster --mds-plot 5 --double-id --out All_M
```


Do the same for the children 


```
#Run plink PCA and MDS African
cat MetadataFinal.C.tsv | grep -v "False" | grep "African" | awk -F'\t' '{print $11}'  > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_C.vcf > outntDNA_C.Africa.vcf
plink --vcf outntDNA_C.Africa.vcf --pca --double-id --out Africa_C
plink --vcf outntDNA_C.Africa.vcf --cluster --mds-plot 5 --double-id --out Africa_C

#Run plink PCA and MDS South Asian
cat MetadataFinal.C.tsv | grep -v "False" | grep "South_Asian" | awk -F'\t' '{print $11}'  > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_C.vcf > outntDNA_C.SouthAsian.vcf
plink --vcf outntDNA_C.SouthAsian.vcf --pca --double-id --out SouthAsian_C
plink --vcf outntDNA_C.SouthAsian.vcf --cluster --mds-plot 5 --double-id --out SouthAsian_C



#######################
#Run plink PCA and MDS all populations
cat MetadataFinal.C.tsv | grep -v "False" | grep -v "SampleID" | awk -F'\t' '{print $11}' > list
#Extract nt DNA SNPs for each sample in list
bcftools view -S list outntDNA_C.vcf > outntDNA.All.C.vcf
plink --vcf outntDNA.All.C.vcf --pca --double-id --out All_C
plink --vcf outntDNA.All.C.vcf --cluster --mds-plot 5 --double-id --out All_C
```

## Combinbe MDS/PCA data with mother and child metadata.
### Combine.py:
Takes MDS/PCS files generated above and adds it into the metadata (MetadataFinal.M.tsv,MetadataFinal.C.tsv) for mother and child. Outputs MetadataFinal.C.2.tsv and MetadataFinal.M.2.tsv.

## Multiple linear regression (MLR)
Use MetadataFinal.M.2.tsv (mother) and MetadataFinal.C.2.tsv (child) for the MLR. 

Below is an example using the mother dataset, but the child dataset can be done the exact same way. Just simply swap the ".M" for ".C" .  
```
> library(ISLR)
> library(ggplot2)
> 
> #Read in mother dataset of south asian and african 
> df=read.table("MetadataFinal.M.2.tsv",header=TRUE,sep = '\t',quote="")
> #Remove any samples with a main and/or sub haplogroup <10.
> df=df[!grepl("False", df$IsAtLeast10MainHap),]
> df=df[!grepl("False", df$IsAtLeast10SubHap),]
> 
> 
> #Set reference haplogroups
> df$MainHap= relevel(factor(df$MainHap), ref="H")
> df$SubHap= relevel(factor(df$SubHap), ref="H2")
> 
```
Fit models predicting gestational age (days) using main/sub haplogroups, PCA/MDS comps calculated from all populations, and sex. 

### Predicting gestational age (days) using (PCA All Populations, Main Haplogropup, and sex)
```
> glm.fit=glm(gday~MainHap + sex + PC1.M_All  + PC2.M_All  + PC3.M_All  + PC4.M_All  + PC5.M_All , data=df  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + PC1.M_All + PC2.M_All + 
    PC3.M_All + PC4.M_All + PC5.M_All, data = df)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-94.457  -10.454    4.141   11.819   44.828  

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 267.9276     1.4449 185.434  < 2e-16 ***
MainHapD     -1.6030     2.8962  -0.553  0.58000    
MainHapF    -12.9461     4.8532  -2.668  0.00770 ** 
MainHapL0    -0.6492     2.4596  -0.264  0.79187    
MainHapL1     0.6404     2.4887   0.257  0.79695    
MainHapL2    -0.8550     2.2700  -0.377  0.70646    
MainHapL3    -2.3278     2.1182  -1.099  0.27192    
MainHapL4     0.8958     4.4443   0.202  0.84028    
MainHapM      0.4226     1.0688   0.395  0.69261    
MainHapR     -2.8044     4.0581  -0.691  0.48961    
MainHapU      1.5249     1.9627   0.777  0.43729    
sex           0.8026     0.7825   1.026  0.30518    
PC1.M_All   113.0969    41.9538   2.696  0.00708 ** 
PC2.M_All   -13.4768    17.4783  -0.771  0.44076    
PC3.M_All    30.2465    17.4609   1.732  0.08339 .  
PC4.M_All    10.3081    17.3318   0.595  0.55208    
PC5.M_All    29.6608    17.3209   1.712  0.08698 .  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 299.6098)

    Null deviance: 599470  on 1968  degrees of freedom
Residual deviance: 584838  on 1952  degrees of freedom
  (5 observations deleted due to missingness)
AIC: 16835

Number of Fisher Scoring iterations: 2

```
### Predicting gestational age (days) using (PCA All Populations, Sub Haplogropup, and sex)

```
> glm.fit=glm(gday~SubHap + sex  + PC1.M_All  + PC2.M_All  + PC3.M_All  + PC4.M_All  + PC5.M_All , data=df  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + PC1.M_All + PC2.M_All + PC3.M_All + 
    PC4.M_All + PC5.M_All, data = df)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-95.779  -10.393    3.868   11.831   45.095  

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 268.5179     1.4992 179.108  < 2e-16 ***
SubHapD4     -1.6402     2.8927  -0.567  0.57077    
SubHapF1    -13.0270     4.8457  -2.688  0.00724 ** 
SubHapH3     -7.6304     4.7511  -1.606  0.10843    
SubHapH9     -1.9716     5.2643  -0.375  0.70805    
SubHapL0a    -2.1482     2.6429  -0.813  0.41642    
SubHapL1b    -6.8487     4.4878  -1.526  0.12716    
SubHapL1c     0.3121     2.7807   0.112  0.91063    
SubHapL2a    -2.3390     2.4653  -0.949  0.34286    
SubHapL3b    -4.1333     3.1914  -1.295  0.19543    
SubHapL3d    -3.7120     2.8856  -1.286  0.19846    
SubHapL3e    -3.8671     2.6209  -1.475  0.14025    
SubHapL3f    -2.8148     4.5635  -0.617  0.53744    
SubHapL4b    -0.5093     4.5332  -0.112  0.91055    
SubHapM1     -0.4576     4.8519  -0.094  0.92487    
SubHapM2     -4.3582     3.0619  -1.423  0.15480    
SubHapM3      0.1086     1.2599   0.086  0.93130    
SubHapM4      1.6397     2.7333   0.600  0.54864    
SubHapM5      4.8349     2.1862   2.212  0.02711 *  
SubHapM6     -5.7326     3.8378  -1.494  0.13541    
SubHapR6     -2.8662     4.0522  -0.707  0.47945    
SubHapU2      4.0884     2.4558   1.665  0.09612 .  
SubHapU7     -2.5290     3.0193  -0.838  0.40235    
sex           0.8049     0.7833   1.028  0.30430    
PC1.M_All   144.6579    46.9235   3.083  0.00208 ** 
PC2.M_All   -14.2080    17.4565  -0.814  0.41580    
PC3.M_All    34.4422    17.8562   1.929  0.05389 .  
PC4.M_All    10.4036    17.3045   0.601  0.54777    
PC5.M_All    28.0154    17.3485   1.615  0.10650    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 298.5672)

    Null deviance: 599470  on 1968  degrees of freedom
Residual deviance: 579220  on 1940  degrees of freedom
  (5 observations deleted due to missingness)
AIC: 16840

Number of Fisher Scoring iterations: 2

```
### Predicting gestational age (days) using (MDS All Populations, Main Haplogropup, and sex)
```
> glm.fit=glm(gday~MainHap + sex + C1.M_All  + C2.M_All  + C3.M_All  + C4.M_All  + C5.M_All , data=df  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + C1.M_All + C2.M_All + C3.M_All + 
    C4.M_All + C5.M_All, data = df)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-92.595  -10.462    4.052   12.008   46.835  

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  267.2386     1.4553 183.626  < 2e-16 ***
MainHapD      -1.8913     2.8889  -0.655  0.51275    
MainHapF     -12.6572     4.8492  -2.610  0.00912 ** 
MainHapL0      0.7155     2.4938   0.287  0.77421    
MainHapL1      1.8958     2.5155   0.754  0.45114    
MainHapL2      0.7908     2.3321   0.339  0.73456    
MainHapL3     -0.5172     2.1976  -0.235  0.81397    
MainHapL4      3.3680     4.5142   0.746  0.45571    
MainHapM       0.2849     1.0696   0.266  0.78998    
MainHapR      -2.9301     4.0371  -0.726  0.46805    
MainHapU       1.3708     1.9605   0.699  0.48449    
sex            0.9285     0.7813   1.188  0.23479    
C1.M_All     -42.9714    22.9335  -1.874  0.06112 .  
C2.M_All     102.9092    61.1207   1.684  0.09240 .  
C3.M_All      -4.2903    74.5401  -0.058  0.95411    
C4.M_All    -266.6388    98.9496  -2.695  0.00711 ** 
C5.M_All     113.1990    97.0298   1.167  0.24350    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 298.9775)

    Null deviance: 599470  on 1968  degrees of freedom
Residual deviance: 583604  on 1952  degrees of freedom
  (5 observations deleted due to missingness)
AIC: 16831

Number of Fisher Scoring iterations: 2

```
### Predicting gestational age (days) using (MDS All Populations, Sub Haplogropup, and sex)
```
> glm.fit=glm(gday~SubHap + sex  + C1.M_All  + C2.M_All  + C3.M_All  + C4.M_All  + C5.M_All  , data=df  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + C1.M_All + C2.M_All + C3.M_All + 
    C4.M_All + C5.M_All, data = df)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-94.265  -10.475    3.828   11.890   47.187  

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  267.7544     1.5106 177.252  < 2e-16 ***
SubHapD4      -1.9299     2.8854  -0.669  0.50367    
SubHapF1     -12.7283     4.8415  -2.629  0.00863 ** 
SubHapH3      -7.0192     4.7467  -1.479  0.13937    
SubHapH9      -1.8549     5.2611  -0.353  0.72445    
SubHapL0a     -0.6343     2.6777  -0.237  0.81278    
SubHapL1b     -5.1252     4.5131  -1.136  0.25625    
SubHapL1c      1.6489     2.8039   0.588  0.55655    
SubHapL2a     -0.5397     2.5274  -0.214  0.83094    
SubHapL3b     -2.4353     3.2293  -0.754  0.45085    
SubHapL3d     -1.6985     2.9535  -0.575  0.56531    
SubHapL3e     -1.8480     2.6926  -0.686  0.49260    
SubHapL3f     -0.6565     4.6137  -0.142  0.88686    
SubHapL4b      2.1284     4.6053   0.462  0.64402    
SubHapM1      -0.5226     4.8482  -0.108  0.91416    
SubHapM2      -4.4391     3.0711  -1.445  0.14850    
SubHapM3      -0.0205     1.2605  -0.016  0.98703    
SubHapM4       1.8158     2.7291   0.665  0.50592    
SubHapM5       4.6094     2.1862   2.108  0.03513 *  
SubHapM6      -6.6462     3.8454  -1.728  0.08409 .  
SubHapR6      -3.0385     4.0312  -0.754  0.45110    
SubHapU2       4.0155     2.4526   1.637  0.10174    
SubHapU7      -2.8027     3.0168  -0.929  0.35298    
sex            0.9442     0.7821   1.207  0.22750    
C1.M_All     -58.0031    25.5582  -2.269  0.02335 *  
C2.M_All     116.7350    62.4808   1.868  0.06187 .  
C3.M_All       4.8884    74.8037   0.065  0.94790    
C4.M_All    -273.8129    99.3082  -2.757  0.00588 ** 
C5.M_All      97.2147    97.2714   0.999  0.31772    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 297.9139)

    Null deviance: 599470  on 1968  degrees of freedom
Residual deviance: 577953  on 1940  degrees of freedom
  (5 observations deleted due to missingness)
AIC: 16836

Number of Fisher Scoring iterations: 2

```

Subset to South Asian and run same models. Use PCA/MDS comps specific to South Asian.
```
> dfSA=df[grepl("South_Asian", df$Population),]
```
### Predicting gestational age (days) using (PCA South Asian Populations, Main Haplogropup, and sex)
```
> glm.fit=glm(gday~MainHap + sex + PC1.M_SouthAsian  + PC2.M_SouthAsian  + PC3.M_SouthAsian  + PC4.M_SouthAsian  + PC5.M_SouthAsian , data=dfSA  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + PC1.M_SouthAsian + PC2.M_SouthAsian + 
    PC3.M_SouthAsian + PC4.M_SouthAsian + PC5.M_SouthAsian, data = dfSA)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-94.019  -11.701    5.162   12.669   34.230  

Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
(Intercept)      265.5422     1.6933 156.815  < 2e-16 ***
MainHapD          -1.6918     2.9935  -0.565  0.57208    
MainHapF         -12.9959     5.0081  -2.595  0.00957 ** 
MainHapM           0.1627     1.1057   0.147  0.88304    
MainHapR          -2.4755     4.1963  -0.590  0.55535    
MainHapU           1.2006     2.0285   0.592  0.55403    
sex                1.2550     1.0201   1.230  0.21885    
PC1.M_SouthAsian  -7.1842    17.9617  -0.400  0.68925    
PC2.M_SouthAsian  25.2185    18.0516   1.397  0.16266    
PC3.M_SouthAsian   9.7024    17.8837   0.543  0.58755    
PC4.M_SouthAsian -27.0271    17.8668  -1.513  0.13062    
PC5.M_SouthAsian   2.2359    17.8787   0.125  0.90050    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 318.9905)

    Null deviance: 393604  on 1230  degrees of freedom
Residual deviance: 388849  on 1219  degrees of freedom
  (2 observations deleted due to missingness)
AIC: 10604

Number of Fisher Scoring iterations: 2

```

### Predicting gestational age (days) using (PCA South Asian Populations, Sub Haplogropup, and sex)
```
> glm.fit=glm(gday~SubHap + sex  + PC1.M_SouthAsian  + PC2.M_SouthAsian  + PC3.M_SouthAsian  + PC4.M_SouthAsian  + PC5.M_SouthAsian , data=dfSA  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + PC1.M_SouthAsian + PC2.M_SouthAsian + 
    PC3.M_SouthAsian + PC4.M_SouthAsian + PC5.M_SouthAsian, data = dfSA)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-95.458  -11.700    5.127   12.604   34.404  

Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
(Intercept)      265.4502     1.6936 156.737  < 2e-16 ***
SubHapD4          -1.7098     2.9885  -0.572  0.56733    
SubHapF1         -13.0602     4.9980  -2.613  0.00908 ** 
SubHapH9          -2.3382     5.4308  -0.431  0.66687    
SubHapM1          -0.3830     5.0045  -0.077  0.93901    
SubHapM2          -4.3088     3.1580  -1.364  0.17269    
SubHapM3          -0.1404     1.3018  -0.108  0.91411    
SubHapM4           1.5331     2.8191   0.544  0.58668    
SubHapM5           4.3366     2.2625   1.917  0.05551 .  
SubHapM6          -6.1082     3.9597  -1.543  0.12319    
SubHapR6          -2.5359     4.1882  -0.605  0.54496    
SubHapU2           3.9443     2.5334   1.557  0.11976    
SubHapU7          -3.0953     3.1215  -0.992  0.32158    
sex                1.3424     1.0204   1.316  0.18856    
PC1.M_SouthAsian -10.5709    18.3796  -0.575  0.56530    
PC2.M_SouthAsian  25.9015    18.0184   1.437  0.15083    
PC3.M_SouthAsian   9.7955    17.8472   0.549  0.58321    
PC4.M_SouthAsian -25.1552    17.8840  -1.407  0.15981    
PC5.M_SouthAsian   2.6264    17.8556   0.147  0.88309    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 317.5781)

    Null deviance: 393604  on 1230  degrees of freedom
Residual deviance: 384905  on 1212  degrees of freedom
  (2 observations deleted due to missingness)
AIC: 10606

Number of Fisher Scoring iterations: 2

```
### Predicting gestational age (days) using (MDS South Asian Populations, Main Haplogropup, and sex)
```
> glm.fit=glm(gday~MainHap + sex + C1.M_SouthAsian  + C2.M_SouthAsian  + C3.M_SouthAsian  + C4.M_SouthAsian  + C5.M_SouthAsian , data=dfSA  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + C1.M_SouthAsian + C2.M_SouthAsian + 
    C3.M_SouthAsian + C4.M_SouthAsian + C5.M_SouthAsian, data = dfSA)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-93.569  -11.625    5.154   12.585   34.504  

Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
(Intercept)     265.56341    1.68190 157.895  < 2e-16 ***
MainHapD         -2.28021    2.96899  -0.768  0.44263    
MainHapF        -13.16000    4.97955  -2.643  0.00833 ** 
MainHapM         -0.03748    1.09992  -0.034  0.97282    
MainHapR         -3.01119    4.14348  -0.727  0.46753    
MainHapU          0.90162    2.01711   0.447  0.65496    
sex               1.32384    1.01312   1.307  0.19156    
C1.M_SouthAsian  18.95891   64.62013   0.293  0.76927    
C2.M_SouthAsian -28.25840   80.68810  -0.350  0.72624    
C3.M_SouthAsian -63.79492  100.42753  -0.635  0.52540    
C4.M_SouthAsian -61.33993  102.55817  -0.598  0.54989    
C5.M_SouthAsian 472.17552  106.38348   4.438 9.88e-06 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 314.8756)

    Null deviance: 393604  on 1230  degrees of freedom
Residual deviance: 383833  on 1219  degrees of freedom
  (2 observations deleted due to missingness)
AIC: 10588

Number of Fisher Scoring iterations: 2

```
### Predicting gestational age (days) using (MDS South Asian Populations, Sub Haplogropup, and sex)

```
> glm.fit=glm(gday~SubHap + sex  + C1.M_SouthAsian  + C2.M_SouthAsian  + C3.M_SouthAsian  + C4.M_SouthAsian  + C5.M_SouthAsian  , data=dfSA  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + C1.M_SouthAsian + C2.M_SouthAsian + 
    C3.M_SouthAsian + C4.M_SouthAsian + C5.M_SouthAsian, data = dfSA)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-95.097  -11.481    4.925   12.589   32.595  

Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
(Intercept)     265.46925    1.68175 157.853  < 2e-16 ***
SubHapD4         -2.26267    2.96327  -0.764  0.44527    
SubHapF1        -13.15973    4.96804  -2.649  0.00818 ** 
SubHapH9         -1.45370    5.39929  -0.269  0.78779    
SubHapM1          0.03093    4.97508   0.006  0.99504    
SubHapM2         -4.50375    3.16561  -1.423  0.15508    
SubHapM3         -0.36977    1.29503  -0.286  0.77529    
SubHapM4          1.40318    2.80382   0.500  0.61685    
SubHapM5          4.11361    2.24974   1.828  0.06772 .  
SubHapM6         -6.40724    3.93359  -1.629  0.10360    
SubHapR6         -3.11784    4.13467  -0.754  0.45095    
SubHapU2          3.94959    2.51772   1.569  0.11697    
SubHapU7         -3.77455    3.10293  -1.216  0.22405    
sex               1.40531    1.01321   1.387  0.16570    
C1.M_SouthAsian  29.88568   66.17511   0.452  0.65163    
C2.M_SouthAsian -41.57179   80.96229  -0.513  0.60772    
C3.M_SouthAsian -40.65350  100.70681  -0.404  0.68652    
C4.M_SouthAsian -87.11976  102.91583  -0.847  0.39743    
C5.M_SouthAsian 473.78069  106.30015   4.457 9.08e-06 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 313.307)

    Null deviance: 393604  on 1230  degrees of freedom
Residual deviance: 379728  on 1212  degrees of freedom
  (2 observations deleted due to missingness)
AIC: 10589

Number of Fisher Scoring iterations: 2

```


Subset to African and run same models.  Use PCA/MDS comps specific to African.
```
> dfAFR=df[grepl("African", df$Population),]
```
### Predicting gestational age (days) using (PCA African Populations, Main Haplogropup, and sex)
```
> glm.fit=glm(gday~MainHap + sex + PC1.M_Africa  + PC2.M_Africa  + PC3.M_Africa  + PC4.M_Africa  + PC5.M_Africa , data=dfAFR  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + PC1.M_Africa + PC2.M_Africa + 
    PC3.M_Africa + PC4.M_Africa + PC5.M_Africa, data = dfAFR)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-92.541   -7.603    2.113    9.800   47.908  

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  269.1589     2.6090 103.167  < 2e-16 ***
MainHapL0      2.1810     2.3953   0.911  0.36285    
MainHapL1      2.5697     2.3940   1.073  0.28346    
MainHapL2      1.9899     2.2335   0.891  0.37325    
MainHapL3      0.9492     2.1186   0.448  0.65426    
MainHapL4      4.9490     4.2432   1.166  0.24387    
sex            0.2869     1.1945   0.240  0.81027    
PC1.M_Africa -76.8409    16.3119  -4.711 2.96e-06 ***
PC2.M_Africa  19.9759    16.2094   1.232  0.21821    
PC3.M_Africa -30.9407    16.1995  -1.910  0.05653 .  
PC4.M_Africa -51.7693    16.1264  -3.210  0.00138 ** 
PC5.M_Africa -22.1242    16.1702  -1.368  0.17167    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 258.8672)

    Null deviance: 199027  on 737  degrees of freedom
Residual deviance: 187938  on 726  degrees of freedom
  (3 observations deleted due to missingness)
AIC: 6208.8

Number of Fisher Scoring iterations: 2

```
### Predicting gestational age (days) using (PCA African Populations, Sub Haplogropup, and sex)

```
> glm.fit=glm(gday~SubHap + sex  + PC1.M_Africa  + PC2.M_Africa  + PC3.M_Africa  + PC4.M_Africa  + PC5.M_Africa , data=dfAFR  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + PC1.M_Africa + PC2.M_Africa + 
    PC3.M_Africa + PC4.M_Africa + PC5.M_Africa, data = dfAFR)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-93.681   -7.718    1.835    9.785   47.909  

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  270.6655     2.8192  96.009  < 2e-16 ***
SubHapH3      -6.2609     4.4479  -1.408  0.15968    
SubHapL0a      0.7982     2.5874   0.308  0.75780    
SubHapL1b     -3.9454     4.2521  -0.928  0.35379    
SubHapL1c      2.2477     2.6862   0.837  0.40301    
SubHapL2a      0.6170     2.4351   0.253  0.80005    
SubHapL3b     -1.5214     3.0553  -0.498  0.61867    
SubHapL3d     -0.3125     2.8546  -0.109  0.91285    
SubHapL3e     -0.1149     2.6011  -0.044  0.96477    
SubHapL3f      0.2709     4.3375   0.062  0.95021    
SubHapL4b      3.5939     4.3491   0.826  0.40888    
sex            0.2007     1.1991   0.167  0.86710    
PC1.M_Africa -78.9112    16.4627  -4.793 1.99e-06 ***
PC2.M_Africa  18.9101    16.2433   1.164  0.24474    
PC3.M_Africa -30.2603    16.2297  -1.864  0.06266 .  
PC4.M_Africa -52.1842    16.2127  -3.219  0.00135 ** 
PC5.M_Africa -20.1253    16.2663  -1.237  0.21640    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 259.0031)

    Null deviance: 199027  on 737  degrees of freedom
Residual deviance: 186741  on 721  degrees of freedom
  (3 observations deleted due to missingness)
AIC: 6214.1

Number of Fisher Scoring iterations: 2

```
### Predicting gestational age (days) using (MDS African Populations, Main Haplogropup, and sex)

```

> glm.fit=glm(gday~MainHap + sex + C1.M_Africa  + C2.M_Africa  + C3.M_Africa  + C4.M_Africa  + C5.M_Africa , data=dfAFR  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + C1.M_Africa + C2.M_Africa + 
    C3.M_Africa + C4.M_Africa + C5.M_Africa, data = dfAFR)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-92.537   -7.357    2.019    9.849   47.393  

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  269.4697     2.5826 104.339  < 2e-16 ***
MainHapL0      1.7144     2.3804   0.720 0.471632    
MainHapL1      3.0875     2.3790   1.298 0.194766    
MainHapL2      1.5087     2.2244   0.678 0.497823    
MainHapL3      1.0830     2.1119   0.513 0.608240    
MainHapL4      4.5702     4.2262   1.081 0.279873    
sex            0.1239     1.1865   0.104 0.916850    
C1.M_Africa -259.4526    53.3250  -4.865  1.4e-06 ***
C2.M_Africa  -94.0510   106.2891  -0.885 0.376525    
C3.M_Africa  293.5220   108.8393   2.697 0.007162 ** 
C4.M_Africa  -32.5256   110.5803  -0.294 0.768738    
C5.M_Africa -406.7765   112.2942  -3.622 0.000312 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 256.9104)

    Null deviance: 199027  on 737  degrees of freedom
Residual deviance: 186517  on 726  degrees of freedom
  (3 observations deleted due to missingness)
AIC: 6203.2

Number of Fisher Scoring iterations: 2

```
### Predicting gestational age (days) using (MDS African Populations, Sub Haplogropup, and sex)

```
> glm.fit=glm(gday~SubHap + sex  + C1.M_Africa  + C2.M_Africa  + C3.M_Africa  + C4.M_Africa  + C5.M_Africa  , data=dfAFR  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + C1.M_Africa + C2.M_Africa + 
    C3.M_Africa + C4.M_Africa + C5.M_Africa, data = dfAFR)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-93.336   -7.733    1.988    9.852   47.375  

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  270.91356    2.78756  97.187  < 2e-16 ***
SubHapH3      -6.38565    4.42969  -1.442 0.149862    
SubHapL0a      0.32246    2.56969   0.125 0.900174    
SubHapL1b     -2.64535    4.34421  -0.609 0.542758    
SubHapL1c      2.50876    2.65737   0.944 0.345446    
SubHapL2a      0.13434    2.42327   0.055 0.955804    
SubHapL3b     -1.83729    3.04034  -0.604 0.545830    
SubHapL3d     -0.52076    2.84245  -0.183 0.854686    
SubHapL3e      0.30974    2.59775   0.119 0.905122    
SubHapL3f      1.60438    4.34626   0.369 0.712131    
SubHapL4b      3.21088    4.32998   0.742 0.458604    
sex            0.08436    1.19050   0.071 0.943529    
C1.M_Africa -267.47449   53.83973  -4.968 8.46e-07 ***
C2.M_Africa -121.31302  107.41147  -1.129 0.259095    
C3.M_Africa  301.95999  109.61400   2.755 0.006022 ** 
C4.M_Africa  -46.71536  112.28621  -0.416 0.677506    
C5.M_Africa -388.48763  115.04872  -3.377 0.000773 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 257.0866)

    Null deviance: 199027  on 737  degrees of freedom
Residual deviance: 185359  on 721  degrees of freedom
  (3 observations deleted due to missingness)
AIC: 6208.6

Number of Fisher Scoring iterations: 2
```

Make plots for South Asian data. Main haplogroup vs (maternal height in cm, birth weight in gram, sex, maternal age, gestational duration in days).
```


png(paste("MainHaplogroupVSMaternalHeightCM.M.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= ht)) + geom_boxplot()
dev.off() 

png(paste("MainHaplogroupVSBirthWeightGrams.M.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= bwt)) + geom_boxplot()
dev.off() 

png(paste("MainHaplogroupVSSex.M.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= sex)) + geom_bar()
dev.off() 

png(paste("MainHaplogroupVSMaternalAge.M.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= age)) + geom_boxplot()
dev.off() 

png(paste("MainHaplogroupVSGestationalDays.M.png",sep=""),width=10,height=5,units="in",res=1200)
ggplot(dfSA, aes(x = MainHap, y= gday)) + geom_boxplot()
dev.off() 


```

### The MLR.C.r script generates the above for the child data. 

