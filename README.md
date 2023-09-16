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
* outmtDNA.vcf: Only mtDNA (chr26)
* outntDNA_C.vcf and outntDNA_M.vcf: Only autosomes (chr 1-22)
  


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
Takes MDS/PCS files generated above and adds it into the metadata for mother and child. Outputs MetadataFinal.C.2.tsv and MetadataFinal.M.2.tsv.

## Multiple linear regression
Use MetadataFinal.M.2.tsv and MetadataFinal.C.2.tsv for the MLR.

```
> library(ISLR)
> library(ggplot2)

> #Read in mother dataset of south asian and african 
> df=read.table("MetadataFinal.M.2.tsv",header=TRUE,sep = '\t',quote="")
> #Remove any samples with a main and/or sub haplogroup <10.
> df=df[!grepl("False", df$IsAtLeast10MainHap),]
> df=df[!grepl("False", df$IsAtLeast10SubHap),]

> #Set reference haplogroups
> df$MainHap= relevel(factor(df$MainHap), ref="H")
> df$SubHap= relevel(factor(df$SubHap), ref="H2")
```
 
Fit models predicting gestational age using maim/sub haplogroups, PCA/MDS comps, and sex. 
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
Subset to south asian
```
> dfSA=df[grepl("South_Asian", df$Population),]
> 
> #Fit models predicting gestational age using maim/sub haplogroups, PCA/MDS comps, and sex. 
> 
> glm.fit=glm(gday~MainHap + sex + PC1.M_All  + PC2.M_All  + PC3.M_All  + PC4.M_All  + PC5.M_All , data=dfSA  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + PC1.M_All + PC2.M_All + 
    PC3.M_All + PC4.M_All + PC5.M_All, data = dfSA)

Deviance Residuals: 
   Min      1Q  Median      3Q     Max  
-93.98  -11.68    5.21   12.66   34.25  

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 266.9525    12.9662  20.588  < 2e-16 ***
MainHapD     -1.7612     2.9994  -0.587  0.55719    
MainHapF    -13.0224     5.0089  -2.600  0.00944 ** 
MainHapM      0.1588     1.1057   0.144  0.88581    
MainHapR     -2.5143     4.1955  -0.599  0.54909    
MainHapU      1.1902     2.0279   0.587  0.55737    
sex           1.2571     1.0199   1.233  0.21796    
PC1.M_All    81.7783   743.9653   0.110  0.91249    
PC2.M_All   -27.3637    20.8440  -1.313  0.18950    
PC3.M_All    12.0606    23.9932   0.503  0.61529    
PC4.M_All     9.6772    17.9276   0.540  0.58944    
PC5.M_All    27.7779    18.6367   1.490  0.13635    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 319.0448)

    Null deviance: 393604  on 1230  degrees of freedom
Residual deviance: 388916  on 1219  degrees of freedom
  (2 observations deleted due to missingness)
AIC: 10604

Number of Fisher Scoring iterations: 2

```
```
> glm.fit=glm(gday~SubHap + sex  + PC1.M_All  + PC2.M_All  + PC3.M_All  + PC4.M_All  + PC5.M_All , data=dfSA  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + PC1.M_All + PC2.M_All + PC3.M_All + 
    PC4.M_All + PC5.M_All, data = dfSA)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-95.424  -11.739    5.125   12.607   34.415  

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 266.7850    12.9684  20.572  < 2e-16 ***
SubHapD4     -1.7801     2.9942  -0.594  0.55229    
SubHapF1    -13.0885     4.9988  -2.618  0.00895 ** 
SubHapH9     -2.3228     5.4346  -0.427  0.66916    
SubHapM1     -0.3926     5.0057  -0.078  0.93750    
SubHapM2     -4.3035     3.1606  -1.362  0.17358    
SubHapM3     -0.1440     1.3019  -0.111  0.91196    
SubHapM4      1.5356     2.8197   0.545  0.58613    
SubHapM5      4.3158     2.2625   1.908  0.05668 .  
SubHapM6     -6.0952     3.9640  -1.538  0.12440    
SubHapR6     -2.5746     4.1874  -0.615  0.53877    
SubHapU2      3.9383     2.5338   1.554  0.12036    
SubHapU7     -3.1150     3.1199  -0.998  0.31827    
sex           1.3437     1.0202   1.317  0.18805    
PC1.M_All    77.4248   743.9573   0.104  0.91713    
PC2.M_All   -27.5891    20.7991  -1.326  0.18494    
PC3.M_All    15.7292    24.4458   0.643  0.52007    
PC4.M_All     9.7926    17.8914   0.547  0.58425    
PC5.M_All    25.6493    18.6625   1.374  0.16958    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 317.6445)

    Null deviance: 393604  on 1230  degrees of freedom
Residual deviance: 384985  on 1212  degrees of freedom
  (2 observations deleted due to missingness)
AIC: 10606

Number of Fisher Scoring iterations: 2

```
```

> glm.fit=glm(gday~MainHap + sex + C1.M_All  + C2.M_All  + C3.M_All  + C4.M_All  + C5.M_All , data=dfSA  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + C1.M_All + C2.M_All + C3.M_All + 
    C4.M_All + C5.M_All, data = dfSA)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-94.516  -11.600    5.209   12.773   33.638  

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 263.8104    12.3930  21.287  < 2e-16 ***
MainHapD     -2.0164     2.9978  -0.673  0.50132    
MainHapF    -13.1844     5.0212  -2.626  0.00875 ** 
MainHapM      0.2168     1.1078   0.196  0.84490    
MainHapR     -3.1181     4.1780  -0.746  0.45561    
MainHapU      1.1380     2.0307   0.560  0.57529    
sex           1.3268     1.0211   1.299  0.19406    
C1.M_All     51.0096   377.5416   0.135  0.89255    
C2.M_All     -9.8141    92.6273  -0.106  0.91564    
C3.M_All     18.3741    84.2625   0.218  0.82742    
C4.M_All    116.6495   167.1698   0.698  0.48544    
C5.M_All     30.4021   106.0856   0.287  0.77448    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 320.0044)

    Null deviance: 393604  on 1230  degrees of freedom
Residual deviance: 390085  on 1219  degrees of freedom
  (2 observations deleted due to missingness)
AIC: 10608

Number of Fisher Scoring iterations: 2

```
```
> glm.fit=glm(gday~SubHap + sex  + C1.M_All  + C2.M_All  + C3.M_All  + C4.M_All  + C5.M_All  , data=dfSA  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + C1.M_All + C2.M_All + C3.M_All + 
    C4.M_All + C5.M_All, data = dfSA)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-95.936  -11.576    5.067   12.706   33.875  

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 263.8130    12.3980  21.279   <2e-16 ***
SubHapD4     -2.0175     2.9929  -0.674   0.5004    
SubHapF1    -13.1896     5.0115  -2.632   0.0086 ** 
SubHapH9     -1.8783     5.4466  -0.345   0.7303    
SubHapM1     -0.4280     5.0157  -0.085   0.9320    
SubHapM2     -4.3220     3.1811  -1.359   0.1745    
SubHapM3     -0.1024     1.3048  -0.078   0.9375    
SubHapM4      1.7030     2.8234   0.603   0.5465    
SubHapM5      4.3544     2.2683   1.920   0.0551 .  
SubHapM6     -5.9930     3.9933  -1.501   0.1337    
SubHapR6     -3.1966     4.1704  -0.766   0.4435    
SubHapU2      3.9085     2.5371   1.541   0.1237    
SubHapU7     -3.1587     3.1251  -1.011   0.3123    
sex           1.4142     1.0216   1.384   0.1665    
C1.M_All     47.5649   377.6103   0.126   0.8998    
C2.M_All     11.3790    94.2559   0.121   0.9039    
C3.M_All     34.3874    84.6681   0.406   0.6847    
C4.M_All     72.1234   167.9160   0.430   0.6676    
C5.M_All     18.0595   106.2525   0.170   0.8651    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 318.638)

    Null deviance: 393604  on 1230  degrees of freedom
Residual deviance: 386189  on 1212  degrees of freedom
  (2 observations deleted due to missingness)
AIC: 10610

Number of Fisher Scoring iterations: 2

```


Subset to African

```
> dfAFR=df[grepl("African", df$Population),]
> 
> #Fit models predicting gestational age using maim/sub haplogroups, PCA/MDS comps, and sex. 
> 
> glm.fit=glm(gday~MainHap + sex + PC1.M_All  + PC2.M_All  + PC3.M_All  + PC4.M_All  + PC5.M_All , data=dfAFR  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + PC1.M_All + PC2.M_All + 
    PC3.M_All + PC4.M_All + PC5.M_All, data = dfAFR)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-91.478   -7.749    2.204    9.743   47.074  

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 261.0540    20.2498  12.892   <2e-16 ***
MainHapL0     1.9358     2.4213   0.799    0.424    
MainHapL1     2.8329     2.4187   1.171    0.242    
MainHapL2     1.7652     2.2625   0.780    0.436    
MainHapL3     0.4677     2.1406   0.218    0.827    
MainHapL4     4.4391     4.2835   1.036    0.300    
sex           0.3757     1.2111   0.310    0.756    
PC1.M_All   287.6330   697.9856   0.412    0.680    
PC2.M_All    46.1668    37.6447   1.226    0.220    
PC3.M_All   125.7464   279.5384   0.450    0.653    
PC4.M_All   221.8328   402.3328   0.551    0.582    
PC5.M_All   113.3673    78.0562   1.452    0.147    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 263.8406)

    Null deviance: 199027  on 737  degrees of freedom
Residual deviance: 191548  on 726  degrees of freedom
  (3 observations deleted due to missingness)
AIC: 6222.9

Number of Fisher Scoring iterations: 2

```
```
> glm.fit=glm(gday~SubHap + sex  + PC1.M_All  + PC2.M_All  + PC3.M_All  + PC4.M_All  + PC5.M_All , data=dfAFR  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + PC1.M_All + PC2.M_All + PC3.M_All + 
    PC4.M_All + PC5.M_All, data = dfAFR)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-92.704   -7.808    1.822    9.925   47.114  

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 262.4042    20.2992  12.927   <2e-16 ***
SubHapH3     -6.4186     4.4966  -1.427    0.154    
SubHapL0a     0.5146     2.6162   0.197    0.844    
SubHapL1b    -4.4886     4.2953  -1.045    0.296    
SubHapL1c     2.5986     2.7063   0.960    0.337    
SubHapL2a     0.3570     2.4658   0.145    0.885    
SubHapL3b    -1.8907     3.0879  -0.612    0.541    
SubHapL3d    -0.4713     2.8840  -0.163    0.870    
SubHapL3e    -0.9869     2.6235  -0.376    0.707    
SubHapL3f     0.2327     4.3798   0.053    0.958    
SubHapL4b     3.0685     4.3906   0.699    0.485    
sex           0.2741     1.2152   0.226    0.822    
PC1.M_All   295.3073   698.7750   0.423    0.673    
PC2.M_All    44.1477    37.7360   1.170    0.242    
PC3.M_All   129.8381   280.1314   0.463    0.643    
PC4.M_All   243.8550   404.1526   0.603    0.546    
PC5.M_All   121.3559    78.1890   1.552    0.121    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 263.7092)

    Null deviance: 199027  on 737  degrees of freedom
Residual deviance: 190134  on 721  degrees of freedom
  (3 observations deleted due to missingness)
AIC: 6227.4

Number of Fisher Scoring iterations: 2

```
```
> glm.fit=glm(gday~MainHap + sex + C1.M_All  + C2.M_All  + C3.M_All  + C4.M_All  + C5.M_All , data=dfAFR  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ MainHap + sex + C1.M_All + C2.M_All + C3.M_All + 
    C4.M_All + C5.M_All, data = dfAFR)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-91.474   -7.811    1.973    9.515   47.660  

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  262.1691    18.3446  14.291   <2e-16 ***
MainHapL0      1.5107     2.4050   0.628   0.5301    
MainHapL1      2.5799     2.4023   1.074   0.2832    
MainHapL2      1.7984     2.2519   0.799   0.4248    
MainHapL3      0.6947     2.1322   0.326   0.7447    
MainHapL4      5.0724     4.2751   1.186   0.2358    
sex            0.2719     1.2005   0.226   0.8209    
C1.M_All    -137.3149   336.7501  -0.408   0.6836    
C2.M_All    -101.6920   832.1197  -0.122   0.9028    
C3.M_All    -316.4648   209.3724  -1.511   0.1311    
C4.M_All    -507.3941   207.8133  -2.442   0.0149 *  
C5.M_All      73.4847   439.0484   0.167   0.8671    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 261.5529)

    Null deviance: 199027  on 737  degrees of freedom
Residual deviance: 189887  on 726  degrees of freedom
  (3 observations deleted due to missingness)
AIC: 6216.4

Number of Fisher Scoring iterations: 2

```
```
> glm.fit=glm(gday~SubHap + sex  + C1.M_All  + C2.M_All  + C3.M_All  + C4.M_All  + C5.M_All  , data=dfAFR  )
> summary (glm.fit )
```
```
Call:
glm(formula = gday ~ SubHap + sex + C1.M_All + C2.M_All + C3.M_All + 
    C4.M_All + C5.M_All, data = dfAFR)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-92.583   -7.725    2.065    9.518   47.657  

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)  263.4712    18.3678  14.344   <2e-16 ***
SubHapH3      -7.4719     4.4789  -1.668   0.0957 .  
SubHapL0a     -0.1277     2.5983  -0.049   0.9608    
SubHapL1b     -4.3890     4.2652  -1.029   0.3038    
SubHapL1c      2.0153     2.6949   0.748   0.4548    
SubHapL2a      0.1693     2.4531   0.069   0.9450    
SubHapL3b     -1.7727     3.0740  -0.577   0.5643    
SubHapL3d     -0.8188     2.8749  -0.285   0.7759    
SubHapL3e     -0.8742     2.6035  -0.336   0.7371    
SubHapL3f      0.8053     4.3672   0.184   0.8538    
SubHapL4b      3.4931     4.3749   0.798   0.4249    
sex            0.1694     1.2040   0.141   0.8881    
C1.M_All    -146.2600   336.8893  -0.434   0.6643    
C2.M_All    -103.8826   832.0641  -0.125   0.9007    
C3.M_All    -335.3253   210.8770  -1.590   0.1122    
C4.M_All    -511.2213   209.6822  -2.438   0.0150 *  
C5.M_All      87.2053   442.4292   0.197   0.8438    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 261.3311)

    Null deviance: 199027  on 737  degrees of freedom
Residual deviance: 188420  on 721  degrees of freedom
  (3 observations deleted due to missingness)
AIC: 6220.7

Number of Fisher Scoring iterations: 2



```






















