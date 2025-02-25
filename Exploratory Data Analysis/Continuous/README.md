# ContinuousEDA.py
- Takes in Metadata.M.Final.tsv.  Defines continuous columns and drop rows with values < 0 (in any). Standardize continuous variables with StandardScaler.


- Compute and visualize correlation matrix for all variables and PTB and GAGEBRTH. (See plots)


- Analyze Point-Biserial Correlation for PTB (Binary Target)
	- Apply Bonferroni Correction for PTB
```PW_AGE: raw p = 4.3905e-01, corrected p = 1.0000e+00, significant = False
SNIFF_FREQ: raw p = 6.4280e-01, corrected p = 1.0000e+00, significant = False
PW_EDUCATION: raw p = 2.9997e-01, corrected p = 1.0000e+00, significant = False
MAT_HEIGHT: raw p = 3.3894e-05, corrected p = 9.8293e-04, significant = True
PC1: raw p = 4.5784e-19, corrected p = 1.3277e-17, significant = True
PC2: raw p = 9.1147e-02, corrected p = 1.0000e+00, significant = False
PC3: raw p = 1.6933e-02, corrected p = 4.9106e-01, significant = False
PC4: raw p = 5.4418e-02, corrected p = 1.0000e+00, significant = False
PC5: raw p = 3.6000e-02, corrected p = 1.0000e+00, significant = False
PC6: raw p = 3.9585e-03, corrected p = 1.1480e-01, significant = False
PC7: raw p = 7.5677e-01, corrected p = 1.0000e+00, significant = False
PC8: raw p = 5.6856e-01, corrected p = 1.0000e+00, significant = False
PC9: raw p = 2.7423e-01, corrected p = 1.0000e+00, significant = False
PC10: raw p = 8.6467e-01, corrected p = 1.0000e+00, significant = False
PC11: raw p = 4.2865e-01, corrected p = 1.0000e+00, significant = False
PC12: raw p = 4.6764e-01, corrected p = 1.0000e+00, significant = False
PC13: raw p = 6.3004e-01, corrected p = 1.0000e+00, significant = False
PC14: raw p = 1.0237e-01, corrected p = 1.0000e+00, significant = False
PC15: raw p = 3.5818e-01, corrected p = 1.0000e+00, significant = False
PC16: raw p = 3.6794e-01, corrected p = 1.0000e+00, significant = False
PC17: raw p = 8.6324e-01, corrected p = 1.0000e+00, significant = False
PC18: raw p = 8.5892e-01, corrected p = 1.0000e+00, significant = False
PC19: raw p = 9.1979e-01, corrected p = 1.0000e+00, significant = False
PC20: raw p = 8.9468e-01, corrected p = 1.0000e+00, significant = False
C1: raw p = 6.1046e-19, corrected p = 1.7703e-17, significant = True
C2: raw p = 5.1332e-01, corrected p = 1.0000e+00, significant = False
C3: raw p = 2.0252e-03, corrected p = 5.8731e-02, significant = False
C4: raw p = 7.5120e-02, corrected p = 1.0000e+00, significant = False
C5: raw p = 1.3661e-02, corrected p = 3.9615e-01, significant = False
```



- Analyze Pearson Correlation for GAGEBRTH (Continuous Target)
	- Apply Bonferroni Correction for GAGEBRTH
```
PW_AGE: raw p = 1.3234e-03, corrected p = 3.8378e-02, significant = True
SNIFF_FREQ: raw p = 8.7382e-02, corrected p = 1.0000e+00, significant = False
PW_EDUCATION: raw p = 3.3315e-03, corrected p = 9.6612e-02, significant = False
MAT_HEIGHT: raw p = 6.8695e-19, corrected p = 1.9921e-17, significant = True
PC1: raw p = 4.2515e-78, corrected p = 1.2329e-76, significant = True
PC2: raw p = 1.8525e-02, corrected p = 5.3724e-01, significant = False
PC3: raw p = 4.6908e-04, corrected p = 1.3603e-02, significant = True
PC4: raw p = 6.3238e-02, corrected p = 1.0000e+00, significant = False
PC5: raw p = 1.2287e-01, corrected p = 1.0000e+00, significant = False
PC6: raw p = 8.1472e-02, corrected p = 1.0000e+00, significant = False
PC7: raw p = 5.3192e-01, corrected p = 1.0000e+00, significant = False
PC8: raw p = 8.2800e-02, corrected p = 1.0000e+00, significant = False
PC9: raw p = 8.0420e-01, corrected p = 1.0000e+00, significant = False
PC10: raw p = 5.0144e-01, corrected p = 1.0000e+00, significant = False
PC11: raw p = 2.7466e-01, corrected p = 1.0000e+00, significant = False
PC12: raw p = 1.4902e-01, corrected p = 1.0000e+00, significant = False
PC13: raw p = 3.4654e-01, corrected p = 1.0000e+00, significant = False
PC14: raw p = 1.7929e-01, corrected p = 1.0000e+00, significant = False
PC15: raw p = 5.3257e-01, corrected p = 1.0000e+00, significant = False
PC16: raw p = 2.1883e-01, corrected p = 1.0000e+00, significant = False
PC17: raw p = 8.8791e-01, corrected p = 1.0000e+00, significant = False
PC18: raw p = 6.0489e-01, corrected p = 1.0000e+00, significant = False
PC19: raw p = 3.7248e-01, corrected p = 1.0000e+00, significant = False
PC20: raw p = 5.1737e-01, corrected p = 1.0000e+00, significant = False
C1: raw p = 8.9533e-78, corrected p = 2.5965e-76, significant = True
C2: raw p = 3.0205e-01, corrected p = 1.0000e+00, significant = False
C3: raw p = 1.0632e-05, corrected p = 3.0832e-04, significant = True
C4: raw p = 5.4549e-02, corrected p = 1.0000e+00, significant = False
C5: raw p = 1.2587e-01, corrected p = 1.0000e+00, significant = False
```
- Visualize relationships for significant variables (form PTB and GAGEBRTH targets) (See plots)

- VIF Multicollinearity Check
```        Variable          VIF
24            C1  8154.723872
4            PC1  8148.913076
25            C2   207.174707
5            PC2   182.508399
6            PC3   108.253264
7            PC4    78.750677
27            C4    76.646508
26            C3    76.448014
28            C5     2.435633
8            PC5     1.869016
9            PC6     1.290191
2   PW_EDUCATION     1.205701
3     MAT_HEIGHT     1.171760
23          PC20     1.110164
0         PW_AGE     1.093525
1     SNIFF_FREQ     1.063170
18          PC15     1.042610
19          PC16     1.036088
12           PC9     1.033276
13          PC10     1.033208
17          PC14     1.030830
21          PC18     1.020113
14          PC11     1.015443
16          PC13     1.013285
15          PC12     1.009518
22          PC19     1.006936
20          PC17     1.006743
10           PC7     1.004131
11           PC8     1.003318
```
