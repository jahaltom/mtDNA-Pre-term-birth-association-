# ContinuousEDA.py
- Takes in Metadata.Final.tsv. Defines continuous columns and drop rows with values < 0 (in any). Standardize continuous variables with StandardScaler.


- Compute and visualize correlation matrix for continuous variables (including haplogroup, PTB and GAGEBRTH). (See plots)


- Analyze Point-Biserial Correlation for PTB (Binary Target)
	- Apply Bonferroni Correction for PTB
```
PW_AGE: raw p = 4.5349e-01, corrected p = 1.0000e+00, significant = False
PW_EDUCATION: raw p = 1.5073e-01, corrected p = 1.0000e+00, significant = False
MAT_HEIGHT: raw p = 1.1632e-06, corrected p = 3.3733e-05, significant = True
MAT_WEIGHT: raw p = 1.7579e-01, corrected p = 1.0000e+00, significant = False
PC1: raw p = 4.2603e-20, corrected p = 1.2355e-18, significant = True
PC2: raw p = 3.4349e-02, corrected p = 9.9612e-01, significant = False
PC3: raw p = 8.6886e-03, corrected p = 2.5197e-01, significant = False
PC4: raw p = 6.4067e-02, corrected p = 1.0000e+00, significant = False
PC5: raw p = 2.1788e-02, corrected p = 6.3185e-01, significant = False
PC6: raw p = 4.0194e-03, corrected p = 1.1656e-01, significant = False
PC7: raw p = 8.8525e-01, corrected p = 1.0000e+00, significant = False
PC8: raw p = 4.1182e-01, corrected p = 1.0000e+00, significant = False
PC9: raw p = 5.6154e-01, corrected p = 1.0000e+00, significant = False
PC10: raw p = 4.2747e-01, corrected p = 1.0000e+00, significant = False
PC11: raw p = 3.3874e-01, corrected p = 1.0000e+00, significant = False
PC12: raw p = 7.5768e-01, corrected p = 1.0000e+00, significant = False
PC13: raw p = 3.1362e-01, corrected p = 1.0000e+00, significant = False
PC14: raw p = 1.1394e-01, corrected p = 1.0000e+00, significant = False
PC15: raw p = 3.8547e-01, corrected p = 1.0000e+00, significant = False
PC16: raw p = 7.0891e-01, corrected p = 1.0000e+00, significant = False
PC17: raw p = 6.9101e-01, corrected p = 1.0000e+00, significant = False
PC18: raw p = 9.3641e-01, corrected p = 1.0000e+00, significant = False
PC19: raw p = 9.1051e-01, corrected p = 1.0000e+00, significant = False
PC20: raw p = 8.4637e-01, corrected p = 1.0000e+00, significant = False
C1: raw p = 4.7995e-20, corrected p = 1.3919e-18, significant = True
C2: raw p = 4.1308e-01, corrected p = 1.0000e+00, significant = False
C3: raw p = 5.9847e-04, corrected p = 1.7356e-02, significant = True
C4: raw p = 8.5255e-02, corrected p = 1.0000e+00, significant = False
C5: raw p = 3.0052e-02, corrected p = 8.7152e-01, significant = False
```



- Analyze Pearson Correlation for GAGEBRTH (Continuous Target)
	- Apply Bonferroni Correction 
```
PW_AGE: raw p = 5.3964e-02, corrected p = 1.0000e+00, significant = False
PW_EDUCATION: raw p = 3.9130e-02, corrected p = 1.0000e+00, significant = False
MAT_HEIGHT: raw p = 3.6012e-19, corrected p = 1.0444e-17, significant = True
MAT_WEIGHT: raw p = 7.5754e-18, corrected p = 2.1969e-16, significant = True
PC1: raw p = 2.1727e-65, corrected p = 6.3010e-64, significant = True
PC2: raw p = 1.7099e-08, corrected p = 4.9587e-07, significant = True
PC3: raw p = 5.1557e-06, corrected p = 1.4952e-04, significant = True
PC4: raw p = 2.5542e-03, corrected p = 7.4072e-02, significant = False
PC5: raw p = 4.1478e-02, corrected p = 1.0000e+00, significant = False
PC6: raw p = 2.2304e-02, corrected p = 6.4683e-01, significant = False
PC7: raw p = 2.4366e-01, corrected p = 1.0000e+00, significant = False
PC8: raw p = 6.6890e-02, corrected p = 1.0000e+00, significant = False
PC9: raw p = 4.1314e-01, corrected p = 1.0000e+00, significant = False
PC10: raw p = 1.4243e-01, corrected p = 1.0000e+00, significant = False
PC11: raw p = 7.3696e-02, corrected p = 1.0000e+00, significant = False
PC12: raw p = 1.9032e-01, corrected p = 1.0000e+00, significant = False
PC13: raw p = 8.8386e-02, corrected p = 1.0000e+00, significant = False
PC14: raw p = 1.6750e-01, corrected p = 1.0000e+00, significant = False
PC15: raw p = 3.5712e-01, corrected p = 1.0000e+00, significant = False
PC16: raw p = 3.7748e-01, corrected p = 1.0000e+00, significant = False
PC17: raw p = 9.4997e-01, corrected p = 1.0000e+00, significant = False
PC18: raw p = 5.9827e-01, corrected p = 1.0000e+00, significant = False
PC19: raw p = 4.1950e-01, corrected p = 1.0000e+00, significant = False
PC20: raw p = 3.4582e-01, corrected p = 1.0000e+00, significant = False
C1: raw p = 3.5295e-65, corrected p = 1.0236e-63, significant = True
C2: raw p = 1.2956e-03, corrected p = 3.7574e-02, significant = True
C3: raw p = 1.1823e-10, corrected p = 3.4287e-09, significant = True
C4: raw p = 1.5224e-03, corrected p = 4.4150e-02, significant = True
C5: raw p = 1.6640e-01, corrected p = 1.0000e+00, significant = False
```
- Visualize relationships for significant variables (from PTB and GAGEBRTH targets) (See plots)

- VIF Multicollinearity Check
```
        Variable          VIF
4            PC1  8448.897392
24            C1  8396.317163
25            C2   191.419851
5            PC2   152.924873
6            PC3   105.063749
7            PC4    78.330912
27            C4    75.571074
26            C3    68.917836
28            C5     2.435518
8            PC5     1.941188
11           PC8     1.919410
3     MAT_WEIGHT     1.457882
14          PC11     1.373982
2     MAT_HEIGHT     1.369308
9            PC6     1.348236
16          PC13     1.326099
12           PC9     1.184667
0         PW_AGE     1.148878
1   PW_EDUCATION     1.145333
13          PC10     1.128220
23          PC20     1.110716
19          PC16     1.063892
18          PC15     1.052214
17          PC14     1.046639
15          PC12     1.029486
21          PC18     1.024181
10           PC7     1.013295
20          PC17     1.012730
22          PC19     1.010089
```
