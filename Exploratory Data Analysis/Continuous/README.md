- Takes in Metadata.M.Final.tsv

- Replace SNIFF_FREQ (-88: 0, -77) with  0.


- Define continuous columns and drop rows with values < 0 (in any).

- Standardize continuous variables with StandardScaler.


- Compute and visualize correlation matrix for all variables and PTB and GAGEBRTH.############################


- Analyze Point-Biserial Correlation for PTB (Binary Target)
	- Apply Bonferroni Correction for PTB




- Analyze Pearson Correlation for GAGEBRTH (Continuous Target)
	- Apply Bonferroni Correction for GAGEBRTH

- Visualize relationships for significant variables (for predicting PTB and GAGEBRTH)

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
