- Takes in Metadata.M.Final.tsv

- Replace SNIFF_FREQ (-88: 0, -77) with  0.


- Define continuous columns and drop rows with values < 0 (in any).

- Standardize continuous variables with StandardScaler.


- Compute and visualize correlation matrix for all variables and PTB and GAGEBRTH.


- Analyze Point-Biserial Correlation for PTB (Binary Target)
	- Apply Bonferroni Correction for PTB




- Analyze Pearson Correlation for GAGEBRTH (Continuous Target)
	- Apply Bonferroni Correction for GAGEBRTH

- Visualize relationships for significant variables (for predicting PTB and GAGEBRTH)

- VIF Multicollinearity Check
