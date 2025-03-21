# ContinuousEDA.py
- Takes in Metadata.Final.tsv and continuous columns. Standardize continuous variables with StandardScaler.


- Compute and visualize correlation matrix for continuous variables (including haplogroup, PTB and GAGEBRTH). (See ContinuousCorrelationHeatmap.png in plots)


- Analyze Point-Biserial Correlation for PTB (Binary Target)
	- Apply Bonferroni Correction (Point-Biserial-PTB.csv)


- Analyze Pearson Correlation for GAGEBRTH (Continuous Target)
	- Apply Bonferroni Correction (PearsonCorr-GAGEBRTH.csv)

- Visualize relationships for significant variables (from PTB and GAGEBRTH targets) (See plots)

- VIF Multicollinearity Check (Continuous_Multicollinearity_VIF.csv)

