# CategoricalEDA.py

- Takes in Metadata.Final.tsv and categorical variables
- For each categorical column:
    - For PTB; 
        - Performs Chi-Square and cramers v.
        
        - Fisher's Exact Test(if contingency_table.values < 5 and contingency_table.shape == (2, 2)  e.g. TB)(DRINKING_SOURCE has a value < 5 but not 2x2. This gets excluded). 
    - For GAGEBRTH;
        - ANOVA and Kruskal-Wallis
          
- Output results:Separate Bonferroni correction for each test type (Categorical_Analysis_Results.csv).
  
- VIF is used to asses each variable for multicollinearity. MainHap and FUEL_FOR_COOK are one-hot encoded and the 1st is dropped. Outputs results in Categorical_Multicollinearity_VIF.csv.
- Pearson correlation: Using same df from above except 1st is not droped. (CategoricalCorrelationHeatmap.png)

- For each categorical variable class, determine the number of pre-term births and normal births (PTB=1 normal=0) and the % of PTB=1.  (Categorical_counts.csv)

  



