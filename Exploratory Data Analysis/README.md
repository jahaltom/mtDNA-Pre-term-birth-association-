![alt text](https://github.com/jahaltom/mtDNA-Pre-term-birth-association-/blob/main/Exploratory%20Data%20Analysis/MissingDataHeatmap.png?raw=true)





Takes in Metadata.M.Final.tsv 

For each catigorical column, removes missing data.

For PTB; 
Performs Chi-Square and cramers v.

Fisher's Exact Test(if contingency_table.values < 5 and contingency_table.shape == (2, 2)  e.g. TB)
    (DRINKING_SOURCE has a value < 5 but not 2x2. This gets excluded). 



For GAGEBRTH; ANOVA and Kruskal-Wallis



Subsets df to specific columns and removes missing data across the board. 


# Multicollinearity Check

