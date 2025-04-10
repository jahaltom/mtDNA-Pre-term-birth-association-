import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, f_oneway, kruskal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import sys

# Define functions
def cramers_v(contingency_table):
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

# Read dataset
df = pd.read_csv("Metadata.Final.tsv", sep='\t')

# Clean and select categorical variables
categorical_columns = sys.argv[1].split(',')

# Initialize results storage
results = []

# Categorical Analysis
for column in categorical_columns:
    contingency_table = pd.crosstab(df[column], df['PTB'])
    # Chi-Square and Fisher's Exact Test
    if (contingency_table.values < 5).any():
        if contingency_table.shape == (2, 2):
            odds_ratio, p_value = fisher_exact(contingency_table)
            results.append((column, 'Fisher', p_value, None))
        else:
            results.append((column, 'Chi2', None, 'Low counts'))
    else:
        chi2, p, _, _ = chi2_contingency(contingency_table)
        cramers_v_value = cramers_v(contingency_table)
        results.append((column, 'Chi2', p, cramers_v_value))
    # ANOVA and Kruskal-Wallis
    groups = [df[df[column] == level]['GAGEBRTH'].dropna() for level in df[column].unique()]
    if len(groups) > 1:
        anova_p = f_oneway(*groups).pvalue if all(len(g) > 1 for g in groups) else None
        kruskal_p = kruskal(*groups).pvalue if all(len(g) > 1 for g in groups) else None
        results.append((column, 'ANOVA', anova_p, None))
        results.append((column, 'Kruskal-Wallis', kruskal_p, None))


# Output results
results_df = pd.DataFrame(results, columns=['Variable', 'Test','P-Value', 'Effect Size'])
# Separate Bonferroni correction for each test type
for test_type in ['Chi2', 'ANOVA', 'Kruskal-Wallis','Fisher']:
    test_mask = (results_df['Test'] == test_type) & (results_df['P-Value'].notna())
    num_tests = test_mask.sum()  # Number of tests for this type
    results_df.loc[test_mask, 'Significant'] = results_df.loc[test_mask, 'P-Value'] < (0.05 / num_tests)
results_df.to_csv("Categorical_Analysis_Results.csv", index=False)


df=df[categorical_columns  + ["PTB","GAGEBRTH"]]



# Multicollinearity Check
df["FUEL_FOR_COOK"] = df["FUEL_FOR_COOK"].astype(str)
df_encoded = pd.get_dummies(df,drop_first=True,dtype=int)
vif_data = pd.DataFrame()
vif_data["Variable"] = df_encoded.columns
vif_data["VIF"] = [variance_inflation_factor(add_constant(df_encoded).values, i) for i in range(len(df_encoded.columns))]
vif_data.to_csv("Categorical_Multicollinearity_VIF.csv", index=False)

# Visualizations
plt.figure(figsize=(12, 10))
df_encoded = pd.get_dummies(df,drop_first=False,dtype=int)
sns.heatmap(df_encoded.corr(), cmap='coolwarm', center=0, annot=False)
plt.title("Correlation Heatmap for Encoded Categorical Variables")
plt.savefig("CategoricalCorrelationHeatmap.png", bbox_inches="tight")
plt.clf()



#### Get counts
# List to store results
results = []

# Loop through each column
for col in categorical_columns:
    unique_values = df[col].drop_duplicates().to_list()
    for value in unique_values:
        # Calculate counts
        ptb_counts = df[df[col] == value]["PTB"].value_counts()
        # Ensure there are no missing categories (0 or 1)
        ptb_counts = ptb_counts.reindex([0, 1], fill_value=0)       
        # Calculate percentage of PTB = 1 relative to all
        percentage = (ptb_counts[1] / (ptb_counts[0] + ptb_counts[1] ) * 100) 
        # Append results as a row
        results.append({
            "Column": col,
            "Value": value,
            "PTB_0_Count": ptb_counts[0],
            "PTB_1_Count": ptb_counts[1],
            "PTB_1_Percentage": percentage
        })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Display the table
results_df.to_csv("Categorical_counts.csv", index=False)


