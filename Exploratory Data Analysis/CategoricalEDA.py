
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, f_oneway, kruskal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Define functions
def cramers_v(contingency_table):
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

# Read dataset
df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')
df = df.dropna(subset=["PTB", "GAGEBRTH"])
df.set_index('Sample_ID', inplace=True)
df['GAGEBRTH'] = pd.to_numeric(df['GAGEBRTH'], errors='coerce')

# Clean and select categorical variables
categorical_columns = ['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
                       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'CHRON_HTN',
                       'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
                       "SNIFF_TOBA", "SMOKE_HIST"]

# Initialize results storage
results = []

# Categorical Analysis
for column in categorical_columns:
    df_cleaned = df[(df[column] != "-77") & (df[column] != "-88")]
    contingency_table = pd.crosstab(df_cleaned[column], df_cleaned['PTB'])

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
    groups = [df_cleaned[df_cleaned[column] == level]['GAGEBRTH'].dropna() for level in df_cleaned[column].unique()]
    if len(groups) > 1:
        anova_p = f_oneway(*groups).pvalue if all(len(g) > 1 for g in groups) else None
        kruskal_p = kruskal(*groups).pvalue if all(len(g) > 1 for g in groups) else None
        results.append((column, 'ANOVA', anova_p, None))
        results.append((column, 'Kruskal-Wallis', kruskal_p, None))

# Remove rows containing -88 or -77 in any column
df=df[['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
                       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'CHRON_HTN',
                       'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
                       "SNIFF_TOBA", "SMOKE_HIST"]]
df = df[~df.isin([-88, -77]).any(axis=1)]

# Multicollinearity Check
df_encoded = pd.get_dummies(df[categorical_columns], drop_first=True).astype(int)
vif_data = pd.DataFrame()
vif_data["Variable"] = df_encoded.columns
vif_data["VIF"] = [variance_inflation_factor(add_constant(df_encoded).values, i) for i in range(len(df_encoded.columns))]

# Visualizations
sns.heatmap(df_encoded.corr(), cmap='coolwarm', center=0, annot=False)
plt.title("Correlation Heatmap for Encoded Categorical Variables")
plt.show()
plt.savefig("CategoricalCorrelationHeatmap.png", bbox_inches="tight")
plt.clf()

# Output results
results_df = pd.DataFrame(results, columns=['Variable', 'Test', 'P-Value', 'Effect Size'])
results_df['Bonferroni_Corrected_P'] = results_df['P-Value'] * len(results_df)
results_df['Significant'] = results_df['Bonferroni_Corrected_P'] < 0.05

# Select Important Variables
results_df['Effect Size'] = results_df['Effect Size'].replace(['None', 'Low counts'], 0)
important_vars = results_df[(results_df['Significant'] == True) ]['Variable'].unique()   # results_df[(results_df['Significant'] == True) & (results_df['Effect Size'] > 0.1)]['Variable'].unique()

# Remove multicollinear variables
threshold_vif = 5
non_multicollinear_vars = vif_data[vif_data['VIF'] < threshold_vif]['Variable'].unique()

# Final Selected Variables
final_vars = [var for var in important_vars if var in non_multicollinear_vars]





# Output Results
print("Significant Variables After Bonferroni Correction and Multicollinearity Check:")
print(final_vars)
results_df.to_csv("Categorical_Analysis_Results.csv", index=False)
vif_data.to_csv("Categorical_Multicollinearity_VIF.csv", index=False)
















