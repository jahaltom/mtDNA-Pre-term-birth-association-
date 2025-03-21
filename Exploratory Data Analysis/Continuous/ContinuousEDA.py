import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import os
import sys
 
# Load the dataset
df = pd.read_csv("Metadata.Final.tsv", sep='\t')



# Define continuous variables
continuous_vars = sys.argv[1].split(',') + ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","C1","C2","C3","C4","C5"]

# Subset  continuous variables. Add in haplogroups (one-hot encoded)
haplogroups= pd.get_dummies(df["MainHap"], drop_first=False).columns.to_list()
df = pd.concat([df[continuous_vars + ['PTB', 'GAGEBRTH']], pd.get_dummies(df["MainHap"], drop_first=False).astype(int)],axis=1)



# Create a deep copy of the DataFrame
dfCont = df.copy(deep=True)

# Standardize continuous variables
scaler = StandardScaler()
dfCont[continuous_vars] = scaler.fit_transform(dfCont[continuous_vars])

# Output directory for plots
output_dir = "plots/"
os.makedirs(output_dir, exist_ok=True)

# Compute and visualize correlation matrix for GAGEBRTH
plt.figure(figsize=(12, 10))
corr_matrix = dfCont.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Matrix for Continuous Variables")
plt.tight_layout()
plt.savefig(f"{output_dir}ContinuousCorrelationHeatmap.png")
plt.close()


# Drop the haplogroup columns
dfCont.drop(columns=haplogroups, axis=1, inplace=True) 



# Analyze Point-Biserial Correlation for PTB (Binary Target)
print("\nPoint-Biserial Correlation with PTB (Binary Target):")
p_values_ptb = []
r_values_ptb = []
for col in continuous_vars:
    r, p = pointbiserialr(dfCont['PTB'], dfCont[col])
    p_values_ptb.append(p)
    r_values_ptb.append(r)


# Apply Bonferroni Correction for PTB
corrected_ptb = multipletests(p_values_ptb, alpha=0.05, method='bonferroni')
print("\nCorrected P-Values (Bonferroni):")
df_results = pd.DataFrame({
    'Column': continuous_vars,
    'r_value': r_values_ptb,
    'raw_p_value': p_values_ptb,
    'corrected_p_value': corrected_ptb[1],
    'significant': corrected_ptb[0]
})
df_results.to_csv("Point-Biserial-PTB.csv", index=False)

# Analyze Pearson Correlation for GAGEBRTH (Continuous Target)
print("\nPearson Correlation with GAGEBRTH (Continuous Target):")
p_values_ga = []
r_values_ga = []
for col in continuous_vars:
    r, p = pearsonr(dfCont['GAGEBRTH'], dfCont[col])
    p_values_ga.append(p)
    r_values_ga.append(r)


# Apply Bonferroni Correction for GAGEBRTH
corrected_ga = multipletests(p_values_ga, alpha=0.05, method='bonferroni')
print("\nCorrected P-Values (Bonferroni):")
df_results = pd.DataFrame({
    'Column': continuous_vars,
    'r_value': r_values_ga,
    'raw_p_value': p_values_ga,
    'corrected_p_value': corrected_ga[1],
    'significant': corrected_ga[0]
})
df_results.to_csv("PearsonCorr-GAGEBRTH.csv", index=False)



# Visualize relationships for significant variables
print("\nVisualizing Significant Variables...")
significant_vars = [col for col, sig in zip(continuous_vars, corrected_ptb[0]) if sig]
significant_vars2 = [col for col, sig in zip(continuous_vars, corrected_ga[0]) if sig]
significant_vars=significant_vars+significant_vars2
significant_vars=set(significant_vars)


for col in significant_vars:
    # Scatter plots for GAGEBRTH
    plt.figure(figsize=(6, 4))
    sns.regplot(x=df[col], y=df['GAGEBRTH'], scatter_kws={'alpha': 0.6})
    plt.title(f"{col} vs. GAGEBRTH")
    plt.xlabel(col)
    plt.ylabel("GAGEBRTH (Gestational Age in Days)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}GAGEBRTHScatter_{col}.png")
    plt.close()
    # Box plots for PTB
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['PTB'], y=df[col])
    plt.title(f"{col} vs. PTB")
    plt.xlabel("PTB (0 = Full-term, 1 = Pre-term)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"{output_dir}PTBBox_{col}.png")
    plt.close()

# Multicollinearity Check
print("\nChecking for Multicollinearity using Variance Inflation Factor (VIF):")
vif_data = pd.DataFrame()
vif_data['Variable'] = continuous_vars
vif_data['VIF'] = [variance_inflation_factor(dfCont[continuous_vars].values, i)
                   for i in range(len(continuous_vars))]
vif_data.sort_values(by='VIF', ascending=False).to_csv("Continuous_Multicollinearity_VIF.csv", index=False)


