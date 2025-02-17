import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import os

# Load the dataset
df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')


# Replace specific missing value codes with NaN for cleaning
df['SNIFF_FREQ'] = df['SNIFF_FREQ'].replace({-88: 0, -77: 0})

# Define continuous variables
continuous_vars = ['PW_AGE', "SNIFF_FREQ", 'PW_EDUCATION', 'MAT_HEIGHT',
                   "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                   "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19",
                   "PC20", "C1", "C2", "C3", "C4", "C5"]

# Filter for continuous variables
df = df[continuous_vars + ['PTB', 'GAGEBRTH']]

# Drop rows with values < 0 for specific columns
columns_to_check = ['PW_AGE', "SNIFF_FREQ", 'PW_EDUCATION', 'MAT_HEIGHT']
df = df[df[columns_to_check].ge(0).all(axis=1)]

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

# Analyze Point-Biserial Correlation for PTB (Binary Target)
print("\nPoint-Biserial Correlation with PTB (Binary Target):")
p_values_ptb = []
for col in continuous_vars:
    r, p = pointbiserialr(dfCont['PTB'], dfCont[col])
    p_values_ptb.append(p)
    print(f"{col}: r = {r:.4f}, p = {p:.4e}")

# Apply Bonferroni Correction for PTB
corrected_ptb = multipletests(p_values_ptb, alpha=0.05, method='bonferroni')
print("\nCorrected P-Values (Bonferroni) for PTB:")
for col, p, p_corr, sig in zip(continuous_vars, p_values_ptb, corrected_ptb[1], corrected_ptb[0]):
    print(f"{col}: raw p = {p:.4e}, corrected p = {p_corr:.4e}, significant = {sig}")

# Analyze Pearson Correlation for GAGEBRTH (Continuous Target)
print("\nPearson Correlation with GAGEBRTH (Continuous Target):")
p_values_ga = []
for col in continuous_vars:
    r, p = pearsonr(dfCont['GAGEBRTH'], dfCont[col])
    p_values_ga.append(p)
    print(f"{col}: r = {r:.4f}, p = {p:.4e}")

# Apply Bonferroni Correction for GAGEBRTH
corrected_ga = multipletests(p_values_ga, alpha=0.05, method='bonferroni')
print("\nCorrected P-Values (Bonferroni) for GAGEBRTH:")
for col, p, p_corr, sig in zip(continuous_vars, p_values_ga, corrected_ga[1], corrected_ga[0]):
    print(f"{col}: raw p = {p:.4e}, corrected p = {p_corr:.4e}, significant = {sig}")

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
print(vif_data.sort_values(by='VIF', ascending=False))

# Filter out high-VIF variables (>10 suggests multicollinearity issues)
high_vif = vif_data[vif_data['VIF'] > 10]
print("\nHigh VIF Variables (Consider Dropping or Combining):")
print(high_vif)

# Final filtered variables
final_vars = [col for col in continuous_vars if col not in high_vif['Variable'].tolist()]
print("\nFinal Set of Variables for Modeling:")
print(final_vars)
