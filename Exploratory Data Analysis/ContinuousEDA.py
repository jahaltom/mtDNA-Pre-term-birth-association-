import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr, pearsonr

# Load the dataset
df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')

# Drop rows with missing target variables
df = df.dropna(subset=["PTB", "GAGEBRTH"])

# Replace specific missing value codes with NaN for cleaning
df['SNIFF_FREQ'] = df['SNIFF_FREQ'].replace({-88: 0})
df['SNIFF_FREQ'] = df['SNIFF_FREQ'].replace({-77: 0})
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

scaler = StandardScaler()
dfCont[continuous_vars] = scaler.fit_transform(dfCont[continuous_vars])

# Compute and visualize correlation matrix for GA (GAGEBRTH)
plt.figure(figsize=(12, 10))
corr_matrix = dfCont.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Matrix for Continuous Variables")
plt.show()
plt.savefig("GAGEBRTHCorr.png", bbox_inches="tight")
plt.clf()

# Analyze Point-Biserial Correlation for PTB (Binary Target)
print("\nPoint-Biserial Correlation with PTB (Binary Target):")
for col in continuous_vars:
    r, p = pointbiserialr(dfCont['PTB'], dfCont[col])
    print(f"{col}: r = {r:.4f}, p = {p:.4e}")
# MAT_HEIGHT: r = -0.0432, p = 2.9228e-05  *
# PC1: r = 0.0920, p = 5.0557e-19  * 
# PC3: r = -0.0272, p = 8.4650e-03  *
# PC6: r = -0.0309, p = 2.8589e-03 *
# C1: r = -0.0918, p = 6.5128e-19 *
# C3: r = -0.0328, p = 1.5392e-03 *

# Analyze Pearson Correlation for GA (Continuous Target)
print("\nPearson Correlation with GAGEBRTH (Continuous Target):")
ga_corr = dfCont.corr()['GAGEBRTH'].sort_values(ascending=False)
print(ga_corr)
# C1              0.192777  *
# MAT_HEIGHT      0.092833 *
# C3              0.046377 *
# PC3             0.038147 *
# PW_AGE          0.036376 *
# PW_EDUCATION    0.031422 *
# PC1            -0.193098  *


continuous_vars=["MAT_HEIGHT","PW_AGE","PW_EDUCATION","PC1","PC3","PC6","C1","C3"]
# Visualize relationships for significant variables
# Scatter plots for GAGEBRTH (continuous target)
print("\nVisualizing Scatter Plots for GAGEBRTH...")
for col in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df['GAGEBRTH'])
    plt.title(f"{col} vs. GAGEBRTH")
    plt.xlabel(col)
    plt.ylabel("GAGEBRTH (Gestational Age in Days)")
    plt.show()
    plt.savefig("GAGEBRTHScatter"+col+".png", bbox_inches="tight")
    plt.clf()

# Box plots for PTB (binary target)
print("\nVisualizing Box Plots for PTB...")
for col in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['PTB'], y=df[col])
    plt.title(f"{col} vs. PTB")
    plt.xlabel("PTB (0 = Full-term, 1 = Pre-term)")
    plt.ylabel(col)
    plt.show()
    plt.savefig("PTBBox"+col+".png", bbox_inches="tight")
    plt.clf()
