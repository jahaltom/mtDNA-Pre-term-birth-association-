import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json


with open("../scripts/features.json") as f:
    features = json.load(f)

continuous_columns = features["continuous"]
continuous_columns=continuous_columns + ["PC1", "PC2", "PC3", "PC4", "PC5"]

with open("../scripts/CategoricalVariables.json") as f:
    features = json.load(f)

categorical_columns = features["categorical_to_keep"]


# Load the dataset
df = pd.read_csv("Metadata.Final.tsv", sep='\t')

# Clean the dataset
df=df[categorical_columns + continuous_columns +["PTB", "GAGEBRTH_NEW"]]

df["FUEL_FOR_COOK"] = df["FUEL_FOR_COOK"].astype(str)
# Mixed Feature Correlation: One-Hot Encode categorical features
encoded_df = pd.get_dummies(df[categorical_columns], drop_first=False)
encoded_df = encoded_df.astype(int)
mixed_df = pd.concat([df[continuous_columns + ['GAGEBRTH_NEW','PTB']], encoded_df], axis=1)



# Compute and visualize correlation matrix  (Pearson correlation).
plt.figure(figsize=(20, 15))
corr_matrix = mixed_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Matrix for All Variables")
plt.tight_layout()
plt.show()
plt.savefig("CorrAll.png")
plt.close()
