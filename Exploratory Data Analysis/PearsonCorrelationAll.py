import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# Load the dataset
df = pd.read_csv("Metadata.Final.tsv", sep='\t')

# Clean the dataset
df=df[sys.argv[1].split(',')+sys.argv[2].split(',') + ["PC1", "PC2", "PC3", "PC4", "PC5"]+["PTB", "GAGEBRTH"]]
categorical_columns=sys.argv[1].split(',')
continuous_columns=sys.argv[2].split(',') + ["PC1", "PC2", "PC3", "PC4", "PC5"]


# Mixed Feature Correlation: One-Hot Encode categorical features
encoded_df = pd.get_dummies(df[categorical_columns], drop_first=False)
encoded_df = encoded_df.astype(int)
mixed_df = pd.concat([df[continuous_columns + ['GAGEBRTH','PTB']], encoded_df], axis=1)



# Compute and visualize correlation matrix  (Pearson correlation).
plt.figure(figsize=(20, 15))
corr_matrix = mixed_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Matrix for All Variables")
plt.tight_layout()
plt.show()
plt.savefig("CorrAll.png")
plt.close()
