import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


#################################################Multivariable Logistic Regression

dfIN = pd.read_csv("HetroplasmyNN.C.tsv",sep='\t')


continuous_columns=dfIN.drop(["SampleID"], axis=1).columns.to_list()

# Assuming `vaf_matrix` is your matrix of variant allele frequencies
scaler = StandardScaler()
dfIN[continuous_columns] = scaler.fit_transform(dfIN[continuous_columns])

 
# Load metadata
md = pd.read_csv("Metadata.C.Final.tsv",sep='\t')[["SampleID","PTB"]]
md=md.dropna(subset=["PTB"])
md[["PTB"]]=md[["PTB"]].astype(int)


data_encoded=pd.merge(dfIN,md,on=["SampleID"])

# Features (X) and target (y)
X = data_encoded.drop(["PTB","SampleID"], axis=1)
y = data_encoded["PTB"]




# Prepare to store results
results = []
variant_names = X.columns

# Perform logistic regression for each variant
for variant in variant_names:
    # Define independent variables (allele frequency + PCs)
    X_ = sm.add_constant(pd.concat([X[variant]], axis=1))  # Add constant for intercept    X_ = sm.add_constant(pd.concat([X[variant], pcs_df_standardized], axis=1))
    # Logistic regression model
    model = sm.Logit(y, X_)
    try:
        result = model.fit(disp=False)
        # Extract p-value, effect size (coefficient), and other statistics
        p_value = result.pvalues[variant]
        coef = result.params[variant]
        results.append({"variant": variant, "coef": coef, "p_value": p_value})
    except Exception as e:
        # Handle cases where regression fails (e.g., perfect separation)
        results.append({"variant": variant, "coef": np.nan, "p_value": np.nan})
        print(f"Regression failed for {variant}: {e}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.dropna(subset=["p_value"])

# Adjust p-values for multiple testing using FDR
results_df["adjusted_p_value"] = multipletests(results_df["p_value"], method="fdr_bh")[1]

# Sort by adjusted p-value
results_df = results_df.sort_values("adjusted_p_value")

# Save results to a CSV file
results_df.to_csv("logistic_regression_results.csv", index=False)

# Print top 10 results
print(results_df.head(10))
