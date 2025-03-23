import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import sys
import os
import seaborn as sns

# Load metadata
md = pd.read_csv(sys.argv[1],sep='\t')
md=md.dropna(subset=["GAGEBRTH","PTB"])



#All columns we want analized
wantedCol = sys.argv[2].split(',') + sys.argv[3].split(',')

# Apply filtering only in the wantedCol columns. Remove missing data rows. 
md = md[~md[wantedCol].isin([-88, -77,-99]).any(axis=1)]




# Step 1: Fit the Weibull distribution to the data
shape, loc, scale = weibull_min.fit(md["GAGEBRTH"], floc=0)  # Fix location to 0
print(f"Weibull Parameters: Shape={shape}, Scale={scale}, Location={loc}")

# Step 2: Define cutoff thresholds for outlier detection
lower_cutoff = weibull_min.ppf(0.01, shape, loc=loc, scale=scale)  # 1st percentile
upper_cutoff = weibull_min.ppf(0.99, shape, loc=loc, scale=scale)  # 99th percentile


print(f"Lower Cutoff: {lower_cutoff}, Upper Cutoff: {upper_cutoff}")

# Step 3: Filter the data
filtered_data = md[(md["GAGEBRTH"] >= lower_cutoff) & (md["GAGEBRTH"] <= upper_cutoff)]

filtered_data = filtered_data[filtered_data['MainHap'].map(filtered_data['MainHap'].value_counts()) >= 25]


#############################      More filtering
# For each categorical variable class, determine the number of pre-term births and normal births (PTB=1 normal=0) and the % of PTB=1. 
# Remove rows(samples) corresponding to a class from a categorical variable that total counts (PTB=1 normal=0) < 25. If only 1 class would remain after the prior filtering, don't exclude any samples and simply exclude the categorical variable from any future model.
# Reports categorical variables to keep/exclude for future models. Also reports those classes removed due to low counts. 



results = []
troubleClass=[]

# Loop through each column
for col in sys.argv[2].split(','):
    unique_values = filtered_data[col].drop_duplicates().to_list()
    for value in unique_values:
        # Calculate counts
        ptb_counts = filtered_data[filtered_data[col] == value]["PTB"].value_counts()
        # Ensure there are no missing categories (0 or 1)
        ptb_counts = ptb_counts.reindex([0, 1], fill_value=0)
        total_counts = ptb_counts.sum() 
        if total_counts < 25:
            percentage = (ptb_counts[1] / total_counts * 100) 
            troubleClass.append({
                "Column": col,
                "Value": value,
                "PTB_0_Count": ptb_counts[0],
                "PTB_1_Count": ptb_counts[1],
                "PTB_1_Percentage": percentage
            })
        # Only add to results if total counts are >= 25
        if total_counts >= 25:
            percentage = (ptb_counts[1] / total_counts * 100) 
            results.append({
                "Column": col,
                "Value": value,
                "PTB_0_Count": ptb_counts[0],
                "PTB_1_Count": ptb_counts[1],
                "PTB_1_Percentage": percentage
            })
#Unfiltered Categorical Variables
print("Unfiltered Categorical Variables")
print(pd.concat([pd.DataFrame(troubleClass),pd.DataFrame(results)]).sort_values(by=['Column','Value']))
#df for troublesome classes
troubleClass= pd.DataFrame(troubleClass)
print("Troublesome classes") 
print(troubleClass)
##Construct final table for categorical variables
results= pd.DataFrame(results)
# Identify categorical variables where there is only 1 class. These will not be used for future model.
featToExclude = results[~results["Column"].duplicated(keep=False)]["Column"].to_list()
# Remove categorical variables where there is only 1 class. 
final=results[results["Column"].duplicated(keep=False)].sort_values(by=['Column','Value']) 
final.to_csv('CategoricalVariablesToKeepTable.tsv', index=False, sep="\t") 
#df of troublesome classes to remove 
classToRemove = troubleClass[~troubleClass['Column'].isin(featToExclude)]
# Remove unwanted class's
for idx, row in classToRemove.iterrows():
    filtered_data = filtered_data[filtered_data[row['Column']] != row['Value']]
#categorical variables to keep
print("Categorical variables to keep for future model")
print(set(results[results["Column"].duplicated(keep=False)]["Column"].to_list()))
print("Categorical variables excluded from future model")
print(featToExclude)
print("Categorical variable classes removed from data")
print(classToRemove) 



# Count the number of unique classes for each categorical variable
class_counts = results.groupby('Column')['Value'].nunique()
# Identify columns with exactly two unique classes
columns_with_two_classes = class_counts[class_counts == 2].index.tolist()
print("Categorical variables with exactly two classes and continuous variables.  Will be used as continuous variables for Feature selection:", columns_with_two_classes + + sys.argv[3].split(','))
columns_with_moreThantwo_classes = class_counts[class_counts > 2].index.tolist()
print("Categorical variables for Feature selection:", columns_with_moreThantwo_classes)





filtered_data.to_csv('Metadata.Weibull.tsv', index=False, sep="\t") 
filtered_data[["Sample_ID"]].to_csv("IDs.txt", index=False,header=False) 





# Step 4: Plot the original data, filtered data, and Weibull distribution
x = np.linspace(min(md["GAGEBRTH"]), max(md["GAGEBRTH"]), 1000)
weibull_pdf = weibull_min.pdf(x, shape, loc=loc, scale=scale)

plt.figure(figsize=(12, 6))

# Original Data
plt.hist(md["GAGEBRTH"], bins=50, density=True, alpha=0.5, label="Original Data", color="blue")

# Filtered Data
plt.hist(filtered_data["GAGEBRTH"], bins=50, density=True, alpha=0.5, label="Filtered Data", color="green")

# Weibull PDF Line
plt.plot(x, weibull_pdf, 'r-', label="Weibull Fit (PDF)", linewidth=2)

# Add labels, legend, and title
plt.xlabel("Gestational Age (days)")
plt.ylabel("Density")
plt.title("Gestational Age Distribution with Weibull Fit")
plt.legend()
plt.grid(True)
# Add the cutoffs to the plot
plt.axvline(lower_cutoff, color='orange', linestyle='--', label="1st Percentile Cutoff")
plt.axvline(upper_cutoff, color='purple', linestyle='--', label="99th Percentile Cutoff")
plt.show()
plt.savefig("weibullFiltering.png", bbox_inches="tight")
plt.clf()


############
##########Plot continuous features 
# Output directory for plots
output_dir = "plotsAll/"
os.makedirs(output_dir, exist_ok=True)

for col in sys.argv[3].split(','):
    # Scatter plots for GAGEBRTH
    plt.figure(figsize=(6, 4))
    sns.regplot(x=filtered_data[col], y=filtered_data['GAGEBRTH'], scatter_kws={'alpha': 0.6})
    plt.title(f"{col} vs. GAGEBRTH")
    plt.xlabel(col)
    plt.ylabel("GAGEBRTH (Gestational Age in Days)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}GAGEBRTHScatter_{col}.All.png")
    plt.close()
    # Box plots for PTB
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=filtered_data['PTB'], y=filtered_data[col])
    plt.title(f"{col} vs. PTB")
    plt.xlabel("PTB (0 = Full-term, 1 = Pre-term)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"{output_dir}PTBBox_{col}.All.png")
    plt.close()
