import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import sys
import os

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


###########################Look at catigorical features PTB 0/1 counts and %s.
#### Get counts
df = filtered_data

# Columns to analyze
columns = sys.argv[2].split(',')

df=df[columns+["PTB"]]


# List to store results
results = []

# Loop through each column
for col in columns:
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

############
##########Look at continuous columns
# Output directory for plots
output_dir = "plotsAll/"
os.makedirs(output_dir, exist_ok=True)

for col in sys.argv[3].split(','):
    # Scatter plots for GAGEBRTH
    plt.figure(figsize=(6, 4))
    sns.regplot(x=df[col], y=df['GAGEBRTH'], scatter_kws={'alpha': 0.6})
    plt.title(f"{col} vs. GAGEBRTH")
    plt.xlabel(col)
    plt.ylabel("GAGEBRTH (Gestational Age in Days)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}GAGEBRTHScatter_{col}.All.png")
    plt.close()
    # Box plots for PTB
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['PTB'], y=df[col])
    plt.title(f"{col} vs. PTB")
    plt.xlabel("PTB (0 = Full-term, 1 = Pre-term)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"{output_dir}PTBBox_{col}.All.png")
    plt.close()
