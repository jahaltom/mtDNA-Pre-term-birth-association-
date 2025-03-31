import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import sys
import matplotlib.pyplot as plt







def plot(df, ref):
    # Data for the plot
    df["Haplogroup"] = df.index
    df = df[df['Haplogroup'].str.contains('MainHap', na=False)]
    df['Haplogroup'] = df['Haplogroup'].str.replace(r'MainHap\[T\.([A-Z0-9]+)\]', r'\1', regex=True)
    # Plot setup
    fig, ax = plt.subplots()
    ax.errorbar(df['Haplogroup'], df['Odds Ratios'], 
                yerr=[df['Odds Ratios'] - df['95% CI Lower (OR)'], df['95% CI Upper (OR)'] - df['Odds Ratios']],
                fmt='o', color='blue', ecolor='black', elinewidth=1, capsize=0)
    ax.set_xlabel('Haplogroup')
    ax.set_ylabel('Odds Ratios')
    ax.set_title('Odds Ratios of PTB relative to Haplogroup ' + ref + ' (n= )')
    # Add asterisks for significant adjusted p-values
    for i, p in enumerate(df['Adjusted P-Values']):
        if p < 0.05:
            ax.text(df['Haplogroup'][i], df['Odds Ratios'][i] + 0.1, '**', color='red', ha='center', fontsize=12)
    # Adding a key for significance
    ax.text(1.0, -0.2, '** = (p < 0.05)', transform=ax.transAxes, ha='right', color='red', fontsize=12)
    # Adding sample sizes at the very bottom of the plot
    for i, haplogroup in enumerate(df['Haplogroup']):
        ax.text(i / len(df['Haplogroup']), -0.25, f'n={df["n"][i]}', transform=ax.transAxes, ha='center', color='black', fontsize=10)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)  # Increase the bottom margin to ensure space for n= labels
    plt.savefig('high_resolution_plot.' + ref + '.png', dpi=300)
    plt.clf()

# Example usage
# Assuming you have a DataFrame 'df' and a reference haplogroup 'ref'
# plot(df, 'M')
    
    


def relevel_category(series, ref):
    """Relevels a pandas categorical series by setting 'ref' as the first category."""
    current_categories = series.cat.categories.tolist()
    if ref in current_categories:
        new_categories = [ref] + [cat for cat in current_categories if cat != ref]
        series = series.cat.reorder_categories(new_categories, ordered=True)
    return series

def detailed_model_summary(model, data, group_var, is_logistic=False):
    # Extract model summary data
    params = model.params
    pvals = model.pvalues
    std_err = model.bse
    conf_int = model.conf_int()
    # Adjust p-values using FDR
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Coefficients': params,
        'Standard Errors': std_err,
        'Original P-Values': pvals,
        'Adjusted P-Values': pvals_corrected,
        '95% CI Lower': conf_int[0],
        '95% CI Upper': conf_int[1]
    })
    # Calculate group sizes and add to the summary
    group_sizes = data[group_var].value_counts()
    # Map the group sizes to the coefficients, ensure the order of levels
    for level in data[group_var].cat.categories:
        if f"{group_var}[T.{level}]" in summary_df.index:
            summary_df.loc[f"{group_var}[T.{level}]", 'n'] = group_sizes.get(level, 0)
    summary_df.loc['Intercept', 'n'] = len(data)  # For the intercept
    # If the model is logistic, calculate odds ratios and confidence intervals for them
    if is_logistic:
        summary_df['Odds Ratios'] = np.exp(params)
        summary_df['95% CI Lower (OR)'] = np.exp(conf_int[0])
        summary_df['95% CI Upper (OR)'] = np.exp(conf_int[1])
    # Round the DataFrame for better readability
    summary_df = summary_df.round(4)
    return summary_df

# Load your data
df = pd.read_csv("Metadata.Final.tsv", sep='\t', quotechar='"')
df['MainHap'] = pd.Categorical(df['MainHap'], categories=np.unique(df['MainHap']), ordered=True)


print(df['MainHap'].value_counts())


features = sys.argv[1].split(',') + sys.argv[2].split(',') 
features=[item for item in features if item != ''] 

formula_PTB = 'PTB ~ ' + ' + '.join(features) + ' + PC1 + PC2 + PC3'

formula_GA = 'GAGEBRTH ~ ' + ' + '.join(features) + ' + PC1 + PC2 + PC3'


print(formula_PTB)
print(formula_GA)

# Processing the haplogroups
haplogroups = ['M','L3']
for ref in haplogroups:
    df['MainHap'] = relevel_category(df['MainHap'], ref)
    # Fit models for GAGEBRTH
    print(f"Hapologroup and CoVar -> GA, Hapologroup Ref={ref}")
    glm_fit_ga = smf.glm(formula_GA, data=df).fit()
    summary_df_ga = detailed_model_summary(glm_fit_ga, df, 'MainHap')
    print(summary_df_ga)
    summary_df_ga.to_csv('summary_df_ga.'+ref+'.tsv', index=True, sep="\t") 
    # Fit models for PTB
    print(f"Hapologroup and CoVar -> PTB, Hapologroup Ref={ref}")
    glm_fit_ptb = smf.glm(formula_PTB, family=sm.families.Binomial(), data=df).fit()
    summary_df_ptb = detailed_model_summary(glm_fit_ptb, df, 'MainHap', is_logistic=True)
    print(summary_df_ptb)
    summary_df_ptb.to_csv('summary_df_ptb.'+ref+'.tsv', index=True, sep="\t") 
    plot(summary_df_ptb,ref)











ref="African"
df['population'] = pd.Categorical(df['population'], categories=np.unique(df['population']), ordered=True)
df['population'] = relevel_category(df['population'], ref)
# Fit models for GAGEBRTH
print(f"Population and CoVar -> GA, Ref={ref}")
glm_fit_ga = smf.glm("GAGEBRTH ~ population + DIABETES + PW_AGE + BMI", data=df).fit()
summary_df_ga = detailed_model_summary(glm_fit_ga, df, 'population')
print(summary_df_ga)
# Fit models for PTB
print(f"Population and CoVar -> PTB, Ref={ref}")
glm_fit_ptb = smf.glm("PTB ~ population  + DIABETES + PW_AGE + BMI", family=sm.families.Binomial(), data=df).fit()
summary_df_ptb = detailed_model_summary(glm_fit_ptb, df, 'population', is_logistic=True)
print(summary_df_ptb)




ref="GAPPS-Bangladesh"
df['site'] = pd.Categorical(df['site'], categories=np.unique(df['site']), ordered=True)
df['site'] = relevel_category(df['site'], ref)
# Fit models for GAGEBRTH
print(f"site and CoVar -> GA, Ref={ref}")
glm_fit_ga = smf.glm("GAGEBRTH ~ site + DIABETES + PW_AGE + BMI", data=df).fit()
summary_df_ga = detailed_model_summary(glm_fit_ga, df, 'site')
print(summary_df_ga)
# Fit models for PTB
print(f"site and CoVar -> PTB, Ref={ref}")
glm_fit_ptb = smf.glm("PTB ~ site  + DIABETES + PW_AGE + BMI", family=sm.families.Binomial(), data=df).fit()
summary_df_ptb = detailed_model_summary(glm_fit_ptb, df, 'site', is_logistic=True)
print(summary_df_ptb)











