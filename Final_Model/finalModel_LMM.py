import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from patsy import dmatrices

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Function to plot odds ratios for PTB models
def plot(df, ref):
    df = df[df.index.str.contains("MainHap")]
    df["Haplogroup"] = df.index.to_series().str.extract(r'MainHap\[T\.([A-Z0-9]+)\]')
    fig, ax = plt.subplots()
    ax.errorbar(df['Haplogroup'], df['Odds Ratios'],
                yerr=[df['Odds Ratios'] - df['95% CI Lower (OR)'], df['95% CI Upper (OR)'] - df['Odds Ratios']],
                fmt='o', color='blue', ecolor='black', elinewidth=1, capsize=0)
    ax.set(xlabel='Haplogroup', ylabel='Odds Ratios',
           title=f'Odds Ratios of PTB relative to Haplogroup {ref} (n= )')
    for i, p in enumerate(df['Adjusted P-Values']):
        if p < 0.05:
            ax.text(i, df['Odds Ratios'].iloc[i] + 0.1, '**', color='red', ha='center')
    for i, n in enumerate(df['n']):
        ax.text(i / len(df), -0.25, f'n={int(n)}', transform=ax.transAxes, ha='center')
    ax.text(1.0, -0.2, '** = (p < 0.05)', transform=ax.transAxes, ha='right', color='red')
    plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
    plt.savefig(f'high_resolution_plot.{ref}.png', dpi=300)
    plt.clf()

# Reorder categorical levels to set the reference
def relevel_category(series, ref):
    categories = series.cat.categories.tolist()
    if ref in categories:
        return series.cat.reorder_categories([ref] + [x for x in categories if x != ref], ordered=True)
    return series

# Function to format and annotate model output
def detailed_model_summary(model, data, group_var, is_logistic=False):
    conf_int = model.conf_int()
    summary_df = pd.DataFrame({
        'Coefficients': model.params,
        'Standard Errors': model.bse,
        'Original P-Values': model.pvalues,
        'Adjusted P-Values': multipletests(model.pvalues, alpha=0.05, method='fdr_bh')[1],
        '95% CI Lower': conf_int[0],
        '95% CI Upper': conf_int[1]
    })
    group_sizes = data[group_var].value_counts()
    for level in data[group_var].cat.categories:
        idx = f"{group_var}[T.{level}]"
        if idx in summary_df.index:
            summary_df.loc[idx, 'n'] = group_sizes.get(level, 0)
    summary_df.loc['Intercept', 'n'] = len(data)
    if is_logistic:
        summary_df['Odds Ratios'] = np.exp(summary_df['Coefficients'])
        summary_df['95% CI Lower (OR)'] = np.exp(summary_df['95% CI Lower'])
        summary_df['95% CI Upper (OR)'] = np.exp(summary_df['95% CI Upper'])
    return summary_df.round(4)

# Load data
df = pd.read_csv("Metadata.Final.tsv", sep='\t', quotechar='"')
df['MainHap'] = pd.Categorical(df['MainHap'])
df['site'] = pd.Categorical(df['site'])
df['MainHap_cat'] = df['MainHap'].cat.codes
cat_map = dict(enumerate(df['MainHap'].cat.categories))



# Loop through haplogroup references
haplogroups = ['M', 'L3']
quantiles = [0.25, 0.5, 0.75]
qr_results = []

for ref in haplogroups:
    df['MainHap'] = relevel_category(df['MainHap'], ref)
    X_res['MainHap'] = relevel_category(X_res['MainHap'], ref)
    # 1. Linear regression of GA ~ PCs
    model_ga = smf.glm('GAGEBRTH ~ MainHap + BMI + PW_AGE + PC1 + PC2 + PC3', data=df).fit()
    detailed_model_summary(model_ga, df, 'MainHap').to_csv(f'summary_df_ga.PC.{ref}.tsv', sep='\t')
    # 2. Logistic regression of PTB ~ PCs
    model_ptb = smf.glm('PTB ~ MainHap + BMI + PW_AGE + PC1 + PC2 + PC3', family=sm.families.Binomial(), data=df).fit()
    df_ptb = detailed_model_summary(model_ptb, df, 'MainHap', is_logistic=True)
    df_ptb.to_csv(f'summary_df_ptb.PC.{ref}.tsv', sep='\t')
    plot(df_ptb, ref  + "PC")
    # 3. Linear regression of GA ~ site (no PCs)
    model_ga_site = smf.glm('GAGEBRTH ~ MainHap + BMI + PW_AGE + site', data=df).fit()
    detailed_model_summary(model_ga_site, df, 'MainHap').to_csv(f'summary_df_ga.site.{ref}.tsv', sep='\t')
    # 4. Logistic regression of PTB ~ site (no PCs)
    model_ptb_site = smf.glm('PTB ~ MainHap + BMI + PW_AGE + site', family=sm.families.Binomial(), data=df).fit()
    df_ptb_site = detailed_model_summary(model_ptb_site, df, 'MainHap', is_logistic=True)
    df_ptb_site.to_csv(f'summary_df_ptb.site.{ref}.tsv', sep='\t')
    plot(df_ptb_site, ref + "_site")
    # 5. Linear regression of GA ~ site + PCs
    model_ga_site = smf.glm('GAGEBRTH ~ MainHap + BMI + PW_AGE + site + PC2 + PC3', data=df).fit()
    detailed_model_summary(model_ga_site, df, 'MainHap').to_csv(f'summary_df_ga.site.PC.{ref}.tsv', sep='\t')
    # 6. Logistic regression of PTB ~ site + PCs
    model_ptb_site = smf.glm('PTB ~ MainHap + BMI + PW_AGE + site + PC2 + PC3', family=sm.families.Binomial(), data=df).fit()
    df_ptb_site = detailed_model_summary(model_ptb_site, df, 'MainHap', is_logistic=True)
    df_ptb_site.to_csv(f'summary_df_ptb.site.PC.{ref}.tsv', sep='\t')
    plot(df_ptb_site, ref + "_site_PC")
    # 7. Quantile regression with PCs and with site for GA
    for q in quantiles:
        model_qr = smf.quantreg('GAGEBRTH ~ MainHap + BMI + PW_AGE + PC1 + PC2 + PC3', data=df).fit(q=q)
        qsum = model_qr.summary2().tables[1]
        qsum['Quantile'] = q
        qsum['Ref'] = ref
        qsum['Model'] = 'PC'
        qr_results.append(qsum)
        model_qr_site = smf.quantreg('GAGEBRTH ~ MainHap + BMI + PW_AGE + site', data=df).fit(q=q)
        qsum_site = model_qr_site.summary2().tables[1]
        qsum_site['Quantile'] = q
        qsum_site['Ref'] = ref
        qsum_site['Model'] = 'site_only'
        qr_results.append(qsum_site)
        model_qr_site = smf.quantreg('GAGEBRTH ~ MainHap + BMI + PW_AGE + site + PC2 + PC3', data=df).fit(q=q)
        qsum_site = model_qr_site.summary2().tables[1]
        qsum_site['Quantile'] = q
        qsum_site['Ref'] = ref
        qsum_site['Model'] = 'site_AND_PC'
        qr_results.append(qsum_site)

# Save quantile regression summary
pd.concat(qr_results).to_csv("quantile_regression_summary.tsv", sep="\t")







# Load data
df = pd.read_csv("Metadata.Final.tsv", sep='\t', quotechar='"')
df['MainHap'] = pd.Categorical(df['MainHap'], categories=np.unique(df['MainHap']), ordered=True)
df['site'] = pd.Categorical(df['site'], categories=np.unique(df['site']), ordered=True)

features = ['MainHap','BMI','PW_AGE']
covariates = ' + '.join(features) 

for ref in ['M','L3']:
    df['MainHap'] = relevel_category(df['MainHap'], ref)
    # Model 1: LMM for GA with site as random effect
    print(f"GA ~ {covariates} + (1|site), Ref={ref}")
    md1 = smf.mixedlm("GAGEBRTH ~ " + covariates, df, groups=df['site'])
    mdf1 = md1.fit()
    summary_ga_site = detailed_model_summary(mdf1, df, 'MainHap')
    summary_ga_site.to_csv(f'summary_GA_site_RE_{ref}.tsv', sep='\t')
    # Model 2: LMM for GA with PC1 binned as random effect
    df['PC1_bin'] = pd.qcut(df['PC1'], q=5, duplicates='drop')
    df['PC1_bin'] = pd.Categorical(df['PC1_bin'])
    print(f"GA ~ {covariates} + (1|PC1_bin), Ref={ref}")
    md2 = smf.mixedlm("GAGEBRTH ~ " + covariates, df, groups=df['PC1_bin'])
    mdf2 = md2.fit()
    summary_ga_pc1 = detailed_model_summary(mdf2, df, 'MainHap')
    summary_ga_pc1.to_csv(f'summary_GA_PC1_RE_{ref}.tsv', sep='\t')
    
    
