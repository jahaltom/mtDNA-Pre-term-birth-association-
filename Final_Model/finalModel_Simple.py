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

# # Function to plot odds ratios for PTB models
# def plot(df, ref):
#     df = df[df.index.str.contains("MainHap")]
#     df["Haplogroup"] = df.index.to_series().str.extract(r'MainHap\[T\.([A-Z0-9]+)\]')
#     fig, ax = plt.subplots()
#     ax.errorbar(df['Haplogroup'], df['Odds Ratios'],
#                 yerr=[df['Odds Ratios'] - df['95% CI Lower (OR)'], df['95% CI Upper (OR)'] - df['Odds Ratios']],
#                 fmt='o', color='blue', ecolor='black', elinewidth=1, capsize=0)
#     ax.set(xlabel='Haplogroup', ylabel='Odds Ratios',
#            title=f'Odds Ratios of PTB relative to Haplogroup {ref} (n= )')
#     for i, p in enumerate(df['Adjusted P-Values']):
#         if p < 0.05:
#             ax.text(i, df['Odds Ratios'].iloc[i] + 0.1, '**', color='red', ha='center')
#     for i, n in enumerate(df['n']):
#         ax.text(i / len(df), -0.25, f'n={int(n)}', transform=ax.transAxes, ha='center')
#     ax.text(1.0, -0.2, '** = (p < 0.05)', transform=ax.transAxes, ha='right', color='red')
#     plt.xticks(rotation=45)
#     plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
#     plt.savefig(f'high_resolution_plot.{ref}.png', dpi=300)
#     plt.clf()

# Reorder categorical levels to set the reference
def relevel_category(series, ref):
    categories = series.cat.categories.tolist()
    if ref in categories:
        return series.cat.reorder_categories([ref] + [x for x in categories if x != ref], ordered=True)
    return series

# Function to format and annotate model output
def detailed_model_summary(model, data, group_var, is_logistic=False):
    conf_int = model.conf_int()
    
    
    p = model.pvalues
    mask = p.index.str.startswith("MainHap[T.")
    q = pd.Series(np.nan, index=p.index)
    q.loc[mask] = multipletests(p.loc[mask], alpha=0.05, method="fdr_bh")[1]
    
    
    summary_df = pd.DataFrame({
        'Coefficients': model.params,
        'Standard Errors': model.bse,
        'Original P-Values': p,
        'Adjusted P-Values': q,
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


# Loop through haplogroup references
haplogroups = ['M', 'L3']
quantiles = [0.25, 0.5, 0.75]
qr_results = []
pcs='PC1 + PC2 + PC3 + PC4 +PC5'
dfFinal=[]

features = ['MainHap','BMI','PW_AGE']
covariates = ' + '.join(features) 

for ref in haplogroups:
    df['MainHap'] = relevel_category(df['MainHap'], ref)
    # 1. Linear regression of GA ~ PCs
    model_ga = smf.glm('GAGEBRTH ~ '+ covariates +' +' + pcs, data=df).fit()
    test1=detailed_model_summary(model_ga, df, 'MainHap')
    test1['Test']='GA ~ ' + covariates + "+" + pcs
    
    # 2. Logistic regression of PTB ~ PCs
    model_ptb = smf.glm('PTB ~ '+ covariates +' +' + pcs, family=sm.families.Binomial(), data=df).fit()
    test2=detailed_model_summary(model_ptb, df, 'MainHap', is_logistic=True)
    test2['Test']='PTB ~ ' + covariates + "+" + pcs
    
    # 3. Linear regression of GA ~ site (no PCs)
    model_ga_site = smf.glm('GAGEBRTH ~ '+ covariates +' + site', data=df).fit()
    test3=detailed_model_summary(model_ga_site, df, 'MainHap')
    test3['Test']='GA ~ '+ covariates +'+ site (no ' + pcs + ")"
    
    # 4. Logistic regression of PTB ~ site (no PCs)
    model_ptb_site = smf.glm('PTB ~ '+ covariates +' + site', family=sm.families.Binomial(), data=df).fit()
    test4=detailed_model_summary(model_ptb_site, df, 'MainHap', is_logistic=True)
    test4['Test']='PTB ~ '+ covariates +'+ site (no ' + pcs + ")"
    
    # 5. Linear regression of GA ~ site + PCs
    model_ga_site = smf.glm('GAGEBRTH ~ '+ covariates +' + site +' + pcs.replace("PC1 + ", ""), data=df).fit()
    test5=detailed_model_summary(model_ga_site, df, 'MainHap')
    test5['Test']='GA ~ '+covariates+'+ site +' + pcs.replace("PC1 + ", "")
    
    # 6. Logistic regression of PTB ~ site + PCs
    model_ptb_site = smf.glm('PTB ~ '+ covariates +' + site +' + pcs.replace("PC1 + ", ""), family=sm.families.Binomial(), data=df).fit()
    test6=detailed_model_summary(model_ptb_site, df, 'MainHap', is_logistic=True)
    test6['Test']='PTB ~ '+covariates+'+ site +' + pcs.replace("PC1 + ", "")
    
    
    # 7. LMM for GA with site as random effect
    md1 = smf.mixedlm("GAGEBRTH ~ " + covariates, df, groups=df['site'])
    mdf1 = md1.fit()
    test7 = detailed_model_summary(mdf1, df, 'MainHap')
    test7['Test']=(f"GA ~ {covariates} + (1|site)")
    
    
    
    
    dfTemp=pd.concat([test1,test2,test3,test4,test5,test6,test7])
    dfTemp["Ref"]=ref
    dfFinal.append(dfTemp)
    
dfFinal=pd.concat(dfFinal)   
dfFinal.to_csv('testtest.tsv', sep='\t')   
    
