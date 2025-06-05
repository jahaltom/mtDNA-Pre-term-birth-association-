#### finalModel.py
- Takes in Metadata.Final.tsv and categorical and continuous variables.(set in workflow4.sh)
- Returns n for each Haplogroup.
- Returns formula
- Sets refrence haplpogroup
- Fits models for GAGEBRTH and PTB. Hapologroup and Co-variates
- Returns detailed model summary (detailedModelSummary.txt)
  
- Fits models for GAGEBRTH and PTB. (population and site) and Co-variates. Returns detailed model summary (detailedModelSummary.txt). Only relavent for Meta analysis and African/SouthAsian. 










```python
import pandas as pd
df = pd.read_csv("Metadata.Final.tsv", sep='\t', quotechar='"')




#Test for correlation: mtDNA haplogroup vs. site
from scipy.stats import chi2_contingency

hap_site_table = pd.crosstab(df['MainHap'], df['site'])
chi2, pval, dof, expected = chi2_contingency(hap_site_table)
print(f"Chi-squared p-value for MainHap ~ site: {pval:.4g}")


#Test for correlation: mtDNA haplogroup vs. nDNA PCs
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

for i in range(1, 6):  # Adjust range if using more PCs
    model = smf.ols(f"PC{i} ~ C(MainHap)", data=df).fit()
    anova_results = anova_lm(model, typ=2)
    pval = anova_results["PR(>F)"]["C(MainHap)"]
    print(f"PC{i} ~ MainHap: p-value = {pval:.4g}")

#Test for correlation: site vs. nDNA PCs
for i in range(1, 6):
    model = smf.ols(f"PC{i} ~ C(site)", data=df).fit()
    anova_results = anova_lm(model, typ=2)
    pval = anova_results["PR(>F)"]["C(site)"]
    print(f"PC{i} ~ site: p-value = {pval:.4g}")
```
