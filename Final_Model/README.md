

  


# PCA cumulative explained variance
```python
import numpy as np

eigenval_path = "PCA/cleaned.eigenval"

# --- Load Eigenvalues and compute variance explained ---
eigenvals = np.loadtxt(eigenval_path)
pct_var = eigenvals / eigenvals.sum()
cum_var = np.cumsum(pct_var)
cum_var
```
```
array([0.8414837 , 0.8601727 , 0.87277484, 0.88431504, 0.89527309,
       0.90431597, 0.91249483, 0.92040924, 0.927812  , 0.93497141,
       0.94192632, 0.94875054, 0.95550194, 0.96205185, 0.96855387,
       0.97498064, 0.98131876, 0.98760783, 0.99382569, 1.        ])
```





# Chi-squared test for correlation: mtDNA haplogroup vs. site
```python
import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("Metadata.Final.tsv", sep='\t', quotechar='"')

hap_site_table = pd.crosstab(df['MainHap'], df['site'])
chi2, pval, dof, expected = chi2_contingency(hap_site_table)
print(f"Chi-squared p-value for MainHap ~ site: {pval:.4g}")
```
```
Chi-squared p-value for MainHap ~ site: 0
```

# ANOVA test for correlation: mtDNA haplogroup vs. nDNA PCs
```python

from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

for i in range(1, 6):  # Adjust range if using more PCs
    model = smf.ols(f"PC{i} ~ C(MainHap)", data=df).fit()
    anova_results = anova_lm(model, typ=2)
    pval = anova_results["PR(>F)"]["C(MainHap)"]
    print(f"PC{i} ~ MainHap: p-value = {pval:.4g}")
```
```
PC1 ~ MainHap: p-value = 0
PC2 ~ MainHap: p-value = 2.734e-124
PC3 ~ MainHap: p-value = 7.321e-59
PC4 ~ MainHap: p-value = 3.047e-38
PC5 ~ MainHap: p-value = 2.913e-34
```
# ANOVA test for correlation: site vs. nDNA PCs
```python

for i in range(1, 6):
    model = smf.ols(f"PC{i} ~ C(site)", data=df).fit()
    anova_results = anova_lm(model, typ=2)
    pval = anova_results["PR(>F)"]["C(site)"]
    print(f"PC{i} ~ site: p-value = {pval:.4g}")
```
```
PC1 ~ site: p-value = 0
PC2 ~ site: p-value = 0
PC3 ~ site: p-value = 0
PC4 ~ site: p-value = 5.11e-213
PC5 ~ site: p-value = 2.317e-24
```
