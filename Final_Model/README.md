

  


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


# GA & PTB Modeling Pipeline

This pipeline models **Gestational Age (GA)** and **Preterm Birth (PTB)** across all cohorts (sites pooled).  
It includes **frequentist GLMMs** (`glmmTMB`) and **Bayesian models** (`brms`), with diagnostics, posterior probabilities, and prior sensitivity analyses.

---

## Overview

- **Goal:** Estimate haplogroup associations with GA and PTB.  
- **Frequentist:** `glmmTMB` for GA (Gaussian) and PTB (logit).  
- **Bayesian:** `brms` for GA (Student-t) and PTB (Bernoulli), including prior sensitivity and site fixed vs random effects.  
- **Covariates:** scaled BMI, scaled maternal age; site as random intercept.  
- **Haplogroup reference:** defaults to `"R"`, falls back to most frequent if absent.

---

## 1. Setup & Configuration

### Libraries
- `readr`, `dplyr`, `stringr`, `forcats`, `tidyr`  
- `ggplot2`  
- `broom`, `broom.mixed`  
- `glmmTMB`  
- `emmeans`  
- `brms`  
- `DHARMa`, `loo`, `posterior`

### Reproducibility
```r
set.seed(2025)
```

### Paths
```r
INFILE <- "Metadata.Final.tsv"
OUTDIR <- "model_outputs/All"
```

### Model covariates
```r
BMI_s + AGE_s + (1|site)
```
*(Alternative commented version includes PCs 1–5.)*

### Haplogroup reference
```r
DEFAULT_REF <- "R"
```
Falls back to modal haplogroup if missing; warns if n < 100.

---

## 2. Data Preprocessing

- Load file: `Metadata.Final.tsv`  
- Convert to factors: `MainHap`, `site`  
- Standardize covariates:
  - `BMI_s = scale(BMI)`
  - `AGE_s = scale(PW_AGE)`
  - `GAGEBRTH_s = scale(GAGEBRTH)` (used in Bayesian GA)

---

## 3. Helper Functions

- `hap_mask()` → flag haplogroup terms  
- `bh_on_hap()` → BH adjust only hap terms  
- `bh_on_hap_wald()` → Wald p-values + BH adjust  
- `to_or()` → log-odds → odds ratio  
- `robust_hap_label()` → clean hap labels  
- `save_forest_ptb()` → save PTB forest plot of hap ORs

---

## 4. Bayesian Priors & Controls

- **GA priors (`pri_ga`):**
  - Fixed effects: `Normal(0, 0.5)`
  - Random SDs & residual: `Student-t(3, 0, 2.5)`

- **Sampler controls:**
  - GA: `adapt_delta = 0.999`, `max_treedepth = 15`
  - PTB: `adapt_delta = 0.99`, `max_treedepth = 13`

---

## 5. Frequentist Models (`glmmTMB`)

### 5.1 GA (Gaussian)
```r
ga_tmb <- glmmTMB(GAGEBRTH ~ MainHap + BMI_s + AGE_s + (1|site),
                  family = gaussian(), data = df)
```
- Output: `ga_glmmtmb.csv`  
- Diagnostics: `ga_glmmtmb_DHARMa.png`

### 5.2 PTB (Binomial logit)
```r
ptb_tmb <- glmmTMB(PTB ~ MainHap + BMI_s + AGE_s + (1|site),
                   family = binomial(), data = df)
```
- Outputs:
  - `ptb_glmmtmb.csv` (log-odds + ORs)  
  - `ptb_glmmtmb_site_forest.png` (forest plot)  
  - `ptb_glmmtmb_DHARMa.png` (residuals)  
  - `ptb_glmmtmb_emmeans_probs.csv` (predicted PTB probs by hap)  
  - `ptb_glmmtmb_emmeans_pairs_BH.csv` (pairwise hap tests)

---

## 6. Bayesian GA (`brms`, Student-t)

```r
brm_ga <- brm(
  GAGEBRTH_s ~ MainHap + BMI_s + AGE_s + (1|site),
  family  = student(),
  prior   = pri_ga,
  chains  = 4, iter = 4000, cores = 4,
  control = ctrl_ga, inits = 0, seed = 2025,
  data    = df
)
```

- Outputs:
  - `ga_brm_summary.txt`  
  - `ga_brm.csv` (back-transformed to **days**)  
  - `ga_brm_pp_check.png`  
  - `ga_brm_bayesR2.txt`

### Posterior probabilities
- `Pr_beta_gt0` → probability β > 0  
- `p_two` → two-sided sign probability  
- `Pr_days_gt_1` → probability effect > +1 day  
- `Pr_days_lt_m1` → probability effect < −1 day  
- Output: `ga_brm_posterior_probs.csv`

---

## 7. Bayesian PTB: Prior Sensitivity & Site Spec

### 7.1 Hap coefficient names
- Fit quick model to extract `MainHap[...]` names.

### 7.2 Priors
- `Normal(0,0.5)` (shrink)  
- `Normal(0,1.0)` (mild)  
- `Normal(0,2.5)` (weak)  
- `flat` (default)

### 7.3 Fit under each prior
- Extract: ORs, 95% CrIs, `Pr_OR_gt_1`, `p_two`  
- Output: `ptb_brm_prior_sensitivity_haps.csv`

### 7.4 Site sensitivity
- Fixed site:  
  ```r
  PTB ~ MainHap + BMI_s + AGE_s + site
  ```
- Random site:  
  ```r
  PTB ~ MainHap + BMI_s + AGE_s + (1|site)
  ```

⚠️ Bug: script uses `write_csv(ptb_FE, ...)`.  
Use `saveRDS()` or export `summary(fit)$fixed`.

---

## 8. Outputs

### Frequentist
- `ga_glmmtmb.csv` — GA fixed effects  
- `ga_glmmtmb_DHARMa.png` — GA diagnostics  
- `ptb_glmmtmb.csv` — PTB fixed effects + ORs  
- `ptb_glmmtmb_site_forest.png` — PTB forest plot  
- `ptb_glmmtmb_DHARMa.png` — PTB diagnostics  
- `ptb_glmmtmb_emmeans_probs.csv` — predicted PTB probs  
- `ptb_glmmtmb_emmeans_pairs_BH.csv` — hap comparisons

### Bayesian
- `ga_brm_summary.txt` — GA Student-t model summary  
- `ga_brm.csv` — GA back-transformed effects (days)  
- `ga_brm_pp_check.png` — posterior predictive check  
- `ga_brm_bayesR2.txt` — Bayes R²  
- `ga_brm_posterior_probs.csv` — GA posterior sign/threshold probs  
- `ptb_brm_prior_sensitivity_haps.csv` — PTB prior sensitivity results  
- *(fix required)* site-spec results: save as `.rds` or CSV summaries

---

## Notes

- Haplogroup reference (`DEFAULT_REF`) drives all comparisons.  
- BH correction applies **only across hap terms**.  
- `emmeans` predictions are **population-level** (random effects set to mean).  
- Use `inits=0` consistently for `brms`.  
- Save full `brmsfit` objects with `saveRDS()` for reproducibility.  
- Conservative sampler settings (`adapt_delta`, `max_treedepth`) are chosen for robustness.















