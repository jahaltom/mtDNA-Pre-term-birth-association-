

  


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


# mtDNA Pre-Term Birth Association Pipeline

A reproducible statistical framework for modeling associations between mitochondrial DNA (mtDNA) haplogroups and **gestational age (GA)** and **pre-term birth (PTB)** across pooled multi-site cohorts using both **frequentist** and **Bayesian** methods. Supports **dynamic covariate selection**, **fixed or random site effects**, **prior sensitivity analyses**, and **robust convergence diagnostics**.

## Overview

This pipeline evaluates whether **mtDNA haplogroups** are associated with:

- **Gestational age (GA)** — continuous outcome  
- **Pre-term birth (PTB)** — binary outcome  

while adjusting for configurable **clinical**, **environmental**, **socioeconomic**, and **genetic covariates**.

### Features

- Dynamic covariate specification at runtime
- Fixed or random site effects
- Gaussian and Student-t frequentist GA models
- Bayesian Student-t GA modeling
- Bayesian PTB logistic regression
- Prior sensitivity analyses
- Multiple-testing correction (Benjamini–Hochberg)
- Sparse-cell detection
- Automated convergence diagnostics

## Statistical Framework

### Frequentist Models (`glmmTMB`)

#### Gestational Age (GA)

Two models are fit:

- **Gaussian model**
- **Student-t model**

Model fit is compared using **AIC**.

#### Pre-Term Birth (PTB)

Binary logistic regression:

`PTB ~ MainHap + covariates`

Outputs include:

- Odds ratios (OR)
- Confidence intervals
- Forest plots
- Estimated marginal means (EMMs)
- Pairwise haplogroup comparisons

### Bayesian Models (`brms`)

#### Bayesian Gestational Age Model

Family:

`student()`

Posterior probabilities include:

- `Pr(effect > +1 day)`
- `Pr(effect < -1 day)`
- `Pr(beta > 0)`

Effects are estimated on a standardized scale and back-transformed into **gestational days**.

#### Bayesian PTB Model

Family:

`bernoulli()`

Posterior probabilities include:

`Pr(OR > 1)`

which directly estimates the probability that a haplogroup increases PTB odds.

## Prior Sensitivity Analysis

PTB models are fit under multiple priors:

| Prior | Description |
|-------|-------------|
| Normal(0,0.5) | Strong shrinkage |
| Normal(0,1.0) | Moderate shrinkage |
| Normal(0,2.5) | Weakly informative |
| Flat | Minimal prior structure |

## Input Requirements

Required input file:

`Metadata.Final.tsv`

Expected columns include:

### Outcomes

- `GAGEBRTH`
- `PTB`

### mtDNA Variables

- `MainHap`

### Site Variable

- `site`

### Continuous / Ordinal Covariates

- `PW_AGE`
- `PW_EDUCATION`
- `MAT_HEIGHT`
- `MAT_WEIGHT`
- `BMI`
- `TOILET`
- `WEALTH_INDEX`
- `DRINKING_SOURCE`
- `PC1–PC5`

### Binary Covariates

- `BABY_SEX`
- `CHRON_HTN`
- `DIABETES`
- `HH_ELECTRICITY`
- `TB`
- `THYROID`
- `TYP_HOUSE`

## Running the Pipeline

### Command Line Usage

```bash
Rscript finalModel.R REF "COVARIATES"
```

### Example: Fixed Site Effect

```bash
Rscript finalModel.R M "PW_AGE + MAT_HEIGHT + site"
```

### Example: Random Site Effect

```bash
Rscript finalModel.R M "PW_AGE + BMI + (1 | site)"
```

### Example with PCs

```bash
Rscript finalModel.R M "PW_AGE + BMI + PC1 + PC2 + PC3 + (1 | site)"
```

## Output Structure

Example:

```text
model_outputs/
└── All_M_PW_AGE_BMI_siteRE/
```

Folder names automatically encode:

- reference haplogroup
- covariates
- fixed/random site effect

## Key Output Files

### Frequentist Results

| File | Description |
|------|-------------|
| `ga_glmmtmb_gaussian.csv` | Gaussian GA model |
| `ga_glmmtmb_student_t.csv` | Student-t GA model |
| `ga_glmmtmb_gaussian_vs_student_t_AIC.csv` | Model comparison |
| `ptb_glmmtmb.csv` | PTB logistic model |
| `ptb_glmmtmb_site_forest.png` | Forest plot |

### Bayesian Results

| File | Description |
|------|-------------|
| `ga_brm.csv` | Bayesian GA model |
| `ga_brm_posterior_probs.csv` | Posterior probabilities |
| `ptb_brm_summary.txt` | PTB Bayesian summary |
| `ptb_brm_prior_sensitivity_haps.csv` | Prior sensitivity |

### Diagnostics

| File | Description |
|------|-------------|
| `*_traceplot.png` | Chain mixing |
| `*_pp_check.png` | Posterior predictive checks |
| `*_diagnostics.txt` | Divergences & treedepth |
| `*_bad_rhat.csv` | Rhat > 1.01 |
| `*_low_ess_bulk.csv` | Low ESS |

## Interpretation Guide

### Bayesian GA

`Pr(effect > +1 day)`

Probability a haplogroup increases gestational age by **more than one day**.

### Bayesian PTB

`Pr(OR > 1)`

Probability a haplogroup increases PTB risk.

### Multiple Testing

Benjamini–Hochberg (BH) correction is applied across **MainHap effects only**.

## Convergence Diagnostics

Targets:

- **Divergences:** 0
- **Rhat:** < 1.01
- **ESS:** > 400
- **Treedepth hits:** minimal

## Reproducibility

Random seed:

`set.seed(2025)`

Bayesian models use:

`seed = 2025`

## Author

**Jeffrey Haltom, PhD**  
Bioinformatics / Statistical Genetics / Mitochondrial Genomics
