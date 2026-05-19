

  


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

## Overview
This pipeline evaluates associations between **mitochondrial DNA (mtDNA) haplogroups** and:

- **Gestational Age (GA)** — continuous outcome
- **Pre-Term Birth (PTB)** — binary outcome

using both **frequentist** (`glmmTMB`) and **Bayesian** (`brms`) frameworks across pooled cohorts. The pipeline supports:

- Dynamic covariate selection at runtime
- Fixed (`site`) or random (`(1 | site)`) site effects
- Gaussian and Student-t GA models
- Bayesian posterior probability estimation
- Prior sensitivity analyses
- Multiple testing correction (Benjamini–Hochberg)
- Sparse-cell detection
- Automated convergence diagnostics

---

## Statistical Framework

### Frequentist Models (`glmmTMB`)

#### Gestational Age (GA)
Two models are fit:

1. **Gaussian**
2. **Student-t**

Model fit comparison:

- `AIC`

Formula:

```r
GAGEBRTH ~ MainHap + covariates
```

#### Pre-Term Birth (PTB)

Logistic regression:

```r
PTB ~ MainHap + covariates
```

Outputs:

- Odds ratios (OR)
- Confidence intervals
- Forest plots
- Estimated marginal means (EMMs)
- Pairwise comparisons

---

### Bayesian Models (`brms`)

#### Gestational Age

Family:

```r
student()
```

Posterior probabilities:

- `Pr(effect > +1 day)`
- `Pr(effect < -1 day)`
- `Pr(beta > 0)`

Effects are back-transformed to **days**.

#### Pre-Term Birth

Family:

```r
bernoulli()
```

Posterior probabilities:

- `Pr(OR > 1)`

---

## Prior Sensitivity Analysis

PTB models are fit under:

| Prior | Meaning |
|---|---|
| `Normal(0,0.5)` | Strong shrinkage |
| `Normal(0,1.0)` | Moderate shrinkage |
| `Normal(0,2.5)` | Weakly informative |
| `flat` | Minimal prior assumptions |

---

## Input File

Required input:

```text
Metadata.Final.tsv
```

### Required Variables

#### Outcomes
- `GAGEBRTH`
- `PTB`

#### mtDNA
- `MainHap`

#### Site
- `site`

#### Continuous / Ordinal Covariates
- `PW_AGE`
- `PW_EDUCATION`
- `MAT_HEIGHT`
- `MAT_WEIGHT`
- `BMI`
- `TOILET`
- `WEALTH_INDEX`
- `DRINKING_SOURCE`
- `PC1–PC5`

#### Binary Covariates
- `BABY_SEX`
- `CHRON_HTN`
- `DIABETES`
- `HH_ELECTRICITY`
- `TB`
- `THYROID`
- `TYP_HOUSE`

---

## Running the Pipeline

### Syntax

```bash
Rscript finalModel.R REF "COVARIATES"
```

### Examples

Fixed site:

```bash
Rscript finalModel.R M "PW_AGE + MAT_HEIGHT + site"
```

Random site:

```bash
Rscript finalModel.R M "PW_AGE + BMI + (1 | site)"
```

With PCs:

```bash
Rscript finalModel.R M "PW_AGE + BMI + PC1 + PC2 + PC3 + (1 | site)"
```

---

## Output Directory Naming

Example:

```text
model_outputs/
└── All_M_PW_AGE_BMI_siteRE/
```

Encodes:

- Reference haplogroup
- Covariates
- Fixed vs random site effect

---

## Output Files

### Frequentist Results

| File | Description |
|---|---|
| `ga_glmmtmb_gaussian.csv` | Gaussian GA model coefficients |
| `ga_glmmtmb_student_t.csv` | Student-t GA coefficients |
| `ga_glmmtmb_gaussian_vs_student_t_AIC.csv` | AIC comparison |
| `ptb_glmmtmb.csv` | PTB logistic model |
| `ptb_glmmtmb_site_forest.png` | PTB odds-ratio forest plot |
| `ptb_glmmtmb_emmeans_probs.csv` | Predicted PTB probabilities |
| `ptb_glmmtmb_emmeans_pairs_BH.csv` | Pairwise BH-adjusted comparisons |

### Bayesian Results

| File | Description |
|---|---|
| `ga_brm.csv` | Bayesian GA fixed effects |
| `ga_brm_summary.txt` | Full GA model summary |
| `ga_brm_bayesR2.txt` | Bayesian R² |
| `ga_brm_posterior_probs.csv` | Posterior probability table |
| `ptb_brm_summary.txt` | PTB Bayesian model summary |
| `ptb_brm_sensitivity.csv` | Final PTB coefficient table |
| `ptb_brm_prior_sensitivity_haps.csv` | Prior sensitivity results |

### Diagnostics

| File | Description |
|---|---|
| `*_traceplot.png` | Chain mixing plots |
| `*_pp_check.png` | Posterior predictive checks |
| `*_diagnostics.txt` | Divergences, treedepth hits, BFMI |
| `*_draws_summary.csv` | Posterior draw statistics |
| `*_fixed_effects_summary.csv` | Fixed effect summary |
| `*_nuts_params.csv` | Raw NUTS diagnostics |
| `*_bad_rhat.csv` | Parameters with Rhat > 1.01 |
| `*_low_ess_bulk.csv` | Low bulk ESS |
| `*_low_ess_tail.csv` | Low tail ESS |
| `*_glmmtmb_DHARMa.png` | residual diagnostics |


### Cohort Summary Outputs

| File | Description |
|---|---|
| `site_summary.csv` | Site-level descriptive statistics |
| `hap_ptb_counts.csv` | PTB counts by haplogroup |
| `hap_site_ptb_table.csv` | Haplogroup × site PTB table |
| `hap_site_ptb_flags.csv` | Sparse/problematic cell flags |
| `hap_site_ptb_problem_cells.csv` | Problematic cells only |
| `model_formula_used.txt` | Exact formulas and parameters |

---

## Interpretation

### GA

`Pr(effect > +1 day)`

Probability a haplogroup increases gestational age by >1 day.

Example:

`0.97`

→ 97% posterior probability of >1 gestational day increase.

### PTB

`Pr(OR > 1)`

Probability a haplogroup increases PTB odds.

Example:

`0.98`

→ 98% posterior probability of elevated PTB odds.

### Multiple Testing

Benjamini–Hochberg correction is applied to **MainHap effects only**.

---

## Convergence Guidelines

Recommended thresholds:

| Metric | Target |
|---|---|
| Divergences | 0 |
| Rhat | < 1.01 |
| ESS | > 400 |
| Treedepth hits | Minimal |

If convergence is poor:

```r
adapt_delta = 0.999
iter = 6000
warmup = 2000
```

---

## DHARMa evaluates:

- Residual uniformity
- Overdispersion
- Outliers
- Zero inflation
- Model misspecification
- Heteroscedasticity

#### Interpretation Guidelines

**Good model fit:**

- Residuals appear randomly distributed
- QQ plot approximately follows expected line
- No strong residual trends

**Potential issues:**

- Systematic residual patterns → model misspecification
- Strong QQ deviations → poor distributional assumptions
- Overdispersion → variance not adequately modeled
- Clustering or structure → missing covariates or interactions

The **Student-t GA model** is often preferred when Gaussian residual assumptions are violated due to heavy tails or outliers.

## Reproducibility

Random seed:

```r
set.seed(2025)
seed = 2025
```

---

## Author

**Jeffrey Haltom, PhD**  
Bioinformatics • Statistical Genetics • Mitochondrial Genomics
