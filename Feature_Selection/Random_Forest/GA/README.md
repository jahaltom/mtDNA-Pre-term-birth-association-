
# GA Covariate Screening Pipeline  
### Random Forest Regression With Site-Aware Cross‑Validation & Full‑Data SHAP Interpretation

This repository contains a complete workflow for **exploratory covariate screening** for *Gestational Age at Birth (GAGEBRTH)* across multiple international study sites.

The purpose of this pipeline is **not prediction**, but **interpretation** — specifically:

- Identify covariates that explain **within‑site variation in gestational age**
- Extract nonlinear effects using **PDP** and **SHAP**
- Identify interaction effects using SHAP interactions
- Provide a principled set of covariates for downstream **linear mixed‑effects regression** or **Bayesian GLMM** with site as a random effect

This approach ensures that biological, demographic, and environmental predictors are interpreted **independently of between‑site differences**, which is critical for multisite clinical datasets.

---

## 🎯 Key Principle

### **Site is *never* used as a predictor.**  
### **Site is only used as a grouping factor for cross‑validation.**

This design ensures:

- No leakage of between‑site information  
- Hyperparameter tuning penalizes models that rely on site distributions  
- SHAP values reflect **true covariate effects within each site**  
- Results align perfectly with downstream models like:

```r
GAGEBRTH ~ covariate_1 + covariate_2 + ... + (1 | site)
```

---

# 📐 Workflow Overview

## **Stage 1 — Hyperparameter Tuning (Site‑Aware)**

1. Load `Metadata.Final.tsv`
2. Remove `"site"` from categorical predictors automatically
3. Preprocess using:
   - StandardScaler for continuous variables  
   - OneHotEncoder (dense) for categorical variables  
   - Passthrough for binary variables  
4. Perform an **outer split**:
   - If `site` exists → `GroupShuffleSplit` (hold out entire sites)
   - Otherwise → simple random split  
5. Perform **inner CV**:
   - If ≥2 training sites → `GroupKFold`
   - Otherwise → fallback to row‑level `KFold`  
6. Run **GridSearchCV** over RandomForest hyperparameters  
7. Select the best model using `neg_mean_squared_error`

---

## **Stage 2 — Full Dataset Fit & Interpretation**

1. Clone the best hyperparameters  
2. Refit the model on **all samples from all sites**  
3. Run:

```python
run_common_reports(...)
```

This generates:

- **SHAP summary plots**
- **SHAP interaction heatmaps**
- **PDP curves (nonlinear effects)**
- **RFE variable ranking**
- **CSV and PNG outputs for publication-ready analysis**

---

# 🧬 Why SHAP Gives “Within‑Site” Effects

- `site` is not a predictor  
- GroupKFold prevents learning site distributions  
- Hyperparameter search selects models that generalize across sites  

Therefore, SHAP importance represents covariates that matter **within each population**, not site differences.

---

# 🚀 How to Run

### Example:

```
python RF_GA_covariate_screen.py \
    "TYP_HOUSE,MainHap" \
    "MAT_HEIGHT,MAT_WEIGHT,BMI,PW_AGE" \
    "TOILET"
```

- `"site"` is automatically removed  
- Supports mtDNA haplogroups as categorical variables  
- Produces dense feature matrices compatible with SHAP

---

# 📦 Output Files

```
GA.shap_summary.png
GA.shap_importance.csv
GA.interaction_heatmap.png
GA.shap_interactions.csv
GA.pdp_<feature>.png
GA.rfe_results.csv
```

QC metrics:

```
GA_model_mse.txt
GA_model_r2.txt
```

---

# 🧠 Recommended Downstream Use

### Mixed‑Effects Regression

```r
lmer(GAGEBRTH ~ cov1 + cov2 + (1 | site), data=df)
```

### Bayesian GAMM (ideal)

```r
brm(
  GAGEBRTH ~ s(BMI) + s(PW_AGE) + cov1 + cov2 + (1 | site),
  data=df,
  family=gaussian(),
  cores=4
)
```

Use SHAP importance, PDPs, and interactions to choose covariates and functional forms.

---

# 📚 Citation

- Breiman (2001) — Random Forest  
- Lundberg & Lee (2017) — SHAP  
- scikit‑learn documentation  

---

# 📩 Contact

**Jeff Haltom, PhD**  
Bioinformatics Scientist  
Children’s Hospital of Philadelphia  
