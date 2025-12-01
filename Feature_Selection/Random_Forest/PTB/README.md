
# PTB Covariate Screening Pipeline  
### Random Forest With Site-Aware Cross-Validation & Full-Data SHAP Interpretation

This repository contains a full workflow for **exploratory covariate screening** for *Preterm Birth (PTB)* in a multi‑site, multi‑population study.  

The primary goal is **NOT predictive modeling**.  
Instead, this workflow identifies:

- Important covariates that affect PTB **within sites**
- Nonlinear relationships (via PDP/SHAP)
- Interaction effects
- Features to carry into a final **GLMM / brms** mixed-effects model

This pipeline is optimized for **interpretability**, **generalization across sites**, and **biological plausibility**, not accuracy alone.

---

# 📌 Key Idea

### **Site is used ONLY as a grouping variable for cross-validation. It is NOT included as a predictor.**

This design ensures that:

- The model captures **within-site effects**, not between-site differences.
- No site leakage inflates feature importance.
- SHAP values reflect **true covariate signals**, not population structure.
- Results align perfectly with downstream mixed-effects modeling:

```
PTB ~ covariate_1 + covariate_2 + ... + (1 | site)
```

---

# 📁 Workflow Overview

The pipeline executes in two main stages:

---

## **Stage 1 — Site-Aware Hyperparameter Tuning (QC)**

1. Load covariates and PTB labels from `Metadata.Final.tsv`
2. Remove `"site"` from the feature list automatically
3. Preprocess:
   - Standardize continuous variables
   - OHE categorical features (`handle_unknown="ignore"`)
   - Pass binary features as-is
4. Outer split:
   - If site exists → **GroupShuffleSplit** (hold out entire sites)
   - Otherwise → Stratified split
5. Compute class weights to handle imbalance
6. Use **GroupKFold** for hyperparameter tuning:
   - Ensures every validation fold is made of different sites
7. Grid-search best Random Forest parameters using **Average Precision (PR AUC)**

This prevents the model from cheating by memorizing site differences.

---

## **Stage 2 — Full-Data Fit + SHAP Interpretation**

1. Clone tuned model and refit it on **all samples** using recalculated class weights  
2. Run `run_common_reports` to generate:
   - SHAP global importance
   - SHAP interaction effects
   - SHAP heatmap
   - PDPs for nonlinear relationships
   - RFE summary
   - CSV + PNG exports for all diagnostics

This is the model you use for selecting covariates to include in the final mixed model.

---

# 🧬 Why This Workflow Produces “Within-Site SHAP”

Because:

- **GroupKFold** makes validation folds contain *unseen sites*
- `OneHotEncoder(handle_unknown="ignore")` zeros out unseen categories  
- Any model relying on `"site"` performs poorly during tuning  
- Hyperparameter search therefore picks models that generalize **across sites**
- The final model ends up learning **within-site** patterns

Thus, SHAP importance reflects predictors whose effects are **robust in every site**, not confounded by site identity.

This is *exactly* the interpretation needed before fitting a mixed-effects model.

---

# ⚙️ How to Run the Script

### **Command-line:**

```
python RF_PTB_covariate_screen.py \
    "cat1,cat2,MainHap,SubHap,..." \
    "MAT_HEIGHT,MAT_WEIGHT,BMI,..." \
    "TOILET,ELECTRICITY,..."
```

Note:
- `"site"` is automatically removed from the categorical columns
- You may include mtDNA haplogroups as categorical variables (MainHap, SubHap)

### Example:

```
python RF_PTB_covariate_screen.py \
    "TYP_HOUSE,TOILET,MainHap" \
    "MAT_HEIGHT,BMI,PW_AGE" \
    "ELECTRICITY,CHRON_HTN"
```

---

# 📦 Output Files

After running, you will see files like:

```
PTB.shap_summary.png
PTB.shap_importance.csv
PTB.shap_interactions.csv
PTB.interaction_heatmap.png
PTB.pdp_<feature>.png
PTB.rfe_results.csv
roc_auc.png
pr_auc.png
```

These include:
- ranked feature importance
- nonlinear effect plots
- interaction rankings
- variable selection suggestions
- quality-control performance plots

---

# 🧠 How to Use the Results in a Mixed Model

From SHAP/PDP/RFE, choose:

- Covariates with strong main effects
- Covariates with strong nonlinearities → consider splines
- Covariates with interactions worth modeling or testing
- Add site as a **random intercept**:

### **BRMS example:**

```r
brm(
  PTB ~ cov1 + cov2 + s(BMI) + (1 | site),
  data = df,
  family = bernoulli(),
  prior = ...,
  cores = 4
)
```

This guarantees:
- Site-specific differences are correctly modeled
- Covariate effects reflect **within-site** relationships

---

# 📚 Citation

If you use this workflow in a publication, consider citing:

- Lundberg & Lee (2017) — SHAP  
- Breiman (2001) — Random Forests  
- scikit-learn  
- Any relevant PTB consortium datasets

---

# ✉️ Contact

**Jeff Haltom, PhD**  
Bioinformatics Scientist  
Children’s Hospital of Philadelphia  

