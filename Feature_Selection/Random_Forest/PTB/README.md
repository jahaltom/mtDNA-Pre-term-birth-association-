
# PTB Covariate Screening Pipeline  
### Random Forest + Site-Aware Cross-Validation + Full-Data SHAP Interpretation

This repository contains a complete and rigorous workflow for **covariate screening** for **Preterm Birth (PTB)** across multiple international study sites.  

This pipeline is **not intended for predictive modeling**.  
Its purpose is to identify **robust covariates**, evaluate **nonlinear effects**, and generate **interpretation diagnostics** (SHAP, PDP, interactions) that guide the construction of a final **GLMM / brms** inferential model.

---

# 🔍 Overview

The workflow operates in **two stages**:

## **1. Site-aware model tuning (quality control)**  
A Random Forest is tuned using **GroupKFold (groups = site)** so that validation folds contain **entirely unseen sites**.  
This design *forces the model to generalize across sites* and prevents it from memorizing site labels.

Because unseen site categories are encoded as all-zero one-hot vectors (`OneHotEncoder(handle_unknown="ignore")`), hyperparameters that rely heavily on site are **penalized during tuning**.  
This results in a model that focuses on **within-site covariate effects**, not between-site differences.

This stage is only for **QC and stable hyperparameter selection**.

---

## **2. Full-data model fitting + SHAP interpretation (covariate screening)**  
After tuning, the best hyperparameters are cloned and used to fit a Random Forest on **all samples, all sites**, with proper **class weighting** applied.

This final model is passed to `run_common_reports`, which computes:

- SHAP global feature importance  
- SHAP interaction strengths  
- SHAP interaction heatmaps  
- PDP (Partial Dependence Plots)  
- ICE curves (optional)  
- Nonlinearity metrics  
- RFE (Recursive Feature Elimination)  
- Exported CSVs of all rankings and metrics  

These diagnostics provide a rich, stable picture of **which covariates matter**, **how** they matter, and **what functional forms** should be used in a final GLMM/brms model.

---

# 📁 Script Summary

The script performs the following steps:

1. **Load data**  
   - PTB label is binarized  
   - Categorical, continuous, and binary covariates are user-specified  

2. **Preprocessing pipeline**  
   - Continuous → StandardScaler  
   - Binary → passthrough  
   - Categorical → OneHotEncoder (`handle_unknown="ignore"`)  

3. **Random Forest model specification**  
   - RF chosen for interpretability + SHAP support  
   - Hyperparameters grid-searched using AP (Average Precision)  

4. **Outer split (QC)**  
   If `site` column exists and has ≥2 levels:
   - Use GroupShuffleSplit (30% sites held out)  
   Else:
   - Use Stratified shuffle split  

5. **Class imbalance handling**  
   - Compute weights: `neg/pos` ratio  
   - Weights passed through `rf__sample_weight` during fitting  

6. **Inner CV (hyperparameter tuning)**  
   - **If sites available:** use GroupKFold  
   - **Else:** stratified K-fold  

7. **Train best model** and evaluate on held-out outer split  

8. **Refit on full dataset** using the best hyperparameters + recalculated class weights  

9. **Run `run_common_reports`** to produce:  
   - SHAP global importance  
   - SHAP interactions  
   - Top PDP curves  
   - Nonlinearity scores  
   - RFE rankings  
   - CSV outputs  
   - PNG visualizations  

---

# 🧠 Why SHAP Shows *Within-Site* Effects

Because GroupKFold forces the model to validate on unseen sites, any hyperparameter set that relies on “site” as a feature performs poorly.  
Thus, the final tuned hyperparameters produce a model that emphasizes **covariates whose effects generalize across sites**.

This results in:

- Site one-hot encoded columns having **near-zero SHAP importance**  
- Maternal, demographic, and socioeconomic variables rising to the top  
- PDP and SHAP plots that capture **within-site variation**, not merely between-site differences  

This behavior is exactly what we want for a pipeline that ultimately feeds a **mixed-effects final model**:

```
PTB ~ covariate_1 + covariate_2 + ... + (1 | site)
```

---

# 🛠 How to Run

### **Command-Line Usage**

```
python RF_PTB_covariate_screen.py \
    "cat1,cat2,site,..." \
    "height,BMI,age,..." \
    "toilet,electricity,..."
```

Where:
- First argument = comma-separated categorical columns  
- Second argument = continuous columns  
- Third argument = binary columns  

### Example:

```
python RF_PTB_covariate_screen.py \
    "site,TYP_HOUSE,TOILET" \
    "MAT_HEIGHT,MAT_WEIGHT,BMI,PW_AGE" \
    "ELECTRICITY,CHRON_HTN"
```

---

# 📦 Output Files

Outputs from `run_common_reports` follow this naming pattern:

```
PTB.shap_importance.csv
PTB.shap_interactions.csv
PTB.pdp_<feature>.png
PTB.shap_summary.png
PTB.rfe_selected.csv
PTB.interaction_heatmap.png
...
```

A complete report summarizing interpretability diagnostics will be produced.

---

# 🧬 How to Use These Results for Your Final Inferential Model

The final inferential model should be built using **GLMM (e.g., glmmTMB)** or **Bayesian brms**:

- Use SHAP/RFE to select covariates  
- Use PDP and nonlinearity metrics to decide:
  - linear term  
  - spline  
  - cutoff  
  - logistic transform  
- Use interactions flagged by SHAP to test or stratify models  
- Add `(1 | site)` as the random intercept term  

This workflow ensures that the final model:
- is interpretable  
- is biologically plausible  
- captures within-site relationships  
- properly partitions out between-site variation  

---

# 📚 Citation & Credit

If using this workflow in a publication, consider citing:

- Lundberg & Lee (2017) — SHAP  
- Breiman (2001) — Random Forests  
- scikit-learn developers  

And, of course, acknowledge your own PTB dataset sources.

---

# 📬 Contact

For questions or technical issues, contact:

**Jeff Haltom, PhD**  
Bioinformatics Scientist  
Children’s Hospital of Philadelphia  
