
# Random Forest + SHAP Pipeline for Gestational Age Prediction  
### **Full Site-Aware Generalization + Interpretability Workflow**

This repository contains a complete machine‑learning workflow for predicting **gestational age at birth (GAGEBRTH)** using a **Random Forest Regressor** with:

- Full preprocessing pipeline (scaling, OHE, passthrough)
- **Group-aware outer test split by site**
- **GroupKFold inner CV** for unbiased hyperparameter tuning
- Automatic feature importance ranking
- **SHAP summary, interaction, and nonlinearity analysis**
- Partial Dependence Plots (PDPs)
- RFE-based feature subset selection  
- Full interpretability-oriented figures output to `.png`

---

# 🧪 1. Overview

This script predicts gestational age while handling:

- Mixed feature types (categorical, continuous, binary)
- Multi-site population structure  
- Site-level confounding  
- Non-linear feature effects  
- Feature–feature interaction effects  
- Large one-hot encoded feature spaces  

The design answers the scientifically important question:

> **Does the model generalize to new sites, not just within existing sites?**

It does this by performing:

### **Outer split:**  
- **GroupShuffleSplit** using `site` as the grouping variable

### **Inner cross‑validation:**  
- **GroupKFold** on training-only sites for hyperparameter tuning

If `site` is missing or has <2 unique levels, the script automatically falls back to a standard `train_test_split` + `KFold` pipeline.

---

# 📁 2. Required Input File

Your working directory must contain:

```
Metadata.Final.tsv
```

Required columns:

- All categorical, continuous, and binary features you specify via command line
- `GAGEBRTH` (regression target)
- (Optional but recommended) `site` for group-aware splitting

The script will validate column presence and throw a descriptive error if anything is missing.

---

# ▶️ 3. Running the Script

Run from command line as:

```bash
python script.py "cat1,cat2,cat3" "age,bmi,PC1,PC2" "PTB,is_female"
```

Arguments:

1. `categorical_columns` → comma-separated list  
2. `continuous_columns` → comma-separated list  
3. `binary_columns` → comma-separated list  

Example:

```bash
python rf_ga.py   "MainHap,SubHap,site"   "Age,MaternalBMI,PC1,PC2,PC3"   "PTB,is_female"
```

---

# 🔧 4. Preprocessing Pipeline

The script builds a `ColumnTransformer`:

| Feature Type | Transformer |
|--------------|-------------|
| Continuous   | `StandardScaler()` |
| Binary       | passthrough |
| Categorical  | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` |

This ensures:

- No sparse matrices (better SHAP compatibility)
- Consistent preprocessing across CV, training, and test inference

---

# 🌲 5. Random Forest Model

Default estimator:

```python
RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
```

Hyperparameter grid:

```python
{
    "rf__n_estimators": [300, 600, 900],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_leaf": [1, 2, 5],
    "rf__max_features": ["sqrt", 0.5],
}
```

---

# 🧭 6. Site‑Aware Train/Test Split

If `site` exists and has ≥ 2 unique levels:

### **Outer split:**  
- **GroupShuffleSplit(test_size=0.3)**  
- Ensures test set contains **sites unseen during training**

### **Inner split:**  
- **GroupKFold** on training-only sites  
- n_splits = min(5, number of unique training sites)

Else:

- Falls back to `train_test_split` + `KFold(n_splits=5)`

---

# 📊 7. Evaluation Metrics

After training and selecting best hyperparameters:

- **Mean Squared Error (MSE)**
- **R² score**

Printed to console using:

```python
evaluate_model_regression(...)
```

---

# ⭐ 8. Feature Importance

Two methods:

### **Random Forest importance**
- Extracted from `.feature_importances_`
- Printed top 10 features
- Useful for quick ranking

### **Recursive Feature Elimination (RFE)**
- Uses the fitted RF model cloned
- Applied to preprocessed design matrix
- Selects top ~20 most informative features

---

# 🔍 9. SHAP Analysis

The script runs full SHAP interpretability:

### **9.1 SHAP Summary Plot**
Generates:

```
shap.summary_plot.RF.GA.png
```

Shows:

- Global feature ranking
- Directional effects
- Density of contributions per feature

### **9.2 SHAP Interaction Values**
Produces:

- Top pairwise interactions  
- Interaction heatmap for top 30 features  
- Interaction summary plots  
- PDPs for top 5 interaction pairs  

Outputs:

```
shap_interactions_heatmap_top.png
shap_interaction_summary_topk.png
pdp_top_interactions.png
```

### **9.3 Nonlinear Feature Analysis**
Uses ΔR² (cubic – linear) on SHAP to find features whose effects are strongly non-linear.

Outputs:

```
pdp_top_nonlinear.png
shap_dependence_<feature>.png
```

---

# 📤 10. Output Files Summary

You will get:

```
shap.summary_plot.RF.GA.png
shap_interactions_heatmap_top.png
shap_interaction_summary_topk.png
pdp_top_interactions.png
pdp_top_nonlinear.png
shap_dependence_*.png
```

Plus console output:

- Best hyperparameters
- Train/test performance
- Feature importance table
- RFE-selected features
- Top interactions
- Top non-linear features

---

# 🧠 11. Performance Notes

- SHAP interaction values are **O(N × F²)** — the script includes optional subsampling for speed.
- Dense OHE is intentional (SHAP requirement for tree explainer).
- RF is parallelized using `n_jobs=-1`.

---

# 🔬 12. Scientific Use Case

This workflow is ideal for:

- mtDNA heteroplasmy associations  
- Gestational age and PTB modeling  
- Multi-site population studies  
- Controlling site-level artifacts  
- Ensuring generalization to unseen cohorts  
- Deep interpretability of model behavior  

---

# 🛠 13. Dependencies

Install via:

```bash
pip install numpy pandas scikit-learn shap matplotlib seaborn
```

---

# 📚 14. License

MIT License (or adapt to your preferred license).

---

# ✉️ 15. Contact

For support or extensions:  
**Jeff Haltom — Bioinformatics Scientist, CHOP**

