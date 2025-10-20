# 🧠 Gradient Boosting Regression + SHAP Interaction Analysis

## Overview
This pipeline models **Gestational Age at Birth (GAGEBRTH)** using Gradient Boosting Regression, SHAP explainability, and Partial Dependence analysis. It identifies both **main effect** and **interaction features** driving non-linear relationships in complex biological data (e.g., mtDNA, site, BMI, etc.).

---

## ⚙️ Workflow Summary

1. **Load and Filter Data**
   - Reads `Metadata.Final.tsv`
   - Removes invalid codes (-88, -77)
   - Keeps haplogroups with ≥25 samples

2. **Preprocessing**
   - Standardizes continuous variables
   - One-hot encodes categorical variables
   - Leaves binary columns as-is

3. **Model Training**
   - Uses `GradientBoostingRegressor` with grid search
   - Evaluates using **MSE** and **R²**

4. **Feature Selection**
   - Applies Recursive Feature Elimination (RFE)
   - Extracts top 20 predictive features

5. **Explainability**
   - SHAP summary & interaction plots
   - Feature importance ranking
   - Partial Dependence (PDP) and residual diagnostics

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn scipy
```

---

## 🧮 Usage

```bash
python regression_SHAP_pipeline.py "MainHap,Site" "Age,BMI" "Sex,PTB"
```

Arguments:
- 1️⃣ Categorical columns (comma-separated)
- 2️⃣ Continuous columns (comma-separated)
- 3️⃣ Binary columns (comma-separated)

---

## 📈 Output Files

| File | Description |
|------|--------------|
| `shap.summary_plot.GB.GA.png` | SHAP feature importance |
| `shap.summary_plot.Interaction.GB.GA.png` | SHAP interaction summary |
| `FeatureInteractionHeatmap.GB.GA.png` | SHAP interaction heatmap |
| `PDP_Top5.GB.GA.png` | PDP for top 5 interactions |
| `PDP_RFE.GB.GA.png` | PDPs for RFE-selected features |
| `shap.dependence_plot.<feature>.GB.GA.png` | Individual SHAP dependence plots |
| `residuals_vs_fitted.png` | Regression residuals diagnostic |
| `qqplot_residuals.png` | Residuals normality (Q–Q plot) |
| Console output | Includes MSE, R², top features, top interactions |

---

## 📊 Interpretation

- **High SHAP variance** → stronger non-linear influence on GAGEBRTH  
- **Interactions > mean+SD** → statistically significant synergy  
- **Residuals scatter** → random (desired) pattern indicates good model fit  
- **Q–Q plot** → assesses normality of residuals  

---

## 🧠 Notes

- Target (`y`) is not preprocessed — only features are scaled/encoded.  
- Works best on large datasets (>1000 samples).  
- SHAP interactions are memory-intensive; reduce feature count for large models.  
- For reproducibility: `random_state=42` is fixed.

---

## 👨‍💻 Author

**Jeff Haltom, PhD**  
Bioinformatics Scientist II – Children's Hospital of Philadelphia  
GitHub: [jeff-haltom](https://github.com/jeff-haltom)

---
