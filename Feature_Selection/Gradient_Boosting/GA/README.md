# 🧠 Gradient Boosting Regression + SHAP Interaction Pipeline

A reproducible pipeline for modeling **Gestational Age at Birth (GAGEBRTH)** from mixed feature types (categorical, continuous, binary) using **Gradient Boosting**, **SHAP explainability** (main + interaction effects), **PDPs**, and **residual diagnostics**.

> This README documents the exact script you provided and adds best‑practice guidance, caveats, and optional extensions so you (or a collaborator) can run, audit, and extend the workflow with confidence.

---

## 🔎 What this pipeline does

1. **Loads** `Metadata.Final.tsv` (tab‑separated).  
2. **Selects** user‑specified feature columns (categorical, continuous, binary) and the target `GAGEBRTH`.  
3. **Splits** into train/test (70/30).  
4. **Preprocesses**: scales continuous, one‑hot encodes categorical, passes binary through.  
5. **Trains** a **GradientBoostingRegressor** with **GridSearchCV**.  
6. **Evaluates** on the hold‑out test set (MSE, R²).  
7. **Explains** model with SHAP (summary plot, interaction summary, heatmap).  
8. **Finds interactions**: prints top pairs and flags “significant” ones (> mean + 1 SD).  
9. **Visualizes**: PDP for top 5 interactions and for RFE‑selected features; SHAP dependence plots.  
10. **Diagnostics**: residuals vs. fitted and Q–Q plot on test residuals.

---

## 📦 Requirements

```bash
# Recommended: Python 3.10+
pip install pandas numpy scikit-learn shap matplotlib seaborn scipy
```

> **Note**: SHAP interaction computations can be memory‑intensive for high‑dim OHE matrices. 16 GB RAM recommended; see **Troubleshooting** below.

---

## 🗂️ Input data schema

- File: `Metadata.Final.tsv` (TSV)
- **Target**: `GAGEBRTH` (continuous; e.g., days)
- **Predictors**: three lists passed on the command line
  - *Categorical* (e.g., `MainHap,Site`)
  - *Continuous* (e.g., `Age,BMI`)
  - *Binary* (e.g., `Sex,PTB`)
- **Missing/invalid**: This script **does not** impute `NaN` and **does not** strip sentinel codes by default. Ensure your TSV has cleaned values or add imputers (see **Extensions**).

Example minimal header:
```
Sample_ID	GAGEBRTH	MainHap	Site	Age	BMI	Sex	PTB
```

---

## 🚀 How to run

```bash
python regression_SHAP_pipeline.py "MainHap,Site" "Age,BMI" "Sex,PTB"
```

Arguments (in order):
1) **Categorical columns** – comma‑separated (no spaces)  
2) **Continuous columns** – comma‑separated  
3) **Binary columns** – comma‑separated

**What the script does with your args**  
- Subsets the DataFrame to exactly those feature columns **plus** `GAGEBRTH`.  
- Builds a preprocessing pipeline:
  - `StandardScaler` for continuous  
  - `OneHotEncoder(handle_unknown="ignore")` for categorical  
  - passthrough for binary  
- Trains `GradientBoostingRegressor` via `GridSearchCV` with:
  ```python
  {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "max_depth":[3,5]}
  ```
- Evaluates test MSE & R².  
- Computes SHAP values & interactions on the **training matrix**.  
- Produces PDPs for **top interactions** and **RFE‑selected** features.  
- Saves residual diagnostics on the **test set**.

---

## 📈 Outputs (written to the working directory)

| Filename | What it shows |
|---|---|
| `shap.summary_plot.GB.GA.png` | Global SHAP summary (importance & direction) |
| `shap.summary_plot.Interaction.GB.GA.png` | SHAP interaction summary plot |
| `FeatureInteractionHeatmap.GB.GA.png` | Mean |SHAP interaction| heatmap across features |
| `PDP_Top5.GB.GA.png` | Partial dependence for the top 5 SHAP‑interaction pairs |
| `PDP_RFE.GB.GA.png` | PDPs for the RFE‑selected features |
| `shap.dependence_plot.<feature>.GB.GA.png` | Per‑feature SHAP dependence (non‑linearity & pairwise effects) |
| `residuals_vs_fitted.png` | Residuals vs. fitted (heteroscedasticity / misspecification check) |
| `qqplot_residuals.png` | Q–Q plot of residuals (normality check) |
| **Console** | Best GB params, test MSE/R², top interactions, significant interactions |

