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

> Tip: For provenance, redirect stdout to a log file:  
> `python ... > run.log 2>&1`

---

## 📐 Methodological notes

- **Target scaling**: `y` is not transformed. If you log/Box‑Cox `y` in the future, be sure to inverse‑transform predictions before evaluation.  
- **Train/test hygiene**: the `ColumnTransformer` is fit only on the **training** split and applied to test (prevents leakage).  
- **RFE**: performed on the *preprocessed* training matrix; selected feature names are taken from `preprocessor.get_feature_names_out()`.  
- **Interactions**: significance heuristic is `mean + 1 SD` of mean‑absolute SHAP interaction strengths (exploratory, not inferential).  
- **Metrics**: primary metrics are **MSE** and **R²** on the test set.

---

## 🧩 Known caveats & troubleshooting

1) **Sparse vs. dense matrices & SHAP**  
   - `OneHotEncoder` may yield a sparse matrix; some SHAP plots expect dense arrays. If you see errors like “no attribute `.toarray()`” or plotting crashes, convert:  
     ```python
     import scipy.sparse as sp
     if sp.issparse(X_train_preprocessed):
         X_train_dense = X_train_preprocessed.toarray()
     else:
         X_train_dense = X_train_preprocessed
     ```
     Then pass `X_train_dense` to SHAP plotting calls.

2) **NaNs**  
   - The current script does **not** impute missing values. Add `SimpleImputer` stages if your data contain `NaN` (see **Extensions**).

3) **Large OHE cardinality**  
   - Very high‑cardinality categorical features can expand to thousands of columns → slow SHAP interactions & huge plots. Consider cardinality reduction or compute interactions only for the **top‑K** features by SHAP magnitude.

4) **Feature index/name mismatches for PDP**  
   - PDP requires **indices** into the transformed matrix. The script already maps names → indices for the top‑interaction pairs. Ensure consistency if you change the pipeline.

5) **Reproducibility**  
   - `random_state=42` is set for GB and the train/test split; pin package versions for fully reproducible runs (see below).

---

## 🧪 Optional extensions (recommended for production/rpub)

**A. Add robust missing‑data handling**
```python
from sklearn.impute import SimpleImputer
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), continuous_columns),
    ("bin", Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]), binary_columns),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns)
])
```

**B. Save key artifacts (for auditing)**
```python
gb_importances.to_csv("gb_feature_importances.tsv", sep="\t", index=False)
cleaned_interactions.head(50).to_csv("top_interactions.tsv", sep="\t", index=False)
significant_interactions.to_csv("significant_interactions.tsv", sep="\t", index=False)
```

**C. Cross‑validated performance of the full pipeline**
```python
pipe = Pipeline([("prep", preprocessor),
                 ("gb", GradientBoostingRegressor(random_state=42, **gb_cv.best_params_))])
scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
print("CV R²:", scores.mean(), "+/-", scores.std())
```

**D. Permutation importance on the test set (robustness)**
```python
from sklearn.inspection import permutation_importance
perm = permutation_importance(gb_cv.best_estimator_, X_test_preprocessed, y_test,
                              n_repeats=20, random_state=42)
```

---

## 🧰 Suggested repo layout

```
.
├── regression_SHAP_pipeline.py
├── Metadata.Final.tsv
├── README.md
└── outputs/               # (optional) save plots & TSVs here
```

To save into `outputs/`, prefix all `plt.savefig()` and `.to_csv()` paths accordingly.

---

## 📎 Reproducible environment (optional)

**requirements.txt**
```
pandas
numpy
scikit-learn
shap
matplotlib
seaborn
scipy
```

**Conda example**
```bash
conda create -n gbshap python=3.10 -y
conda activate gbshap
pip install -r requirements.txt
```

---

## ✅ Quick sanity check

After running the script you should see:
- Console: best GB params (e.g., `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}`), **MSE**, **R²**.  
- Files: `shap.summary_plot.GB.GA.png`, `FeatureInteractionHeatmap.GB.GA.png`, residual plots, etc.  
- Optional TSVs if you enable artifact saving.

If something looks off (e.g., R² << 0.1), inspect residual plots and the SHAP dependence plots for nonlinearity/outliers.

---

## 👤 Author

**Jeff Haltom, PhD**  
Bioinformatics Scientist II — Children’s Hospital of Philadelphia  
GitHub: https://github.com/jeff-haltom

---

*This README accompanies the exact script content provided and adds practical guardrails for large‑scale, interpretable regression modeling with SHAP.*
