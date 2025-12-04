# GA Linear Regression Pipeline with Ridge, LASSO, ElasticNet, and SHAP

This README documents a Python script that models **gestational age at birth (GA; `GAGEBRTH`)** as a **continuous outcome** using:

- **Ridge regression** (`RidgeCV`)
- **LASSO regression** (`LassoCV`)
- **Elastic Net regression** (`ElasticNetCV`)

The pipeline includes **site-aware train/test splitting**, a unified preprocessing pipeline, coefficient-based feature importance plots, and **SHAP-based interpretability** for the Elastic Net model.

The script assumes a metadata file named `Metadata.Final.tsv` with one row per individual.

---

## 1. Overview

The script performs the following steps:

1. **Load metadata** from `Metadata.Final.tsv` (tab-delimited).
2. **Define feature sets** from command-line arguments:
   - Categorical features
   - Continuous (numeric) features
   - Binary features
3. **Construct a site-aware train/test split**:
   - Uses `GroupShuffleSplit` when there are ≥3 sites (unseen-site test).
   - Falls back to a standard row-wise split when there are 0–2 sites.
4. **Preprocess features** using a `ColumnTransformer`:
   - Standardize continuous variables.
   - Pass binary variables through unchanged.
   - One-hot encode categorical variables (with `handle_unknown="ignore"`).
5. **Fit three linear regression models**:
   - **Ridge regression** with `RidgeCV` over a grid of alphas.
   - **LASSO regression** with `LassoCV` (internal CV to choose alpha).
   - **Elastic Net regression** with `ElasticNetCV` (tuning alphas and `l1_ratio`).
6. **Evaluate performance** on the test set:
   - Mean Squared Error (MSE).
   - R-squared (R²).
   - Metrics written to text files.
7. **Extract and plot feature coefficients**:
   - Coefficient scatterplots for each model, with readable feature names.
   - Coefficients saved to text files.
8. **Run SHAP analysis** on the Elastic Net model:
   - Global importance (mean |SHAP|) and summary plot.
   - SHAP value summaries written to text.
   - Dependence plots for top numeric/binary features.

This provides a consistent, interpretable framework for modeling GA, parallel to a PTB logistic regression pipeline.

---

## 2. Data Requirements

### 2.1 Input file

The script expects:

```text
Metadata.Final.tsv
```

with:

- **Delimiter**: Tab (`\t`)
- **One row** per individual/sample.
- A continuous outcome column:

```text
GAGEBRTH
```

representing gestational age at birth (in days).

- An optional **site column**:

```text
site
```

used for site-aware train/test splitting. If present, it should contain site identifiers (e.g., `BD1`, `BD2`, `PK`, `PEMBA`, `ZAMBIA`).

### 2.2 Feature columns

At runtime, you specify three comma-separated lists of column names:

1. **Categorical features** (excluding `site`)

   These will be one-hot encoded.

2. **Continuous features**

   These will be standardized with `StandardScaler`.

3. **Binary features**

   These will be passed through unchanged.

Example columns might include:

- Categorical: `MainHap`, `SEX`, `ETHNIC`, etc.
- Continuous: `AGE`, `BMI`, `PC1`, `PC2`, ...
- Binary: `SMOKING`, `DIABETES`, `HYPERTENSION`, etc.

The script explicitly **removes `site` from the modeling features** if you accidentally include it in the categorical list, while still using it for grouping in the split.

---

## 3. Command-Line Usage

Run the script with three command-line arguments:

```bash
python ga_linear_shap.py \
  "MainHap,SEX,ETHNIC,SITE_A,SITE_B" \
  "AGE,BMI,PC1,PC2,PC3" \
  "SMOKING,DIABETES,HYPERTENSION"
```

Arguments:

1. `sys.argv[1]`: comma-separated **categorical** column names  
2. `sys.argv[2]`: comma-separated **continuous** column names  
3. `sys.argv[3]`: comma-separated **binary** column names  

Notes:

- Do **not** include `site` as a feature; if present it will be dropped from the feature matrix and only used for splitting.
- All specified columns must exist in `Metadata.Final.tsv`.

---

## 4. Site-Aware Train/Test Split

The script uses a **site-aware** logic for splitting the data into train and test sets:

```python
if "site" in df.columns:
    n_sites = df["site"].nunique()
else:
    n_sites = 0

if ("site" in df.columns) and (n_sites >= 3):
    # With 3+ sites, a true unseen-site test is meaningful
    groups_all = df["site"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
elif ("site" in df.columns) and (n_sites == 2):
    # With only 2 sites, prefer site-aware CV elsewhere and
    # just do a standard row-wise split here
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
else:
    # No / insufficient site info → standard split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
```

Key points:

- For **≥3 sites**, the test set is composed of **unseen sites**, improving generalizability across cohorts.
- For **2 sites or fewer**, the script uses a standard random split (you can still use site-aware CV separately if desired).

---

## 5. Preprocessing

A `ColumnTransformer` handles all preprocessing:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)
```

- Continuous features: standardized (mean 0, variance 1).
- Binary features: passed through unchanged.
- Categorical features: one-hot encoded with robustness to unseen categories.

The preprocessor is fit on the **training data** only and then applied to both train and test sets:

```python
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed  = preprocessor.transform(X_test)
```

---

## 6. Models and Evaluation

### 6.1 Evaluation helper

A helper function evaluates models on the held-out test set:

```python
def evaluate_model_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    # Writes metrics to a text file named like: LinGA._metrics.<model_name>.txt
    # Contents include:
    # - Mean Squared Error (MSE)
    # - R-squared (R²)
```

Conceptually, the text file will contain:

- `Mean Squared Error (MSE)`
- `R-squared (R²)`

for the given model.

### 6.2 Ridge Regression (RidgeCV)

The script first fits a **Ridge regression** model with cross-validated regularization strength:

```python
ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
ridge.fit(X_train_preprocessed, y_train)
evaluate_model_regression(ridge, X_test_preprocessed, y_test, "RidgeRegression")
```

- `RidgeCV` automatically chooses the best `alpha` from the grid `[1e-3, ..., 1e3]` using 5-fold CV.
- The fitted coefficients are extracted and sorted by absolute value:

  ```python
  ridge_importance = pd.DataFrame({
      "Feature": preprocessor.get_feature_names_out(),
      "Coefficient": ridge.coef_,
  }).sort_values(by="Coefficient", key=abs, ascending=False)
  ```

- Coefficients are:
  - Saved to a text file (e.g., `RidgeImportancee.txt`).
  - Visualized with `plot_feat` as:

    ```text
    Ridge_TopFeature.LinReg.GA.png
    ```

### 6.3 LASSO Regression (LassoCV)

Next, the script fits a **LASSO** model:

```python
lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
lasso.fit(X_train_preprocessed, y_train)
evaluate_model_regression(lasso, X_test_preprocessed, y_test, "LassoRegression")
```

- `LassoCV` performs 5-fold CV over a range of alpha values (scikit-learn default path).
- Coefficients are obtained, sorted, and plotted:

  - Text file: `LassoImportancee.txt`
  - Plot: `Lasso_TopFeature.LinReg.GA.png`

LASSO tends to produce **sparse** solutions (many coefficients exactly zero), helping feature selection and interpretability.

### 6.4 Elastic Net Regression (ElasticNetCV)

Finally, the script fits an **Elastic Net** model, balancing L1 and L2 penalties:

```python
elasticnet = ElasticNetCV(
    alphas=np.logspace(-3, 1, 20),   # 0.001 → 10
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5,
    random_state=42,
)
elasticnet.fit(X_train_preprocessed, y_train)
evaluate_model_regression(elasticnet, X_test_preprocessed, y_test, "ElasticNetRegression")
```

- `ElasticNetCV` jointly tunes:
  - `alpha` (strength of regularization).
  - `l1_ratio` (balance between L1 and L2 penalties).
- Coefficients are saved and plotted:

  - Text file: `ElacticImportancee.txt`
  - Plot: `ElasticNet_TopFeature.LinReg.GA.png`

This model is typically used for SHAP analysis in the script.

---

## 7. Coefficient Plotting

The shared `plot_feat` function:

```python
def plot_feat(coefMat, model_name):
    top_features = coefMat.copy()
    top_features["Feature"] = top_features["Feature"].str.replace("^cat__MainHap", "Haplogroup", regex=True)
    top_features["Feature"] = top_features["Feature"].str.replace("^cat__|^num__",
                                                                 "", regex=True)
    plt.figure(figsize=(12, 6))
    colors = np.where(top_features["Coefficient"] > 0, "blue", "red")
    plt.scatter(top_features["Feature"], top_features["Coefficient"],
                color=colors, s=100, edgecolor="black")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylabel("Coefficient")
    plt.title("Top Significant Features by Coefficients")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(model_name + "_TopFeature.LinReg.GA.png", bbox_inches="tight")
    plt.clf()
```

Key points:

- Renames features starting with `cat__MainHap` to `Haplogroup` for interpretability.
- Strips technical prefixes (`cat__`, `num__`) from feature names.
- Colors:
  - **Blue** for positive coefficients (associated with increased GA).
  - **Red** for negative coefficients (associated with decreased GA).
- Inverts the y-axis so the strongest positive effects appear at the top.

---

## 8. SHAP Analysis (Elastic Net)

The script uses **SHAP** to interpret the Elastic Net model:

```python
feature_names = preprocessor.get_feature_names_out()

explainer = shap.LinearExplainer(elasticnet, X_train_preprocessed)
shap_values = explainer.shap_values(X_test_preprocessed)  # (n_samples, n_features)
```

- `shap_values` is expected to have shape `(N, F)` for a regression model.
- SHAP values represent the **per-feature contribution** to the predicted GA (relative to a baseline).

### 8.1 Global importance and summary plot

Global feature importance is computed as:

```python
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs_shap)[::-1]
top_k = min(30, len(feature_names))
top_idx = order[:top_k]
top_feature_names = feature_names[top_idx]
top_shap_values = shap_values[:, top_idx]
```

The script then creates a **summary plot** of the top 30 features:

```python
shap.summary_plot(
    top_shap_values,
    X_test_preprocessed[:, top_idx],
    feature_names=top_feature_names,
    show=False,
    max_display=top_k,
)
plt.tight_layout()
plt.savefig("shap_summary_top30.ElasticNet.GA.png", dpi=300, bbox_inches="tight")
plt.close()
```

It also writes a text summary of the top 20 features by mean |SHAP| to:

```text
ElasticNetSHAP.txt
```

containing lines like:

```text
<feature_name>: <mean_abs_shap_value>
```

### 8.2 Dependence plots

For the top numeric/binary features (identified by `num__` or `bin__` prefixes), the script generates **SHAP dependence plots**:

```python
num_prefixes = ("num__", "bin__")
top_numeric_feats = [f for f in top_feature_names if f.startswith(num_prefixes)]

for fname in top_numeric_feats[:10]:
    shap.dependence_plot(
        fname,
        shap_values,
        X_test_preprocessed,
        feature_names=feature_names,
        show=False,
    )
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in fname)
    plt.tight_layout()
    plt.savefig(f"shap_dependence.ElasticNet.GA.{safe}.png", dpi=300, bbox_inches="tight")
    plt.close()
```

These plots help visualize:

- The relationship between feature values and their SHAP contributions.
- Potential non-linearities or interactions affecting GA.

---

## 9. Outputs Summary

After running the script, expected outputs include:

- **Metrics files**:
  - `LinGA._metrics.RidgeRegression.txt`
  - `LinGA._metrics.LassoRegression.txt`
  - `LinGA._metrics.ElasticNetRegression.txt`
- **Feature importance tables**:
  - `RidgeImportancee.txt`
  - `LassoImportancee.txt`
  - `ElacticImportancee.txt` (Elastic Net)
- **Coefficient plots**:
  - `Ridge_TopFeature.LinReg.GA.png`
  - `Lasso_TopFeature.LinReg.GA.png`
  - `ElasticNet_TopFeature.LinReg.GA.png`
- **SHAP outputs**:
  - `shap_summary_top30.ElasticNet.GA.png`
  - `ElasticNetSHAP.txt`
  - `shap_dependence.ElasticNet.GA.<feature>.png` (multiple files)

These artifacts provide both **predictive performance** and **interpretability** for GA modeling.

---

## 10. Dependencies

You will need:

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `shap`
- `matplotlib`
- `seaborn`

Install with:

```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn
```

or, with conda:

```bash
conda install pandas numpy scikit-learn matplotlib seaborn
pip install shap
```

---

## 11. Reproducibility Notes

- Random seeds (`random_state=42`) are used for:
  - `GroupShuffleSplit`
  - `train_test_split`
  - `LassoCV`, `ElasticNetCV`
- `RidgeCV` chooses alpha based on 5-fold CV.
- Train and test sets are **site-aware** when there are ≥3 sites; otherwise, a standard random split is used.

---

## 12. Suggested Extensions

Potential extensions:

- Add **cross-validated pipelines** that include both preprocessing and modeling in a single `Pipeline`.
- Experiment with **non-linear models** (e.g., Random Forest, Gradient Boosting) while keeping this linear + SHAP pipeline as a primary interpretable baseline.
- Stratify on binned GA if you want more balanced distributions in train/test for certain GA ranges.

This script provides a robust, interpretable baseline for **continuous GA modeling** across multi-site cohorts.
