# PTB Logistic Regression Pipeline with SHAP

This repository contains a Python script that trains **penalized logistic regression models** to predict **preterm birth (PTB)** from clinical, demographic, and mitochondrial features, with **site-aware splitting**, **regularization (L1/L2)**, and **SHAP-based model interpretability**.

The script is designed to be run from the command line and assumes a metadata file named `Metadata.Final.tsv` with one row per individual.

---

## 1. Overview

The pipeline performs the following steps:

1. **Load metadata** from `Metadata.Final.tsv` (tab-delimited).
2. **Define feature sets** from command-line arguments:
   - Categorical features
   - Continuous (numeric) features
   - Binary features
3. **Construct a site-aware train/test split**:
   - Uses `GroupShuffleSplit` when there are ≥3 sites (unseen-site test).
   - Falls back to stratified row-wise splits when there are 0–2 sites.
4. **Preprocess features** with a `ColumnTransformer`:
   - Standardize continuous variables.
   - Leave binary variables as-is.
   - One-hot encode categorical variables (with `handle_unknown="ignore"`).
5. **Fit two logistic regression models** with class imbalance handling:
   - **L1-penalized logistic regression (LASSO)** via `LogisticRegressionCV`.
   - **L2-penalized logistic regression (ridge)** via `LogisticRegressionCV`.
6. **Evaluate performance** on the test set:
   - Classification report (precision, recall, F1, support).
   - ROC AUC and ROC curves (where applicable).
7. **Extract and plot feature coefficients**:
   - Non-zero coefficients for LASSO and ridge.
   - Coefficient scatterplots with cleaned feature names.
8. **Run SHAP analysis** on the ridge logistic model:
   - Global importance (mean |SHAP|).
   - Summary plot.
   - Dependence plots for top numeric/binary features.
9. **Write results to disk**:
   - Metrics and significant feature lists as text files.
   - Coefficient plots, ROC curves, and SHAP visualizations as PNGs.

---

## 2. Data Requirements

### 2.1 Input file

The script expects a file:

```text
Metadata.Final.tsv
```

with:

- **Delimiter**: Tab (`\t`)
- **One row** per individual/sample.
- A binary outcome column named:

```text
PTB
```

where (by convention):

- `0` = term birth
- `1` = preterm birth

- An optional **site column**:

```text
site
```

used for site-aware splitting. If present, it should contain site identifiers (e.g., `BD1`, `BD2`, `PK`, etc.).

### 2.2 Feature columns

At runtime, you specify three comma-separated lists of column names:

1. **Categorical features** (excluding `site`):

   These will be one-hot encoded.

2. **Continuous features**:

   These will be standardized with `StandardScaler`.

3. **Binary features**:

   These will be passed through without transformation.

Example columns might include mtDNA haplogroup, age, BMI, PCs, etc. The script **removes** the `site` column from the feature list automatically, even if you include it in the categorical list.

---

## 3. Command-Line Usage

Run the script with three command-line arguments:

```bash
python ptb_logistic_shap.py \
  "MainHap,SEX,ETHNIC,SITE_A,SITE_B" \
  "AGE,BMI,PC1,PC2,PC3" \
  "SMOKING,DIABETES,HYPERTENSION"
```

Arguments:

1. `sys.argv[1]`: comma-separated **categorical** column names  
2. `sys.argv[2]`: comma-separated **continuous** column names  
3. `sys.argv[3]`: comma-separated **binary** column names  

Notes:

- Do **not** include `site` in the categorical list; if it is included, the script explicitly drops it from the modeling features but still uses it for grouping.
- All specified columns must exist in `Metadata.Final.tsv`.

---

## 4. Site-Aware Train/Test Split

The script uses a **site-aware** logic for splitting:

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
elif ("site" in df.columns) and (n_sites == 2):
    # With only 2 sites, we do a stratified row-wise split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
else:
    # No / insufficient site info → standard stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
```

This ensures:

- When there are **≥3 sites**, the test set contains completely unseen sites.
- When there are **2 or fewer sites**, the script falls back to a **stratified** split on PTB status.

---

## 5. Preprocessing

A `ColumnTransformer` handles preprocessing:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)
```

- Continuous features are standardized (mean 0, unit variance).
- Binary features are kept as-is.
- Categorical features are one-hot encoded with unknown-category robustness.

The preprocessor is fit on the **training data only**, and then applied to both train and test sets.

---

## 6. Models and Evaluation

### 6.1 L1-penalized Logistic Regression (LASSO)

The script fits an L1-penalized logistic regression via `LogisticRegressionCV`:

```python
lasso = LogisticRegressionCV(
    penalty="l1",
    solver="saga",
    cv=5,
    max_iter=5000,
    class_weight="balanced",
    random_state=42,
)
```

- `class_weight="balanced"` handles PTB class imbalance.
- Cross-validation selects the inverse regularization strength `C`.

The evaluation helper:

```python
evaluate_model(lasso, X_test_preprocessed, y_test, "LassoRegression")
```

writes:

- A **classification report** (precision, recall, F1, support) to a metrics text file.
- A **ROC curve** and `ROC AUC` to PNG output.

Non-zero coefficients are extracted, sorted by absolute value, and plotted:

- Text file: `LassoSigFeat.txt`
- Coefficient plot: `LASSO_TopFeature.LogReg.PTB.png` (scatter plot)

### 6.2 L2-penalized Logistic Regression (Ridge)

Similarly, the ridge logistic model is fit via `LogisticRegressionCV`:

```python
ridge = LogisticRegressionCV(
    penalty="l2",
    solver="saga",
    cv=5,
    max_iter=5000,
    class_weight="balanced",
    random_state=42,
)
```

Evaluation is done via:

```python
evaluate_model(ridge, X_test_preprocessed, y_test, "RidgeRegression")
```

Outputs include:

- Classification metrics text file.
- ROC AUC and ROC curve PNG.
- Ridge significant features in `RidgeSigFeat.txt`.
- Coefficient plot: `Ridge_TopFeature.LogReg.PTB.png`.

### 6.3 Coefficient plotting

The `plot_feat` function:

- Renames `cat__MainHap*` features to `Haplogroup*`.
- Strips `cat__` and `num__` prefixes for readability.
- Produces scatter plots with:
  - Blue = positive coefficient.
  - Red = negative coefficient.
  - Y-axis inverted so strongest positive effects appear at the top.

The plots provide a **quick visual summary** of which features push the log-odds of PTB up or down.

---

## 7. SHAP Analysis

The script runs SHAP on the **ridge logistic** model:

```python
explainer = shap.LinearExplainer(ridge, X_train_preprocessed)
shap_values = explainer.shap_values(X_test_preprocessed)
```

- For some SHAP versions, `shap_values` may be a list; the script normalizes to a `(n_samples, n_features)` NumPy array.
- SHAP values are in **log-odds units** (effect on the decision function).

### 7.1 Global importance

Global feature importance is computed as:

```python
mean_abs_shap = np.abs(shap_values).mean(axis=0)
```

The script then:

- Ranks features by mean |SHAP|.
- Selects the top 30 features.
- Produces a **SHAP summary plot**:

  ```text
  shap_summary_top30.LogReg.Ridge.PTB.png
  ```

- Writes a text summary to:

  ```text
  RidgeSHAP.txt
  ```

with the top 20 features and their mean |SHAP| values.

### 7.2 Dependence plots

For the top numeric/binary features (identified by prefixes `num__` or `bin__`), the script creates **SHAP dependence plots**:

```text
shap_dependence.LogReg.Ridge.PTB.<feature>.png
```

These show:

- The relationship between each feature’s value and its SHAP value.
- Potential non-linear relationships and interactions.

---

## 8. Outputs Summary

After running the script, you should see outputs such as:

- **Metrics files**:
  - `LogitPTB._metrics.LassoRegression.txt`
  - `LogitPTB._metrics.RidgeRegression.txt`
- **Significant feature lists**:
  - `LassoSigFeat.txt`
  - `RidgeSigFeat.txt`
- **Coefficient plots**:
  - `LASSO_TopFeature.LogReg.PTB.png`
  - `Ridge_TopFeature.LogReg.PTB.png`
- **ROC curves**:
  - `ROC_AUC.LogReg.LassoRegression.PTB.png`
  - `ROC_AUC.LogReg.RidgeRegression.PTB.png`
- **SHAP outputs**:
  - `shap_summary_top30.LogReg.Ridge.PTB.png`
  - `RidgeSHAP.txt`
  - `shap_dependence.LogReg.Ridge.PTB.<feature>.png` (multiple files)

These provide both **predictive performance metrics** and **interpretability artifacts** for downstream reporting and figures.

---

## 9. Dependencies

You will need:

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `shap`
- `matplotlib`
- `seaborn`

A typical installation via `pip`:

```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn
```

If you are running in a conda environment:

```bash
conda install pandas numpy scikit-learn matplotlib seaborn
pip install shap
```

---

## 10. Reproducibility Notes

- The script sets `random_state=42` in `GroupShuffleSplit`, `train_test_split`, and `LogisticRegressionCV` for reproducibility.
- Train and test sets are **site-aware** when possible, and **stratified** by PTB to preserve class balance.
- Regularization strengths (`C`) are chosen via 5-fold cross-validation within `LogisticRegressionCV`.

---

## 11. Suggested Extensions

Potential extensions you might add:

- Additional models (e.g., `RandomForestClassifier`, `GradientBoostingClassifier`, `MLPClassifier`) with the same preprocessor.
- PR AUC (precision–recall AUC), which is often informative for imbalanced PTB outcomes.
- Saving the fitted models and preprocessor with `joblib` for later reuse.

For now, this script provides a solid, interpretable baseline for **PTB prediction using penalized logistic regression and SHAP**.
