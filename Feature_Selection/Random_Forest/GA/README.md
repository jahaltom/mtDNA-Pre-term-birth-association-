
# Random Forest Regression for Gestational Age with SHAP Analysis

This script trains a **Random Forest regression model** to predict gestational age at birth (`GAGEBRTH`) from a mix of categorical, continuous, and binary predictors. It performs:

- Preprocessing (scaling, one‑hot encoding)
- Hyperparameter tuning with cross‑validation (optionally **grouped by site**)
- Model evaluation on a held‑out test set
- Feature importance ranking
- **SHAP-based** global and interaction analyses
- Partial Dependence Plots (PDPs)
- A heuristic search for **non‑linear feature effects**

The goal is to understand which features (and feature interactions) are most associated with gestational age and whether their effects appear linear or non‑linear in the fitted Random Forest.

---

## 1. Input Data

The script expects a TSV file named:

```text
Metadata.Final.tsv
```

in the working directory, with at least the following columns:

- **Outcome**
  - `GAGEBRTH`: gestational age at birth (numeric, in days)

- **Predictors**
  - Categorical features (you pass these in via `sys.argv[1]`)
  - Continuous features (you pass these in via `sys.argv[2]`)
  - Binary features (you pass these in via `sys.argv[3]`)

Optionally (for group-wise CV):

- `site`: site identifier used as a **grouping variable** for cross‑validation.  
  - If present and `n_unique(site) ≥ 2`, the script uses **GroupKFold** over sites inside the training set.
  - If `site` is missing or has `< 2` unique values in the training set, the script falls back to ordinary K‑fold CV.

On startup, the script checks that all requested columns plus `GAGEBRTH` are present and will raise a `ValueError` if any are missing.

---

## 2. Command‑Line Arguments

The script is intended to be run from the command line and takes **three positional arguments**, each a comma‑separated list of column names:

```bash
python rf_ga_shap.py   "cat1,cat2,cat3"   "age,bmi,PC1,PC2,PC3"   "sex,smoker,diabetes"
```

Arguments:

1. `sys.argv[1]`: **categorical_columns**  
   - e.g. `"MainHap,SubHap,site"` (or you may choose to drop `site` here and keep it purely as a grouping variable)
2. `sys.argv[2]`: **continuous_columns**  
   - e.g. `"GAGEBRTH_mom,PC1,PC2,PC3,PC4,PC5"`
3. `sys.argv[3]`: **binary_columns**  
   - e.g. `"PTB,is_female,is_smoker"`

Internally, the script constructs:

```python
categorical_columns = sys.argv[1].split(',')
continuous_columns  = sys.argv[2].split(',')
binary_columns      = sys.argv[3].split(',')
```

and then uses these lists to build the model matrix.

---

## 3. Preprocessing Pipeline

A `ColumnTransformer` is used to preprocess features:

- **Continuous columns**
  - Processed with `StandardScaler()`
- **Binary columns**
  - Passed through unchanged (`"passthrough"`)
- **Categorical columns**
  - One‑hot encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

The preprocessor is combined with the Random Forest model in a single `Pipeline`:

```python
pipe = Pipeline([
    ("prep", preprocessor),
    ("rf", rf),
])
```

This ensures that all transformations learned on the training data (scaling and one‑hot encoding) are applied consistently to the test set and throughout cross‑validation and SHAP analyses.

---

## 4. Model: Random Forest Regressor

The core model is a `RandomForestRegressor`, configured for regression on `GAGEBRTH`:

- `n_estimators`: tuned over `[300, 600, 900]`
- `max_depth`: tuned over `[None, 10, 20]`
- `min_samples_leaf`: tuned over `[1, 2, 5]`
- `max_features`: tuned over `["sqrt", 0.5]`
- `random_state = 42`, `n_jobs = -1`

These are arranged into a hyperparameter grid:

```python
param_grid_rf = {
    "rf__n_estimators": [300, 600, 900],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_leaf": [1, 2, 5],
    "rf__max_features": ["sqrt", 0.5],
}
```

---

## 5. Train / Test Split and Cross‑Validation

1. **Train/Test Split**

   The script first does a standard random split:

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
   )
   ```

   - `X` contains all predictors from the three user‑supplied feature lists.
   - `y` is the gestational age (`GAGEBRTH`).

2. **Cross‑Validation Strategy**

   - If a `site` column exists and there are at least 2 unique sites **within the training set**, the script uses **GroupKFold** with sites as groups to tune hyperparameters.
   - Otherwise, it falls back to a standard `KFold` with 5 splits and shuffling.

   In both cases, `GridSearchCV` is run with:

   - `scoring="neg_mean_squared_error"`
   - `n_jobs=-1` (parallelized)
   - The `pipe` (preprocessor + RF) as the estimator
   - The hyperparameter grid `param_grid_rf`

The **best hyperparameters** are printed at the end of the grid search.

---

## 6. Model Evaluation

After hyperparameter tuning, the script evaluates the best model on the held‑out test set using:

- Mean Squared Error (MSE)
- R‑squared (R²)

```python
Mean Squared Error (MSE): <value>
R-squared: <value>
```

This gives a quick quantitative sense of model fit.

---

## 7. Feature Importances and RFE

### 7.1. Random Forest Feature Importances

Using the best fitted pipeline:

- Extract the fitted preprocessor and RF model:

  ```python
  best_pipe   = rf_cv.best_estimator_
  rf_model    = best_pipe.named_steps["rf"]
  fitted_prep = best_pipe.named_steps["prep"]
  ```

- Get the feature names from the preprocessor:

  ```python
  feature_names = fitted_prep.get_feature_names_out()
  ```

- Retrieve and tabulate `rf_model.feature_importances_` aligned with `feature_names`.

The script prints the **top 10 most important features** for quick inspection.

### 7.2. Recursive Feature Elimination (RFE)

The script then:

1. Transforms `X_train` through the fitted preprocessor to get a dense design matrix.
2. Clones the best RF model (`rf_for_rfe`).
3. Runs `RFE` to select up to 20 features (or fewer if fewer features exist).

The list of **RFE‑selected features** is printed, which gives a second, somewhat more conservative view of which features might be most important.

---

## 8. SHAP Analyses

The script uses the **SHAP TreeExplainer** for Random Forest to provide global and interaction‑level explanations.

### 8.1. SHAP Summary Plot

- Fits a `shap.TreeExplainer(rf_model)` on the dense, preprocessed training data.
- Computes SHAP values for all samples and features.
- Generates a **summary plot** (beeswarm) and saves it as:

```text
shap.summary_plot.RF.GA.png
```

This figure ranks features by overall importance and shows the direction and spread of their effects.

### 8.2. SHAP Interaction Values and Heatmap

- Computes SHAP **interaction values**, which quantify pairwise feature interactions.
- Averages their absolute values across samples to form a feature‑by‑feature **interaction matrix**.
- Extracts the top interaction pairs and prints the top 10.
- Restricts to the **top K features** (by main SHAP effect, default `K = 30`) and plots a heatmap:

```text
shap_interactions_heatmap_top.png
```

This gives a high‑level view of the strongest interactions among top features.

### 8.3. SHAP Interaction Summary Plot (Top‑K)

For the same top‑K features, the script:

- Recomputes interaction values only on the top‑K subset.
- Produces a SHAP **interaction summary** plot:

```text
shap_interaction_summary_topk.png
```

This visualizes how interactions distribute across features, focusing on the most important ones.

---

## 9. Partial Dependence Plots (PDPs)

Two kinds of PDPs are generated using `sklearn.inspection.PartialDependenceDisplay`:

1. **2D PDPs for top interaction pairs**

   - Based on the strongest SHAP interaction pairs, the script:
     - Maps feature names to their column indices in the transformed space.
     - Generates 2D PDPs for the top 5 pairs.
   - Output file:

   ```text
   pdp_top_interactions.png
   ```

2. **1D PDPs for top non‑linear features**

   - After scoring non‑linearity (see below), the script generates 1D PDPs for the top non‑linear features.
   - Output file:

   ```text
   pdp_top_nonlinear.png
   ```

These plots complement SHAP by showing the isolated marginal effect of each feature (or feature pair).

---

## 10. Non‑Linearity Scoring and Dependence Plots

To identify which features exhibit **non‑linear relationships** with the prediction:

1. For each feature, the script computes a **nonlinearity score**:

   - Fit a **linear** model of SHAP values vs feature.
   - Fit a **cubic polynomial** model of SHAP values vs feature.
   - The score is `max(0, R²_cubic − R²_linear)`.

2. It ranks features by this non‑linearity gain and prints the **top 10**.

3. For the top non‑linear features (default: top 8), the script produces SHAP **dependence plots**, which:

   - Show the scatter of SHAP values vs the feature’s values.
   - Optionally color points by the strongest interacting partner.

Each dependence plot is saved as:

```text
shap_dependence_<feature_name_sanitized>.png
```

---

## 11. Outputs Summary

After a successful run, you can expect the following files:

- `shap.summary_plot.RF.GA.png`  
  **Global SHAP summary** for all features.

- `shap_interactions_heatmap_top.png`  
  Heatmap of SHAP interaction strengths among top‑K features.

- `shap_interaction_summary_topk.png`  
  SHAP interaction summary for the top‑K features.

- `pdp_top_interactions.png`  
  2D PDPs for the strongest feature interaction pairs.

- `pdp_top_nonlinear.png`  
  1D PDPs for the strongest non‑linear features.

- `shap_dependence_<feature>.png`  
  Individual SHAP dependence plots for top non‑linear features.

You may also see console output including:

- MSE and R² on the test set
- Best hyperparameters from `GridSearchCV`
- Top features by RF importance
- RFE‑selected features
- Top interaction pairs
- Top non‑linear features and their scores

---

## 12. Suggested Usage Patterns

- Use this script when you want both **predictive performance** and deep **interpretability** of how features (and their interactions) relate to gestational age.
- Start with a modest set of features to keep SHAP and interaction computations manageable.
- If your dataset is very large, consider:
  - Subsampling rows before computing SHAP **interaction** values.
  - Reducing the number of categorical levels (or grouping rare levels) to keep the feature space more compact.

---

## 13. Dependencies

You will need at least:

- Python 3.8+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `shap`

You can install the main stack via:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap
```

---

## 14. Reproducibility Notes

- The Random Forest is initialized with a fixed `random_state = 42`.
- `train_test_split` also uses `random_state = 42`.
- This should give you reproducible splits and model fits, assuming the same scikit‑learn and SHAP versions.

---

## 15. Extending the Script

Some natural extensions:

- Swap in **Gradient Boosting**, **XGBoost**, or **LightGBM** models for comparison.
- Add **GroupShuffleSplit** for a fully group‑aware train/test split (e.g., if you want to strictly evaluate generalization to unseen sites).
- Log metrics and plots to a tracking tool (MLflow, W&B, etc.).
- Wrap the logic in functions so you can import and call it from Jupyter/Colab notebooks.
