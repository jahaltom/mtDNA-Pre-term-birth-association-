
# Random Forest Pipeline for PTB Classification (with SHAP, Interactions, PDP/ICE)

This script trains a **RandomForestClassifier** to predict **PTB** (binary) from mixed tabular covariates.  
It uses a leakage‑safe `Pipeline` with `ColumnTransformer`, tunes RF hyperparameters by **Average Precision (PR AUC)**, evaluates on a **stratified train/test split**, and generates **ROC/PR curves, SHAP global/interaction plots, PDP/ICE**, plus a simple **nonlinearity score** computed from PDPs.

---

## TL;DR — What the script does

1. **Load data** from `Metadata.Final.tsv` (tab‑separated) into `df`.
2. **Define features** in three groups:
   - Categorical → one‑hot encoded (`OneHotEncoder(handle_unknown="ignore")`).
   - Continuous → standardized (`StandardScaler`).
   - Binary → passthrough.
   - A safety assert prevents leaking `haplogroup/site` variables into covariates.
3. **Target** is `PTB` coerced to `int`.
4. **Preprocessing** is wrapped in a `ColumnTransformer` and placed **inside** a `Pipeline` to avoid leakage during CV.
5. **Model** is `RandomForestClassifier` with `class_weight="balanced"` to handle imbalance.
6. **Split** the data into train/test with `train_test_split(..., stratify=y, random_state=42)` to **preserve class ratios**.  
7. **Tune** RF hyperparameters via `GridSearchCV` using **`average_precision`** (PR AUC) across a 5‑fold **StratifiedKFold**.
8. **Evaluate** the best model on the held‑out test set:
   - Classification report at threshold 0.5
   - ROC AUC and PR AUC
   - Save `roc_auc.png`, `pr_auc.png`
9. **Explain** the model with SHAP:
   - Compute SHAP values on a test‑set subset (≤2000 rows).
   - **Normalize shapes** so the SHAP matrix is `(n_samples, n_features)` for the **positive class**.
   - Plot `shap_summary_top30.png` (global importance) and SHAP **interaction** plots/heatmap.
10. **Inspect nonlinearity** using PDP/ICE:
    - PDP for the top 12 features (`pdp_top12.png`).
    - ICE + PDP for BMI (`ice_bmi.png`).
    - A **nonlinearity score** per feature (spline R² − linear R²) saved to `nonlinearity_scores.csv`.

---

## Why these choices

- **Pipeline + ColumnTransformer:** keeps preprocessing inside CV → **no data leakage**.
- **Stratified train/test split:** `stratify=y` maintains the PTB case ratio in both train and test → **fair evaluation on imbalanced data**.
- **Average Precision (PR AUC):** better reflects performance on **rare positive labels** than ROC AUC.
- **`class_weight="balanced"`:** RF natively compensates for class imbalance without oversampling.
- **SHAP for trees:** gives consistent, local additivity explanations; **interaction values** help find feature pair effects.
- **PDP/ICE + nonlinearity score:** visualize and quantify curved relationships beyond linear effects.

---

## Important implementation details & gotchas

### 1) OneHotEncoder and versioning
- The script uses `OneHotEncoder(..., sparse_output=True)`, which requires **scikit‑learn ≥ 1.2**.  
  On older versions, change to `sparse=True`.

### 2) Sparse matrices
- The preprocessor emits a **sparse** design matrix (CSR). RF can consume CSR directly.  
- For SHAP figures, the code densifies with `toarray()` when needed.

### 3) SHAP shape normalization (binary classification)
Different SHAP versions return different shapes for a classifier:
- **List of arrays**: `[ (N,F) for each class ]`
- **3‑D array**: `(N, F, C)` or `(N, C, F)`

This script **selects the positive class** along the correct axis and asserts `sv.shape == (N, F)` before plotting.  
This prevents the common beeswarm error caused by `(N,2,...)` shapes.

### 4) Threshold vs. PR AUC optimization
- We tune by **PR AUC** (threshold‑free), but report a **0.5** threshold in `classification_report`.  
- If operating point matters, add a small threshold sweep (on validation or via CV) to maximize F1 or meet a recall target.

### 5) Probability calibration (optional)
- RF probabilities can be mis‑calibrated. If you use probabilities downstream, consider:
  - `CalibratedClassifierCV(best, method="isotonic"|"sigmoid")` on a validation split, or
  - Train with cross‑validated calibration after model selection.

### 6) Interaction plots are **O(N × K²)**
- `shap_interaction_values` can be heavy. The script caps rows (≤2000) and features (top‑k) to stay tractable.

---

## How to run

### 1) Install requirements
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pandas numpy scikit-learn matplotlib seaborn shap
```

> **Note:** for `sparse_output=True` you need scikit‑learn ≥ 1.2.

### 2) Prepare the data
- Put `Metadata.Final.tsv` in the working directory.
- Ensure the following columns exist (or edit the lists in the script):
  - **Categorical:** `DRINKING_SOURCE, FUEL_FOR_COOK, TOILET, WEALTH_INDEX`
  - **Continuous:** `PW_AGE, PW_EDUCATION, MAT_HEIGHT, MAT_WEIGHT, BMI`
  - **Binary:** `BABY_SEX, CHRON_HTN, DIABETES, HH_ELECTRICITY, TB, THYROID, TYP_HOUSE`
  - **Target:** `PTB` (0/1)

### 3) Run the script
```bash
python your_rf_script.py
```

### 4) Outputs
- **Console:**
  - Best hyperparameters
  - Classification report (threshold 0.5)
  - ROC AUC, PR AUC
  - Top SHAP interaction pairs
- **Figures:**
  - `roc_auc.png`, `pr_auc.png`
  - `shap_summary_top30.png`
  - `shap_interaction_summary_topk.png`
  - `shap_interactions_heatmap_topk.png`
  - `dep_<feature>.png` for top features
  - `pdp_top12.png`, `ice_bmi.png`
- **Tables:**
  - `nonlinearity_scores.csv`

---

## Suggested enhancements (optional)

1. **Threshold selection:** choose a threshold that maximizes F1 or attains a target recall/precision on CV folds.
2. **Permutation importance:** add `sklearn.inspection.permutation_importance` on the test set to validate RF importances.
3. **Group‑aware CV (if needed):** if observations cluster by site/family, use `StratifiedGroupKFold` to respect grouping.
4. **Feature stability:** bootstrap feature importances / SHAP to assess ranking stability.
5. **Model zoo:** compare RF to Logistic Regression (`class_weight="balanced"`), Gradient Boosting, XGBoost/LightGBM, or CatBoost.
6. **Calibration:** add a post‑fit calibration step if probabilities are used for risk communication.

---

## Reproducibility checklist

- Fix your package versions (e.g., `requirements.txt`).
- Use `random_state=42` (already set) and `stratify=y` in the train/test split.
- Keep preprocessing inside the `Pipeline`.
- Do not touch the test set until model selection is finalized.
- Save artifacts if needed: `joblib.dump(best, "best_rf_pipeline.joblib")`.

---

## Troubleshooting

- **`TypeError: OneHotEncoder got unexpected argument 'sparse_output'`**  
  → You’re on scikit‑learn < 1.2. Use `sparse=True` instead.

- **`ValueError: operands could not be broadcast together …` in `shap.summary_plot`**  
  → You passed a 3‑D SHAP array. The script already normalizes to `(N,F)` for the positive class; ensure that block is intact.

- **Long runtime / high memory on SHAP interactions**  
  → Reduce the subset size and/or `topk`. Interactions are O(N × K²).

- **`KeyError` in PDP for feature names**  
  → Confirm your `feat_names_full` cleaning and `top_names` are names (strings), not indices.

---

## Interpretation cues for your figure

- **Left** = lower predicted PTB risk; **Right** = higher risk.  
- **Blue → Pink** = low → high raw feature value.  
- Clean monotones (e.g., `MAT_HEIGHT`) show color gradients aligned with left/right.  
- Mixed colors on both sides (e.g., `BMI`, `MAT_WEIGHT`) indicate **nonlinearity** or **interactions** (check interaction heatmap).
