
# Gradient Boosting + Preprocessing + SHAP (PTB classification)

This repository/script trains a **GradientBoostingClassifier** to predict **PTB** (binary) from a mix of continuous, binary, and categorical covariates. It performs proper preprocessing inside a scikit‑learn `Pipeline`, tunes hyperparameters with cross‑validated **Average Precision (PR AUC)**, evaluates on a held‑out test set, and generates **ROC/PR curves, SHAP summaries, SHAP interaction heatmaps, PDP/ICE plots,** and a simple **nonlinearity score** for top features.

---

## What the script does (step‑by‑step)

1. **Imports**
   - Loads core Python/data viz libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `re`, `os`, `sys`).
   - Loads scikit‑learn tools for **splitting**, **preprocessing**, **pipelines**, **tuning**, **metrics**, and **inspection**.
   - Loads **SHAP** for post‑hoc explainability (global and interaction effects).

2. **Data IO**
   - Reads `Metadata.Final.tsv` (tab‑separated) into a DataFrame `df`.
   - You can also wire CLI args to pass column lists; the current code inlines them.

3. **Column definitions**
   - Defines three groups:
     - `categorical_columns`: one‑hot encoded (drinking source, cooking fuel, toilet type, wealth index).
     - `continuous_columns`: standardized via `StandardScaler` (age, education, height, weight, BMI).
     - `binary_columns`: passed through as is (sex, chronic HTN, diabetes, electricity, TB, thyroid, house type).
   - A safety **assert** prevents accidental leakage: it ensures that haplogroup/site columns are **not** included as covariates.

4. **Target**
   - `y = df["PTB"].astype(int)` (binary outcome). The model predicts the probability of PTB=1.

5. **Preprocessing (`ColumnTransformer`)**
   - Numerical features → `StandardScaler`.
   - Binary features → `passthrough` (no change).
   - Categorical features → `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`.
   - `remainder="drop"` to keep only specified columns; `sparse_threshold=1.0` ensures a sparse matrix is produced when possible (efficient with many one‑hot columns).
   - All transforms live **inside** the pipeline so CV and test splits are leak‑free.

6. **Model**
   - `GradientBoostingClassifier(random_state=42, subsample=1.0)` as the base classifier (robust, fairly fast; good baseline for tabular data).

7. **Pipeline**
   - `pipe = Pipeline([("pre", pre), ("clf", gb)])` ties preprocessing + model into one estimator.

8. **Hyperparameter grid + CV**
   - `param_grid = {"clf__n_estimators": [200, 400], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}`.
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` preserves class ratios per fold.
   - Split once into **train/test** (`test_size=0.3`, stratified, reproducible). Grid search runs **only on the training set** to avoid peeking at the test set.
      - The argument stratify=y in train_test_split ensures that both training and test sets preserve the same positive/negative ratio of PTB cases. This prevents sampling bias and stabilizes performance metrics across splits—especially important for imbalanced datasets.

9. **Class imbalance handling (NO SMOTE)**
   - Since `GradientBoostingClassifier` doesn’t have `class_weight`, positives are **up‑weighted** during `fit` via `sample_weight`.
   - Weight = `(N_negatives / N_positives)`. Training samples with `y=1` get this weight; negatives get weight 1.0.
   - The grid search uses **Average Precision** (`scoring="average_precision"`) which is appropriate for imbalanced data.

10. **Model selection + fit**
    - `GridSearchCV` fits the pipeline with the sample weights: `gs.fit(X_tr, y_tr, clf__sample_weight=sample_weight)`.
    - The best estimator is extracted and printed: `gs.best_params_`.

11. **Evaluation on held‑out test set**
    - Predicts probabilities on `X_te`, computes:
      - `classification_report` at a 0.5 threshold (precision/recall/F1 per class).
      - ROC AUC and PR AUC (Average Precision).
    - Saves **ROC** (`roc_auc.png`) and **PR** (`pr_auc.png`) plots.

12. **SHAP explainability**
    - Transforms the test set with the fitted preprocessor (`pre.transform`) and densifies to a NumPy array.
    - Builds `TreeExplainer` on the fitted GBDT; samples up to 2,000 test points for speed.
    - Computes SHAP values → global feature importances via mean |SHAP|.
    - Plots a **SHAP summary** for the **Top‑30** features: `shap_summary_top30.png`.

13. **SHAP interactions (top‑k only)**
    - Computes `shap_interaction_values` for the top features (can be heavy).
    - Prints the **Top 10 interaction pairs** by mean |interaction|.
    - Writes an **upper‑triangle heatmap**: `shap_interactions_heatmap_topk.png`.

14. **Non‑linear behavior diagnostics**
    - **SHAP dependence plots**: For each top feature, saves `dep_<name>.png` showing SHAP value vs raw feature value (curvature indicates nonlinearity in log‑odds contribution).
    - **PDP/ICE** with `PartialDependenceDisplay`:
      - Plots PDPs for the first 12 top features: `pdp_top12.png`.
      - Plots ICE + PDP for **BMI**: `ice_bmi.png`.
    - **Nonlinearity score** from PDP curves:
      - Fits a straight line vs a cubic spline to the PDP and reports `NL_score = R2_spline − R2_linear` (larger → more nonlinearity).
      - Saves all scores to `nonlinearity_scores.csv`.

15. **Housekeeping**
    - Uses `plt.close('all')` to free figures and `gc.collect()` in places.
    - Uses deterministic seeds (`random_state=42`) for reproducibility.

---

## How to run

### 1) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -U pandas numpy scikit-learn matplotlib seaborn shap
```

> **Note on versions:**  
> The code uses `OneHotEncoder(sparse_output=True)`, which requires **scikit‑learn ≥ 1.2**.  
> If you’re on an older scikit‑learn, change it to `sparse=True` and update any references accordingly.

### 2) Prepare your data
- Place `Metadata.Final.tsv` in the working directory.


Outputs will appear in the current directory (see below).

---

## Outputs

- **Model selection & metrics (stdout)**
  - Best hyperparameters
  - `classification_report` for threshold 0.5 on test set
  - ROC AUC and PR AUC
  - Top SHAP interaction pairs

- **Figures**
  - `roc_auc.png`, `pr_auc.png`
  - `shap_summary_top30.png`
  - `shap_interaction_summary_topk.png`
  - `shap_interactions_heatmap_topk.png`
  - `dep_<feature>.png` (one per top feature)
  - `pdp_top12.png`
  - `ice_bmi.png`

- **Tables**
  - `nonlinearity_scores.csv`

---

## Why these choices?

- **Pipeline + ColumnTransformer** keeps preprocessing **inside** CV to avoid data leakage.
- **Average Precision (PR AUC)** is more informative than ROC AUC for **imbalance**.
- **Sample‑weight upweighting** is simple and effective when `class_weight` isn’t available.
- **GBDT** is a strong baseline for structured data; **SHAP** gives consistent feature attributions for trees.
- **Interactions & nonlinearity checks** (SHAP interactions, PDP/ICE, spline gap) reveal how features **combine** and whether relationships are **curved** vs. linear.

---

## Tips, caveats, and troubleshooting

- **Typo in code block fence:** use ```python (not ```pyton) for syntax highlighting.
- **scikit‑learn version:** If you see an error about `sparse_output`, replace with `sparse=True` or upgrade scikit‑learn.
- **GridSearchCV sample weights:** Passing `clf__sample_weight` in `fit(...)` is correct. Older sklearn versions may need `fit_params={"clf__sample_weight": sample_weight}` explicitly.
- **Memory for SHAP interactions:** `shap_interaction_values` scales roughly as _O(n × k²)_. Reduce `topk` or the subset size if you hit memory/time limits.
- **One‑hot feature names:** The script strips `num__/bin__/cat__` prefixes and trailing `_idx` so `PartialDependenceDisplay` receives **raw feature names**.
- **Seeding:** Random seeds are set for reproducibility (`42`). If you need deterministic SHAP visuals across runs, fix the sampled indices list to disk.
- **Metrics beyond 0.5 threshold:** Since PR AUC is optimized, the 0.5 threshold may not be optimal. Consider a threshold that maximizes F1/recall depending on your application.
- **Calibration:** If well‑calibrated probabilities are required, consider `CalibratedClassifierCV` post‑fit or swap to a calibrated model.

---

