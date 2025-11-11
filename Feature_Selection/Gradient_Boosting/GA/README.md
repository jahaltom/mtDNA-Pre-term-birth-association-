
# GA/PTB Covariate Feature Selection & Explainability Pipeline

This repository/script builds a **Gradient Boosting** model on maternal, socioeconomic, and clinical covariates to explain **Gestational Age (GA)**, then performs **model explainability** and **diagnostics** to:
- Rank covariates by importance
- Quantify **pairwise interactions** via SHAP
- Detect **non-linear** features
- Export plots for reporting and to inform a downstream **mixed-effects** association model (with `site` as a random effect and **mtDNA haplogroup** as the fixed effect of interest)

> **Design choice:** This step intentionally **excludes** `site`, population labels, and **nDNA PCs** from feature selection, because those are structural/ancestry controls to be modeled later (e.g., `(1 | site)` in your GLMM). It also **excludes mtDNA haplogroup** since that is the predictor of interest to be tested in the final association model, not a nuisance covariate.

---

## Contents

- [Requirements](#requirements)
- [Data Expectations](#data-expectations)
- [How It Works (Step by Step)](#how-it-works-step-by-step)
- [Usage](#usage)
- [Outputs](#outputs)
- [Customization](#customization)
- [Performance Tips](#performance-tips)
- [Interpretation Notes](#interpretation-notes)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- Python 3.9+
- Packages:
  - `numpy`, `pandas`
  - `scikit-learn` ≥ 1.2 (for `OneHotEncoder(sparse_output=False)`); if older, use `sparse=False`
  - `matplotlib`
  - `seaborn`
  - `shap`

Install (example):
```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap
```

---

## Data Expectations

- Input file: `Metadata.Final.tsv` (tab-separated)
- Must contain at least these columns:
  - **Categorical:** `DRINKING_SOURCE`, `FUEL_FOR_COOK`, `TOILET`, `WEALTH_INDEX`
  - **Continuous:** `PW_AGE`, `PW_EDUCATION`, `MAT_HEIGHT`, `MAT_WEIGHT`, `BMI`
  - **Binary:** `BABY_SEX`, `CHRON_HTN`, `DIABETES`, `HH_ELECTRICITY`, `TB`, `THYROID`, `TYP_HOUSE`
  - **Target:** `GAGEBRTH`

> If you want to use CLI-driven column sets instead of the hardcoded lists, enable those lines and pass comma-separated names.

---

## How It Works (Step by Step)

1. **Load & Validate Data**  
   Reads `Metadata.Final.tsv`, asserts all required columns exist, and selects the covariate matrix `X` and target `y=GAGEBRTH`.

2. **Train/Test Split**  
   70/30 split with `random_state=42` for reproducibility.

3. **Preprocessing (ColumnTransformer)**  
   - `StandardScaler` on continuous features
   - `'passthrough'` for binary features
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical → **dense** matrix to support GradientBoosting + SHAP

4. **Model & Pipeline**  
   - `GradientBoostingRegressor` wrapped in a `Pipeline` with the preprocessor to **avoid leakage**.
   - `GridSearchCV` over `n_estimators`, `learning_rate`, and `max_depth` with 5-fold `KFold(shuffle=True)`.

5. **Evaluation**  
   Prints best hyperparameters and evaluates on held-out test data (MSE, R²).

6. **Feature Importances**  
   Extracts `feature_importances_` from the fitted GB and prints top 10 (using the **expanded** feature names from the fitted preprocessor).

7. **RFE (Recursive Feature Elimination)**  
   Runs RFE on the **transformed** design (post-OHE) to select the top 20 features by model-based importance; prints their names.

8. **SHAP (Main Effects)**  
   - Builds a `TreeExplainer` on the trained GB.
   - Computes SHAP values on the transformed training design.
   - Saves the **SHAP summary plot** of main effects.

9. **SHAP (Interactions)**  
   - Subsamples rows (≤2000) to keep runtime/memory manageable.
   - Computes `shap_interaction_values` and aggregates mean |interaction| across samples, yielding an **interaction matrix**.
   - Prints top-10 interacting pairs.
   - Produces a **heatmap** of interactions for the **top-K** features by main |SHAP|.
   - Draws **2D Partial Dependence** plots for the top 5 pairs.
   - Generates a **SHAP interaction summary** plot for the same **top-K** subset.

10. **Non-Linearity Scoring + Plots**  
    - For each feature, fits **linear** vs **cubic** regression to model *SHAP contribution* from the feature value; the ΔR² is a **nonlinearity score**.
    - Prints the top-10 suspected non-linear features.
    - Saves SHAP **dependence plots** (colored by strongest partner) for the top non-linear features.
    - Saves 1D **PDP** panels for the same set.

---

## Usage

```bash
python your_script_name.py
```

- The script uses **hardcoded column lists** by default.  
  If you prefer **CLI-driven columns**, uncomment the CLI lines and pass comma-separated names for categorical / continuous / binary in `sys.argv[1:3]`.

---

## Outputs

Generated files include (names may vary slightly):

- `shap.summary_plot.GB.GA.png` — SHAP main-effects summary (beeswarm)
- `shap_interactions_heatmap_top.png` — Heatmap of SHAP interactions among top-K important features
- `pdp_top_interactions.png` — PDP grid for the top 5 interaction pairs
- `shap_interaction_summary_topk.png` — SHAP **interaction** summary (beeswarm) for top-K features
- `shap_dependence_<FEATURE>.png` — SHAP dependence plots for top non-linear features
- `pdp_top_nonlinear.png` — PDP grid of the top non-linear candidates
- Console prints:
  - Best GB params
  - Test MSE & R²
  - Top 10 GB importances
  - RFE-selected features
  - Top 10 SHAP interaction pairs
  - Top 10 non-linear features (ΔR² cubic vs linear)

> **Note:** In the code, the print statement says `Saved SHAP summary plot to 'shap_summary_GB_GA.png'` but the actual filename is `shap.summary_plot.GB.GA.png`. You can update either to match.

---

## Customization

- **Top-K for interaction heatmap:** `K = min(30, len(feature_names))`
- **Row subsample for interactions:** `row_sample = min(2000, n_rows)`
- **RFE feature count:** `n_feats = min(20, Xtr.shape[1])`
- **Non-linear plot count:** change `nl_df.head(8)`
- **Grid search space:** edit `param_grid_gb`

---

## Performance Tips

- SHAP **interactions** are `O(F^2 × N)`; keep `K` moderate and sample rows.
- Large OHE can explode feature count; confirm `n_feats` ≤ #columns for RFE.
- Set `n_jobs=-1` for parallel CV; keep memory in mind on large datasets.

---

## Interpretation Notes

- **Positive SHAP** → pushes GA **higher** (longer gestation); **negative** → pushes GA **lower**.
- For **one-hot dummies** (e.g., `cat__FUEL_FOR_COOK_3`), **red** points indicate “category present (1)”. Red on the **right** = category increases predicted GA.
- Interaction patterns often reflect **site/ancestry confounding** when those variables are excluded by design. In the **final mixed model** (with `(1 | site)`), such effects may attenuate or flip direction.
- Use this pipeline to **choose covariates**, then test **mtDNA haplogroup** in a separate GLMM/BRMS model.

---

## Troubleshooting

- `ValueError: Missing columns in input` — Your TSV lacks one or more required fields; adjust the column lists or the data.
- `TypeError: A sparse matrix was passed, but dense data is required` — Ensure `OneHotEncoder(..., sparse_output=False)` (or `sparse=False` on older sklearn).
- `MemoryError` or very slow **interaction** plots — Reduce `row_sample` and `K`.

---

Happy modeling! Use these outputs to lock in your adjustment set, then proceed to your GA/PTB mixed-effects models with **mtDNA haplogroup** as the fixed effect of interest.
