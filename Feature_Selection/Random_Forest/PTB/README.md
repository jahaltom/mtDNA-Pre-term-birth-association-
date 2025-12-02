
# PTB Random Forest Modeling + Interpretation Pipeline

This project implements a **site‑aware machine learning workflow** for modeling Pre‑Term Birth (PTB)
and extracting biologically interpretable signals from epidemiological data.

---

## 🔍 Purpose

The pipeline is *not* focused on maximizing predictive performance.  
Instead, it is designed to:

✔ avoid data leakage  
✔ correctly respect study site structure  
✔ reveal interpretable relationships among covariates  
✔ evaluate interaction structure and nonlinear effects  

---

## ⚙️ Method Overview

### 1. **Site‑Aware Train/Test Split**
- If ≥3 sites → unseen‑site split using `GroupShuffleSplit`
- If 2 sites → stratified split but *site used for site‑aware CV*
- If <2 sites → standard Stratified split

### 2. **Class Imbalance Handling**
- No SMOTE
- Uses analytical **class‑weights**

### 3. **Inner Cross‑Validation**
- `GroupKFold` when site labels exist
- `StratifiedKFold` otherwise

### 4. **Model Type**
Gradient Boosting Classifier wrapped in a preprocessing pipeline:

- StandardScaler (numeric)
- Pass‑through (binary)
- Dense One‑Hot Encoding (categorical)

### 5. **Full‑Dataset Refit**
After tuning, best settings are re‑fit on *all data* to support global interpretation.

---

## 📊 Automatic Outputs Generated

The helper script `run_common_reports()` produces:

| Output Type | Interpretation |
|-------------|----------------|
| SHAP rankings | Which features matter most |
| Feature importance | Tree‑based gain importance |
| RFE‑selected features | Covariate subsets |
| SHAP summary plot | Directionality & spread |
| SHAP interactions | Pairwise dependencies |
| Heatmaps | Visual interaction structure |
| PDP curves | Marginal functional shape |
| Nonlinearity metrics | Linear vs spline response |

All results are emitted with prefix:

```
PTB.*
```

---

## ▶️ Running The Script

```
python RF.PTB.py "CATEGORICAL_COLS" "CONTINUOUS_COLS" "BINARY_COLS"
```

Example:

```
python RF.PTB.py "RACE,EDU" "BMI,AGE" "SMOKER"
```

---

## 📁 Output Directory Contents

You will find files such as:

```
PTB.shap_importance.csv
PTB.importance.csv
PTB.rfe_selected.csv
PTB.shap_summary.png
PTB.shap_interactions.csv
PTB.shap_interactions_heatmap.png
PTB.pdp_<feature>.png
PTB.nonlinearity_scores.csv
```

---

## 🧠 Why This Matters

This architecture gives you:

✔ Honest signal discovery  
✔ Site structure‑aware inference  
✔ Covariate selection usable in downstream models (e.g., mixed models, GLMMs, brms)  

This approach answers **biological questions**, not leaderboard questions.

---

## ✍️ Citation / Attribution

If you use this workflow, cite as:

> Haltom & GPT‑assisted ML interpretability pipeline for PTB modeling (2025)

---

