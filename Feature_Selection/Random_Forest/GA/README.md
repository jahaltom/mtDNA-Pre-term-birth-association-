# Site-Aware Random Forest Regression Pipeline for Gestational Age (GA)

This repository implements a **site-aware machine learning workflow** for predicting gestational age (GAGEBRTH) using clinical, categorical, and binary metadata.

The pipeline:

- Performs **site-aware data splitting** (GroupShuffleSplit or GroupKFold)
- Runs **hyperparameter tuning** via GridSearchCV
- Evaluates generalization performance
- Retrains the best model on the **full dataset**
- Generates interpretability reports via `run_common_reports`

---

## 📌 Features

✔ ColumnTransformer preprocessing  
✔ One‑Hot Encoding for categorical variables  
✔ Standard scaling of continuous variables  
✔ Group-aware cross‑validation using site labels  
✔ Model tuning for Random Forest hyperparameters  
✔ Full‑dataset interpretability output  
✔ Modular design for GA / PTB reuse  

---

## 🏗️ Pipeline Overview

```text
Load Metadata.Final.tsv
│
├── Extract feature groups
│
├── Site‑aware train/test split
│     ├── ≥3 sites → unseen-site test via GroupShuffleSplit
│     ├── 2 sites → standard split + GroupKFold CV
│     └── else → standard CV/no grouping
│
├── GridSearchCV hyperparameter tuning (site-aware if possible)
│
├── Evaluate best model on held‑out test set
│
└── Retrain best model on full data + run interpretability reports
```

---

## ⚙️ Dependencies

- Python 3.10+
- numpy
- pandas
- scikit‑learn
- `common_reports.py` (included in repo)

---

## 📂 Required Data File

```
Metadata.Final.tsv
```

Must contain:

- `GAGEBRTH` (target variable)
- Feature columns
- `site` column (recommended)

---

## 🧠 Running the model

Example execution:

```bash
python run_ga_rf.py     "SITE,SEX,MAINHAP"     "BMI,MAT_HEIGHT,MAT_WEIGHT"     "TOILET,WATER"
```

---

## 📊 Output

### Terminal Metrics

- Best RF hyperparameters
- MSE
- R²

### Generated interpretability artifacts

- Feature rankings
- RFE stability analysis
- PDP plots
- Interaction importance
- Report files prefixed with `GA_*`

---

## ✨ Notes

- Works even if `site` column is missing — falls back to standard CV.
- Fully compatible with your PTB classification infrastructure.
- Extendable — swap RF for Gradient Boosting, XGBoost, or GLMM wrappers.

---

## 👨‍💻 Author

Jeff Haltom  
Bioinformatics Scientist II, CHOP  

---

## 📎 License

MIT

