# Site-Aware Gradient Boosting Model for Preterm Birth (PTB)

This repository implements a **site-aware machine learning pipeline** that predicts **preterm birth (PTB)** using Gradient Boosting Classification and interpretable model diagnostics.

The script is designed for real clinical population data with **imbalanced outcomes** and **multi‑site sampling structure**, enabling non‑leaky evaluation and biologically interpretable output.

---

## 🔍 What This Pipeline Does

✔ Loads metadata from `Metadata.Final.tsv`  
✔ Extracts categorical, continuous, and binary features  
✔ Applies preprocessing:
- Standard scaling for numeric features
- One‑hot encoding for categorical features
- Pass‑through for binary indicators

✔ Handles **site structure** explicitly:
- ≥3 sites → unseen‑site evaluation  
- 2 sites → stratified row split + site‑aware inner CV  
- <2 sites → standard stratified CV  

✔ Trains a **Gradient Boosting Classifier**  
✔ Uses **class weighting instead of SMOTE**  
✔ Optimizes hyperparameters via `GridSearchCV`  
✔ Evaluates performance using:
- ROC AUC
- Precision‑Recall AUC
- Threshold‑based classification report  

✔ Produces **ROCAUC and PRAUC plots**  
✔ Retrains best model on all samples  
✔ Generates deep interpretability outputs via `run_common_reports`

---

## 📂 Input Requirements

File required:

```
Metadata.Final.tsv
```

Must include:

- `PTB` — binary target (1 = preterm birth)
- Feature columns
- `site` column recommended

---

## ▶️ Running the Script

Example execution:

```bash
python GB.PTB.py \
    "MAINHAP,SEX" \
    "MAT_HEIGHT,MAT_WEIGHT,BMI" \
    "TOILET,WATER"
```

Argument positions:

1. Comma‑separated categorical variables  
2. Comma‑separated continuous variables  
3. Comma‑separated binary variables  

⚠ **Do NOT include `site` in the categorical list** — script handles it automatically.

---

## 🔧 Model Details

Classifier used:

```
GradientBoostingClassifier(random_state=42)
```

Hyperparameters tuned:

```python
n_estimators: [200, 400]
learning_rate: [0.05, 0.1]
max_depth: [2, 3]
```

Evaluation metrics:

- Precision / Recall / F1
- ROC AUC
- Average Precision (PR AUC)

Plots saved:

```
roc_auc.png
pr_auc.png
```

---

## 📊 Outputs Generated

### Console

- Best hyperparameters
- Classification report
- ROC AUC & PR AUC values

### Files from `run_common_reports`

Prefixed with:

```
PTB_*
```

Examples include:

- PTB_top_features.tsv
- PTB_interaction_scores.tsv
- PTB_rfe.txt
- PTB_pdp_*.png

---

## 🤝 Why This Matters

This pipeline is suitable for:

- Clinical ML research
- Genetic / demographic risk modeling
- Multi‑site effect correction
- Biological signal discovery

It is particularly aligned with PTB analysis pipelines where **interpretability, bias control, and generalization testing** are key.

---

## 👤 Author

Jeff Haltom  
Bioinformatics Scientist  

---

## 📄 License

MIT
