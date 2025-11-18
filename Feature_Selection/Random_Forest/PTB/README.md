# PTB Random Forest Classification Pipeline with SHAP & Site-Aware Cross-Validation

This project implements a **Random Forest classifier** to model **Preterm Birth (PTB)** using maternal, fetal, mtDNA, and demographic metadata.  
It includes **rigorous cross-site modeling**, **class imbalance handling**, and **deep interpretability** via **SHAP**, **interaction SHAP**, **PDP**, **ICE**, and **nonlinearity scoring**.

The script is designed to answer **two key questions**:

1. **Can we generalize to new, unseen sites?**  
2. **What features most strongly drive PTB risk?** (including non-linear & interaction effects)

---

## 🔍 Key Features

### ✔️ **Unseen-Site Evaluation (Outer Split)**
If a `site` column exists with ≥2 unique sites, the script uses:

- **GroupShuffleSplit** (test) → ensures **test set contains sites not seen during training**  
- **GroupKFold** (CV) → prevents site leakage in hyperparameter tuning

This isolates **geographic/site-specific confounding**.

---

### ✔️ **Class Imbalance Handling**
PTB is heavily imbalanced → model uses:

- Manual **sample weighting**  
- No `class_weight` to avoid double-weighting  

Formula:

```
pos_weight = (# negatives) / (# positives)
```

Positives get upweighted in training.

---

### ✔️ **Random Forest Classifier**
Hyperparameters tuned via GridSearchCV:

- `n_estimators`: 300, 600, 900  
- `max_depth`: {None, 10, 20}  
- `min_samples_leaf`: {1, 2, 5}  
- `max_features`: {"sqrt", 0.5}

Scored using **Average Precision (PR AUC)** — standard for imbalanced classification.

---

### ✔️ **Full Feature Preprocessing Pipeline**
Uses `ColumnTransformer`:

- **StandardScaler** for continuous variables  
- **Passthrough** for binary variables  
- **OneHotEncoder** (sparse) for categorical variables  
- Efficient sparse → dense conversion for SHAP

---

### ✔️ **Comprehensive Model Evaluation**
Outputs:

- **Classification report (precision, recall, F1)**
- **ROC AUC**
- **PR AUC**
- Saved plots:  
  - `roc_auc.png`  
  - `pr_auc.png`

---

## 🧠 SHAP-BASED INTERPRETABILITY

### ✔️ SHAP Main Effects
Generates:

- `shap_summary_top30.png`  
- Sorted feature importances (mean |SHAP|)

### ✔️ SHAP Interaction Effects
To reveal feature synergy:

- Interaction SHAP values computed on **top-k features**  
- Results saved as:  
  - `shap_interaction_summary_topk.png`  
  - `shap_interactions_heatmap_topk.png`

Also prints **top 10 strongest interactions**.

---

## 📈 PDP, ICE, and Nonlinearity

### ✔️ PDP for top features
Partial dependence curves (top 12 features):

- `pdp_top12.png`

### ✔️ ICE for BMI
Individual Conditional Expectations:

- `ice_bmi.png`

### ✔️ Nonlinearity scoring
Spline-based curvature analysis:

- Saved to: `nonlinearity_scores.csv`  
- Identifies features with **non-monotonic effects**.

---

## 🧬 Input Format

Run script as:

```
python script.py "cat1,cat2" "cont1,cont2" "bin1,bin2"
```

Where:

- `categorical_columns` → fed to OHE  
- `continuous_columns` → scaled  
- `binary_columns` → passthrough  
- Must include:
  - `PTB` column (binary outcome)
  - `site` column (optional but recommended)

The dataset must be named:

```
Metadata.Final.tsv
```

---

## 📁 Outputs

The script generates:

### **Model Performance**
- `roc_auc.png`
- `pr_auc.png`

### **SHAP Main Effects**
- `shap_summary_top30.png`

### **SHAP Interaction Effects**
- `shap_interaction_summary_topk.png`
- `shap_interactions_heatmap_topk.png`

### **PDP & Nonlinearity**
- `pdp_top12.png`
- `ice_bmi.png`
- `nonlinearity_scores.csv`

---

## 🧪 Scientific Motivation

PTB prediction is influenced by:

- mtDNA variants / haplogroups  
- nDNA ancestry (PCs)  
- demographic covariates  
- environmental & site-specific factors  

This pipeline:

- isolates **site effects**
- evaluates **cross-site generalizability**
- reveals **main drivers** of PTB risk
- highlights **interaction structure**
- analyzes **non-linear biology**

It's suitable for:

- Manuscript supplements  
- Internal reproducibility  
- Exploratory modeling  
- Feature significance screening  

---

## 🏁 Final Notes

This pipeline is computationally heavy due to:

- SHAP interactions  
- PDP + spline nonlinearity  
- full-grid RF tuning  

For fast iteration, comment out:

- SHAP interaction section  
- PDP/ICE section  
- Nonlinearity scoring  

---

## 📜 Author

Jeff Haltom  
Bioinformatics Scientist II  
Children’s Hospital of Philadelphia (CHOP)

---

Feel free to ask for:

- A **lighter version**  
- A **GPU-ready version**  
- A **cluster SLURM script**  
- A **paired regression README**  
