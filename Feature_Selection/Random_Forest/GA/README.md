# GA & PTB Random Forest Modeling Pipeline
This repository contains two major components:

1. **RF.GA.py** – End‑to‑end Random Forest modeling workflow for *Gestational Age (GA)* regression  
2. **common_reports.py** – A unified interpretability/reporting engine shared across GA, PTB, and future models

It is designed for large‑scale epidemiological and genomic datasets, including:
- Continuous covariates  
- Binary clinical/household variables  
- Categorical site and cooking‑fuel variables  
- Nested site structure handled via Group-aware CV  
- SHAP-based interpretability  
- PDP/ICE visualization  
- Nonlinearity scoring  
- Interaction heatmaps  
- RFE feature selection  

---

# 📦 File Overview

## **1. RF.GA.py**
Main script to train a **RandomForestRegressor** for predicting gestational age.

### **Workflow**
1. Load metadata  
2. Validate required columns  
3. Define categorical, continuous, and binary feature groups  
4. Construct preprocessing pipeline:  
   - StandardScaler for continuous variables  
   - Passthrough for binary features  
   - OneHotEncoder for categorical variables  
5. Group-aware **train/test split** using site  
6. Group-aware **cross‑validated hyperparameter tuning**  
7. Fit final best model  
8. Evaluate on test set  
9. Run shared interpretability suite via `run_common_reports()`

### **Main Features**
- Protects against **site leakage** using GroupShuffleSplit  
- Uses **GroupKFold** when multiple sites exist  
- Saves:
  - Best parameters  
  - Test MSE / R²  
  - Full interpretability reports  

---

## **2. common_reports.py**
This module centralizes advanced interpretability for both regression and classification tasks.

### **SHAP Analysis**
✔ Handles all SHAP formats:
- (N, F)  
- (N, F, 2)  
- (N, 2, F)  
- List-of-arrays output  

Outputs:
- **shap_importance.csv**  
- **Top interaction pairs**  
- **SHAP bar plot**  
- **SHAP beeswarm plot**  
- **Interaction summary plot**  
- **Interaction heatmap (top K features)**  

### **Interaction Analysis**
- Computes full SHAP interaction matrix  
- Extracts strongest interacting feature pairs  
- Saves to CSV  
- Produces heatmap & interaction summary  

### **RFE (Recursive Feature Elimination)**
- Performs RFE using the tuned RF estimator  
- Works on **post-transform** feature space  
- Outputs selected features list

### **PDP & ICE**
- Partial Dependence Plots (PDP)  
- Individual Conditional Expectation (ICE)  
- Multiple-grid visualization  

### **Nonlinearity Index**
Evaluates functional form of each main feature via:

- Linear regression fit  
- Spline regression fit  
- Nonlinearity score = R²_spline − R²_linear  
- Saves nonlinearity ranking CSV  

---

# 🧪 Outputs Generated
Running `RF.GA.py` produces:

### **CSV Files**
| File | Description |
|------|-------------|
| `GA.shap_importance.csv` | Mean |SHAP| values for each feature |
| `GA.shap_interactions.csv` | Pairwise interaction strengths |
| `GA.rfe_features.csv` | Features selected by RFE |
| `GA.nonlinearity_scores.csv` | Sorted scores for spline nonlinear behavior |

### **Plots / Images**
| Plot | Purpose |
|------|---------|
| `GA_shap_bar.png` | Ranking of top features |
| `GA_shap_beeswarm.png` | Full SHAP distribution |
| `GA_shap_interaction_heatmap.png` | Heatmap of top-K interactions |
| `GA_shap_interaction_summary.png` | Interaction impact summary |
| `GA_pdp_<FEATURE>.png` | PDP + ICE for each selected feature |

---

# 🧠 Model Architecture

### **Preprocessing**
```
ColumnTransformer(
  num = StandardScaler() → continuous vars
  bin = passthrough        → binary vars
  cat = OneHotEncoder()    → categorical vars
)
```

### **Model**
```
RandomForestRegressor(
    n_estimators=[300–900],
    max_depth=[None, 10, 20],
    min_samples_leaf=[1,2,5],
    max_features=["sqrt", 0.5]
)
```

### **Cross‑Validation**
- Test split: **GroupShuffleSplit** (site-level)
- Hyperparameter CV: **GroupKFold** if ≥2 sites; else KFold

---

# ▶️ How to Run

```
python RF.GA.py
```

Outputs appear in working directory as CSVs and PNGs.

---

# 📁 Directory Structure

```
.
├── RF.GA.py
├── common_reports.py
├── GA.shap_importance.csv
├── GA.shap_interactions.csv
├── GA_nonlinearity_scores.csv
├── GA_rfe_features.csv
├── plots/
│   ├── GA_shap_bar.png
│   ├── GA_shap_beeswarm.png
│   ├── GA_shap_interaction_heatmap.png
│   ├── GA_pdp_*.png
```

---

