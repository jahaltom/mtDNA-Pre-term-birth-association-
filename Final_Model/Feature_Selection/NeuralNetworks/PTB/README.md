
# Neural Network Model for PTB Prediction with SHAP Explainability

## Overview
This repository contains a Keras-based neural network for predicting **Preterm Birth (PTB)**, using site-aware splitting, hyperparameter optimization, and SHAP-based model interpretation.

The pipeline includes:
- Data preprocessing and encoding
- Group-aware train/validation/test splitting
- Class balancing via class weights
- Hyperparameter tuning with Keras-Tuner
- Evaluation with ROC/AUPRC curves
- SHAP DeepExplainer and dependence plots

---

## Workflow Summary

### 1. Dataset Loading  
Reads `Metadata.Final.tsv`, extracting categorical, binary, and continuous features.

### 2. Site-Aware Data Splitting  
If site labels exist:
- ≥3 sites → GroupShuffleSplit (unseen‑site test)
- 2 sites → Stratified split + site-aware tuning
- <2 sites → Standard stratified split

### 3. Preprocessing
A `ColumnTransformer` handles:
- StandardScaler for numeric features
- One-hot encoding for categorical variables
- Pass-through for binary features  

Transformed features are used for training and SHAP interpretation.

### 4. Neural Network Architecture
- Tunable hidden layers
- L2 regularization
- Dropout
- Sigmoid output for binary classification

Hyperparameters are optimized via `RandomSearch`.

### 5. Training Strategy
Includes:
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint tracking best validation AUPRC

### 6. Evaluation
Metrics computed on unseen test set:
- ROC AUC
- PR AUC
- Full classification report

Saved outputs:
```
ROC_AUC_plot.NN.PTB.png  
PR_AUC_plot.NN.PTB.png
NN.PTB_metrics.txt
```

### 7. Model Explainability
SHAP DeepExplainer is applied over **all preprocessed data**, producing:

✔ Top 20 most influential features  
✔ Dependence plots for top predictors  
✔ Mean absolute SHAP rankings  

Outputs:
```
SHAP_summary_top20.NN.PTB.png  
SHAP_dependence_<feature>.NN.PTB.png  
Top20SHAPfeatures.txt
```

---

## Key Outputs

| File | Meaning |
|------|---------|
| `NN.PTB_metrics.txt` | Hyperparameters + performance summary |
| `ROC_AUC_plot.NN.PTB.png` | ROC curve |
| `PR_AUC_plot.NN.PTB.png` | Precision‑Recall curve |
| `SHAP_summary_top20.NN.PTB.png` | Top SHAP feature ranking |
| `Top20SHAPfeatures.txt` | Ranked list of driver variables |
| `SHAP_dependence_*` | Partial SHAP feature effects |

---

## Notes for Practical Use

- The **site-aware split** ensures realistic generalization across study sites.
- The **class imbalance is automatically corrected** via class weighting.
- SHAP DeepExplainer is run using all data for better interpretability consistency.
- Outputs can be fed into downstream modelling (e.g., GLMM, mediation, pathway analysis).

---

## Suggested Extensions
✔ KernelExplainer to cross-validate feature importance  
✔ Permutation importance for robustness  
✔ Export tuned network for deployment  

---

## Citation
If used in publications, please cite this methodology as:

> Haltom et al., Neural Network‑Based Preterm Birth Prediction Using Site‑Aware Learning and SHAP Interpretability, 2025.

---

## Author
Jeff Haltom, PhD – Bioinformatics Scientist II, Children’s Hospital of Philadelphia

