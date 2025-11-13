
# Neural Network for Gestational Age Prediction (Site-Aware, SHAP-Interpretable)

This repository contains a **site-aware neural network pipeline** for predicting **gestational age (GAGEBRTH)** from multivariate metadata.  
It includes:

- **Group-aware train/validation/test splitting** (prevents site leakage)
- **Keras Tuner hyperparameter optimization**
- **Neural network regression model**
- **Full preprocessing pipeline (StandardScaler + OneHotEncoder)**
- **SHAP-based feature interpretation**
- **Model evaluation and visualization**

This pipeline is specifically designed for **multi-site biological datasets**, where avoiding site leakage is essential for honest generalization.

---

## 📌 Why Site-Aware Splitting?

Traditional `train_test_split()` mixes samples across sites, which causes:

- Inflated model performance  
- SHAP falsely identifying site-correlated features  
- Poor generalization to unseen cohorts  
- Hidden batch/population structure influencing predictions  

To avoid this, the workflow uses:

### **GroupShuffleSplit**
- Ensures **entire sites** are held out during:  
  ✔ Train → Test split  
  ✔ Train → Validation split  

This makes both training and evaluation *site-independent*, producing a model that reflects true biological signal rather than site artifacts.

---

## 🧬 Pipeline Overview

### 1. **Load and filter dataset**
The script loads the metadata table and selects the required features plus `GAGEBRTH`.

### 2. **Define feature groups**
You pass three comma-separated lists to the script:
- `categorical_columns`
- `continuous_columns`
- `binary_columns`

### 3. **Site-aware splitting**
Two levels of group-aware splitting:

1. **Train/Test**  
2. **Train/Validation**

This guarantees no overlap in site labels across splits.

### 4. **Preprocessing**
A `ColumnTransformer` handles:
- Standardization of continuous features  
- Passthrough of binary features  
- OneHotEncoding of categorical features  

Preprocessor is **fit only on training data**, then applied to validation and test sets.

### 5. **Neural Network Model**
Hyperparameterized using **Keras Tuner**:

- Dense layers (32–512 units)
- Optional extra layers
- Dropout (0–0.5)
- L2 regularization
- Learning rate search (`1e-4` → `1e-2`)
- Linear output layer for regression

### 6. **SHAP Interpretation**
Using `shap.DeepExplainer`:

- SHAP summary plot  
- Top 20 features ranked by mean |SHAP|  
- SHAP dependence plots  

SHAP is much more reliable with site-aware splitting because the network can no longer cheat using site-level patterns.

### 7. **Model evaluation**
Metrics:

- **MSE**
- **MAE**
- **R²**
- Predicted vs Actual scatterplot

All plots are saved as PNG files.

### 8. **Model saving**
Best model written as:
```
NN.GA_best_model.h5
```

---

## 📂 File Outputs

| Output File | Description |
|------------|-------------|
| `ActualvsPredicted_GestationalAge.NN.GA.png` | Predicted vs actual scatter |
| `shap_summary_plot.NN.GA.png` | Global SHAP feature importance |
| `shap.dependence_plot.NN.GA.<feature>.png` | Dependence plots for top features |
| `NN.GA_best_model.h5` | Trained neural network |
| `model_tuning/` | Keras Tuner trials |

---

## 🚀 Running the Script

Example usage:

```bash
python nn_ga.py \
  "MotherAge,MotherHeight,Education" \
  "BMI,GAGE_PRE,PrenatalVisits" \
  "is_smoker,is_married"
```

This corresponds to:

- Categorical features = `"MotherAge,MotherHeight,Education"`
- Continuous features = `"BMI,GAGE_PRE,PrenatalVisits"`
- Binary features = `"is_smoker,is_married"`

All features must exist in `Metadata.Final.tsv`.

---

## 🧠 Best Practices & Notes

### ✔ Always use `GroupShuffleSplit` for multi-site data  
This prevents the model from memorizing site effects.

### ✔ Fit preprocessing ONLY on the training set  
Avoids future information leaking into training.

### ✔ SHAP improves dramatically  
Without site leakage, SHAP reveals **true biological signal** instead of site artifacts.

### ✔ Keras Tuner uses proper validation  
Since validation is site-aware, hyperparameter tuning reflects generalization ability.

---

## 📌 Requirements

```
pandas
numpy
tensorflow
keras-tuner
scikit-learn
matplotlib
shap
```

---

## 🧩 Future Extensions

- Incorporate **GroupKFold** analog for repeated CV in deep learning  
- Add **per-site SHAP decomposition**  
- Implement **bootstrap SHAP stability analysis**  
- Compare against RandomForest, GradientBoosting, XGBoost  
- Add logging and reproducibility utilities  

---

## ✨ Citation / Acknowledgment

If you use this workflow in a publication, cite:

**"Site-aware neural network modeling with SHAP interpretation for gestational age prediction."**

---

## 📞 Contact
For questions or improvements, feel free to contact the author.

---

Enjoy the analysis! 🚀  
This README fully documents the site-aware neural network pipeline, preprocessing, SHAP interpretation, and rationale.
