# Gestational Age Prediction Pipeline (Neural Network + Site-Aware Splits + SHAP)

This repository contains a complete machine‑learning pipeline for predicting **gestational age at birth (GAGEBRTH)** using a neural network with **hyperparameter tuning**, **site‑aware data splitting**, **one‑hot encoding**, **scaling**, and **SHAP feature interpretation**.

The workflow is optimized for biomedical metadata containing categorical, binary, and continuous variables as well as site‑specific structure that must be respected to avoid data leakage.

---

# 📌 Key Features of This Pipeline

### ✅ **1. Site‑aware Train/Val/Test Splits**
- Uses `GroupShuffleSplit` to ensure that **no site appears in more than one dataset**.
- Properly prevents leakage of site‑specific artifacts.

### ✅ **2. Fully Integrated Preprocessing**
- `StandardScaler` for continuous features  
- Pass‑through for binary features  
- `OneHotEncoder` for categorical variables  
- Clean `ColumnTransformer` pipeline  
- Ensures preprocessing is fit only on training data.

### ✅ **3. Hyperparameter‑Tuned Neural Network**
- Uses **Keras Tuner RandomSearch**
- Tunable:
  - Number of layers
  - Units per layer
  - Dropout
  - Learning rate
- Early stopping with `restore_best_weights=True`.

### ✅ **4. Full SHAP Explainability**
- Uses `DeepExplainer`
- Summary plots
- Top‑feature ranking
- Dependence plots for each top feature

### ✅ **5. Saved Model**
- Best model saved as:  
  `NN.GA_best_model.h5`

---

# 📁 Input Data Requirements

The script expects:

```
Metadata.Final.tsv
```

This file **must contain**:

- `GAGEBRTH` – continuous target variable  
- `site` – grouping variable  
- Categorical, binary, and continuous features (specified via CLI)

Example columns:

| Type | Example |
|------|---------|
| Categorical | `SubHap`, `MainHap` |
| Continuous | `Age`, `BMI` |
| Binary | `PTB`, `Sex_binary` |

---

# ▶️ How to Run

## Command‑line invocation

```bash
python train_nn_ga.py "<categorical_columns>" "<continuous_columns>" "<binary_columns>"
```

Example:

```bash
python train_nn_ga.py "SubHap,MainHap,site" "Age,BMI" "PTB,Sex_binary"
```

The script automatically removes `site` from categorical features and uses it exclusively as the `groups=` variable for site‑aware splitting.

---

# 🧠 Data Splitting Logic

## **If ≥2 unique sites:**
- 1st split → **Train vs Test** (GroupShuffleSplit 70/30)
- 2nd split → **Train vs Validation** (GroupShuffleSplit on training portion)

Final approximate proportions:

- 56% train  
- 14% validation  
- 30% test  

## **If only 1 site:**
Falls back to:
- 70% train  
- 15% validation  
- 15% test  

---

# 🛠️ Preprocessing Details

### **ColumnTransformer structure**

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_columns),
        ('bin', 'passthrough', binary_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)
```

Training:

```python
X_tr_preprocessed = preprocessor.fit_transform(X_tr)
```

Then:

```python
X_val_preprocessed  = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)
```

⚠️ **Note about sparseness**  
If using SHAP or some TensorFlow versions, convert to dense:

```python
if hasattr(X_tr_preprocessed, "toarray"):
    X_tr_preprocessed = X_tr_preprocessed.toarray()
    X_val_preprocessed = X_val_preprocessed.toarray()
    X_test_preprocessed = X_test_preprocessed.toarray()
```

---

# 🤖 Neural Network Model + Hyperparameter Tuning

Model built in a `HyperModel`:

### Tuned Parameters:
- Units per layer (32–512)
- Number of layers (1–3)
- Dropout rate (0–0.5)
- Learning rate (1e‑4 to 1e‑2, log‑sampled)

### Optimization:
- Loss: `mean_squared_error`
- Metric: `mean_squared_error`
- Optimizer: Adam

### Early stopping:
```python
EarlyStopping(
    monitor="val_mean_squared_error",
    patience=5,
    restore_best_weights=True
)
```

Hyperparameter tuning search:

```python
tuner = RandomSearch(
    hypermodel,
    objective='val_mean_squared_error',
    max_trials=10,
    executions_per_trial=2,
    directory='model_tuning',
    project_name='gestational_age_prediction'
)
```

---

# 📊 Evaluation Metrics

After selecting the best model:

- **MSE** – Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **R²** – Coefficient of Determination  

Plot saved as:

```
ActualvsPredicted_GestationalAge.NN.GA.png
```

---

# 🔍 SHAP Explainability

### SHAP initialization:

```python
explainer = shap.DeepExplainer(best_model, X_tr_background)
shap_values = explainer.shap_values(X_test_preprocessed)
```

### Correct handling of SHAP output:

```python
if isinstance(shap_values, list):
    shap_values_squeezed = shap_values[0]
else:
    shap_values_squeezed = shap_values
```

### Outputs created:
- `shap_summary_plot.NN.GA.png`
- `shap.dependence_plot.NN.GA.<feature>.png` for each top feature

### Ranking top features:

```python
mean_abs = np.abs(shap_values_squeezed).mean(axis=0)
sorted_idx = np.argsort(mean_abs)[::-1]
top_features = feature_names[sorted_idx[:20]]
```

---

# 💾 Saved Model

The best model is saved here:

```
NN.GA_best_model.h5
```

When reloading, you **must** also reuse the same preprocessing object.

---

# 🚀 Quick Start

1. Place `Metadata.Final.tsv` in working directory  
2. Identify your feature columns  
3. Run:

```bash
python train_nn_ga.py "<cats>" "<conts>" "<bins>"
```

4. View outputs:
   - Plots  
   - SHAP interpretations  
   - Tuned model  
   - All metrics  
   - Preprocessing pipeline behavior  

---

# 📈 Suggested Improvements

- Add `argparse` for robust CLI parsing  
- Add automated missing-data imputation  
- Store preprocessor with model (pickle, joblib)  
- Add logging output for traceability  
- Add version pinning to avoid SHAP/TensorFlow mismatch  

---

# ✔️ Summary

This pipeline is a **robust and production‑grade** approach for biomedical regression settings that require site‑aware modeling, neural networks, explainability, and strict leakage prevention.

It is optimized for your mtDNA/PTB/GA workflows but is generalizable to ANY similar biomedical structure.

If you'd like:
- A **PDF** version  
- A **GitHub‑ready folder structure**
- A **diagram of the ML flow**
- A **version with argparse**
- A **module‑based (importable) refactor**

Just tell me — I’ll generate it.

