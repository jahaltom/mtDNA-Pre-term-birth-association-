
# Neural Network for Gestational Age (GA) Prediction — Working Version

This document summarizes the **current TensorFlow/Keras neural network pipeline** for predicting gestational age (GAGEBRTH). It describes what the code does, how to run it, what files it generates, and how to interpret its output — **exactly as it currently works**, without suggesting modifications.

---

## 🧩 Overview

This script builds and evaluates a **feedforward neural network (regression)** to predict gestational age using a mix of maternal and environmental covariates.

It includes:
- Full preprocessing with **StandardScaler** (continuous), **OneHotEncoder** (categorical), and passthrough for binary features.
- A **tunable deep neural network** built via `keras_tuner.RandomSearch`.
- Evaluation using **MSE** , **MAE** and **R²**.
- **SHAP explainability** (DeepExplainer) for feature importance and dependence plots.
- Automatic plotting of results and saving to file.

---

## ⚙️ Pipeline summary

1. **Input**  
   Reads `Metadata.Final.tsv` and filters for valid entries (removes `-88`, `-77` values).  
   Keeps only selected columns passed through command line:
   ```bash
   python script.py cat_cols cont_cols bin_cols
   ```
   Example:
   ```bash
   python NN_GA.py DRINKING_SOURCE,FUEL_FOR_COOK,TOILET,WEALTH_INDEX PW_AGE,PW_EDUCATION,MAT_HEIGHT,MAT_WEIGHT,BMI BABY_SEX,CHRON_HTN,DIABETES,HH_ELECTRICITY,TB,THYROID,TYP_HOUSE
   ```

2. **Feature groups**
   - `categorical_columns`: e.g., water source, toilet type, fuel type, wealth index  
   - `continuous_columns`: maternal age, education, height, weight, BMI  
   - `binary_columns`: baby sex, chronic hypertension, diabetes, etc.  

3. **Preprocessing**
   Uses a `ColumnTransformer` with:
   - `StandardScaler` for continuous variables  
   - passthrough for binaries  
   - `OneHotEncoder(handle_unknown='ignore')` for categoricals  

4. **Model architecture**
   The `HyperModel` defines a flexible multi-layer ReLU network with optional dropout layers.  
   - Tunable parameters: number of units per layer, number of layers (1–3), dropout rate, learning rate.  
   - Compiled with `Adam` optimizer, loss = MSE.

5. **Hyperparameter tuning**
   - Conducted using `keras_tuner.RandomSearch`
   - Up to `max_trials=10`, each executed twice (`executions_per_trial=2`)
   - Objective: minimize validation MSE

6. **Evaluation**
   - Computes **MSE** and **R²** on the test set  
   - Generates scatter plot of **Actual vs Predicted Gestational Age** (`ActualvsPredicted_GestationalAge.NN.GA.png`)

7. **Explainability**
   - SHAP DeepExplainer computes feature attributions.  
   - Creates SHAP **summary plot** (`shap_summary_plot.NN.GA.png`) and **dependence plots** for top features.  
   - Prints top 10 features with highest mean |SHAP| values.

8. **Output**
   - `NN.GA_best_model.h5` — best model (Keras HDF5 format)  
   - `shap_summary_plot.NN.GA.png` — global importance plot  
   - `shap.dependence_plot.NN.GA.<feature>.png` — feature-level SHAP plots  
   - `ActualvsPredicted_GestationalAge.NN.GA.png` — performance visualization  

---

## 📊 Interpretation guide

| Output | Meaning |
|--------|----------|
| **MSE** | Mean Squared Error — smaller is better (average squared prediction deviation). |
| **R²** | Coefficient of determination — how much variance in GA the model explains. |
| **SHAP summary plot** | Each point = one sample × feature. Color = feature value, x-position = contribution to GA. |
| **Dependence plots** | Show how a feature’s value influences predicted GA (nonlinearity or saturation indicates complex effects). |
| **Top SHAP features** | Features with highest average influence across the test set. |

---

## 📁 Files generated

| File | Description |
|------|-------------|
| `NN.GA_best_model.h5` | Saved Keras model for reuse. |
| `ActualvsPredicted_GestationalAge.NN.GA.png` | Scatter plot of actual vs predicted GA. |
| `shap_summary_plot.NN.GA.png` | SHAP summary (global importance). |
| `shap.dependence_plot.NN.GA.<feature>.png` | Individual dependence plots for each top SHAP feature. |

---

## 🧠 Notes on DeepSHAP

- Uses `shap.DeepExplainer(best_model, X_train_preprocessed)` directly.  
- SHAP values are squeezed for plotting (`np.squeeze(..., axis=2)`).  
- Works fine as-is; no manual shape handling needed since results are stable.  
- Dependence plots loop over the top 20 SHAP-ranked features.

---

## 💾 Model usage

Reload the model later:
```python
from tensorflow.keras.models import load_model
model = load_model("NN.GA_best_model.h5")
preds = model.predict(X_new_preprocessed)
```

---

## ✅ Summary

This script is **fully functional and stable**.  
It performs end-to-end neural network regression with preprocessing, hyperparameter tuning, explainability via SHAP, and automated visualization.  

It’s well-suited for exploratory nonlinear modeling of gestational age and complements your existing RF/GB pipelines.

