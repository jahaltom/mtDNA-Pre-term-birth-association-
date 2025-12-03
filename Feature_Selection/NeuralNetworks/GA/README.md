# Neural Network Regression for Gestational Age (GA) with SHAP Explanations

This script trains a **Keras neural network regressor** to predict **gestational age at birth (GAGEBRTH)** from maternal and household covariates, using **site-aware data splitting**, **target scaling**, **hyperparameter tuning with Keras Tuner**, and **SHAP-based model interpretation** (DeepExplainer + KernelExplainer).

The workflow is designed to be:
- **Site-aware** (outer + inner splits respect the `site` column when available)
- **Numerically stable** (GA target is standardized for NN training)
- **Explainable** (global feature importance + dependence plots via SHAP)
- **Reproducible** (fixed random seed and explicit feature definitions)

---

## 1. Data Input

The script expects a tab-delimited metadata file in the current working directory:

- **File**: `Metadata.Final.tsv`
- **Separator**: `\t` (tab)

### Required columns

- **Outcome**
  - `GAGEBRTH` — Gestational age at birth (continuous; e.g., days)

- **Categorical predictors**
  - `FUEL_FOR_COOK`

- **Continuous predictors**
  - `PW_AGE`
  - `PW_EDUCATION`
  - `BMI`
  - `TOILET`
  - `WEALTH_INDEX`
  - `DRINKING_SOURCE`

- **Binary predictors**
  - `BABY_SEX`
  - `CHRON_HTN`
  - `DIABETES`
  - `HH_ELECTRICITY`
  - `TB`
  - `THYROID`
  - `TYP_HOUSE`

- **Optional grouping variable**
  - `site` — site label for **site-aware splits**. If present and has:
    - ≥3 unique levels: outer split is a **GroupShuffleSplit** with **unseen-site test set**.
    - 2 unique levels: row-level outer split, but `site` is kept to enable **group-aware inner split**.
    - <2 or missing: standard random splits (no grouping).

---

## 2. Environment & Dependencies

Python packages used:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
  - `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`
  - `GroupShuffleSplit`, `train_test_split`
  - `mean_squared_error`, `r2_score`, `mean_absolute_error`
- `tensorflow` / `keras` (TF 2.x)
  - `Sequential`, `Dense`, `Dropout`, `Input`, `regularizers`
  - `tf.random.set_seed`
- `keras-tuner`
  - `RandomSearch`, `HyperModel`, `Objective`
- `shap`

You can install them with e.g.:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow keras-tuner shap
```

> **Note:** Exact TensorFlow/SHAP compatibility can matter for DeepExplainer. KernelExplainer is also included as a more model-agnostic fallback.

---

## 3. Reproducibility

The script sets a global seed:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

This controls:
- Numpy sampling for background / SHAP subsets
- Train/val/test splits
- Keras weight initialization (within TF's constraints)

Perfect bitwise reproducibility is not always guaranteed on GPU, but this makes runs substantially more stable.

---

## 4. Data Splitting Strategy (Site-Aware)

The script performs a two-stage split:

1. **Outer split**: `X_train_full` / `X_test`, `y_train_full` / `y_test`
   - If `site` exists and `n_sites >= 3`: use `GroupShuffleSplit` with `groups=df["site"]` for an **unseen-site test set**.
   - If `site` exists and `n_sites == 2`: use standard `train_test_split`, but keep `site` labels for group-aware inner split.
   - Else: simple `train_test_split` with no grouping.

2. **Inner split**: `X_train` / `X_val`, `y_train` / `y_val`
   - If ≥2 training sites: `GroupShuffleSplit` on the training portion.
   - Else: random `train_test_split` of the training data.

This preserves **site structure** during tuning and evaluation when possible.

---

## 5. Target Scaling for GA

To improve neural network training stability, the outcome `GAGEBRTH` is standardized:

```python
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled   = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled  = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
```

- The **NN is trained and tuned** on the **scaled GA** (approximately mean 0, variance 1).
- After prediction, results are **inverse-transformed** back to original GA units for interpretation.

---

## 6. Feature Preprocessing

Predictors are preprocessed via `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ],
    remainder="drop",
)
```

- Continuous features → standardized.
- Binary features → passthrough (0/1 coding assumed).
- Categorical (`FUEL_FOR_COOK`) → one-hot encoded with dense output (`sparse_output=False`).

Transformed matrices:

- `X_train_preprocessed`
- `X_val_preprocessed`
- `X_test_preprocessed`
- `X_all_preprocessed` (full dataset, used for SHAP)

Feature names are extracted via:

```python
feature_names = preprocessor.get_feature_names_out()
```

---

## 7. Neural Network HyperModel (Keras Tuner)

A custom `HyperModel` class defines the architecture search space:

- Input: shape = `(n_features,)`
- Hidden layers:
  - First hidden layer:
    - Units: 64–512 (step=64)
    - Activation: ReLU
    - L2 regularization: `{0.0, 1e-6, 1e-5, 1e-4}`
    - Dropout: 0.0–0.5 (step=0.1)
  - Additional hidden layers:
    - `n_hidden` in `{0, 1, 2}`
    - Same pattern: units 64–512, L2 regularization, dropout
- Output:
  - Single linear unit: `Dense(1, activation="linear")`
- Optimizer:
  - Adam with learning rate `lr` ∈ `[1e-4, 3e-3]` (log-sampled)
- Loss / metric:
  - `loss="mean_squared_error"`
  - `metrics=["mean_squared_error"]`

Hyperparameter tuning is done via `RandomSearch`:

```python
tuner = RandomSearch(
    hypermodel,
    objective=Objective("val_mean_squared_error", direction="min"),
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory=os.path.join("tuning"),
)
```

Callbacks:

- `EarlyStopping(monitor="val_mean_squared_error", patience=10, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor="val_mean_squared_error", patience=5, factor=0.5, min_lr=1e-5)`
- `ModelCheckpoint("checkpoint.best.keras", save_best_only=True, monitor="val_mean_squared_error")`

Training:

- `BATCH_SIZE = 256`
- `EPOCHS = 100`

---

## 8. Evaluation: Scaled and Original GA Units

After tuning, the best model is retrieved:

```python
best_model = tuner.get_best_models(1)[0]
best_hps   = tuner.get_best_hyperparameters(1)[0]
```

Predictions on the **test set**:

- In scaled space:
  - `y_pred_scaled = best_model.predict(X_test_preprocessed).flatten()`
  - Compare to `y_test_scaled`

- In original GA units:
  - `y_pred_orig = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()`
  - Compare to `y_test_orig = y_test.values`

Metrics are computed for both spaces:

- `MSE`
- `MAE`
- `R²`

Output is written to:

- **`NN.GA_metrics.txt`**

A scatter plot of **Actual vs Predicted GA** in original units is saved as:

- **`ActualvsPredicted_GestationalAge.NN.GA.png`**

---

## 9. SHAP Explanations

Two SHAP approaches are used for global feature importance:

### 9.1 DeepExplainer (Full Data)

- Background: `X_train_preprocessed`
- Data to explain: `X_all_preprocessed` (all rows)

```python
explainer = shap.DeepExplainer(best_model, X_train_preprocessed)
shap_raw  = explainer.shap_values(X_for_shap)
```

The script handles different SHAP API shapes and then creates:

- **Global SHAP summary bar/scatter plot** (top 20 features):

  - File: `SHAP_summary_top20.NN.GA.png`

- **Top-20 features by mean |SHAP|** written to:

  - `Top20SHAPfeatures.txt`

- **Dependence plots** for top 5 features:

  - Files: `SHAP_dependence_<feature>.NN.GA.png`

> Depending on TensorFlow/SHAP versions, DeepExplainer may behave oddly. That’s why KernelExplainer is also included as a model-agnostic check.

### 9.2 KernelExplainer (Subset)

To validate and cross-check feature importance:

1. Sample a background set (up to 300 rows) from the full preprocessed matrix.
2. Sample up to 2000 rows as the explanation set.
3. Define a prediction function:

   ```python
   def predict_fn(data):
       return best_model.predict(data).ravel()
   ```

4. Use `shap.KernelExplainer`:
   ```python
   explainer = shap.KernelExplainer(predict_fn, background)
   shap_values = explainer.shap_values(X_exp, nsamples="auto")
   ```

5. Generate a SHAP summary plot (top 20 features) for the subset:

   - File: `SHAP_summary_top20.NN.GA.Kernel.png`

This gives a more model-agnostic view of feature importance that can be compared to the DeepExplainer output and to tree-based models (e.g., GB/RF) on the same features.

---

## 10. Outputs Summary

After a successful run, you should have:

- **Metrics**
  - `NN.GA_metrics.txt` — best hyperparameters + scaled and original-unit metrics

- **Performance Plot**
  - `ActualvsPredicted_GestationalAge.NN.GA.png` — actual vs predicted GA (original units)

- **SHAP (DeepExplainer)**
  - `SHAP_summary_top20.NN.GA.png`
  - `Top20SHAPfeatures.txt`
  - `SHAP_dependence_<feature>.NN.GA.png` for top 5 features

- **SHAP (KernelExplainer)**
  - `SHAP_summary_top20.NN.GA.Kernel.png`

These files together document:
- How well the NN predicts GA.
- Which covariates are most important according to SHAP.
- How GA changes as key features vary (dependence plots).

---

## 11. How to Run

From the directory containing `Metadata.Final.tsv` and this script (e.g., `nn_ga_shap.py`), run:

```bash
python nn_ga_shap.py
```

Make sure your environment has all required dependencies installed, and that the `site` column is present if you want site-aware splitting.

---

## 12. Notes & Tips

- If DeepExplainer SHAP plots look suspicious (e.g., unexpected top features), rely more on the **KernelExplainer** plots for GA.
- You can plug the SHAP-derived top features into downstream models (e.g., GLMMs) as a **data-driven feature selection** step.
- If you later add or change covariates, update the lists:
  - `categorical_columns`
  - `continuous_columns`
  - `binary_columns`

This README describes the exact behavior of the provided script so it can be understood, rerun, and extended without re-reading the source code line by line.
