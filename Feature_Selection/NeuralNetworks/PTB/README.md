# PTB Neural Network Classifier Pipeline

This repository implements a **deep-learning pipeline** using **TensorFlow/Keras**, **Keras Tuner**, and **SHAP** to model **Preterm Birth (PTB)** from clinical, demographic, and genetic features.  
It includes **site-aware splitting**, **feature preprocessing**, **hyperparameter search**, **model evaluation**, and **interpretability outputs**.

---

# 📌 Overview

This script:

1. Loads `Metadata.Final.tsv`
2. Performs **site-aware train/validation/test splitting** using `GroupShuffleSplit`
3. Applies preprocessing:
   - Standardization for continuous variables  
   - Passthrough for binary variables  
   - One-hot encoding for categorical variables  
4. Computes heuristic **class weights** to address imbalance  
5. Builds & tunes a fully connected neural network via **Keras Tuner RandomSearch**
6. Evaluates the best model on an untouched test set
7. Generates:
   - ROC curve  
   - Precision–Recall curve  
   - SHAP summary plot  
   - SHAP dependence plots for top features  
8. Saves predictions, metrics, and the final model

All outputs are stored in:

```
ptb_nn_outputs/
```

---

# ⚙️ Requirements

Install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras-tuner shap
```

Recommended Python version: **3.9+**

---

# 📁 Input File Requirements

The script expects:

### **Input TSV file:**  
`Metadata.Final.tsv` (tab-delimited)

### Required columns:

| Column | Description |
|--------|-------------|
| `PTB` | Binary outcome (0/1) |
| `site` | Site identifier (used for grouped splitting) |
| Feature columns | Provided by user on command line |

### Command line arguments define the feature sets:

1. Categorical columns (comma-separated)
2. Continuous columns (comma-separated)
3. Binary columns (comma-separated)

Example:

```bash
python ptb_nn.py "HAPLOGROUP,SUBHAP,SMOKING" "AGE,BMI,PC1,PC2,PC3" "CHRON_HTN,GEST_DIAB"
```

`site` is automatically excluded from categorical inputs.

---

# 🔀 Train / Validation / Test Splitting

### If ≥2 sites:
- 70% train+val  
- 30% test  
- Train split into 50% train / 50% validation  
- **No site appears in more than one split**

### If only 1 site:
Stratified splitting is used instead.

This ensures **no data leakage** between sites.

---

# 🧪 Preprocessing

### OneHotEncoder
Used for categorical variables:

```python
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

### ColumnTransformer

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", ohe, categorical_columns),
    ]
)
```

This produces a fully numerical, dense feature matrix ready for NN training.

---

# 🧮 Handling Class Imbalance

The script computes balanced heuristic class weights:

```python
class_weight = {
    0: (total/(2*n0)),
    1: (total/(2*n1))
}
```

These are passed into model training to compensate for imbalanced PTB labels.

---

# 🧱 Neural Network Architecture (Tuned)

The model is defined using a **HyperModel**:

- Tunable hidden layers (0–2)
- Units per layer: 64 → 512
- L2 regularization (0 → 1e-4)
- Dropout (0 → 0.5)
- Tunable learning rate

Final output layer:

```python
Dense(1, activation="sigmoid")
```

Trained with:

- Loss: `binary_crossentropy`
- Metrics: ROC AUC, PR AUC

---

# 🔍 Hyperparameter Search

The script uses:

```python
Keras Tuner → RandomSearch
```

### Objective:
Maximize **validation AUPRC**

### Search settings:

- `max_trials = 12`
- Up to 100 epochs per trial
- Early stopping and learning rate reduction

Results saved under:

```
ptb_nn_outputs/my_dir/ptb_hyperopt/
```

---

# 🧪 Final Evaluation (Untouched Test Set)

The best model is evaluated on the true test set:

- ROC AUC
- PR AUC
- Classification report
- Predictions saved to:
  ```
  ptb_nn_outputs/NN.PTB_test_predictions.tsv
  ```

---

# 📈 Plots Generated

Saved as PNG:

| Plot | File |
|------|------|
| ROC Curve | `ROC_AUC_plot.NN.PTB.png` |
| Precision–Recall Curve | `PR_AUC_plot.NN.PTB.png` |
| SHAP Summary | `shap_summary_plot.NN.PTB.png` |
| SHAP Dependence Plots | One per top feature |

---

# 🧠 SHAP Interpretability

The script uses **DeepExplainer** with:

- 200-sample background
- 500-sample test subset

Outputs:

- **Top 20 SHAP features**
- Dependence plots for numerical & binary features

---

# 💾 Model Saving

The final tuned model is written to:

```
ptb_nn_outputs/NN.PTB_best_model.keras
```

Load later using:

```python
import tensorflow as tf
model = tf.keras.models.load_model("ptb_nn_outputs/NN.PTB_best_model.keras")
```

---

# 📂 Output Directory Summary

After running the script, expect:

```
ptb_nn_outputs/
  ├── NN.PTB_best_model.keras
  ├── checkpoint.best.keras
  ├── NN.PTB_test_predictions.tsv
  ├── NN.PTB_metrics.txt
  ├── ROC_AUC_plot.NN.PTB.png
  ├── PR_AUC_plot.NN.PTB.png
  ├── shap_summary_plot.NN.PTB.png
  ├── shap.dependence_plot.NN.PTB.<feature>.png
  └── my_dir/
       └── ptb_hyperopt/...
```

---

# 🧩 Troubleshooting

### → **ValueError: could not convert string to float**
Check that continuous and binary columns contain strictly numeric values.

### → **KeyError for feature columns**
Make sure column names match exactly what is passed from the command line.

### → **SHAP too slow**
Reduce:
```python
bg_n, ts_n
```
inside SHAP section.

### → **Inconsistent results on GPU**
GPU introduces nondeterminism.  
Use CPU for strict reproducibility.

---

# ✔️ Citation (Optional)

If publishing, cite:

- TensorFlow/Keras  
- scikit-learn  
- Keras Tuner  
- SHAP  

---

# 🎉 You're Ready to Run!

1. Prepare `Metadata.Final.tsv`  
2. Choose feature lists  
3. Run the script  
4. Inspect:
   - Metrics  
   - Plots  
   - SHAP  
5. Use the saved model for downstream inference  

Enjoy your fully documented PTB neural network pipeline!
