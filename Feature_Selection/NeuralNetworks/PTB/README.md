# Neural Network for Preterm Birth (PTB) Prediction

## Overview

This repository contains a full deep-learning pipeline for predicting **Preterm Birth (PTB)** as a binary outcome using:

- TensorFlow/Keras fully-connected neural networks  
- KerasTuner hyperparameter optimization  
- Class-weighted training for imbalanced data  
- Dense one-hot encoding + feature scaling via scikit-learn  
- SHAP-based model explainability (global + local feature importance)  
- ROC and Precision–Recall curve evaluation  
- Automated saving of predictions, metrics, plots, and the final model

The main script is designed to be run from the command line, taking lists of **categorical**, **continuous**, and **binary** predictor variables as arguments.

---

## Input Data

The script expects a tab-delimited file:

```text
Metadata.Final.tsv
```

### Required Columns

The file **must** contain:

- All variables listed as:
  - `categorical_columns`
  - `continuous_columns`
  - `binary_columns`
- A binary target column:
  - `PTB`  (0 = term, 1 = preterm)

Any rows with missing values in these columns are dropped before modeling.

---

## Script: `NN_PTB_final.py`

This script implements the full PTB modeling pipeline you see in the code.

### Command-Line Usage

```bash
python NN_PTB_final.py \
  "CAT1,CAT2,CAT3" \
  "CONT1,CONT2,CONT3" \
  "BIN1,BIN2"
```

#### Example

```bash
python NN_PTB_final.py   "MainHap,SubHap,Site"   "Age,BMI,PC1,PC2,PC3"   "Diabetes,CHRON_HTN"
```

Arguments:

1. `sys.argv[1]`: comma-separated categorical feature names  
2. `sys.argv[2]`: comma-separated continuous feature names  
3. `sys.argv[3]`: comma-separated binary feature names  

These names **must** match column names in `Metadata.Final.tsv`.

---

## Data Processing Workflow

1. **Load data**

   ```python
   DATA_PATH = "Metadata.Final.tsv"
   df = pd.read_csv(DATA_PATH, sep="\t")
   ```

2. **Column validation**

   The script verifies that all requested feature columns and the `PTB` column exist in the dataframe. If not, it raises a `ValueError`.

3. **Filtering and cleaning**

   - Subset to:
     ```python
     needed_cols = categorical_columns + continuous_columns + binary_columns + ["PTB"]
     df = df[needed_cols].dropna()
     ```
   - Coerce `PTB` to integer 0/1.

4. **Train/Validation/Test split**

   - First split: 70% train / 30% temp (stratified by PTB)
   - Second split: temp → 50% validation / 50% test (stratified)

   Final proportions:
   - Train: 70%
   - Validation: 15%
   - Test: 15%

5. **Preprocessing**

   Uses `ColumnTransformer`:

   - `StandardScaler` on continuous features  
   - `passthrough` for binary features  
   - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categoricals  

   Returns a **dense** NumPy array suitable for Keras and SHAP.

6. **Class weighting**

   To handle class imbalance, class weights are computed as:

   ```python
   class_weight[c] = total_samples / (n_classes * count_c)
   ```

   This gives approximately equal total weight to each class and is passed into `model.fit()`.

---

## Model & Hyperparameter Tuning

### Architecture (HyperModel)

The network is built with a tunable number of layers and units:

- Input layer with dimension = number of preprocessed features
- First hidden layer:
  - Dense
  - Units: 64–512 (step 64), ReLU
  - L2 regularization: {0, 1e-6, 1e-5, 1e-4}
  - Dropout: 0.0 – 0.5 (step 0.1)
- Additional hidden layers:
  - 0 to 2 extra hidden layers
  - Each with tunable units, L2 regularization, and dropout
- Output layer:
  - Dense(1, activation = "sigmoid")  → binary PTB prediction

### Loss, Optimizer, Metrics

- Loss: `binary_crossentropy`
- Optimizer: `Adam` with tunable learning rate (1e-4 to 3e-3, log scale)
- Metrics:
  - `AUC` (ROC)
  - `AUC` (PR) as `AUPRC`

### Tuning Objective

The KerasTuner `RandomSearch` is configured to **maximize**:

```python
objective = "val_AUPRC"
```

This focuses on performance for the minority PTB class.

### Callbacks

During tuning and training, the script uses:

- `EarlyStopping` on `val_AUPRC` (patience = 10, restore best weights)
- `ReduceLROnPlateau` on `val_AUPRC` (patience = 5, factor = 0.5)
- `ModelCheckpoint`:
  - Saves the best model by `val_AUPRC` as `checkpoint.best.keras` inside the output directory.

---

## Reproducibility

To make results as stable as possible:

```python
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

This stabilizes random splits, network initialization, and sampling.

---

## Evaluation

After tuning, the best model is evaluated on the **held-out test set**:

- Classification report (@ threshold 0.5):
  - Precision, recall, F1 for each class
- ROC AUC
- PR AUC (Average Precision)

### Curves

The script generates:

1. **ROC curve**

   Saved as:
   ```text
   ptb_nn_outputs/ROC_AUC_plot.NN.PTB.png
   ```

2. **Precision–Recall curve**

   Saved as:
   ```text
   ptb_nn_outputs/PR_AUC_plot.NN.PTB.png
   ```

---

## SHAP Explainability

The script uses **SHAP DeepExplainer** to interpret the trained neural network.

### Steps:

1. **Background sample**  
   A subset of up to 200 training examples is used as the background:

   ```python
   bg_n = min(200, X_train_p.shape[0])
   ```

2. **Test sample for explanation**  
   A subset of up to 500 test examples is used for SHAP plotting.

3. **Explainer & SHAP values**

   ```python
   explainer = shap.DeepExplainer(best_model, background)
   shap_values = explainer(X_test_sample)
   ```

   The script then robustly handles different SHAP return types and shapes, ensuring a 2D array `(n_samples, n_features)`.

4. **Summary plot**

   Global feature importance (top 20 features) is saved as:

   ```text
   ptb_nn_outputs/shap_summary_plot.NN.PTB.png
   ```

5. **Top 20 features by mean |SHAP|**

   The script prints the top 20 most important features (by mean absolute SHAP value) to stdout.

6. **Dependence plots**

   For selected features (numeric or binary, e.g., `num__Age`, `bin__Diabetes`), dependence plots are generated and saved as:

   ```text
   ptb_nn_outputs/shap.dependence_plot.NN.PTB.<feature>.png
   ```

---

## Saved Outputs

All outputs are written under:

```text
ptb_nn_outputs/
```

### Files Produced

- **Model:**
  - `NN.PTB_best_model.keras` — Saved best Keras model.
- **Curves:**
  - `ROC_AUC_plot.NN.PTB.png`
  - `PR_AUC_plot.NN.PTB.png`
- **SHAP:**
  - `shap_summary_plot.NN.PTB.png`
  - `shap.dependence_plot.NN.PTB.<feature>.png` (multiple files)
- **Predictions & Metrics:**
  - `NN.PTB_test_predictions.tsv`  
    - Columns: `y_test`, `y_prob`, `y_pred`
  - `NN.PTB_metrics.txt`  
    - ROC AUC  
    - PR AUC  
    - Full text classification report at 0.5 threshold

---

## Dependencies

Install required packages (example):

```bash
pip install   pandas   numpy   scikit-learn   matplotlib   tensorflow   keras-tuner   shap
```

> Note:  
> - If you use scikit-learn < 1.2, change `sparse_output=False` to `sparse=False` in `OneHotEncoder`.  
> - Some SHAP + TensorFlow versions may require compatible versions (e.g., SHAP ≥ 0.41+).

---

## Recommended Folder Structure

```text
├── Metadata.Final.tsv
├── NN_PTB_final.py
├── ptb_nn_outputs/
│   ├── NN.PTB_best_model.keras
│   ├── ROC_AUC_plot.NN.PTB.png
│   ├── PR_AUC_plot.NN.PTB.png
│   ├── shap_summary_plot.NN.PTB.png
│   ├── shap.dependence_plot.NN.PTB.<feature>.png
│   ├── NN.PTB_test_predictions.tsv
│   └── NN.PTB_metrics.txt
└── README.md
```

---

## Notes and Future Extensions

- You can adapt this pipeline to other binary phenotypes by changing the target column name and feature lists.
- To support multi-class outcomes, you would:
  - Change output layer & activation (e.g., softmax)
  - Use `categorical_crossentropy` or `sparse_categorical_crossentropy`
  - Adjust SHAP interpretation accordingly.
- If the dataset is extremely imbalanced, you could experiment with:
  - SMOTE on the preprocessed input (commented in code)
  - Focal loss instead of binary cross-entropy
  - Alternative thresholding strategies beyond 0.5.

---

## Citation (Example)

If you use this pipeline in a manuscript, you can describe it along the lines of:

> "We trained a class-weighted feed-forward neural network to predict preterm birth using clinical and demographic features. Hyperparameters were optimized via KerasTuner RandomSearch with validation AUPRC as the objective. Feature importance and effect directions were assessed using SHAP DeepExplainer on the final model."

