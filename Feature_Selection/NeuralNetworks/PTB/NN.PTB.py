import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from keras_tuner import RandomSearch, HyperModel, Objective

# Optional SMOTE (off by default; NNs often do better with class_weight)
# from imblearn.over_sampling import SMOTE

import shap

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)




categorical_columns = sys.argv[1].split(',')
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')



# -----------------------------
# IO
# -----------------------------
DATA_PATH = "Metadata.Final.tsv"
OUTDIR = "ptb_nn_outputs"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep="\t")

# Keep only needed columns
needed_cols = categorical_columns + continuous_columns + binary_columns + ["PTB"]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in the dataset: {missing}")

df = df[needed_cols].dropna()
# Ensure PTB is int {0,1}
df["PTB"] = df["PTB"].astype(int)

X = df[categorical_columns + continuous_columns + binary_columns]
y = df["PTB"]

# -----------------------------
# Split: train / val / test
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# Force numpy arrays for y to avoid pandas indexing quirks with class_weight
y_train = y_train.to_numpy().astype(int)
y_val   = y_val.to_numpy().astype(int)
y_test  = y_test.to_numpy().astype(int)

# -----------------------------
# Preprocessing (dense OHE)
# -----------------------------
# For sklearn >= 1.2 use 'sparse_output=False'. If older, replace with 'sparse=False'.
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", ohe, categorical_columns),
    ],
    remainder="drop",
)

X_train_p = preprocessor.fit_transform(X_train)
X_val_p   = preprocessor.transform(X_val)
X_test_p  = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

# OPTIONAL: SMOTE on X_train_p (commented out by default)
# sm = SMOTE(random_state=SEED)
# X_train_p, y_train = sm.fit_resample(X_train_p, y_train)

# -----------------------------
# Class weights (balanced heuristic)
# -----------------------------
classes, counts = np.unique(y_train, return_counts=True)
n_classes = len(classes)
total = counts.sum()
class_weight = {int(c): float(total / (n_classes * n)) for c, n in zip(classes, counts)}
# Example outcome for binary: {0: ~0.5/Pr(y=0), 1: ~0.5/Pr(y=1)}

# -----------------------------
# HyperModel
# -----------------------------
class PTBHyperModel(HyperModel):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        # First hidden
        model.add(
            Dense(
                units=hp.Int("units0", min_value=64, max_value=512, step=64),
                activation="relu",
                kernel_regularizer=regularizers.l2(hp.Choice("l2_0", [0.0, 1e-6, 1e-5, 1e-4])),
            )
        )
        model.add(Dropout(hp.Float("dropout0", 0.0, 0.5, step=0.1)))
        # Additional hidden layers
        for i in range(hp.Int("n_hidden", 0, 2)):
            model.add(
                Dense(
                    units=hp.Int(f"units{i+1}", min_value=64, max_value=512, step=64),
                    activation="relu",
                    kernel_regularizer=regularizers.l2(
                        hp.Choice(f"l2_{i+1}", [0.0, 1e-6, 1e-5, 1e-4])
                    ),
                )
            )
            model.add(Dropout(hp.Float(f"dropout{i+1}", 0.0, 0.5, step=0.1)))
        # Output
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float("lr", 1e-4, 3e-3, sampling="log")
            ),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name="AUC"),
                tf.keras.metrics.AUC(curve="PR", name="AUPRC"),
            ],
        )
        return model

# -----------------------------
# Tuner
# -----------------------------
hypermodel = PTBHyperModel(input_dim=X_train_p.shape[1])

tuner = RandomSearch(
    hypermodel,
    objective=Objective("val_AUPRC", direction="max"),
    max_trials=12,
    executions_per_trial=1,
    overwrite=True,
    directory=os.path.join(OUTDIR, "my_dir"),
    project_name="ptb_hyperopt",
)

callbacks = [
    EarlyStopping(monitor="val_AUPRC", mode="max", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_AUPRC", mode="max", patience=5, factor=0.5, min_lr=1e-5),
    ModelCheckpoint(
        filepath=os.path.join(OUTDIR, "checkpoint.best.keras"),
        monitor="val_AUPRC",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
    ),
]

BATCH_SIZE = 256
EPOCHS = 100

tuner.search(
    X_train_p,
    y_train,
    validation_data=(X_val_p, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1,
)

best_model = tuner.get_best_models(1)[0]

best_hps = tuner.get_best_hyperparameters(1)[0]
print("Best hyperparameters:")
for k, v in best_hps.values.items():
    print(f"  {k}: {v}")


# -----------------------------
# Final evaluation on untouched test set
# -----------------------------
y_prob = best_model.predict(X_test_p).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification report @0.5 threshold:")
print(classification_report(y_test, y_pred, digits=3))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR  AUC: {average_precision_score(y_test, y_prob):.4f}")

# -----------------------------
# Curves: ROC + PR
# -----------------------------
# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"Best NN (ROC AUC={roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - PTB")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ROC_AUC_plot.NN.PTB.png"), dpi=150)
plt.close()

# PR
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(rec, prec, label=f"Best NN (PR AUC={average_precision_score(y_test, y_prob):.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall - PTB")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "PR_AUC_plot.NN.PTB.png"), dpi=150)
plt.close()

# -----------------------------
# SHAP (robust, small background)
# -----------------------------
# Use a small background sample to keep DeepExplainer sane
bg_n = min(200, X_train_p.shape[0])
bg_idx = np.random.choice(X_train_p.shape[0], size=bg_n, replace=False)
background = X_train_p[bg_idx]

# Test sample for SHAP plotting
ts_n = min(500, X_test_p.shape[0])
ts_idx = np.random.choice(X_test_p.shape[0], size=ts_n, replace=False)
X_test_sample = X_test_p[ts_idx]

# Some TF/SHAP combos work best with TF functions disabled eager; modern SHAP usually OK.
explainer = shap.DeepExplainer(best_model, background)
shap_values = explainer(X_test_sample)

# Handle SHAP API differences
vals = getattr(shap_values, "values", shap_values)
if isinstance(vals, list):
    # binary sigmoid can return list of arrays; take first
    vals = vals[0]


# Fix shape: ensure (n_samples, n_features)
if vals.ndim == 3:
    vals = np.squeeze(vals, axis=-1)
elif vals.ndim == 1:
    vals = vals.reshape(-1, 1)


# Summary plot (top 20)
shap.summary_plot(vals, X_test_sample, feature_names=feature_names, show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "shap_summary_plot.NN.PTB.png"), dpi=150)
plt.close()

# Top 20 features & dependence plots
mean_abs = np.abs(vals).mean(axis=0)
topk_idx = np.argsort(mean_abs)[::-1][:20]
top_feats = [feature_names[i] for i in topk_idx]

# Print top 20 to stdout
print("\nTop 20 features by mean |SHAP|:")
for fn, mv in zip(top_feats, mean_abs[topk_idx]):
    print(f"{fn}: {mv:.6f}")

num_feats = [f for f in top_feats if f.startswith("num__") or f.startswith("bin__")]




# Dependence plots for top 20
for feat in num_feats:
    shap.dependence_plot(feat, vals, X_test_sample, feature_names=feature_names, show=False)
    safe = feat.replace("/", "_").replace(" ", "_").replace("[", "").replace("]", "")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"shap.dependence_plot.NN.PTB.{safe}.png"), dpi=150)
    plt.close()












# -----------------------------
# Save model
# -----------------------------
best_model.save(os.path.join(OUTDIR, "NN.PTB_best_model.keras"))
print(f"\nSaved model to {os.path.join(OUTDIR, 'NN.PTB_best_model.keras')}")
print(f"Outputs in: {OUTDIR}")
