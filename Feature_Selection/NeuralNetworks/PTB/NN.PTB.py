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
from sklearn.metrics import ( classification_report, roc_auc_score,average_precision_score,  roc_curve, precision_recall_curve)
from keras_tuner import RandomSearch, HyperModel, Objective
import shap
import os


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



df = pd.read_csv("Metadata.Final.tsv", sep="\t")

categorical_columns=['FUEL_FOR_COOK']
continuous_columns=['PW_AGE','PW_EDUCATION','BMI','TOILET','WEALTH_INDEX','DRINKING_SOURCE']
binary_columns=['BABY_SEX','CHRON_HTN','DIABETES','HH_ELECTRICITY','TB','THYROID','TYP_HOUSE']






X = df[categorical_columns + continuous_columns + binary_columns]
y = df["PTB"]



from sklearn.model_selection import GroupShuffleSplit, train_test_split

# -----------------------------
# Outer split: site-aware if possible
#   - ≥3 sites: unseen-site test via GroupShuffleSplit
#   - 2 sites: stratified row split + keep site labels for group-aware inner split
#   - <2 sites: standard stratified split, no groups
# -----------------------------
if "site" in df.columns:
    n_sites = df["site"].nunique()
else:
    n_sites = 0
if ("site" in df.columns) and (n_sites >= 3):
    groups_all = df["site"].values
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, test_idx = next(gss_outer.split(X, y, groups=groups_all))
    X_train_full = X.iloc[train_idx]
    y_train_full = y.iloc[train_idx]
    X_test       = X.iloc[test_idx]
    y_test       = y.iloc[test_idx]
    groups_train = groups_all[train_idx]
elif ("site" in df.columns) and (n_sites == 2):
    # Prefer site-aware inner tuning over a strict unseen-site test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=SEED
    )
    groups_train = df.loc[X_train_full.index, "site"].values
else:
    # No usable site structure
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=SEED
    )
    groups_train = None
# -----------------------------
# Inner split: train vs val
#   - GroupShuffleSplit if we still have ≥2 training sites
#   - Else Stratified split (classification)
# -----------------------------
if (groups_train is not None) and (len(np.unique(groups_train)) >= 2):
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 1)
    tr_idx, val_idx = next(
        gss_inner.split(X_train_full, y_train_full, groups=groups_train)
    )
    X_train = X_train_full.iloc[tr_idx]
    y_train = y_train_full.iloc[tr_idx]
    X_val   = X_train_full.iloc[val_idx]
    y_val   = y_train_full.iloc[val_idx]
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        stratify=y_train_full,
        random_state=SEED
    )

# Ensure numpy int arrays as before
y_train = np.asarray(y_train).astype(int)
y_val   = np.asarray(y_val).astype(int)
y_test  = np.asarray(y_test).astype(int)






preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ],
    remainder="drop",
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed   = preprocessor.transform(X_val)
X_test_preprocessed  = preprocessor.transform(X_test)


# Transform the full dataset (train + val + test) with the same preprocessor
X_all_preprocessed = preprocessor.transform(X)
# Feature names for SHAP / plotting
feature_names = preprocessor.get_feature_names_out()


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
class HyperModel(HyperModel):
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
hypermodel = HyperModel(input_dim=X_train_preprocessed.shape[1])

tuner = RandomSearch(
    hypermodel,
    objective=Objective("val_AUPRC", direction="max"),
    max_trials=12,
    executions_per_trial=1,
    overwrite=True,
    directory=os.path.join("tuning"),
)

callbacks = [
    EarlyStopping(monitor="val_AUPRC", mode="max", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_AUPRC", mode="max", patience=5, factor=0.5, min_lr=1e-5),
    ModelCheckpoint(
        filepath=os.path.join("checkpoint.best.keras"),
        monitor="val_AUPRC",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
    ),
]

BATCH_SIZE = 256
EPOCHS = 100

tuner.search(
    X_train_preprocessed,
    y_train,
    validation_data=(X_val_preprocessed, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1,
)

best_model = tuner.get_best_models(1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]

# -----------------------------
# Final evaluation on untouched test set
# -----------------------------
y_prob = best_model.predict(X_test_preprocessed).ravel()
y_pred = (y_prob >= 0.5).astype(int)

with open(os.path.join("NN.PTB_metrics.txt"), "w") as f:
    f.write("Best hyperparameters:")
    for k, v in best_hps.values.items():
        f.write(f"  {k}: {v}")
    f.write(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}\n")
    f.write(f"PR  AUC: {average_precision_score(y_test, y_prob):.4f}\n")
    f.write("\nClassification report @0.5:\n")
    f.write(classification_report(y_test, y_pred, digits=3))


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
plt.savefig(os.path.join("ROC_AUC_plot.NN.PTB.png"), dpi=150)
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
plt.savefig(os.path.join("PR_AUC_plot.NN.PTB.png"), dpi=150)
plt.close()

# ---------




# -----------------------------
# SHAP DeepExplainer on FULL DATA (train + val + test)
# -----------------------------
import shap
import matplotlib.pyplot as plt




# 2) Data to explain: entire dataset
X_for_shap = np.asarray(X_all_preprocessed)

# 3) Build DeepExplainer
explainer = shap.DeepExplainer(best_model, X_train_preprocessed)

# 4) Compute SHAP values on FULL DATA
shap_raw = explainer.shap_values(X_for_shap)

# Handle SHAP API variants
vals = getattr(shap_raw, "values", shap_raw)
if isinstance(vals, list):
    # For binary sigmoid TF/Keras, DeepExplainer often returns [array]
    vals = vals[0]

vals = np.asarray(vals)


# 5) Fix shape so it's (n_samples, n_features)
n_samples, n_features_expected = X_for_shap.shape

if vals.ndim == 1:
    # (n_samples,) -> (n_samples, 1)
    vals = vals.reshape(-1, 1)
elif vals.ndim == 2:
    # (n_samples, n_features?) or (n_features, n_samples?)
    if vals.shape[0] == n_features_expected and vals.shape[1] == n_samples:
        # Looks transposed -> fix
        vals = vals.T
elif vals.ndim == 3:
    # Common weird case: (n_samples, n_features, 1) or (1, n_samples, n_features)
    if vals.shape[-1] == 1:
        vals = np.squeeze(vals, axis=-1)
    elif vals.shape[0] == 1 and vals.shape[2] == n_features_expected:
        vals = vals[0, :, :]  # (1, n_samples, n_features) -> (n_samples, n_features)



# 6) SHAP summary plot (top 20 features, full data)
plt.figure()
shap.summary_plot(
    vals,
    X_for_shap,
    feature_names=feature_names,
    max_display=20,
    show=False,
)
plt.tight_layout()

summary_name = "SHAP_summary_top20.NN.PTB.png"  # or .GA for the GA script
plt.savefig(summary_name, dpi=150)
plt.close()



# 6) Top 20 features by mean |SHAP| over ALL rows
mean_abs_shap = np.mean(np.abs(vals), axis=0)
top20_idx = np.argsort(mean_abs_shap)[::-1][:20]
top20_features = [feature_names[i] for i in top20_idx]
top20_values = mean_abs_shap[top20_idx]


with open(os.path.join("Top20SHAPfeatures.txt"), "w") as ff:
    ff.write("\nTop 20 features by mean |SHAP| over full dataset:")
    for rank, (fname, val) in enumerate(zip(top20_features, top20_values), start=1):
        ff.write(f"{rank:2d}. {fname:40s}  mean|SHAP| = {val:.6f}")




# Dependence plots for top 5 features
for fname in top20_features[:5]:
    plt.figure()
    shap.dependence_plot(
        fname,
        vals,
        X_for_shap,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"SHAP_dependence_{fname.replace(os.sep, '_')}.NN.PTB.png", dpi=150)
    plt.close()
