import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras_tuner import RandomSearch, HyperModel
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import sys
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import random

SEED=42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




df = pd.read_csv("Metadata.Final.tsv", sep='\t')

# Define features
categorical_columns = [c for c in sys.argv[1].split(',') if c != "site"]
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')




X = df[categorical_columns + continuous_columns+ binary_columns]
y = df['GAGEBRTH']  





from sklearn.model_selection import GroupShuffleSplit, train_test_split

# -----------------------------
# Outer split: site-aware if possible
#   - ≥3 sites: unseen-site test via GroupShuffleSplit
#   - 2 sites: row-level split, keep site labels for group-aware inner split
#   - <2 sites: standard split, no groups
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
        X, y, test_size=0.30, random_state=SEED
    )
    groups_train = df.loc[X_train_full.index, "site"].values

else:
    # No usable site structure
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    groups_train = None

# -----------------------------
# Inner split: train vs val
#   - GroupShuffleSplit if we still have ≥2 training sites
#   - Else plain random split (regression → no stratify)
# -----------------------------
if (groups_train is not None) and (len(np.unique(groups_train)) >= 2):
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 1)
    tr_idx, val_idx = next(
        gss_inner.split(X_train_full, y_train_full, groups=groups_train)
    )
    X_train  = X_train_full.iloc[tr_idx]
    y_train  = y_train_full.iloc[tr_idx]
    X_val = X_train_full.iloc[val_idx]
    y_val = y_train_full.iloc[val_idx]
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20,
        random_state=SEED
    )




# Preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_columns),
        ('bin', 'passthrough', binary_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

X_train_preprocessed   = preprocessor.fit_transform(X_train)
X_val_preprocessed  = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)





# Define hypermodel class with dropout and regularization, suitable for regression
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from keras_tuner import HyperModel

class HyperModel(HyperModel):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))

        # First hidden layer (mirrors PTB style)
        units0 = hp.Int("units0", min_value=64, max_value=512, step=64)
        l2_0   = hp.Choice("l2_0", [0.0, 1e-6, 1e-5, 1e-4])
        model.add(
            Dense(
                units=units0,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_0),
            )
        )
        model.add( Dropout( hp.Float("dropout0", min_value=0.0, max_value=0.5, step=0.1)))
        # Additional hidden layers (0–2), same pattern as PTB
        n_hidden = hp.Int("n_hidden", min_value=0, max_value=2)
        for i in range(n_hidden):
            units_i = hp.Int(f"units{i+1}", min_value=64, max_value=512, step=64)
            l2_i    = hp.Choice(f"l2_{i+1}", [0.0, 1e-6, 1e-5, 1e-4])
            model.add(
                Dense(
                    units=units_i,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2_i),
                )
            )
            model.add(  Dropout(hp.Float(f"dropout{i+1}", min_value=0.0, max_value=0.5, step=0.1)))
        # Regression output: linear
        model.add(Dense(1, activation="linear"))
        # Same lr concept as PTB, just name it 'lr' to match
        lr = hp.Float("lr", min_value=1e-4, max_value=3e-3, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mean_squared_error",
            metrics=["mean_squared_error"],
        )
        return model


from keras_tuner import RandomSearch, Objective
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Initialize hypermodel
hypermodel = HyperModel(input_dim=X_train_preprocessed.shape[1])

# Tuner (similar spirit to PTB)
tuner = RandomSearch(
    hypermodel,
    objective=Objective("val_mean_squared_error", direction="min"),
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory=os.path.join(OUTDIR, "my_dir"),
    project_name="ga_hyperopt",
)

# Callbacks – mirror PTB logic, but for val MSE (minimize)
callbacks = [
    EarlyStopping(  monitor="val_mean_squared_error",  mode="min",patience=10, restore_best_weights=True,),
    ReduceLROnPlateau( monitor="val_mean_squared_error",mode="min", patience=5, factor=0.5, min_lr=1e-5,),
    ModelCheckpoint( filepath=os.path.join(OUTDIR, "checkpoint.best.keras"), monitor="val_mean_squared_error",mode="min",save_best_only=True,save_weights_only=False,
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
    verbose=1,
)

# Best model + evaluation (same as before)
best_model = tuner.get_best_models(1)[0]

print("Best hyperparameters:")
best_hps = tuner.get_best_hyperparameters(1)[0]
for k, v in best_hps.values.items():
    print(f"  {k}: {v}")

best_model.evaluate(X_test_preprocessed, y_test, verbose=1)
y_pred = best_model.predict(X_test_preprocessed).flatten()

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")







# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
y_min, y_max = y_test.min(), y_test.max()
plt.plot([y_min, y_max], [y_min, y_max], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Gestational Age')
plt.show()
plt.savefig("ActualvsPredicted_GestationalAge.NN.GA.png")
plt.clf()



explainer = shap.DeepExplainer(best_model, X_train_preprocessed) 
shap.summary_plot(shap_values_squeezed, X_test_preprocessed, feature_names=preprocessor.get_feature_names_out(), show=True)
#Top 20mean_abs_shap_values =
top_feature - names:shap.dependence_plot
