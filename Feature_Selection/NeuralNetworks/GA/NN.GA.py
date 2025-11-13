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


import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# Set to display all columns
pd.set_option('display.max_columns', None)
# Set to display all rows
pd.set_option('display.max_rows', None)

# Load the dataset
df = pd.read_csv("Metadata.Final.tsv", sep='\t')

# Define features
categorical_columns = sys.argv[1].split(',')
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')

df = df[categorical_columns + continuous_columns + binary_columns + ["GAGEBRTH"]]




X = df[categorical_columns + continuous_columns+ binary_columns]
y = df['GAGEBRTH']  





# Train-test split
from sklearn.model_selection import GroupShuffleSplit

# After loading df and defining X, y:
groups = df["site"].values  # even if site is not in X

# --- 1) Site-aware train vs test split ---
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(gss1.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_test  = y.iloc[test_idx]

groups_train = groups[train_idx]

# --- 2) Site-aware train vs val split within the train set ---
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=43)
train_idx2, val_idx = next(
    gss2.split(X_train, y_train, groups=groups_train)
)

X_tr = X_train.iloc[train_idx2]
y_tr = y_train.iloc[train_idx2]
X_val = X_train.iloc[val_idx]
y_val = y_train.iloc[val_idx]






# Preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_columns),
        ('bin', 'passthrough', binary_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Fit preprocessor only on training part (X_tr)
X_tr_preprocessed   = preprocessor.fit_transform(X_tr)
X_val_preprocessed  = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)












# Define hypermodel class with dropout and regularization, suitable for regression
class PTBHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def build(self, hp):
        model = Sequential([
            Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                  activation='relu', input_shape=(self.input_shape,)),
            Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.05))
        ])
        for i in range(hp.Int('layers', 1, 3)):
            model.add(Dense(units=hp.Int(f'units_{i}', 32, 512, 32), activation='relu'))
            model.add(Dropout(rate=hp.Float(f'dropout_{i}', 0.0, 0.5, 0.05)))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model

# Initialize and run the tuner
hypermodel = PTBHyperModel(input_shape=X_tr_preprocessed.shape[1])

tuner = RandomSearch(
    hypermodel,
    objective='val_mean_squared_error',
    max_trials=10,
    executions_per_trial=2,
    directory='model_tuning',
    project_name='gestational_age_prediction'
)

tuner.search(
    X_tr_preprocessed,
    y_tr,
    epochs=50,
    validation_data=(X_val_preprocessed, y_val),  # <-- key change
    verbose=1
)


# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
best_model.evaluate(X_test_preprocessed, y_test)
y_pred = best_model.predict(X_test_preprocessed).flatten()

mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared: {r_squared:.4f}")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
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




# SHAP values and summary plot
explainer = shap.DeepExplainer(best_model, X_tr_preprocessed) 
shap_values = explainer.shap_values(X_test_preprocessed)


# Squeeze the SHAP values to remove the unnecessary dimension
shap_values_squeezed = np.squeeze(np.array(shap_values), axis=2)

# Now try plotting with the corrected shape
shap.summary_plot(shap_values_squeezed, X_test_preprocessed, feature_names=preprocessor.get_feature_names_out(), show=True)
plt.savefig("shap_summary_plot.NN.GA.png")
plt.clf()

#Top 20
mean_abs_shap_values = np.abs(shap_values_squeezed).mean(axis=0)
# Get the feature names from the preprocessor
feature_names = preprocessor.get_feature_names_out()
# Sort the features by mean absolute SHAP value
sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
top_feature_names = np.array(feature_names)[sorted_indices[:20]]
top_shap_values = mean_abs_shap_values[sorted_indices[:20]]
# Output top 10 features and their SHAP values
print("Top 10 SHAP Values and Corresponding Features:")
for feature, value in zip(top_feature_names, top_shap_values):
    print(f"{feature}: {value:.4f}")

for feature in top_feature_names:
    shap.dependence_plot(feature, shap_values_squeezed, X_test_preprocessed, feature_names=preprocessor.get_feature_names_out())
    plt.savefig("shap.dependence_plot.NN.GA."+feature+".png")
    plt.clf()

#shap.force_plot(explainer.expected_value, shap_values_squeezed[sample_idx, :], X_test_preprocessed[sample_idx, :], feature_names=preprocessor.get_feature_names_out())





# Save the best model
best_model.save("NN.GA_best_model.h5")
