import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from keras_tuner import RandomSearch, HyperModel
from sklearn.neural_network import MLPClassifier
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import sys

# Helper function for evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred_proba = model.predict(X_test).ravel()  # Get probabilities
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold of 0.5
    # Now use y_pred for classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    # If you have probabilities and they are applicable, calculate AUC
    if y_pred_proba is not None:
        auc_score = roc_auc_score(y_test, y_pred_proba)  # Ensure y_test and y_pred_proba are correct
        print(f"ROC AUC Score for {model_name}: {auc_score:.4f}")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.show()
        plt.savefig("ROC_AUC_plot.NN.PTB.png")
        plt.clf()





# Load the dataset
df = pd.read_csv("Metadata.Final.tsv", sep='\t')


df = df[sys.argv[1].split(',') + sys.argv[2].split(',') + sys.argv[3].split(',') + ["PTB"]]

df = df[~df.isin([-88, -77]).any(axis=1)]  # Remove rows with invalid entries (-88, -77)
df = df[df['MainHap'].map(df['MainHap'].value_counts()) >= 25]

# Define features
categorical_columns = sys.argv[1].split(',')
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')

X = df[categorical_columns + continuous_columns+ binary_columns]
y = df['PTB'] 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_columns),
        ('bin', 'passthrough', binary_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# Ensure dense matrix for SMOTE
if hasattr(X_train_preprocessed, "toarray"):
    X_train_preprocessed = X_train_preprocessed.toarray()

# Step 2: Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)

# Convert balanced data to DataFrame
X_train_balanced = pd.DataFrame(X_train_balanced, columns=preprocessor.get_feature_names_out())









# Define hypermodel class with dropout and regularization
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
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

hypermodel = PTBHyperModel(input_shape=X_train_balanced.shape[1])

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='ptb_hyperopt'
)

tuner.search(X_train_balanced, y_train_balanced, epochs=50, validation_data=(X_test_preprocessed, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
loss, accuracy = best_model.evaluate(X_test_preprocessed, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

evaluate_model(best_model, X_test_preprocessed, y_test, "NN")





# SHAP values and summary plot
explainer = shap.DeepExplainer(best_model, X_train_balanced)
shap_values = explainer.shap_values(X_test_preprocessed)
# Squeeze the SHAP values to remove the unnecessary dimension
shap_values_squeezed = np.squeeze(np.array(shap_values), axis=2)

# Now try plotting with the corrected shape
shap.summary_plot(shap_values_squeezed, X_test_preprocessed, feature_names=preprocessor.get_feature_names_out(), show=True)
plt.savefig("shap_summary_plot.NN.PTB.png")
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
    plt.savefig("shap.dependence_plot.NN.PTB."+feature+".png")
    plt.clf()

#shap.force_plot(explainer.expected_value, shap_values_squeezed[sample_idx, :], X_test_preprocessed[sample_idx, :], feature_names=preprocessor.get_feature_names_out())




# Save the best model
best_model.save("NN.PTB_best_model.h5")






















