import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import keras_tuner as kt




dfIN = pd.read_csv("HetroplasmyNN.C.tsv",sep='\t')

 
# Load metadata
md = pd.read_csv("Metadata.C.Final.tsv",sep='\t')[["SampleID","PTB"]]
md[["PTB"]]=md[["PTB"]].astype(int)





data_encoded=pd.merge(dfIN,md,on=["SampleID"])


# Features (X) and target (y)
X = data_encoded.drop(["PTB","SampleID"], axis=1)
y = data_encoded["PTB"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)






# Define the model for tuning
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(hp.Int('units1', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(hp.Int('units2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Instantiate tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    project_name='haplogroup_tuning'
)

# Perform search
tuner.search(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]






# Define the final model using the best hyperparameters
final_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(best_hps.get('units1'), activation='relu'),
    tf.keras.layers.Dropout(best_hps.get('dropout1')),
    tf.keras.layers.Dense(best_hps.get('units2'), activation='relu'),
    tf.keras.layers.Dropout(best_hps.get('dropout2')),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the final model on the full training data
history = final_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)





# Evaluate the final model
loss, accuracy= final_model.evaluate(X_test, y_test)
print(f"Final Model Test Loss: {loss}")
print(f"Final Model Test Accuracy: {accuracy}")

#print(f"Test Accuracy: {accuracy:.2f}, AUC: {auc:.2f}")






#DeepExplainer
explainer = shap.Explainer(final_model, X_train)
shap_values = explainer.shap_values(X_test)




# Plot feature importance
shap.summary_plot(shap_values, X_test)
plt.savefig("shap_summary_plot.C.sub.png", bbox_inches="tight")
plt.clf()




#Cross-Validation

