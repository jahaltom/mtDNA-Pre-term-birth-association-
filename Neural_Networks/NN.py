import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import keras_tuner as kt
import pickle
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense, Dropout




# Load metadata
md = pd.read_csv("Metadata.M.Final.tsv",sep='\t')

md=md.dropna(subset=["PTB","GAGEBRTH"])



md.set_index('Sample_ID', inplace=True)
md=md.astype(str)

md=md[["PW_AGE",        "PW_EDUCATION", "TYP_HOUSE",    "HH_ELECTRICITY",       "FUEL_FOR_COOK",        "DRINKING_SOURCE",      "TOILET",       "WEALTH_INDEX", "SNIFF_TOBA",   "SNIFF_FREQ",   "SMOKE_HIST",   "SMOK_FREQ",    "PASSIVE_SMOK", "ALCOHOL",
    "ALCOHOL_FREQ",     "CHRON_HTN",    "DIABETES",     "TB",   "THYROID",      "EPILEPSY",     "MAT_HEIGHT",   "BIRTH_WEIGHT", "GAGEBRTH",  "PTB",  "Sex",  "MainHap",      "PC1_All",      "PC2_All",      "PC3_All",      "PC4_All",      "PC5_All",      "PC6_All",      "PC7_All",
    "PC8_All",  "PC9_All",      "PC10_All",     "PC11_All",     "PC12_All",     "PC13_All",     "PC14_All",     "PC15_All",     "PC16_All",     "PC17_All",     "PC18_All",     "PC19_All",     "PC20_All",     "C1_All",       "C2_All",       "C3_All",       "C4_All",       "C5_All"]]



#Catigorical
# One-hot encode haplogroups and gender
data_encoded = pd.get_dummies(md, columns=['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'ALCOHOL', 'CHRON_HTN',
       'DIABETES', 'TB', 'THYROID', 'EPILEPSY',
        'Sex', 'MainHap',"SNIFF_TOBA","SMOKE_HIST"])



# Select continuous columns
continuous_columns = ['PW_AGE','ALCOHOL_FREQ',"SNIFF_FREQ", "SMOK_FREQ",'PW_EDUCATION', 'BIRTH_WEIGHT', 'MAT_HEIGHT','GAGEBRTH',"PC1_All",   "PC2_All",      "PC3_All",      "PC4_All",      "PC5_All",      "PC6_All",      "PC7_All",  "PC8_All",  "PC9_All",      "PC10_All",     "PC11_All",     "PC12_All",     "PC13_All",  "PC14_All",     "PC15_All",     "PC16_All",     "PC17_All",     "PC18_All",     "PC19_All",     "PC20_All",     "C1_All",       "C2_All",       "C3_All",       "C4_All",       "C5_All"]


# Scale data to range [0, 1]
scaler = MinMaxScaler()
data_encoded[continuous_columns] = scaler.fit_transform(data_encoded[continuous_columns])


 

cols=data_encoded.drop(columns=continuous_columns+["PTB"]).columns
data_encoded[cols]=data_encoded[cols].astype(int)






##########
data_encoded[["PTB"]]= data_encoded[["PTB"]].astype(float).astype(int)



# Features (X) and target (y)
X = data_encoded.drop(["PTB","GAGEBRTH"], axis=1)
y = data_encoded["PTB"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)






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
loss, accuracy = final_model.evaluate(X_test, y_test)
print(f"Final Model Test Loss: {loss}")
print(f"Final Model Test Accuracy: {accuracy}")




explainer = shap.Explainer(final_model, X_train)
shap_values = explainer.shap_values(X_test)




# Plot feature importance
shap.summary_plot(shap_values, X_test)
plt.savefig("shap_summary_plot.sub.M.png", bbox_inches="tight")
plt.clf()








# Explain the model predictions using KernelExplainer
explainer = shap.KernelExplainer(final_model, X_train)
# Compute SHAP values for the test set.   Use a lower nsamples value (e.g., 100 or 200) to save time, 
shap_values = explainer.shap_values(X_test, nsamples=100)


with open("shap_values.M.pkl", "wb") as file:
    pickle.dump(shap_values, file)



# # Load shap_values back
# with open("shap_values.pkl", "rb") as file:
#     loaded_shap_values = pickle.load(file)


# Plot feature importance
shap.summary_plot(shap_values, X_test)
plt.savefig("shap_summary_plot.sub.Kernal.M.png", bbox_inches="tight")
plt.clf()






# shap_values = shap_values.squeeze()

# #For individual predictions, use shap.force_plot:
# shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
# plt.savefig("shap_force_plot.sub.png", bbox_inches="tight")
# plt.clf()


# #Understand interactions between features and predictions:
# shap.dependence_plot("HH_ELECTRICITY_0", shap_values, X_test)
# plt.savefig("shap_dependence_plot.sub.png", bbox_inches="tight")
# plt.clf()





#Cross-Validation
