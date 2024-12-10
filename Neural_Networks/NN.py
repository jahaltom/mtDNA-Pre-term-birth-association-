import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import keras_tuner as kt

#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense, Dropout




# Load metadata
md = pd.read_csv("MetadataFinal.C.tsv",sep='\t')
md.set_index('Sample_ID', inplace=True)
md=md.astype(str)


#Catigorical 
# One-hot encode haplogroups and gender
data_encoded = pd.get_dummies(md, columns=['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'ALCOHOL', 'CHRON_HTN',
       'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'FETAL_ANOMALIES', 'BABY_SEX',
       'site', 'Sex', 'MainHap','SubHap'])

print(data_encoded)

cols=['TYP_HOUSE_-77', 'TYP_HOUSE_-88', 'TYP_HOUSE_1', 'TYP_HOUSE_2', 'HH_ELECTRICITY_-88', 'HH_ELECTRICITY_0', 'HH_ELECTRICITY_1', 'FUEL_FOR_COOK_-77', 'FUEL_FOR_COOK_-88', 'FUEL_FOR_COOK_1', 'FUEL_FOR_COOK_2', 'FUEL_FOR_COOK_3', 'FUEL_FOR_COOK_4', 'FUEL_FOR_COOK_5', 'DRINKING_SOURCE_-77', 'DRINKING_SOURCE_-88', 'DRINKING_SOURCE_1', 'DRINKING_SOURCE_2', 'DRINKING_SOURCE_3', 'DRINKING_SOURCE_4', 'TOILET_-88', 'TOILET_1', 'TOILET_2', 'TOILET_3', 'TOILET_4', 'WEALTH_INDEX_-77', 'WEALTH_INDEX_-88', 'WEALTH_INDEX_1', 'WEALTH_INDEX_2', 'WEALTH_INDEX_3', 'WEALTH_INDEX_4', 'WEALTH_INDEX_5', 'PASSIVE_SMOK_-77', 'PASSIVE_SMOK_0', 'PASSIVE_SMOK_1', 'ALCOHOL_-77', 'ALCOHOL_1', 'ALCOHOL_2', 'ALCOHOL_4', 'CHRON_HTN_-77', 'CHRON_HTN_0', 'CHRON_HTN_1', 'DIABETES_-77', 'DIABETES_0', 'DIABETES_1', 'TB_-77', 'TB_0', 'TB_1', 'THYROID_-77', 'THYROID_0', 'THYROID_1', 'EPILEPSY_-77', 'EPILEPSY_0', 'EPILEPSY_1', 'FETAL_ANOMALIES_-77', 'FETAL_ANOMALIES_-88', 'FETAL_ANOMALIES_0', 'BABY_SEX_1', 'BABY_SEX_2', 'site_AMANHI-Bangladesh', 'site_AMANHI-Pakistan', 'site_AMANHI-Pemba', 'site_GAPPS-Bangladesh', 'site_GAPPS-Zambia', 'Sex_1.0', 'Sex_2.0', 'Sex_nan', 'MainHap_A', 'MainHap_D', 'MainHap_E', 'MainHap_F', 'MainHap_G', 'MainHap_H', 'MainHap_HV', 'MainHap_J', 'MainHap_K', 'MainHap_L0', 'MainHap_L1', 'MainHap_L2', 'MainHap_L3', 'MainHap_L4', 'MainHap_M', 'MainHap_N', 'MainHap_R', 'MainHap_T', 'MainHap_U', 'MainHap_W', 'MainHap_X', 'MainHap_Z', 'SubHap_A+', 'SubHap_A1', 'SubHap_A2', 'SubHap_A6', 'SubHap_D4', 'SubHap_D5', 'SubHap_E1', 'SubHap_F1', 'SubHap_G1', 'SubHap_G2', 'SubHap_H1', 'SubHap_H2', 'SubHap_H4', 'SubHap_H5', 'SubHap_H6', 'SubHap_H7', 'SubHap_HV1', 'SubHap_HV2', 'SubHap_HV6', 'SubHap_J1', 'SubHap_J2', 'SubHap_K1', 'SubHap_K2', 'SubHap_L0a', 'SubHap_L0d', 'SubHap_L0f', 'SubHap_L1b', 'SubHap_L1c', 'SubHap_L2a', 'SubHap_L2b', 'SubHap_L2c', 'SubHap_L2d', 'SubHap_L3a', 'SubHap_L3b', 'SubHap_L3d', 'SubHap_L3e', 'SubHap_L3f', 'SubHap_L3h', 'SubHap_L3i', 'SubHap_L3k', 'SubHap_L3x', 'SubHap_L4b', 'SubHap_M1', 'SubHap_M2', 'SubHap_M3', 'SubHap_M4', 'SubHap_M5', 'SubHap_M6', 'SubHap_M7', 'SubHap_M8', 'SubHap_M9', 'SubHap_N1', 'SubHap_N2', 'SubHap_N5', 'SubHap_N8', 'SubHap_R', 'SubHap_R0', 'SubHap_R2', 'SubHap_R3', 'SubHap_R5', 'SubHap_R6', 'SubHap_R7', 'SubHap_R8', 'SubHap_T1', 'SubHap_T2', 'SubHap_U1', 'SubHap_U2', 'SubHap_U3', 'SubHap_U4', 'SubHap_U5', 'SubHap_U6', 'SubHap_U7', 'SubHap_U8', 'SubHap_U9', 'SubHap_W', 'SubHap_W3', 'SubHap_W4', 'SubHap_W6', 'SubHap_X2', 'SubHap_Z+', 'SubHap_Z3', 'SubHap_Z7']
data_encoded[cols]=data_encoded[cols].astype(int)



# Select continuous columns
continuous_columns = ['PW_AGE', 'PW_EDUCATION', 'ROOMS_HOUSE', 'BIRTH_WEIGHT', 'MAT_HEIGHT','GAGEBRTH_IN']

# Scale data to range [0, 1]
scaler = MinMaxScaler()
data_encoded[continuous_columns] = scaler.fit_transform(data_encoded[continuous_columns])

print(data_encoded)




data_encoded[["PTB"]]=data_encoded[["PTB"]].astype(int)



# Features (X) and target (y)
X = data_encoded.drop(["PTB","GAGEBRTH_IN"], axis=1)
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









# Explain the model predictions using KernelExplainer
explainer = shap.KernelExplainer(final_model, X_train)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test, nsamples=100)

# Plot feature importance
shap.summary_plot(shap_values, X_test)
plt.savefig("shap_summary_plot.sub.png", bbox_inches="tight")



shap_values = shap_values.squeeze() 

#For individual predictions, use shap.force_plot:
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
plt.savefig("shap_force_plot.sub.png", bbox_inches="tight")



#Understand interactions between features and predictions:
shap.dependence_plot("HH_ELECTRICITY_0", shap_values, X_test)
plt.savefig("shap_dependence_plot.sub.png", bbox_inches="tight")









#Cross-Validation












