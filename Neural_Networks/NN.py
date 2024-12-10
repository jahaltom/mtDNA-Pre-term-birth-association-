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
md = pd.read_csv("Metadata.M.Final.tsv",sep='\t')
md.set_index('Sample_ID', inplace=True)
md=md.astype(str)

md=md[["PW_AGE",	"PW_EDUCATION",	"TYP_HOUSE",	"HH_ELECTRICITY",	"FUEL_FOR_COOK",	"DRINKING_SOURCE",	"TOILET",	"WEALTH_INDEX",	"SNIFF_TOBA",	"SNIFF_FREQ",	"SMOKE_HIST",	"SMOK_FREQ",	"PASSIVE_SMOK",	"ALCOHOL",	
    "ALCOHOL_FREQ",	"CHRON_HTN",	"DIABETES",	"TB",	"THYROID",	"EPILEPSY",	"MAT_HEIGHT",	"BIRTH_WEIGHT",	"GAGEBRTH_IN", 	"PTB",	"Sex",	"MainHap",	"PC1_All",	"PC2_All",	"PC3_All",	"PC4_All",	"PC5_All",	"PC6_All",	"PC7_All",	
    "PC8_All",	"PC9_All",	"PC10_All",	"PC11_All",	"PC12_All",	"PC13_All",	"PC14_All",	"PC15_All",	"PC16_All",	"PC17_All",	"PC18_All",	"PC19_All",	"PC20_All",	"C1_All",	"C2_All",	"C3_All",	"C4_All",	"C5_All"]]





#Catigorical 
# One-hot encode haplogroups and gender
data_encoded = pd.get_dummies(md, columns=['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'ALCOHOL', 'CHRON_HTN',
       'DIABETES', 'TB', 'THYROID', 'EPILEPSY',
        'Sex', 'MainHap',"SNIFF_TOBA","SMOKE_HIST"])

print(data_encoded)

cols=[ 'TYP_HOUSE_-77.0', 'TYP_HOUSE_-88.0', 'TYP_HOUSE_1.0', 'TYP_HOUSE_2.0', 'HH_ELECTRICITY_-77.0', 'HH_ELECTRICITY_-88.0', 'HH_ELECTRICITY_0.0', 'HH_ELECTRICITY_1.0', 'FUEL_FOR_COOK_-77.0', 'FUEL_FOR_COOK_-88.0', 'FUEL_FOR_COOK_1.0', 'FUEL_FOR_COOK_2.0', 'FUEL_FOR_COOK_3.0', 'FUEL_FOR_COOK_4.0', 'FUEL_FOR_COOK_5.0', 'DRINKING_SOURCE_-77.0', 'DRINKING_SOURCE_-88.0', 'DRINKING_SOURCE_1.0', 'DRINKING_SOURCE_2.0', 'DRINKING_SOURCE_3.0', 'DRINKING_SOURCE_4.0', 'TOILET_-77.0', 'TOILET_-88.0', 'TOILET_1.0', 'TOILET_2.0', 'TOILET_3.0', 'TOILET_4.0', 'WEALTH_INDEX_-77.0', 'WEALTH_INDEX_-88.0', 'WEALTH_INDEX_1.0', 'WEALTH_INDEX_2.0', 'WEALTH_INDEX_3.0', 'WEALTH_INDEX_4.0', 'WEALTH_INDEX_5.0', 'PASSIVE_SMOK_-77.0', 'PASSIVE_SMOK_0.0', 'PASSIVE_SMOK_1.0', 'ALCOHOL_-77.0', 'ALCOHOL_1.0', 'ALCOHOL_2.0', 'ALCOHOL_4.0', 'CHRON_HTN_-77.0', 'CHRON_HTN_-88.0', 'CHRON_HTN_0.0', 'CHRON_HTN_1.0', 'DIABETES_-77.0', 'DIABETES_-88.0', 'DIABETES_0.0', 'DIABETES_1.0', 'TB_-77.0', 'TB_-88.0', 'TB_0.0', 'TB_1.0', 'THYROID_-77.0', 'THYROID_-88.0', 'THYROID_0.0', 'THYROID_1.0', 'EPILEPSY_-77.0', 'EPILEPSY_-88.0', 'EPILEPSY_0.0', 'EPILEPSY_1.0', 'Sex_2.0', 'Sex_nan', 'MainHap_A', 'MainHap_B', 'MainHap_C', 'MainHap_D', 'MainHap_E', 'MainHap_F', 'MainHap_G', 'MainHap_H', 'MainHap_HV', 'MainHap_I', 'MainHap_J', 'MainHap_K', 'MainHap_L0', 'MainHap_L1', 'MainHap_L2', 'MainHap_L3', 'MainHap_L4', 'MainHap_L5', 'MainHap_M', 'MainHap_N', 'MainHap_R', 'MainHap_T', 'MainHap_U', 'MainHap_W', 'MainHap_X', 'MainHap_Z', 'SNIFF_TOBA_-77.0', 'SNIFF_TOBA_-88.0', 'SNIFF_TOBA_1.0', 'SNIFF_TOBA_2.0', 'SNIFF_TOBA_3.0', 'SNIFF_TOBA_4.0', 'SMOKE_HIST_-77.0', 'SMOKE_HIST_-88.0', 'SMOKE_HIST_1.0', 'SMOKE_HIST_2.0', 'SMOKE_HIST_3.0', 'SMOKE_HIST_4.0']
data_encoded[cols]=data_encoded[cols].astype(int)



# Select continuous columns
continuous_columns = ['PW_AGE','ALCOHOL_FREQ',"SNIFF_FREQ", "SMOK_FREQ",'PW_EDUCATION', 'BIRTH_WEIGHT', 'MAT_HEIGHT','GAGEBRTH_IN',"PC1_All",	"PC2_All",	"PC3_All",	"PC4_All",	"PC5_All",	"PC6_All",	"PC7_All",  "PC8_All",	"PC9_All",	"PC10_All",	"PC11_All",	"PC12_All",	"PC13_All",	"PC14_All",	"PC15_All",	"PC16_All",	"PC17_All",	"PC18_All",	"PC19_All",	"PC20_All",	"C1_All",	"C2_All",	"C3_All",	"C4_All",	"C5_All"]

# Scale data to range [0, 1]
scaler = MinMaxScaler()
data_encoded[continuous_columns] = scaler.fit_transform(data_encoded[continuous_columns])

print(data_encoded)


data_encoded = data_encoded[data_encoded["PTB"] != "nan"]

data_encoded[["PTB"]]= data_encoded[["PTB"]].astype(float).astype(int)



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












