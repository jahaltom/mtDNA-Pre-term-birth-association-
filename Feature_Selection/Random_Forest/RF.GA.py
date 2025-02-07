import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegressionCV, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns



# Helper function for evaluation
def evaluate_model_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R-squared: {r2_score(y_test, y_pred):.4f}")



# Load the dataset
df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')

#df['GAGEBRTH'] = pd.to_numeric(df['GAGEBRTH'], errors='coerce')  # Ensure GAGEBRTH is numeric
df = df[['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
                       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'CHRON_HTN',
                       'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
                       "SNIFF_TOBA","GAGEBRTH",'PW_AGE', 'MAT_HEIGHT', "PC1", "PC2", "PC3", "PC4", "PC5"]]
df = df[~df.isin([-88, -77]).any(axis=1)]  # Remove rows with invalid entries (-88, -77)
df = df[df['MainHap'].map(df['MainHap'].value_counts()) >= 25]

# Define features
categorical_columns = ['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
                       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'CHRON_HTN',
                       'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
                       "SNIFF_TOBA"]
continuous_columns = ['PW_AGE', 'MAT_HEIGHT', "PC1", "PC2", "PC3", "PC4", "PC5"]

X = df[categorical_columns + continuous_columns]
y = df['GAGEBRTH']  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)







# Random Forest Regression with hyperparamiter tuning. 
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_cv = GridSearchCV(rf, param_grid_rf, cv=5)
rf_cv.fit(X_train_preprocessed, y_train)

print("\nBest Parameters for Random Forest:", rf_cv.best_params_)
evaluate_model_regression(rf_cv.best_estimator_, X_test_preprocessed, y_test, "Random Forest")

# Random Forest Feature Importances. Reports top 10 feature importances.
rf_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': rf_cv.best_estimator_.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nRandom Forest Feature Importances:")
print(rf_importances.head(10))



#Recursive Feature Elimination (RFE) for feature selection
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestRegressor(n_estimators=100)
rfe = RFE(model, n_features_to_select=20)
rfe.fit(X_train_preprocessed, y_train)
selected_features = preprocessor.get_feature_names_out()[rfe.support_]
print("\nRandom Forest RFE Feature Importances:")
print(selected_features)


















# Compute SHAP  values
explainer = shap.TreeExplainer(rf_cv.best_estimator_)  # Use the trained Random Forest
shap_values_rf = explainer.shap_values(X_train_preprocessed)

# SHAP Summary Plot
shap.summary_plot(shap_values_rf, X_train_preprocessed, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shapSummary.RF.GA.png", bbox_inches="tight")
plt.clf()



########################################################This is for interactions ##############################################
# Compute SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X_train_preprocessed)



#Interaction plot
plt.figure(figsize=(20, 20))
shap.summary_plot(shap_interaction_values, X_train_preprocessed,feature_names=preprocessor.get_feature_names_out())
plt.savefig("shapSummary.Interaction.RF.GA.png", bbox_inches="tight")
plt.clf()

#Finds top interactions. Reports top 10
feature_names = preprocessor.get_feature_names_out()  
interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)  # Take mean absolute interaction values
interaction_df = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names) # Convert to DataFrame
np.fill_diagonal(interaction_df.values, 0) # Mask diagonal (self-interactions)
top_interactions = interaction_df.unstack().sort_values(ascending=False).reset_index()
top_interactions.columns = ["Feature_1", "Feature_2", "Interaction_Strength"]
cleaned_interactions = top_interactions.iloc[::2].reset_index(drop=True)  #Drop repeats
print("Top 10 Feature Interactions:")
print(cleaned_interactions.head(10))   #Print top 10
 

#Find significant interactions based on filtering  ( Interaction_Strength >  mean + 1 std deviation)
mean_interaction = cleaned_interactions["Interaction_Strength"].mean()
std_interaction = cleaned_interactions["Interaction_Strength"].std()
threshold = mean_interaction + std_interaction  # Define a threshold: mean + 1 std deviation
significant_interactions = cleaned_interactions[cleaned_interactions["Interaction_Strength"] > threshold]  # Select significant interactions. Interactions above threshold
print("Significant Feature Interactions:")
print(significant_interactions)




#PDP plots for top 5 interactions
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

top_pairs = cleaned_interactions.head(5)[["Feature_1", "Feature_2"]].values  # Select top 5 interactions from SHAP results

feature_indices = [ # Convert feature names to index for PDP
    (np.where(feature_names == pair[0])[0][0], np.where(feature_names == pair[1])[0][0])
    for pair in top_pairs
]
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    rf_cv.best_estimator_,
    X_train_preprocessed,
    features=feature_indices,
    feature_names=feature_names,
    grid_resolution=20,
    ax=ax
)
plt.savefig("PDP_Top5.png", bbox_inches="tight")
plt.clf()





#Total interaction heatmap
plt.figure(figsize=(40, 40))
sns.heatmap(interaction_df, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Interaction Heatmap (SHAP)")
plt.savefig("FeatureInteractionHeatmap.png", bbox_inches="tight")
plt.clf()














###########################################################This is for finding non-linear features predicting GA#############################
from sklearn.inspection import PartialDependenceDisplay



# Creating Partial Dependence Plots for all relevant features (selected_features from  Recursive Feature Elimination (RFE)) 
features = list(range(len(selected_features))) 
fig, ax = plt.subplots(figsize=(25, 25))
display = PartialDependenceDisplay.from_estimator(
    rf_cv.best_estimator_,
    X_train_preprocessed,
    features=features,
    feature_names=selected_features,
    grid_resolution=20,
    n_cols=5,
    ax=ax
)
plt.subplots_adjust(top=0.9)
plt.savefig("PDP_RFE.png", bbox_inches="tight")
plt.clf()




# Compute variance of SHAP values for each feature
shap_variances = np.var(shap_values_rf, axis=0)
shap_variance_df = pd.DataFrame(shap_variances, index=preprocessor.get_feature_names_out(), columns=["SHAP Variance"])
print("\nSHAP Variance by Feature:")
print(shap_variance_df.sort_values(by="SHAP Variance", ascending=False))


# SHAP Dependence Plots to detect non-linear relationships and interactions.Plots for all relevant features (selected_features from  Recursive Feature Elimination (RFE)) 
for name in selected_features:
    shap.dependence_plot(name, shap_values_rf, X_train_preprocessed, feature_names=preprocessor.get_feature_names_out(), show=False)
    plt.savefig("Dep_Plot."+name+".png", bbox_inches="tight")
    plt.clf()
    
    
    
    
    
    
    
    
    
