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

##For LASSO, Ridge and ElasticNet
def plot_feat(coefMat, model_name):
    top_features = coefMat.copy()  # Adjusted to show top features
    top_features['Feature'] = top_features['Feature'].str.replace('^cat__MainHap', 'Haplogroup', regex=True)
    top_features['Feature'] = top_features['Feature'].str.replace('^cat__|^num__', '', regex=True)
    plt.figure(figsize=(12, 6))
    colors = np.where(top_features['Coefficient'] > 0, 'blue', 'red')
    plt.scatter(top_features['Feature'], top_features['Coefficient'], color=colors, s=100, edgecolor='black')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel('Coefficient')
    plt.title('Top Significant Features by Coefficients')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.savefig(model_name+"_TopFeature.GA.png", bbox_inches="tight")
    plt.clf()


# Helper function for evaluation
def evaluate_model_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R-squared: {r2_score(y_test, y_pred):.4f}")



# Load the dataset
df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')
df = df.dropna(subset=["GAGEBRTH"])  # Drop rows with missing targets
df['GAGEBRTH'] = pd.to_numeric(df['GAGEBRTH'], errors='coerce')  # Ensure GAGEBRTH is numeric
df = df[[  'DIABETES', 'PW_AGE', 'MAT_HEIGHT', "PC1", "PC2", "PC3", "PC4", "PC5", "MainHap", "GAGEBRTH"]]
df = df[~df.isin([-88, -77]).any(axis=1)]  # Remove rows with invalid entries (-88, -77)
df = df[df['MainHap'].map(df['MainHap'].value_counts()) >= 25]

# Define features
categorical_columns = ['DIABETES', 'MainHap']
continuous_columns = ['PW_AGE', 'MAT_HEIGHT', "PC1", "PC2", "PC3", "PC4", "PC5"]

X = df[categorical_columns + continuous_columns]
y = df['GAGEBRTH']  # Predict GAGEBRTH instead of PTB

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




# Step 1: Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_preprocessed, y_train)

evaluate_model_regression(ridge, X_test_preprocessed, y_test, "Ridge Regression")

# Plot Ridge feature importance
ridge_importance = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Coefficient': ridge.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)
plot_feat(ridge_importance, "Ridge")
print(ridge_importance)




# Step 2: Lasso Regression
lasso = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, max_iter=5000, random_state=42)
lasso.fit(X_train_preprocessed, y_train)
evaluate_model_regression(lasso, X_test_preprocessed, y_test, "Lasso Regression")

# Plot Lasso feature importance
lasso_importance = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Coefficient': lasso.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)
plot_feat(lasso_importance, "Lasso")
print(lasso_importance)



# Step 3: ElasticNet Regression
elasticnet = ElasticNetCV(alphas=[0.01], l1_ratio=0.5, cv=5, random_state=42)
elasticnet.fit(X_train_preprocessed, y_train)

evaluate_model_regression(elasticnet, X_test_preprocessed, y_test, "ElasticNet Regression")

# Plot ElasticNet feature importance
elasticnet_importance = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Coefficient': elasticnet.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)
plot_feat(elasticnet_importance, "ElasticNet")
print(elasticnet_importance)





# Step 4: Random Forest Regression for Feature Importance
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

# Random Forest Feature Importances
rf_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': rf_cv.best_estimator_.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest Feature Importances:")
print(rf_importances.head(10))

# SHAP for Random Forest
explainer_rf = shap.TreeExplainer(rf_cv.best_estimator_)
shap_values_rf = explainer_rf.shap_values(X_train_preprocessed)

# SHAP Summary Plot for Random Forest
shap.summary_plot(shap_values_rf, X_train_preprocessed, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shap.RF.png", bbox_inches="tight")
plt.clf()

# Step 5: Gradient Boosting for Feature Importance
gb = GradientBoostingRegressor(random_state=42)
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
gb_cv = GridSearchCV(gb, param_grid_gb, cv=5)
gb_cv.fit(X_train_preprocessed, y_train)

print("\nBest Parameters for Gradient Boosting:", gb_cv.best_params_)
evaluate_model_regression(gb_cv.best_estimator_, X_test_preprocessed, y_test, "Gradient Boosting")

# Gradient Boosting Feature Importances
gb_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': gb_cv.best_estimator_.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nGradient Boosting Feature Importances:")
print(gb_importances.head(10))

# SHAP for Gradient Boosting
explainer_gb = shap.TreeExplainer(gb_cv.best_estimator_)
shap_values_gb = explainer_gb.shap_values(X_train_preprocessed)

# SHAP Summary Plot for Gradient Boosting
shap.summary_plot(shap_values_gb, X_train_preprocessed, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shap.GB.png", bbox_inches="tight")
plt.clf()







# Step 6: Neural Network for GAGEBRTH (NN)
from sklearn.neural_network import MLPRegressor

# Hyperparameter tuning for Neural Network using GridSearchCV
param_grid_nn = {
    'hidden_layer_sizes': [(50, 50), (100,), (100, 100)],
    'max_iter': [500, 1000],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive']
}

nn = MLPRegressor(random_state=42)
grid_search_nn = GridSearchCV(nn, param_grid_nn, cv=5, n_jobs=-1)
grid_search_nn.fit(X_train_preprocessed, y_train)

print(f"\nBest Parameters for Neural Network: {grid_search_nn.best_params_}")

# Train best NN model from grid search
best_nn = grid_search_nn.best_estimator_
best_nn.fit(X_train_preprocessed, y_train)
evaluate_model_regression(best_nn, X_test_preprocessed, y_test, "Neural Network")

# SHAP analysis for NN
explainer = shap.KernelExplainer(best_nn.predict, X_train_preprocessed[:100])
shap_values = explainer.shap_values(X_test_preprocessed[:100])

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_test_preprocessed[:100], feature_names=preprocessor.get_feature_names_out())
plt.savefig("SHAP_NN.png", bbox_inches="tight")
plt.clf()

# Feature Importances for NN (using permutation importance)
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(best_nn, X_test_preprocessed, y_test, n_repeats=10, random_state=42)
nn_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', key=abs, ascending=False)

print("\nNeural Network Feature Importances (Permutation):")
print(nn_importances)

# Plot feature importances for Neural Network
plt.figure(figsize=(10, 6))
sns.barplot(data=nn_importances.head(10), x='Importance', y='Feature')
plt.title("Neural Network Feature Importances (Top 10)")
plt.tight_layout()
plt.savefig("NN_FeatureImportance.png", bbox_inches="tight")
plt.clf()
