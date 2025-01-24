import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

##For LASSO, Ridge and Elactic
def plot_feat(coefMat, model_name):
    # Plot top 10 significant features
    top_features = coefMat#.head(10)
    # Rename 'cat__MainHap' to 'Haplogroup'
    top_features['Feature'] = top_features['Feature'].str.replace('^cat__MainHap', 'Haplogroup', regex=True)###########################################################
    # Remove 'cat__' and 'num__' prefixes
    top_features['Feature'] = top_features['Feature'].str.replace('^cat__|^num__', '', regex=True)
    plt.figure(figsize=(12, 6))
    # Scatter plot with color coding for positive/negative coefficients
    colors = np.where(top_features['Coefficient'] > 0, 'blue', 'red')
    plt.scatter(top_features['Feature'], top_features['Coefficient'], color=colors, s=100, edgecolor='black')
    # Add labels and title
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Add a horizontal line at 0
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate feature names
    plt.ylabel('Coefficient')
    plt.title('Top Significant Features by Coefficients')
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.tight_layout()
    plt.show()
    plt.savefig(model_name+"_TopFeature.PTB.png", bbox_inches="tight")
    plt.clf()


# Helper function for evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    if y_prob is not None:
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC Score for {model_name}: {auc_score:.4f}")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.show()



# Load the dataset
df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')
# Clean the dataset
df = df.dropna(subset=["PTB", "GAGEBRTH"])  # Drop rows with missing targets
df['GAGEBRTH'] = pd.to_numeric(df['GAGEBRTH'], errors='coerce')  # Ensure GAGEBRTH is numeric
df=df[[  'DIABETES','PW_AGE', 'MAT_HEIGHT',"PC1", "PC2", "PC3", "PC4", "PC5","MainHap","PTB", "GAGEBRTH"]]
df = df[~df.isin([-88, -77]).any(axis=1)]  # Remove rows with invalid entries (-88, -77)
df = df[df['MainHap'].map(df['MainHap'].value_counts()) >= 25]

# Define features
categorical_columns = [ 'DIABETES', 'MainHap']
                       
continuous_columns = ['PW_AGE', 'MAT_HEIGHT',"PC1", "PC2", "PC3", "PC4", "PC5"]

X = df[categorical_columns + continuous_columns]
y = df['PTB']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Step 1: Preprocess the training data
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










# Step 2a: Penalized Logistic Regression (Lasso)
lasso = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, max_iter=5000, class_weight='balanced', random_state=42)
lasso.fit(X_train_balanced, y_train_balanced)

evaluate_model(lasso, X_test_preprocessed, y_test, "Lasso Regression")

# Extract non-zero coefficients
lasso_coefs = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Coefficient': lasso.coef_[0]
}).query("Coefficient != 0").sort_values(by='Coefficient', key=abs, ascending=False)
print("\nLasso Significant Features:")
print(lasso_coefs)
plot_feat(lasso_coefs,"LASSO")







# Step 2a: Ridge Regression (Ridge)
ridge = LogisticRegressionCV(penalty='l2', solver='saga', cv=5, max_iter=5000, class_weight='balanced', random_state=42)
ridge.fit(X_train_balanced, y_train_balanced)

evaluate_model(ridge, X_test_preprocessed, y_test, "Ridge Regression")

# Extract non-zero coefficients
ridge_coefs = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Coefficient': ridge.coef_[0]
}).query("Coefficient != 0").sort_values(by='Coefficient', key=abs, ascending=False)
print("\nRidge Significant Features:")
print(ridge_coefs)
plot_feat(ridge_coefs,"Ridge")







# # Step 2d: ElasticNet Regression
# elasticnet = ElasticNetCV(alphas=[0.01], l1_ratio=0.5, cv=5, random_state=42)  # l1_ratio=0.5 for balanced mix
# elasticnet.fit(X_train_preprocessed, y_train)

# evaluate_model_regression(elasticnet, X_test_preprocessed, y_test, "ElasticNet Regression")

# elasticnet_coefs = pd.DataFrame({
#     'Feature': preprocessor.get_feature_names_out(),
#     'Coefficient': elasticnet.coef_
# }).query("Coefficient != 0").sort_values(by='Coefficient', key=abs, ascending=False)
# print("\nElasticNet Significant Features:")
# print(elasticnet_coefs)
# plot_feat(elasticnet_coefs,"Elastic")






# Step 2b: Random Forest for Feature Importance
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_cv = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc')
rf_cv.fit(X_train_balanced, y_train_balanced)

print("\nBest Parameters for Random Forest:", rf_cv.best_params_)
evaluate_model(rf_cv.best_estimator_, X_test_preprocessed, y_test, "Random Forest")

rf_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': rf_cv.best_estimator_.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nRandom Forest Feature Importances:")
print(rf_importances.head(10))

explainer = shap.TreeExplainer(rf_cv.best_estimator_)
shap_values = explainer.shap_values(X_train_balanced)
# Extract SHAP values for the positive class (index 1) for binary classification
shap_values_positive_class = shap_values[:, :, 1]  # shape will be (11734, 29)

# Plot the SHAP summary plot for the positive class
shap.summary_plot(shap_values_positive_class, X_train_balanced.values, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shap.RF.PTB.png", bbox_inches="tight")
plt.clf()

shap.dependence_plot('num__PW_AGE', shap_values_positive_class, X_train_balanced.values, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shapDep.RF.PTB.png", bbox_inches="tight")
plt.clf()









# Step 2c: Gradient Boosting for Feature Importance
gb = GradientBoostingClassifier(random_state=42)
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
gb_cv = GridSearchCV(gb, param_grid_gb, cv=5, scoring='roc_auc')
gb_cv.fit(X_train_balanced, y_train_balanced)

print("\nBest Parameters for Gradient Boosting:", gb_cv.best_params_)
evaluate_model(gb_cv.best_estimator_, X_test_preprocessed, y_test, "Gradient Boosting")

gb_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': gb_cv.best_estimator_.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nGradient Boosting Feature Importances:")
print(gb_importances.head(10))

explainer = shap.TreeExplainer(gb_cv.best_estimator_)
shap_values = explainer.shap_values(X_train_balanced)
# Extract SHAP values for the positive class (index 1) for binary classification


# Plot the SHAP summary plot for the positive class
shap.summary_plot(shap_values, X_train_balanced.values, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shap.GB.PTB.png", bbox_inches="tight")
plt.clf()

shap.dependence_plot('num__PW_AGE', shap_values, X_train_balanced.values, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shapDep.GB.PTB.png", bbox_inches="tight")
plt.clf()


# shap_interaction_values = explainer.shap_interaction_values(X_train_balanced)
# shap.summary_plot(shap_interaction_values, X_train_balanced.values, feature_names=preprocessor.get_feature_names_out())
# plt.savefig("shapInt.GB.PTB.png", bbox_inches="tight")
# plt.clf()














# Step 4: Neural Network for PTB
from sklearn.inspection import permutation_importance

# Step 4: Neural Network for PTB
nn = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
nn.fit(X_train_balanced, y_train_balanced)
evaluate_model(nn, X_test_preprocessed, y_test, "Neural Network")

# Step 4a: Permutation Importance for Neural Network
perm_importance = permutation_importance(nn, X_test_preprocessed, y_test, n_repeats=10, random_state=42)
nn_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', key=abs, ascending=False)

print("\nNeural Network Feature Importances (Permutation):")
print(nn_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=nn_importances.head(10), x='Importance', y='Feature')
plt.title("Neural Network Feature Importances (Top 10)")
plt.tight_layout()
plt.savefig("NN_FeatureImportance.PTB.png", bbox_inches="tight")
plt.clf()


# Step 4b: SHAP Values for Neural Network                                         DeepExplainer
explainer_nn = shap.Explainer(nn.predict_proba, X_train_balanced) #explainer_nn = shap.KernelExplainer(nn.predict_proba, X_train_balanced)
shap_values_nn = explainer_nn.shap_values(X_train_balanced)


# Extract SHAP values for the positive class (index 1) for binary classification
#shap_values_positive_class = shap_values_nn[:, :, 1]  # shape will be (11734, 29)

# Plot the SHAP summary plot for the positive class
shap.summary_plot(shap_values, X_train_balanced.values, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shap.NN.PTB.png", bbox_inches="tight")
plt.clf()

shap.dependence_plot('num__PW_AGE', shap_values, X_train_balanced.values, feature_names=preprocessor.get_feature_names_out())
plt.savefig("shapDep.NN.PTB.png", bbox_inches="tight")
plt.clf()


