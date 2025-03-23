import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import sys

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
    plt.savefig(model_name+"_TopFeature.LinReg.GA.png", bbox_inches="tight")
    plt.clf()


# Helper function for evaluation
def evaluate_model_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R-squared: {r2_score(y_test, y_pred):.4f}")



# Load the dataset
df = pd.read_csv("Metadata.Final.tsv", sep='\t')

df = df[sys.argv[1].split(',') + sys.argv[2].split(',') + ["GAGEBRTH"]]
df = df[~df.isin([-88, -77]).any(axis=1)]  # Remove rows with invalid entries (-88, -77)
df = df[df['MainHap'].map(df['MainHap'].value_counts()) >= 25]

# Define features
categorical_columns = sys.argv[1].split(',')
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')

X = df[categorical_columns + continuous_columns]
y = df['GAGEBRTH']  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_columns),
        ('bin', binary_transformer, binary_columns),
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




# Lasso Regression for continuous prediction
lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
lasso.fit(X_train_preprocessed, y_train)  # y_train should be continuous, not binary
evaluate_model_regression(lasso, X_test_preprocessed, y_test, "Lasso Regression")

# Lasso Feature Importance
lasso_importance = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Coefficient': lasso.coef_
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


