import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, GroupShuffleSplit

from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Set to display all columns
pd.set_option('display.max_columns', None)
# Set to display all rows
pd.set_option('display.max_rows', None)

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




# Define features
categorical_columns = sys.argv[1].split(',')
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')

X = df[categorical_columns + continuous_columns+ binary_columns]
y = df['GAGEBRTH']  


# Train-test split: unseen-site if possible, else random
# -----------------------------
if "site" in df.columns and df["site"].nunique() >= 2:
    # Site-aware split: test set contains sites not seen during training
    groups_all = df["site"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
else:
    # Fallback: standard random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
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
elasticnet = ElasticNetCV(
    alphas=np.logspace(-3, 1, 20),   # 0.001 → 10
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5,
    random_state=42
)

elasticnet.fit(X_train_preprocessed, y_train)

evaluate_model_regression(elasticnet, X_test_preprocessed, y_test, "ElasticNet Regression")

# Plot ElasticNet feature importance
elasticnet_importance = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Coefficient': elasticnet.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)
plot_feat(elasticnet_importance, "ElasticNet")
print(elasticnet_importance)

# -----------------------------
# SHAP analysis for ElasticNet (linear model)
# -----------------------------
# Use LinearExplainer (fast, exact for linear models)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.LinearExplainer(elasticnet, X_train_preprocessed)
shap_values = explainer.shap_values(X_test_preprocessed)  # (n_samples, n_features) for regression

shap_values = np.asarray(shap_values)
assert shap_values.ndim == 2 and shap_values.shape[1] == len(feature_names), \
    f"Expected (N,F) SHAP matrix, got {shap_values.shape}"

# Global importance: mean |SHAP|
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs_shap)[::-1]
top_k = min(30, len(feature_names))
top_idx = order[:top_k]

top_feature_names = feature_names[top_idx]
top_shap_values = shap_values[:, top_idx]

# Summary plot for top-K features
plt.figure(figsize=(10, 6))
shap.summary_plot(
    top_shap_values,
    X_test_preprocessed[:, top_idx],
    feature_names=top_feature_names,
    show=False,
    max_display=top_k,
)
plt.tight_layout()
plt.savefig("shap_summary_top30.ElasticNet.GA.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nTop 20 features by mean |SHAP| (ElasticNet):")
for name, val in zip(top_feature_names[:20], mean_abs_shap[top_idx][:20]):
    print(f"{name}: {val:.6f}")

# Optional: SHAP dependence plots for the top numeric features
num_prefixes = ("num__", "bin__")
top_numeric_feats = [f for f in top_feature_names if f.startswith(num_prefixes)]

for fname in top_numeric_feats[:10]:  # limit to first 10 numeric features
    shap.dependence_plot(
        fname,
        shap_values,
        X_test_preprocessed,
        feature_names=feature_names,
        show=False,
    )
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in fname)
    plt.tight_layout()
    plt.savefig(f"shap_dependence.ElasticNet.GA.{safe}.png", dpi=300, bbox_inches="tight")
    plt.close()

