import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, GroupShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Set to display all columns
pd.set_option('display.max_columns', None)
# Set to display all rows
pd.set_option('display.max_rows', None)

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
    plt.savefig(model_name+"_TopFeature.LogReg.PTB.png", bbox_inches="tight")
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
        plt.savefig("ROC_AUC_plot.LogReg.PTB.png")
        plt.clf()



# Load the dataset
df = pd.read_csv("Metadata.Final.tsv", sep='\t')


# Define features
categorical_columns = [c for c in sys.argv[1].split(',') if c != "site"]
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')

X = df[categorical_columns + continuous_columns + binary_columns]
y = df["PTB"]

# -----------------------------
# Site-aware train/test split
# -----------------------------
if "site" in df.columns:
    n_sites = df["site"].nunique()
else:
    n_sites = 0

if ("site" in df.columns) and (n_sites >= 3):
    # With 3+ sites, a true unseen-site test is meaningful
    groups_all = df["site"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

elif ("site" in df.columns) and (n_sites == 2):
    # With only 2 sites, we just do a stratified row-wise split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

else:
    # No / insufficient site info → standard stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

# -----------------------------
# Preprocessing pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)

# Preprocess the data (no SMOTE, keep as sparse/dense as returned)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed  = preprocessor.transform(X_test)










# Step 2a: Penalized Logistic Regression (Lasso)
lasso = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, max_iter=5000, class_weight='balanced', random_state=42)
lasso.fit(X_train_preprocessed, y_train)

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
ridge.fit(X_train_preprocessed, y_train)

evaluate_model(ridge, X_test_preprocessed, y_test, "Ridge Regression")

# Extract non-zero coefficients
ridge_coefs = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Coefficient': ridge.coef_[0]
}).query("Coefficient != 0").sort_values(by='Coefficient', key=abs, ascending=False)
print("\nRidge Significant Features:")
print(ridge_coefs)
plot_feat(ridge_coefs,"Ridge")








