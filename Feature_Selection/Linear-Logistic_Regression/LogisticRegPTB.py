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
import sys

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

df = df[sys.argv[1].split(',') + sys.argv[2].split(',') + ["PTB"]]
df = df[~df.isin([-88, -77]).any(axis=1)]  # Remove rows with invalid entries (-88, -77)
df = df[df['MainHap'].map(df['MainHap'].value_counts()) >= 25]

# Define features
categorical_columns = sys.argv[1].split(',')
continuous_columns = sys.argv[2].split(',')

X = df[categorical_columns + continuous_columns]
y = df['PTB']  

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








