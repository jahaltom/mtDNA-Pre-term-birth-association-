import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor   # <-- RF instead of GB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def evaluate_model_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R-squared: {r2_score(y_test, y_pred):.4f}")

categorical_columns=['FUEL_FOR_COOK','site']
continuous_columns  = ['PW_AGE','PW_EDUCATION','MAT_HEIGHT','MAT_WEIGHT','BMI','TOILET','WEALTH_INDEX','DRINKING_SOURCE']
binary_columns=['BABY_SEX','CHRON_HTN','DIABETES','HH_ELECTRICITY','TB','THYROID','TYP_HOUSE']

# Columns


df = pd.read_csv("Metadata.Final.tsv", sep="\t")
required = categorical_columns + continuous_columns + binary_columns + ["GAGEBRTH"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input: {missing}")

X = df[categorical_columns + continuous_columns + binary_columns]
y = df["GAGEBRTH"]



# Dense OHE to support tree SHAP & downstream ops
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # or sparse=False on older sklearn
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", ohe, categorical_columns),
    ]
)

# --- Random Forest model (replaces GradientBoostingRegressor) ---
rf = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("pre", preprocessor),
    ("rf", rf),
])

# RF hyperparameter grid
param_grid_rf = {
    "rf__n_estimators": [300, 600, 900],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_leaf": [1, 2, 5],
    "rf__max_features": ["sqrt", 0.5],
}





from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit

if "site" in df.columns and df["site"].nunique() >= 2:
    # Use sites as groups so test contains (mostly) unseen sites
    groups_all = df["site"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups_all.iloc[train_idx]
    # -----------------------------
    # Inner CV: try GroupKFold, fall back if only 1 train site
    # -----------------------------
    n_groups_train = groups_train.nunique()
    if n_groups_train >= 2:
        # safe to use GroupKFold
        n_splits = min(5, n_groups_train)  # cap at 5
        cv = GroupKFold(n_splits=n_splits)
        rf_cv = GridSearchCV(
            pipe,
            param_grid_rf,
            cv=cv,
            n_jobs=-1,
            scoring="neg_mean_squared_error"
        )
        rf_cv.fit(X_train, y_train, groups=groups_train)
    else:
        # Only one site in training data -> GroupKFold impossible.
        # Fall back to standard KFold on rows.
        n_splits = min(5, len(X_train))  # at most number of samples
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rf_cv = GridSearchCV(
            pipe,
            param_grid_rf,
            cv=cv,
            n_jobs=-1,
            scoring="neg_mean_squared_error"
        )
        rf_cv.fit(X_train, y_train)
else:
    # Fallback: no/insufficient site info → standard random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = GridSearchCV(
        pipe,
        param_grid_rf,
        cv=cv,
        n_jobs=-1,
        scoring="neg_mean_squared_error"
    )
    rf_cv.fit(X_train, y_train)


print("\nBest Parameters for Random Forest:", rf_cv.best_params_)
evaluate_model_regression(rf_cv.best_estimator_, X_test, y_test, "Random Forest")




from common_reports import run_common_reports

# -----------------------------
# After CV: best model + eval
# -----------------------------
best_pipe = rf_cv.best_estimator_

print("\nBest Parameters for Random Forest:", rf_cv.best_params_)
evaluate_model_regression(best_pipe, X_test, y_test, "Random Forest (GA)")

# -----------------------------
# Common reports (shared for GA / PTB)
# -----------------------------
run_common_reports(
    pipeline=best_pipe,
    X_raw=X,          # full raw dataframe with GA features
    y=y,              # full GAGEBRTH vector (for RFE etc.)
    task="reg",       # regression mode
    out_prefix="GA",  # prefix for all output files
    n_top_main=10,
    n_top_interactions=10,
    n_top_pdp=10,
    n_rfe=20
)














