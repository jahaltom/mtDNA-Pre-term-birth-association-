import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import seaborn as sns
from common_reports import run_common_reports
from sklearn.base import clone
from sklearn.model_selection import GroupKFold




# Define features
categorical_columns = [c for c in sys.argv[1].split(',') if c != "site"]
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')



df = pd.read_csv("Metadata.Final.tsv", sep="\t")

X = df[categorical_columns + continuous_columns + binary_columns]
y = df["GAGEBRTH"]



pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns),  ])

gb = GradientBoostingRegressor(random_state=42)

pipe = Pipeline([
    ("pre", pre),
    ("reg", gb),
])

param_grid_model = {
    "reg__n_estimators": [100, 200],
    "reg__learning_rate": [0.01, 0.1],
    "reg__max_depth": [3, 5],
}







# --- Site-aware train/test split ---
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
    groups_train = groups_all[train_idx]

elif ("site" in df.columns) and (n_sites == 2):
    # With only 2 sites, we prefer site-aware CV over a pure unseen-site test,
    # so do a row-wise split but keep site labels for GroupKFold.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    groups_train = df.loc[X_train.index, "site"].values

else:
    # No / insufficient site info → standard split, no groups
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    groups_train = None



# --- Inner CV: GroupKFold if possible, else row-level KFold ---
if (groups_train is not None) and (len(np.unique(groups_train)) >= 2):
    n_groups_train = len(np.unique(groups_train))
    n_splits = min(5, n_groups_train)  # cap at 5

    cv = GroupKFold(n_splits=n_splits)
    model_cv = GridSearchCV(
        pipe,
        param_grid_model,   
        cv=cv,
        n_jobs=-1,
        scoring="neg_mean_squared_error"
    )
    model_cv.fit(X_train, y_train, groups=groups_train)
else:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model_cv = GridSearchCV(
        pipe,
        param_grid_model,
        cv=cv,
        n_jobs=-1,
        scoring="neg_mean_squared_error"
    )
    model_cv.fit(X_train, y_train)














# -----------------------------
# After CV: best model + eval
# -----------------------------
best_pipe = model_cv.best_estimator_

with open(os.path.join("GB.GA_metrics.txt"), "w") as f:
    f.write("\nBest Parameters for Gradient Boosting: {model_cv.best_params_}\n")
    y_pred = best_pipe.predict(X_test)
    f.write(f"\n Gradient Boosting Evaluation:\n")
    f.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}\n")
    f.write(f"R-squared: {r2_score(y_test, y_pred):.4f}")


best_pipe_full = clone(best_pipe)
best_pipe_full.fit(X, y)   # X, y = ALL samples, all sites

# -----------------------------
# Common reports (shared for GA / PTB)
# -----------------------------
run_common_reports(
    pipeline=best_pipe_full,
    X_raw=X,          # full raw dataframe with GA features
    y=y,              # full GAGEBRTH vector (for RFE etc.)
    task="reg",       # regression mode
    out_prefix="GA",  # prefix for all output files
    n_top_main=10,
    n_top_interactions=10,
    n_top_pdp=10,
    n_rfe=20
)


