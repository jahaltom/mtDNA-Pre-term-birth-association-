import pandas as pd, numpy as np, re, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.ensemble import RandomForestClassifier  # <-- RF instead of GB
from sklearn.inspection import PartialDependenceDisplay
import shap, os, sys
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from common_reports import run_common_reports


df = pd.read_csv("Metadata.Final.tsv", sep="\t")

categorical_columns = [c for c in sys.argv[1].split(',') if c != "site"]
continuous_columns  = sys.argv[2].split(',')
binary_columns      = sys.argv[3].split(',')


X = df[categorical_columns + continuous_columns + binary_columns].copy()
y = df["PTB"].astype(int)

# --- Preprocess ---
pre = ColumnTransformer([
    ("num", StandardScaler(), continuous_columns),
    ("bin", "passthrough", binary_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_columns)
], remainder="drop", sparse_threshold=1.0)

# --- MODEL: RandomForest ---
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight=None  
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", rf)
])

# --- RF grid (mirrors your style, but with RF params) ---
param_grid = {
    "clf__n_estimators": [300, 600, 900],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 2, 5],
    "clf__max_features": ["sqrt", 0.5],
}








# -----------------------------
# Outer split: site-aware if possible
#   - ≥3 sites: unseen-site test via GroupShuffleSplit
#   - 2 sites: stratified row split + keep site labels for GroupKFold
#   - <2 sites: standard stratified split, no groups
# -----------------------------
if "site" in df.columns:
    n_sites = df["site"].nunique()
else:
    n_sites = 0

if ("site" in df.columns) and (n_sites >= 3):
    groups_all = df["site"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    groups_tr = groups_all[train_idx]

elif ("site" in df.columns) and (n_sites == 2):
    # Prefer site-aware inner CV over a strict unseen-site test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    groups_tr = df.loc[X_tr.index, "site"].values

else:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    groups_tr = None
# -----------------------------
# Class-weighting instead of SMOTE
# -----------------------------
pos_wt = (len(y_tr) - y_tr.sum()) / y_tr.sum()
sample_weight = np.where(y_tr == 1, pos_wt, 1.0)

# -----------------------------
# Inner CV: GroupKFold if we have ≥2 training sites, else StratifiedKFold
# -----------------------------

if (groups_tr is not None) and (len(np.unique(groups_tr)) >= 2):
    n_groups_tr = len(np.unique(groups_tr))
    n_splits = min(5, n_groups_tr)
    cv = GroupKFold(n_splits=n_splits)
    rf_cv = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="average_precision",
    )
    rf_cv.fit(X_tr, y_tr, groups=groups_tr, clf__sample_weight=sample_weight)
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = GridSearchCV(
        pipe,
        param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
    )
    rf_cv.fit(X_tr, y_tr, clf__sample_weight=sample_weight)




best = rf_cv.best_estimator_
proba = best.predict_proba(X_te)[:,1]

with open(os.path.join("RF.PTB_metrics.txt"), "w") as f:
    f.write("Best params: {rf_cv.best_params_}/n")
    f.write("\nClassification report @0.5:\n")
    f.write(classification_report(y_te, (proba>=0.5).astype(int))+"\n")
    f.write("ROC AUC: {roc_auc_score(y_te, proba)} \n")
    f.write("PR AUC : {average_precision_score(y_te, proba)}\n")
            
RocCurveDisplay.from_predictions(y_te, proba)
plt.savefig("roc_auc.png", dpi=200); plt.clf()
PrecisionRecallDisplay.from_predictions(y_te, proba)
plt.savefig("pr_auc.png", dpi=200); plt.clf()





# Recompute weights on FULL data
pos_wt_full = (len(y) - y.sum()) / y.sum()
sample_weight_full = np.where(y == 1, pos_wt_full, 1.0)

best_pipe_full = clone(best)
best_pipe_full.fit(X, y, clf__sample_weight=sample_weight_full)


# ----- Run common interpretation reports on the FULL data -----
run_common_reports(
    pipeline=best_pipe_full,
    X_raw=X,          # raw features (not transformed)
    y=y,
    task="clf",          # classification mode
    pos_label=1,         # PTB = 1
    out_prefix="PTB",    # prefix for all output files
    n_top_main=15,       # you can tune these
    n_top_interactions=10,
    n_top_pdp=10,
    n_rfe=20
)
