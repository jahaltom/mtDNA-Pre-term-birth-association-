import pandas as pd, numpy as np, re, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, GroupShuffleSplit
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

# --- IO ---
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

# --- MODEL: RandomForest (drop GB) ---
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
    ("rf", rf)
])

# --- RF grid (mirrors your style, but with RF params) ---
param_grid = {
    "rf__n_estimators": [300, 600, 900],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_leaf": [1, 2, 5],
    "rf__max_features": ["sqrt", 0.5],
}





from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedKFold, train_test_split

# -----------------------------
# Outer split: by site if possible, else stratified random
# -----------------------------
if "site" in df.columns and df["site"].nunique() >= 2:
    # Use sites as groups so test set contains (mostly) unseen sites
    groups_all = df["site"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    groups_tr = groups_all.iloc[train_idx]
else:
    # Fallback: no/insufficient site info → standard stratified split
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
if (groups_tr is not None) and (groups_tr.nunique() >= 2):
    n_groups_tr = groups_tr.nunique()
    n_splits = min(5, n_groups_tr)  # cap at 5
    skf = GroupKFold(n_splits=n_splits)
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=skf,
        n_jobs=-1,
        scoring="average_precision",
    )
    gs.fit(X_tr, y_tr, groups=groups_tr, rf__sample_weight=sample_weight)
else:
    # No site info or only 1 site in training → use stratified row-level CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring="average_precision",
        cv=skf,
        n_jobs=-1,
    )
    gs.fit(X_tr, y_tr, rf__sample_weight=sample_weight)







best = gs.best_estimator_
print("Best params:", gs.best_params_)

# --- Eval ---
proba = best.predict_proba(X_te)[:,1]
print(classification_report(y_te, (proba>=0.5).astype(int)))
print("ROC AUC:", roc_auc_score(y_te, proba))
print("PR AUC :", average_precision_score(y_te, proba))
RocCurveDisplay.from_predictions(y_te, proba)
plt.savefig("roc_auc.png", dpi=200); plt.clf()
PrecisionRecallDisplay.from_predictions(y_te, proba)
plt.savefig("pr_auc.png", dpi=200); plt.clf()



# Recompute weights on FULL data
pos_wt_full = (len(y) - y.sum()) / y.sum()
sample_weight_full = np.where(y == 1, pos_wt_full, 1.0)

best_pipe_full = clone(best)
best_pipe_full.fit(X, y, rf__sample_weight=sample_weight_full)



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
