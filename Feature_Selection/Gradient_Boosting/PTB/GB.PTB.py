import pandas as pd, numpy as np, re, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
import shap, os, sys
import seaborn as sns


# --- IO ---
df = pd.read_csv("Metadata.Final.tsv", sep="\t")

categorical_columns = sys.argv[1].split(',')
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

gb = GradientBoostingClassifier(
    random_state=42,
    subsample=1.0
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", gb)
])

param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [2, 3],
}



X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Class-weighting instead of SMOTE:
# (GradientBoostingClassifier lacks class_weight; instead upweight positives by sample_weight.)
pos_wt = (len(y_tr) - y_tr.sum()) / y_tr.sum()
sample_weight = np.where(y_tr==1, pos_wt, 1.0)









if df["site"].nunique() >= 2: 
    groups = df.loc[X_tr.index, "site"]
    skf = GroupKFold(n_splits=df["site"].nunique())
    gb_cv = GridSearchCV(
        pipe,
        param_grid,
        cv=skf,
        n_jobs=-1,
        scoring="average_precision"
    )
    gb_cv.fit(X_tr, y_tr, groups=groups,clf__sample_weight=sample_weight)
else:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gb_cv = GridSearchCV(pipe, param_grid, scoring="average_precision", cv=skf, n_jobs=-1)
    gb_cv.fit(X_tr, y_tr, clf__sample_weight=sample_weight)








best = gb_cv.best_estimator_
print("Best params:", gb_cv.best_params_)

# --- Eval ---
proba = best.predict_proba(X_te)[:,1]
print(classification_report(y_te, (proba>=0.5).astype(int)))
print("ROC AUC:", roc_auc_score(y_te, proba))
print("PR AUC :", average_precision_score(y_te, proba))
RocCurveDisplay.from_predictions(y_te, proba)
plt.savefig("roc_auc.png", dpi=200); plt.clf()
PrecisionRecallDisplay.from_predictions(y_te, proba)
plt.savefig("pr_auc.png", dpi=200); plt.clf()

# --- SHAP on a manageable subset ---
X_te_trans = best.named_steps["pre"].transform(X_te)    # sparse ok for TreeExplainer input if we densify
X_te_dense = X_te_trans.toarray()
feat_names = best.named_steps["pre"].get_feature_names_out()

explainer = shap.TreeExplainer(best.named_steps["clf"])
sub_ix = np.random.RandomState(42).choice(X_te_dense.shape[0], size=min(2000, X_te_dense.shape[0]), replace=False)
sv = explainer.shap_values(X_te_dense[sub_ix])

# Mean |SHAP| importances
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
topk = order[:30]
top_names = feat_names[topk]

shap.summary_plot(sv[:, topk], X_te_dense[sub_ix][:, topk], feature_names=top_names, show=False)
plt.savefig("shap_summary_top30.png", bbox_inches="tight", dpi=200); plt.clf()

# Interactions on top-k only
sv_int = explainer.shap_interaction_values(X_te_dense[sub_ix][:, topk])
if isinstance(sv_int, list):  # SHAP may return per-class list
    sv_int = sv_int[0]

names = feat_names[topk]

# 1) SHAP interaction SUMMARY plot (top-k)
shap.summary_plot(
    sv_int,
    X_te_dense[sub_ix][:, topk],
    feature_names=names,
    max_display=min(20, len(names)),
    show=False
)
plt.tight_layout()
plt.savefig("shap_interaction_summary_topk.png", dpi=300, bbox_inches="tight")
#plt.clf()
plt.close('all')

# 2) Mean |interaction| matrix + top pairs
int_mat = np.abs(sv_int).mean(axis=0)                # (k x k)
interaction_df = pd.DataFrame(int_mat, index=names, columns=names)
np.fill_diagonal(interaction_df.values, 0.0)         # ignore self-interactions

M = 10  # how many pairs to print
top_pairs = (
    interaction_df.where(np.triu(np.ones_like(interaction_df), 1).astype(bool))
                  .stack()
                  .sort_values(ascending=False)
                  .head(M)
)
print("\nTop interaction pairs (mean |SHAP interaction|):")
print(top_pairs)

# 3) Upper-triangle heatmap
mask = np.tril(np.ones_like(interaction_df, dtype=bool))
plt.figure(figsize=(max(8, 0.6*len(names)), max(6, 0.6*len(names))))
sns.heatmap(
    interaction_df,
    mask=mask,
    cmap="coolwarm",
    square=True,
    cbar_kws={"label": "Mean |SHAP interaction|"},
    xticklabels=True, yticklabels=True
)
plt.title("SHAP Interaction Heatmap (Top-k)")
plt.tight_layout()
plt.savefig("shap_interactions_heatmap_topk.png", dpi=300)
#plt.clf()
plt.close('all')




























##########################Non -linear stuff

#SHAP dependence plots (curvature = nonlinearity). Interpretation: a curved (non-monotone) relationship between SHAP value (log-odds contribution) and the raw feature indicates non-linearity. A straight line suggests linear or near-linear.
top_names = feat_names[topk]
for name in top_names:
    shap.dependence_plot(name, sv, X_te_dense[sub_ix], feature_names=feat_names, show=False)
    plt.savefig(f"dep_{name}.png", bbox_inches="tight")
    plt.close('all')





# --- PDP / ICE / Nonlinearity: use NAMES, not indices ---
import matplotlib
matplotlib.use("Agg")
import gc
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

# Always fetch the FULL transformed feature list
feat_names_full = best.named_steps["pre"].get_feature_names_out()

raw = []
seen = set()

for t in feat_names_full:
    # remove num__, bin__, cat__
    name = re.sub(r'^(num__|bin__|cat__)', '', t)
    # remove final _digit suffix (but keep underscores in middle)
    name = re.sub(r'_\d+$', '', name)
    if name not in seen:
        raw.append(name)
        seen.add(name)
feat_names_full=raw

for t in top_names:
    # remove num__, bin__, cat__
    name = re.sub(r'^(num__|bin__|cat__)', '', t)
    # remove final _digit suffix (but keep underscores in middle)
    name = re.sub(r'_\d+$', '', name)
    if name not in seen:
        raw.append(name)
        seen.add(name)
top_names=raw












# 1) PDP for first 12 top features (pass NAMES)
disp = PartialDependenceDisplay.from_estimator(
    best,
    X,                                    # raw X; pipeline handles transforms
    features=top_names[:12],              # names, not indices
    feature_names=feat_names_full,
    grid_resolution=101,
    response_method="predict_proba",
    n_cols=6,                             # 6 columns => 2 rows for 12 features
)

# --- Make the plot taller ---
# default is small (~8x6 inches); this doubles the height for readability
disp.figure_.set_size_inches(14, 8)       # width=14, height=8 (adjust to taste)

plt.tight_layout()
plt.savefig("pdp_top12.png", dpi=200, bbox_inches="tight")
plt.close('all')


# 2) Simple nonlinearity score using PDP curves (use NAMES here too)
def nonlinearity_score(est, X_sample, feat_name, K=6):
    pd_res = partial_dependence(
        est, X_sample, features=[feat_name], grid_resolution=50, kind="average"
    )
    xs = pd_res["grid_values"][0].reshape(-1, 1)
    ys = pd_res["average"][0].ravel()
    lin_r2 = LinearRegression().fit(xs, ys).score(xs, ys)
    spline = make_pipeline(SplineTransformer(degree=3, n_knots=K, include_bias=False),
                           LinearRegression())
    spl_r2 = spline.fit(xs, ys).score(xs, ys)
    return {"feature": feat_name, "R2_linear": lin_r2,
            "R2_spline": spl_r2, "NL_score": max(0.0, spl_r2 - lin_r2)}

scores = [nonlinearity_score(best, X, n, K=6) for n in top_names[:30]]
scores_df = pd.DataFrame(scores).sort_values("NL_score", ascending=False)
scores_df.to_csv("nonlinearity_scores.csv", index=False)
print(scores_df.head(10))

# 3) ICE + PDP for BMI (pass NAME)
PartialDependenceDisplay.from_estimator(
    best, X,
    features=["BMI"],                # <-- name, not index
    feature_names=feat_names_full,
    kind="both"                           # PDP + ICE
)
plt.tight_layout()
plt.savefig("ice_bmi.png", dpi=200, bbox_inches="tight")
plt.close('all'); gc.collect()
