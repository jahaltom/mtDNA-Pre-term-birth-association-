import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def evaluate_model_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R-squared: {r2_score(y_test, y_pred):.4f}")



# Define features
categorical_columns = sys.argv[1].split(',')
continuous_columns = sys.argv[2].split(',')
binary_columns = sys.argv[3].split(',')

df = pd.read_csv("Metadata.Final.tsv", sep="\t")
required = categorical_columns + continuous_columns + binary_columns + ["GAGEBRTH"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input: {missing}")

X = df[categorical_columns + continuous_columns + binary_columns]
y = df["GAGEBRTH"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Dense OHE to support GB + SHAP
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # or sparse=False on older sklearn
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_columns),
        ("bin", "passthrough", binary_columns),
        ("cat", ohe, categorical_columns),
    ]
)

gb = GradientBoostingRegressor(random_state=42)

pipe = Pipeline([
    ("prep", preprocessor),
    ("gb", gb),
])

param_grid_gb = {
    "gb__n_estimators": [100, 200],
    "gb__learning_rate": [0.01, 0.1],
    "gb__max_depth": [3, 5],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
gb_cv = GridSearchCV(pipe, param_grid_gb, cv=cv, n_jobs=-1)
gb_cv.fit(X_train, y_train)

print("\nBest Parameters for Gradient Boosting:", gb_cv.best_params_)
evaluate_model_regression(gb_cv.best_estimator_, X_test, y_test, "Gradient Boosting")

# Feature names after fit (use the fitted preprocessor inside the pipeline)
fitted_prep = gb_cv.best_estimator_.named_steps["prep"]
feature_names = fitted_prep.get_feature_names_out()

# Importances from the fitted GB
gb_model = gb_cv.best_estimator_.named_steps["gb"]
importances = getattr(gb_model, "feature_importances_", None)
if importances is not None:
    gb_importances = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
          .sort_values("Importance", ascending=False)
    )
    print("\nGradient Boosting Feature Importances (top 10):")
    print(gb_importances.head(10))
else:
    print("\nModel has no feature_importances_ attribute.")

# RFE over the pipeline's final feature space
# (fit RFE on the transformed design to respect OHE expansion)
from sklearn.base import clone
Xtr = fitted_prep.transform(X_train)  # dense by construction
gb_for_rfe = clone(gb_model)
n_feats = min(20, Xtr.shape[1])
rfe = RFE(gb_for_rfe, n_features_to_select=n_feats)
rfe.fit(Xtr, y_train)
selected_features = feature_names[rfe.support_]
print("\nRFE-selected features:")
print(selected_features)




import shap
import matplotlib.pyplot as plt

# --- SHAP analysis for the trained Gradient Boosting model ---
# Extract trained model and preprocessor from pipeline
best_pipe = gb_cv.best_estimator_
gb_model = best_pipe.named_steps["gb"]
fitted_prep = best_pipe.named_steps["prep"]

# Transform training data to match model’s feature space
X_train_preprocessed = fitted_prep.transform(X_train)
feature_names = fitted_prep.get_feature_names_out()

# Initialize SHAP TreeExplainer for Gradient Boosting
explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_train_preprocessed)

# --- Summary plot ---
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_train_preprocessed,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Summary Plot - Gradient Boosting (Gestational Age)")
plt.tight_layout()
plt.savefig("shap.summary_plot.GB.GA.png", bbox_inches="tight", dpi=300)
plt.close()

print("\nSaved SHAP summary plot to 'shap_summary_GB_GA.png'")



























# -----------------------------
# SHAP INTERACTIONS (tree-only)
# -----------------------------
# WARNING: O(F^2 * N). Use a row subsample and cap features for speed.

X_sub = X_train_preprocessed

# Compute interaction values
interaction_values = shap.TreeExplainer(gb_model).shap_interaction_values(X_sub)
# interaction_values shape: (n_samples, F, F)
# Aggregate to mean absolute strength per pair
interaction_matrix = np.abs(interaction_values).mean(axis=0)  # (F, F)

# Work with upper triangle (no duplicates / no self)
iu = np.triu_indices_from(interaction_matrix, k=1)
pairs = pd.DataFrame({
    "Feature_1": feature_names[iu[0]],
    "Feature_2": feature_names[iu[1]],
    "Interaction_Strength": interaction_matrix[iu]
}).sort_values("Interaction_Strength", ascending=False)

print("\nTop 10 SHAP interactions:")
print(pairs.head(10))

# Save a manageable heatmap for top-K features (based on main effect magnitude)
K = min(30, len(feature_names))
main_effect_strength = np.abs(shap_values).mean(axis=0)   # reuse main shap values
top_idx = np.argsort(main_effect_strength)[-K:]

sub_mat = interaction_matrix[np.ix_(top_idx, top_idx)]
sub_names = feature_names[top_idx]

plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(sub_mat, index=sub_names, columns=sub_names),
            cmap="coolwarm", center=0, square=True, cbar=True)
plt.title("SHAP Interaction Heatmap (Top features)")
plt.tight_layout()
plt.savefig("shap_interactions_heatmap_top.png", dpi=300)
plt.close()

# PDP for top 5 interaction pairs
from sklearn.inspection import PartialDependenceDisplay

top_pairs = [(pairs.iloc[i].Feature_1, pairs.iloc[i].Feature_2)
             for i in range(min(5, len(pairs)))]

# map names -> indices in transformed space
name_to_idx = {n:i for i, n in enumerate(feature_names)}
pair_indices = [(name_to_idx[a], name_to_idx[b]) for (a,b) in top_pairs]

disp=PartialDependenceDisplay.from_estimator(
    gb_model,                      # model expects transformed X
    X_train_preprocessed,          # pass transformed design
    features=pair_indices,
    feature_names=feature_names,
    grid_resolution=20
)
disp.figure_.set_size_inches(14, 8)  
plt.suptitle("PDP — Top SHAP Interaction Pairs", y=1.02)
plt.tight_layout()
plt.savefig("pdp_top_interactions.png", dpi=300, bbox_inches="tight")
plt.close()

# --- SHAP interaction SUMMARY (top-K features you already picked) ---
sv_int = shap.TreeExplainer(gb_model).shap_interaction_values(X_sub[:, top_idx])
if isinstance(sv_int, list):  # for classifiers
    sv_int = sv_int[0]

shap.summary_plot(
    sv_int,
    X_sub[:, top_idx],
    feature_names=sub_names,
    max_display=min(20, len(sub_names)),
    show=False
)
plt.tight_layout()
plt.savefig("shap_interaction_summary_topk.png", dpi=300, bbox_inches="tight")
plt.close()
















# -------------------------------------------
# NON-LINEAR FEATURES: score + dependence plots
# -------------------------------------------
# A simple nonlinearity score:
# Compare linear vs. cubic (poly deg=3) model predicting SHAP from the feature's column.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def nonlinearity_score(x, shap_col):
    x = np.asarray(x).reshape(-1, 1)
    # linear fit
    lr = LinearRegression().fit(x, shap_col)
    r2_lin = lr.score(x, shap_col)
    # cubic fit
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X3 = poly.fit_transform(x)
    lr3 = LinearRegression().fit(X3, shap_col)
    r2_cubic = lr3.score(X3, shap_col)
    # gain in explained variance beyond linear
    return max(0.0, r2_cubic - r2_lin), r2_lin, r2_cubic

nl_rows = []
for j, name in enumerate(feature_names):
    gain, r2_lin, r2_cub = nonlinearity_score(X_train_preprocessed[:, j], shap_values[:, j])
    nl_rows.append((name, gain, r2_lin, r2_cub))

nl_df = (pd.DataFrame(nl_rows, columns=["Feature", "NonlinearityGain", "R2_linear", "R2_cubic"])
           .sort_values("NonlinearityGain", ascending=False))
print("\nTop 10 suspected non-linear features (by cubic gain over linear):")
print(nl_df.head(10))

# SHAP dependence plots for the top non-linear features (directionality + interactions)
top_nonlin = nl_df.head(8)["Feature"].tolist()
for name in top_nonlin:
    shap.dependence_plot(
        name,
        shap_values,
        X_train_preprocessed,
        feature_names=feature_names,
        interaction_index="auto",   # SHAP auto-picks strongest interacting partner
        show=False
    )
    safe_name = "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
    plt.title(f"SHAP dependence: {name}")
    plt.tight_layout()
    plt.savefig(f"shap_dependence_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Optional: PDP (1D) for the same non-linear suspects
one_d_idxs = [name_to_idx[n] for n in top_nonlin]
disp=PartialDependenceDisplay.from_estimator(
    gb_model,
    X_train_preprocessed,
    features=one_d_idxs,
    feature_names=feature_names,
    grid_resolution=25
)
disp.figure_.set_size_inches(14, 8)  
plt.suptitle("PDP — Non-linear Feature Candidates", y=1.02)
plt.tight_layout()
plt.savefig("pdp_top_nonlinear.png", dpi=300, bbox_inches="tight")
plt.close()




