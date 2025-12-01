import re
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import RFE
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline


# ------------------------
# Helpers
# ------------------------

def _to_base_name(t):
    """
    Convert transformed feature name to a raw base name.
    E.g. "num__BMI" -> "BMI", "cat__SITE_A_0" -> "SITE_A".
    Adjust regex if needed.
    """
    name = re.sub(r'^(num__|bin__|cat__)', '', t)
    name = re.sub(r'_\d+$', '', name)
    return name


def compute_shap_main_and_interactions(model, X_trans, feature_names, task="reg", pos_label=1):
    """
    Normalize SHAP main + interaction values for both regression and binary classification.

    Returns
    -------
    shap_main : (N, F)
    shap_int  : (N, F, F)   (if interactions fail / mismatch, zeros with same shape)
    """
    expl = shap.TreeExplainer(model)

    # -----------------------
    # MAIN SHAP VALUES
    # -----------------------
    sv_raw = expl.shap_values(X_trans)

    if task == "reg":
        # Regression: usually a single (N, F) array.
        shap_main = np.asarray(sv_raw)
        if shap_main.ndim == 3:
            # Occasionally (N, 1, F)
            if shap_main.shape[1] == 1:
                shap_main = shap_main[:, 0, :]
            else:
                shap_main = shap_main.reshape(shap_main.shape[0], -1)
        elif shap_main.ndim != 2:
            shap_main = shap_main.reshape(shap_main.shape[0], -1)
    else:
        # Classification: pick SHAP for the positive class.
        classes_ = getattr(model, "classes_", None)
        if classes_ is None:
            raise ValueError("Need classes_ for classification SHAP.")
        pos_idx = int(np.where(classes_ == pos_label)[0][0])

        if isinstance(sv_raw, list):
            # Typical binary case: list [class0, class1]
            shap_main = np.asarray(sv_raw[pos_idx])
        else:
            sv_arr = np.asarray(sv_raw)
            # Newer SHAP: directly (N, F) for binary clf
            if sv_arr.ndim == 2 and sv_arr.shape[1] == len(feature_names):
                shap_main = sv_arr
            elif sv_arr.ndim == 3:
                # (N, C, F) or (N, F, C)
                if sv_arr.shape[2] == len(feature_names):      # (N, C, F)
                    shap_main = sv_arr[:, pos_idx, :]
                elif sv_arr.shape[1] == len(feature_names):    # (N, F, C)
                    shap_main = sv_arr[:, :, pos_idx]
                else:
                    print(f"[WARN] Unexpected SHAP main shape {sv_arr.shape}, reshaping.")
                    shap_main = sv_arr.reshape(sv_arr.shape[0], -1)
            else:
                print(f"[WARN] Unexpected SHAP main shape {sv_arr.shape}, reshaping.")
                shap_main = sv_arr.reshape(sv_arr.shape[0], -1)

    # Final sanity check for main
    shap_main = np.asarray(shap_main)
    if shap_main.ndim != 2:
        shap_main = shap_main.reshape(shap_main.shape[0], -1)

    if shap_main.shape[1] != len(feature_names):
        print(
            f"[WARN] shap_main has {shap_main.shape[1]} features but "
            f"feature_names has {len(feature_names)}; attempting to align."
        )
        # If fewer SHAP cols than names, truncate names; if more, truncate SHAP
        F = min(shap_main.shape[1], len(feature_names))
        shap_main = shap_main[:, :F]
        feature_names = feature_names[:F]

    # -----------------------
    # INTERACTION VALUES
    # -----------------------
    try:
        int_raw = expl.shap_interaction_values(X_trans)
        int_arr = np.asarray(int_raw)

        if task == "reg":
            # Regression: usually (N, F, F)
            shap_int = int_arr
        else:
            # Classification: list-of-classes or (N, C, F, F) / (N, F, F, C)
            pos_idx = int(np.where(model.classes_ == pos_label)[0][0])

            if isinstance(int_raw, list):
                shap_int = np.asarray(int_raw[pos_idx])
            elif int_arr.ndim == 4:
                # (N, C, F, F) or (N, F, F, C)
                if int_arr.shape[2] == len(feature_names):      # (N, C, F, F)
                    shap_int = int_arr[:, pos_idx, :, :]
                elif int_arr.shape[1] == len(feature_names):    # (N, F, F, C)
                    shap_int = int_arr[:, :, :, pos_idx]
                else:
                    print(f"[WARN] Unexpected interaction shape {int_arr.shape}, reshaping.")
                    # Try to pick the first "class-like" dimension
                    shap_int = int_arr.reshape(int_arr.shape[0], len(feature_names), len(feature_names))
            elif int_arr.ndim == 3:
                # Already (N, F, F)
                shap_int = int_arr
            else:
                print(f"[WARN] Unexpected interaction shape {int_arr.shape}; will fallback to zeros.")
                shap_int = None
    except Exception as e:
        print(f"[WARN] Could not compute SHAP interactions: {e}")
        shap_int = None

    # -----------------------
    # Final shape check / fallback
    # -----------------------
    F = len(feature_names)
    N = X_trans.shape[0]

    if shap_int is None:
        print(f"[WARN] No valid SHAP interactions; using zeros of shape (N={N}, F={F}, F={F}).")
        shap_int = np.zeros((N, F, F), dtype=float)
    else:
        shap_int = np.asarray(shap_int)
        if shap_int.ndim != 3:
            print(f"[WARN] SHAP interaction ndim={shap_int.ndim}; reshaping to (N, F, F).")
            shap_int = shap_int.reshape(N, F, F)
        if shap_int.shape[1] != F or shap_int.shape[2] != F:
            print(
                f"[WARN] SHAP interaction shape {shap_int.shape} does not match F={F}; "
                "truncating/padding to (N, F, F)."
            )
            # Truncate or pad along feature dims
            out = np.zeros((N, F, F), dtype=float)
            f1 = min(F, shap_int.shape[1])
            f2 = min(F, shap_int.shape[2])
            out[:, :f1, :f2] = shap_int[:, :f1, :f2]
            shap_int = out

    return shap_main, shap_int


def pdp_nonlinearity_score(est, X_sample, feat_name, K=6, link="identity"):
    """
    PDP-based nonlinearity score for both regression and classification.

    link = "identity" for regression,
           "logit"    for PTB (probability -> log-odds)
    """
    pd_res = partial_dependence(
        est,
        X_sample,
        features=[feat_name],
        grid_resolution=50,
        kind="average"
    )
    xs = pd_res["grid_values"][0].reshape(-1, 1)
    ys = pd_res["average"][0].ravel()

    # optional logit transform
    if link == "logit":
        eps = 1e-6
        ys = np.clip(ys, eps, 1 - eps)
        ys = np.log(ys / (1 - ys))

    xs_c = xs - xs.mean(axis=0, keepdims=True)
    ys_c = ys - ys.mean()

    if np.var(ys_c) < 1e-8:
        return {
            "feature": feat_name,
            "R2_linear": 0.0,
            "R2_spline": 0.0,
            "NL_abs": 0.0,
        }

    lin = LinearRegression().fit(xs_c, ys_c)
    R2_lin = lin.score(xs_c, ys_c)

    spline = make_pipeline(
        SplineTransformer(degree=3, n_knots=K, include_bias=False),
        LinearRegression()
    )
    R2_spl = spline.fit(xs_c, ys_c).score(xs_c, ys_c)

    NL_abs = max(0.0, R2_spl - R2_lin)

    return {
        "feature": feat_name,
        "R2_linear": R2_lin,
        "R2_spline": R2_spl,
        "NL_abs": NL_abs,
    }


# ------------------------
# Main report driver
# ------------------------

def run_common_reports(
    pipeline,
    X_raw,
    y,
    task="reg",              # "reg" or "clf"
    pos_label=1,
    out_prefix="GA",         # e.g. "GA" or "PTB"
    n_top_main=10,
    n_top_interactions=10,
    n_top_pdp=10,
    n_rfe=20
):
    """
    Generate common outputs for both your scripts:

    - Top-10 importance (SHAP + feature_importances_)
    - RFE-selected features
    - SHAP summary
    - SHAP interaction summary
    - SHAP interaction heatmap
    - PDP & nonlinearity scores
    """

    # 1) Extract preprocessor + model and feature names
    pre = pipeline.named_steps["pre"]
    model = list(pipeline.named_steps.values())[-1]

    X_trans = pre.transform(X_raw)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    feature_names = np.asarray(pre.get_feature_names_out())

    # ------------------ TOP IMPORTANCE ------------------
    shap_main, shap_int = compute_shap_main_and_interactions(
        model,
        X_trans,
        feature_names,
        task=task,
        pos_label=pos_label
    )

    mean_abs = np.abs(shap_main).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top_idx = order[:min(n_top_main, len(order))]
    top_feats_trans = feature_names[top_idx]

    # data frame of SHAP importances
    shap_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": mean_abs
    }).sort_values("MeanAbsSHAP", ascending=False)

    shap_imp_df.to_csv(f"{out_prefix}.shap_importance.csv", index=False)
    print(f"\n[{out_prefix}] Top 10 by mean |SHAP|:")
    print(shap_imp_df.head(10))

    # model.feature_importances_ (tree-based)
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        fi_df = (pd.DataFrame({"Feature": feature_names, "Importance": fi})
                 .sort_values("Importance", ascending=False))
        fi_df.to_csv(f"{out_prefix}.importance.csv", index=False)
        print(f"\n[{out_prefix}] Top 10 by  feature_importances_:")
        print(fi_df.head(10))

    # ------------------ RFE ------------------
    from sklearn.base import clone
    mod_for_rfe = clone(model)
    n_feats = min(n_rfe, X_trans.shape[1])
    rfe = RFE(mod_for_rfe, n_features_to_select=n_feats)
    rfe.fit(X_trans, y)
    selected_features = feature_names[rfe.support_]

    rfe_df = pd.DataFrame({"SelectedFeature": selected_features})
    rfe_df.to_csv(f"{out_prefix}.rfe_selected.csv", index=False)
    print(f"\n[{out_prefix}] RFE-selected features (n={len(selected_features)}):")
    print(selected_features[:20])

    # ------------------ SHAP SUMMARY PLOT ------------------
    shap.summary_plot(
        shap_main,
        X_trans,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"{out_prefix} – SHAP Summary")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------ SHAP INTERACTIONS ------------------
    # mean |interaction| matrix
    int_mat = np.abs(shap_int).mean(axis=0)  # (F, F)

    # top interaction pairs (upper triangle)
    iu = np.triu_indices_from(int_mat, k=1)
    pairs_df = pd.DataFrame({
        "Feature_1": feature_names[iu[0]],
        "Feature_2": feature_names[iu[1]],
        "InteractionStrength": int_mat[iu]
    }).sort_values("InteractionStrength", ascending=False)

    pairs_df.to_csv(f"{out_prefix}.shap_interactions.csv", index=False)
    print(f"\n[{out_prefix}] Top {n_top_interactions} SHAP interaction pairs:")
    print(pairs_df.head(n_top_interactions))

    # heatmap for top-K main SHAP features
    K = min(30, len(feature_names))
    top_idx_K = order[:K]
    sub_mat = int_mat[np.ix_(top_idx_K, top_idx_K)]
    sub_names = feature_names[top_idx_K]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pd.DataFrame(sub_mat, index=sub_names, columns=sub_names),
        cmap="coolwarm",
        square=True,
        cbar=True
    )
    plt.title(f"{out_prefix} – SHAP Interaction Heatmap (Top-{K})")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.shap_interactions_heatmap.png", dpi=300)
    plt.close()

    # SHAP interaction summary plot on top-K
    shap.summary_plot(
        shap_int[:, top_idx_K, :][:, :, top_idx_K],
        X_trans[:, top_idx_K],
        feature_names=sub_names,
        max_display=min(20, len(sub_names)),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.shap_interaction_summary_topK.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------ PDP + NONLINEARITY ------------------
    # map transformed names → base names
    top_base = []
    seen = set()
    for t in top_feats_trans:
        b = _to_base_name(t)
        if b not in seen and b in X_raw.columns:
            seen.add(b)
            top_base.append(b)

    link = "identity" if task == "reg" else "logit"

    nl_scores = []
    for feat in top_base[:n_top_pdp]:
        sc = pdp_nonlinearity_score(
            pipeline,
            X_raw,
            feat,
            K=6,
            link=link
        )
        nl_scores.append(sc)

        # PDP/ICE plot

        disp = PartialDependenceDisplay.from_estimator(
            pipeline,
            X_raw,
            features=[feat],
            kind=both,
            grid_resolution=50
        )
        disp.figure_.set_size_inches(6, 4)
        plt.suptitle(f"{out_prefix} – PDP/ICE: {feat}", y=1.02)
        plt.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', feat)
        plt.savefig(f"{out_prefix}.pdp_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close()

    nl_df = pd.DataFrame(nl_scores).sort_values("NL_abs", ascending=False)
    nl_df.to_csv(f"{out_prefix}.nonlinearity_scores.csv", index=False)
    print(f"\n[{out_prefix}] Top 5 by PDP-based nonlinearity:")
    print(nl_df.head(5))

    print(f"\n[{out_prefix}] Common reports generated.")

