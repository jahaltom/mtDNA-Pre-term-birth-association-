import argparse
import itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA
from scipy.spatial.distance import pdist, squareform


def permanova(distance_matrix, groups, n_perm=999, seed=123):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    n = len(groups)
    D2 = distance_matrix ** 2
    total_ss = D2.sum() / n

    within_ss = 0.0
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        within_ss += D2[np.ix_(idx, idx)].sum() / len(idx)

    between_ss = total_ss - within_ss
    df_between = len(unique_groups) - 1
    df_within = n - len(unique_groups)

    F_obs = (between_ss / df_between) / (within_ss / df_within)
    r2 = between_ss / total_ss

    perm_F = []
    for _ in range(n_perm):
        perm_groups = rng.permutation(groups)

        perm_within_ss = 0.0
        for g in unique_groups:
            idx = np.where(perm_groups == g)[0]
            perm_within_ss += D2[np.ix_(idx, idx)].sum() / len(idx)

        perm_between_ss = total_ss - perm_within_ss
        perm_F_val = (perm_between_ss / df_between) / (perm_within_ss / df_within)
        perm_F.append(perm_F_val)

    perm_F = np.asarray(perm_F)
    p_perm = (np.sum(perm_F >= F_obs) + 1) / (n_perm + 1)

    return {
        "permanova_F": F_obs,
        "permanova_R2": r2,
        "permanova_p": p_perm,
        "df_between": df_between,
        "df_within": df_within,
        "n_permutations": n_perm,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test association between study site and nDNA PCs using ANOVA, R2, MANOVA, and PERMANOVA."
    )

    parser.add_argument("--input", required=True, help="Input metadata file, CSV or TSV.")
    parser.add_argument("--site-col", default="site", help="Site column name.")
    parser.add_argument("--pc-prefix", default="PC", help="PC column prefix, e.g. PC for PC1, PC2.")
    parser.add_argument("--n-pcs", type=int, default=5, help="Number of PCs to test.")
    parser.add_argument("--sep", default=None, help="File separator. Default auto-detect.")
    parser.add_argument("--permutations", type=int, default=999, help="Number of PERMANOVA permutations.")
    parser.add_argument("--out-prefix", default="site_pc_association", help="Output prefix.")

    args = parser.parse_args()

    if args.sep is None:
        sep = "\t" if args.input.endswith((".tsv", ".tab")) else ","
    else:
        sep = args.sep

    df = pd.read_csv(args.input, sep=sep)

    pc_cols = [f"{args.pc_prefix}{i}" for i in range(1, args.n_pcs + 1)]

    needed = [args.site_col] + pc_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[needed].dropna().copy()
    df[args.site_col] = df[args.site_col].astype(str)

    print(f"Using {df.shape[0]} samples")
    print(f"Sites: {df[args.site_col].nunique()}")
    print(df[args.site_col].value_counts())

    # -----------------------
    # 1. Per-PC ANOVA + R²
    # -----------------------
    anova_rows = []

    for pc in pc_cols:
        model = smf.ols(f"{pc} ~ C({args.site_col})", data=df).fit()
        aov = anova_lm(model, typ=2)

        site_row = aov.loc[f"C({args.site_col})"]

        anova_rows.append({
            "PC": pc,
            "n": int(model.nobs),
            "site_df": site_row["df"],
            "residual_df": model.df_resid,
            "F": site_row["F"],
            "p_value": site_row["PR(>F)"],
            "R2": model.rsquared,
            "adj_R2": model.rsquared_adj,
        })

    anova_df = pd.DataFrame(anova_rows)
    anova_df.to_csv(f"{args.out_prefix}_anova_r2.csv", index=False)

    print("\nANOVA + R²:")
    print(anova_df)

    # -----------------------
    # 2. MANOVA
    # -----------------------
    lhs = " + ".join(pc_cols)
    formula = f"{lhs} ~ C({args.site_col})"

    manova = MANOVA.from_formula(formula, data=df)
    manova_results = manova.mv_test()

    with open(f"{args.out_prefix}_manova.txt", "w") as f:
        f.write(str(manova_results))

    print("\nMANOVA:")
    print(manova_results)

    # -----------------------
    # 3. PERMANOVA
    # -----------------------
    X = df[pc_cols].to_numpy()

    # Standardize PCs so PC1 does not dominate just because of scale
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    dist_mat = squareform(pdist(X, metric="euclidean"))

    perm_results = permanova(
        distance_matrix=dist_mat,
        groups=df[args.site_col].values,
        n_perm=args.permutations,
        seed=123
    )

    perm_df = pd.DataFrame([perm_results])
    perm_df.to_csv(f"{args.out_prefix}_permanova.csv", index=False)

    print("\nPERMANOVA:")
    print(perm_df)

    # -----------------------
    # 4. Site means for PCs
    # -----------------------
    site_means = df.groupby(args.site_col)[pc_cols].agg(["mean", "std", "count"])
    site_means.to_csv(f"{args.out_prefix}_site_pc_summary.csv")

    print("\nSaved:")
    print(f"{args.out_prefix}_anova_r2.csv")
    print(f"{args.out_prefix}_manova.txt")
    print(f"{args.out_prefix}_permanova.csv")
    print(f"{args.out_prefix}_site_pc_summary.csv")


if __name__ == "__main__":
    main()
    
    
    
    
    
