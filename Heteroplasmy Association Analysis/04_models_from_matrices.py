#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covariates_csv", required=True)
    ap.add_argument("--cov_sep", default="\t")
    ap.add_argument("--presence_matrix", required=True)
    ap.add_argument("--dose_matrix", required=True)
    ap.add_argument("--sample_col", default="Sample_ID")
    ap.add_argument("--site_col", default="site")
    ap.add_argument("--outcome", default="GAGEBRTH")
    ap.add_argument("--covars", default="BMI,PW_AGE")
    ap.add_argument("--pcs", default="")
    ap.add_argument("--min_n_used", type=int, default=50)
    ap.add_argument("--min_carriers_dose", type=int, default=15)
    ap.add_argument("--results_csv", required=True)
    return ap.parse_args()

def main():
    args = parse_args()

    cov = pd.read_csv(args.covariates_csv, sep=args.cov_sep, low_memory=False)
    cov[args.sample_col] = cov[args.sample_col].astype(str)
    cov[args.site_col] = cov[args.site_col].astype(str)

    covars = [c.strip() for c in args.covars.split(",") if c.strip()]
    pcs = [c.strip() for c in args.pcs.split(",") if c.strip()]
    needed = [args.sample_col, args.site_col, args.outcome] + covars + pcs
    missing = [c for c in needed if c not in cov.columns]
    if missing:
        raise ValueError(f"Missing columns in covariates file: {missing}")

    for c in covars + pcs:
        cov[c] = pd.to_numeric(cov[c], errors="coerce")

    cov = cov.set_index(args.sample_col)

    pres = pd.read_parquet(args.presence_matrix)
    dose = pd.read_parquet(args.dose_matrix)

    pres.index = pres.index.astype(str)
    dose.index = dose.index.astype(str)

    common = pres.index.intersection(cov.index)
    pres = pres.loc[common]
    dose = dose.loc[dose.index.intersection(common)]
    cov = cov.loc[common].copy()

    for c in covars + pcs:
        cov[c + "_c"] = cov[c] - cov[c].mean()

    centered_covars = [c + "_c" for c in covars + pcs]

    rows = []

    for v in pres.columns:
        # ---------------------------
        # Presence model
        # ---------------------------
        y = pd.to_numeric(cov[args.outcome], errors="coerce")
        p = pd.to_numeric(pres[v], errors="coerce")

        mask = y.notna() & p.notna()
        if mask.sum() < args.min_n_used or p[mask].nunique() < 2:
            rows.append({
                "variant": v,
                "coef_present": np.nan,
                "se_present": np.nan,
                "p_present": np.nan,
                "ci_present_low": np.nan,
                "ci_present_high": np.nan,
                "coef_dose": np.nan,
                "se_dose": np.nan,
                "p_dose": np.nan,
                "ci_dose_low": np.nan,
                "ci_dose_high": np.nan,
                "n_used": int(mask.sum()),
                "n_carriers": int((p == 1).sum()),
                "status": "skip_presence_low_n_or_no_variation"
            })
            continue

        X = pd.DataFrame({"present": p[mask].astype(float)}, index=cov.index[mask])
        for c in centered_covars:
            X[c] = pd.to_numeric(cov.loc[mask, c], errors="coerce")

        site_dummies = pd.get_dummies(cov.loc[mask, args.site_col], drop_first=True, prefix="site").astype(float)
        X = pd.concat([X, site_dummies], axis=1)
        X = sm.add_constant(X, has_constant="add").replace([np.inf, -np.inf], np.nan)

        valid = X.notna().all(axis=1) & y[mask].notna()
        Xv = X.loc[valid]
        yv = y.loc[mask].loc[valid]


        try:
            res_p = sm.OLS(yv, Xv).fit(cov_type="HC3")
            coef_present = float(res_p.params.get("present", np.nan))
            se_present = float(res_p.bse.get("present", np.nan))
            p_present = float(res_p.pvalues.get("present", np.nan))
            ci_present_low, ci_present_high = res_p.conf_int().loc["present"].tolist() if "present" in res_p.params.index else (np.nan, np.nan)
        except Exception as e:
            rows.append({
                "variant": v,
                "coef_present": np.nan,
                "se_present": np.nan,
                "p_present": np.nan,
                "ci_present_low": np.nan,
                "ci_present_high": np.nan,
                "coef_dose": np.nan,
                "se_dose": np.nan,
                "p_dose": np.nan,
                "ci_dose_low": np.nan,
                "ci_dose_high": np.nan,
                "n_used": int(len(yv)),
                "n_carriers": int((p == 1).sum()),
                "status": f"error_presence_{type(e).__name__}"
            })
            continue

        # ---------------------------
        # Dose model
        # ---------------------------
        if v not in dose.columns:
            rows.append({
                "variant": v,
                "coef_present": coef_present,
                "se_present": se_present,
                "p_present": p_present,
                "ci_present_low": ci_present_low,
                "ci_present_high": ci_present_high,
                "coef_dose": np.nan,
                "se_dose": np.nan,
                "p_dose": np.nan,
                "ci_dose_low": np.nan,
                "ci_dose_high": np.nan,
                "n_used": int(mask.sum()),
                "n_carriers": int((p == 1).sum()),
                "status": "ok_presence_only_not_in_dose_matrix"
            })
            continue

        d = pd.to_numeric(dose[v], errors="coerce")
        mask2 = y.notna() & d.notna()
        n_car = int(mask2.sum())

        if n_car < args.min_carriers_dose or np.nanstd(d[mask2].to_numpy()) == 0:
            rows.append({
                "variant": v,
                "coef_present": coef_present,
                "se_present": se_present,
                "p_present": p_present,
                "ci_present_low": ci_present_low,
                "ci_present_high": ci_present_high,
                "coef_dose": np.nan,
                "se_dose": np.nan,
                "p_dose": np.nan,
                "ci_dose_low": np.nan,
                "ci_dose_high": np.nan,
                "n_used": int(mask.sum()),
                "n_carriers": n_car,
                "status": "ok_presence_only_dose_low_n_or_no_variation"
            })
            continue

        d_centered = d.copy()
        d_centered.loc[mask2] = d_centered.loc[mask2] - d_centered.loc[mask2].mean()

        X2 = pd.DataFrame({"dose": d_centered[mask2].astype(float)}, index=cov.index[mask2])
        for c in centered_covars:
            X2[c] = pd.to_numeric(cov.loc[mask2, c], errors="coerce")

        site_dummies2 = pd.get_dummies(cov.loc[mask2, args.site_col], drop_first=True, prefix="site").astype(float)
        X2 = pd.concat([X2, site_dummies2], axis=1)
        X2 = sm.add_constant(X2, has_constant="add").replace([np.inf, -np.inf], np.nan)

        valid2 = X2.notna().all(axis=1) & y[mask2].notna()
        X2v = X2.loc[valid2]
        y2v = y.loc[mask2].loc[valid2]


        try:
            res_d = sm.OLS(y2v, X2v).fit(cov_type="HC3")
            coef_dose = float(res_d.params.get("dose", np.nan))
            se_dose = float(res_d.bse.get("dose", np.nan))
            p_dose = float(res_d.pvalues.get("dose", np.nan))
            ci_dose_low, ci_dose_high = res_d.conf_int().loc["dose"].tolist() if "dose" in res_d.params.index else (np.nan, np.nan)
        except Exception as e:
            rows.append({
                "variant": v,
                "coef_present": coef_present,
                "se_present": se_present,
                "p_present": p_present,
                "ci_present_low": ci_present_low,
                "ci_present_high": ci_present_high,
                "coef_dose": np.nan,
                "se_dose": np.nan,
                "p_dose": np.nan,
                "ci_dose_low": np.nan,
                "ci_dose_high": np.nan,
                "n_used": int(mask.sum()),
                "n_carriers": n_car,
                "status": f"error_dose_{type(e).__name__}"
            })
            continue

        rows.append({
            "variant": v,
            "coef_present": coef_present,
            "se_present": se_present,
            "p_present": p_present,
            "ci_present_low": ci_present_low,
            "ci_present_high": ci_present_high,
            "coef_dose": coef_dose,
            "se_dose": se_dose,
            "p_dose": p_dose,
            "ci_dose_low": ci_dose_low,
            "ci_dose_high": ci_dose_high,
            "n_used": int(mask.sum()),
            "n_carriers": n_car,
            "status": "ok"
        })

    res = pd.DataFrame(rows)

    pres_ok = res["p_present"].notna()
    dose_ok = res["p_dose"].notna()

    if pres_ok.any():
        res.loc[pres_ok, "fdr_present"] = multipletests(res.loc[pres_ok, "p_present"], method="fdr_bh")[1]
    if dose_ok.any():
        res.loc[dose_ok, "fdr_dose"] = multipletests(res.loc[dose_ok, "p_dose"], method="fdr_bh")[1]

    res.to_csv(args.results_csv, index=False)
    print(f"[write] {args.results_csv}")

    print(f"Variants attempted: {len(res)}")
    print(f"Presence tests run: {pres_ok.sum()}")
    print(f"Dose tests run: {dose_ok.sum()}")
    print(f"Significant presence FDR<0.05: {(res.get('fdr_present', pd.Series(dtype=float)) < 0.05).sum()}")
    print(f"Significant dose     FDR<0.05: {(res.get('fdr_dose', pd.Series(dtype=float)) < 0.05).sum()}")

if __name__ == "__main__":
    main()
    
    
    
    


  
