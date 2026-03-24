#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def parse_args():
    ap = argparse.ArgumentParser(
        description="Variant-level two-part PTB association analysis from presence/dose matrices"
    )
    ap.add_argument("--covariates_csv", required=True)
    ap.add_argument("--cov_sep", default="\t")
    ap.add_argument("--presence_matrix", required=True)
    ap.add_argument("--dose_matrix", required=True)
    ap.add_argument("--sample_col", default="Sample_ID")
    ap.add_argument("--site_col", default="site")
    ap.add_argument("--outcome", default="PTB", help="Binary outcome coded 0/1")
    ap.add_argument("--covars", default="BMI,PW_AGE")
    ap.add_argument("--pcs", default="")
    ap.add_argument("--min_n_used", type=int, default=50)
    ap.add_argument("--min_carriers_dose", type=int, default=15)
    ap.add_argument("--min_cases", type=int, default=10)
    ap.add_argument("--min_controls", type=int, default=10)
    ap.add_argument("--results_csv", required=True)
    return ap.parse_args()


def safe_exp(x: float) -> float:
    return float(np.exp(x)) if np.isfinite(x) else np.nan


def append_skip_row(
    rows,
    variant,
    n_used,
    n_carriers,
    status,
    coef_present=np.nan,
    se_present=np.nan,
    p_present=np.nan,
    ci_present_low=np.nan,
    ci_present_high=np.nan,
    or_present=np.nan,
    or_ci_present_low=np.nan,
    or_ci_present_high=np.nan,
    coef_dose=np.nan,
    se_dose=np.nan,
    p_dose=np.nan,
    ci_dose_low=np.nan,
    ci_dose_high=np.nan,
    or_dose=np.nan,
    or_ci_dose_low=np.nan,
    or_ci_dose_high=np.nan,
):
    rows.append({
        "variant": variant,
        "coef_present": coef_present,
        "se_present": se_present,
        "p_present": p_present,
        "ci_present_low": ci_present_low,
        "ci_present_high": ci_present_high,
        "or_present": or_present,
        "or_ci_present_low": or_ci_present_low,
        "or_ci_present_high": or_ci_present_high,
        "coef_dose": coef_dose,
        "se_dose": se_dose,
        "p_dose": p_dose,
        "ci_dose_low": ci_dose_low,
        "ci_dose_high": ci_dose_high,
        "or_dose": or_dose,
        "or_ci_dose_low": or_ci_dose_low,
        "or_ci_dose_high": or_ci_dose_high,
        "n_used": int(n_used),
        "n_carriers": int(n_carriers),
        "status": status,
    })


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

    # Outcome must be binary 0/1
    cov[args.outcome] = pd.to_numeric(cov[args.outcome], errors="coerce")
    y_unique = set(pd.Series(cov[args.outcome].dropna()).unique())
    if not y_unique.issubset({0, 1}):
        raise ValueError(
            f"{args.outcome} must be coded 0/1 for PTB modeling. "
            f"Observed non-missing values include: {sorted(y_unique)}"
        )

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

    # Center covariates on actual shared analysis set
    for c in covars + pcs:
        cov[c + "_c"] = cov[c] - cov[c].mean()

    centered_covars = [c + "_c" for c in covars + pcs]

    print(f"Samples in covariates: {len(cov):,}")
    print(f"Presence matrix shape: {pres.shape}")
    print(f"Dose matrix shape: {dose.shape}")

    rows = []

    for v in pres.columns:
        # ---------------------------
        # Presence model
        # ---------------------------
        y = pd.to_numeric(cov[args.outcome], errors="coerce")
        p = pd.to_numeric(pres[v], errors="coerce")

        mask = y.notna() & p.notna()
        n_used_presence = int(mask.sum())
        n_carriers_presence = int((p == 1).sum())

        if n_used_presence < args.min_n_used or p[mask].nunique() < 2:
            append_skip_row(
                rows=rows,
                variant=v,
                n_used=n_used_presence,
                n_carriers=n_carriers_presence,
                status="skip_presence_low_n_or_no_variation",
            )
            continue

        X = pd.DataFrame({"present": p[mask].astype(float)}, index=cov.index[mask])
        for c in centered_covars:
            X[c] = pd.to_numeric(cov.loc[mask, c], errors="coerce")

        site_dummies = pd.get_dummies(
            cov.loc[mask, args.site_col],
            drop_first=True,
            prefix="site"
        ).astype(float)

        X = pd.concat([X, site_dummies], axis=1)
        X = sm.add_constant(X, has_constant="add").replace([np.inf, -np.inf], np.nan)

        valid = X.notna().all(axis=1) & y[mask].notna()
        Xv = X.loc[valid]
        yv = y.loc[mask].loc[valid]

        n_cases = int(yv.sum())
        n_controls = int(len(yv) - n_cases)

        if n_cases < args.min_cases or n_controls < args.min_controls:
            append_skip_row(
                rows=rows,
                variant=v,
                n_used=len(yv),
                n_carriers=n_carriers_presence,
                status="skip_presence_low_cases_or_controls",
            )
            continue

        try:
            res_p = sm.GLM(
                yv,
                Xv,
                family=sm.families.Binomial()
            ).fit(cov_type="HC3")

            coef_present = float(res_p.params.get("present", np.nan))
            se_present = float(res_p.bse.get("present", np.nan))
            p_present = float(res_p.pvalues.get("present", np.nan))

            if "present" in res_p.params.index:
                ci_present_low, ci_present_high = res_p.conf_int().loc["present"].tolist()
            else:
                ci_present_low, ci_present_high = np.nan, np.nan

            or_present = safe_exp(coef_present)
            or_ci_present_low = safe_exp(ci_present_low)
            or_ci_present_high = safe_exp(ci_present_high)

        except Exception as e:
            append_skip_row(
                rows=rows,
                variant=v,
                n_used=len(yv),
                n_carriers=n_carriers_presence,
                status=f"error_presence_{type(e).__name__}",
            )
            continue

        # ---------------------------
        # Dose model
        # ---------------------------
        if v not in dose.columns:
            append_skip_row(
                rows=rows,
                variant=v,
                n_used=n_used_presence,
                n_carriers=n_carriers_presence,
                status="ok_presence_only_not_in_dose_matrix",
                coef_present=coef_present,
                se_present=se_present,
                p_present=p_present,
                ci_present_low=ci_present_low,
                ci_present_high=ci_present_high,
                or_present=or_present,
                or_ci_present_low=or_ci_present_low,
                or_ci_present_high=or_ci_present_high,
            )
            continue

        d = pd.to_numeric(dose[v], errors="coerce")
        mask2 = y.notna() & d.notna()
        n_car = int(mask2.sum())

        if n_car < args.min_carriers_dose or np.nanstd(d[mask2].to_numpy()) == 0:
            append_skip_row(
                rows=rows,
                variant=v,
                n_used=n_used_presence,
                n_carriers=n_car,
                status="ok_presence_only_dose_low_n_or_no_variation",
                coef_present=coef_present,
                se_present=se_present,
                p_present=p_present,
                ci_present_low=ci_present_low,
                ci_present_high=ci_present_high,
                or_present=or_present,
                or_ci_present_low=or_ci_present_low,
                or_ci_present_high=or_ci_present_high,
            )
            continue

        d_centered = d.copy()
        d_centered.loc[mask2] = d_centered.loc[mask2] - d_centered.loc[mask2].mean()

        X2 = pd.DataFrame({"dose": d_centered[mask2].astype(float)}, index=cov.index[mask2])
        for c in centered_covars:
            X2[c] = pd.to_numeric(cov.loc[mask2, c], errors="coerce")

        site_dummies2 = pd.get_dummies(
            cov.loc[mask2, args.site_col],
            drop_first=True,
            prefix="site"
        ).astype(float)

        X2 = pd.concat([X2, site_dummies2], axis=1)
        X2 = sm.add_constant(X2, has_constant="add").replace([np.inf, -np.inf], np.nan)

        valid2 = X2.notna().all(axis=1) & y[mask2].notna()
        X2v = X2.loc[valid2]
        y2v = y.loc[mask2].loc[valid2]

        n_cases2 = int(y2v.sum())
        n_controls2 = int(len(y2v) - n_cases2)

        if n_cases2 < args.min_cases or n_controls2 < args.min_controls:
            append_skip_row(
                rows=rows,
                variant=v,
                n_used=n_used_presence,
                n_carriers=n_car,
                status="ok_presence_only_dose_low_cases_or_controls",
                coef_present=coef_present,
                se_present=se_present,
                p_present=p_present,
                ci_present_low=ci_present_low,
                ci_present_high=ci_present_high,
                or_present=or_present,
                or_ci_present_low=or_ci_present_low,
                or_ci_present_high=or_ci_present_high,
            )
            continue

        try:
            res_d = sm.GLM(
                y2v,
                X2v,
                family=sm.families.Binomial()
            ).fit(cov_type="HC3")

            coef_dose = float(res_d.params.get("dose", np.nan))
            se_dose = float(res_d.bse.get("dose", np.nan))
            p_dose = float(res_d.pvalues.get("dose", np.nan))

            if "dose" in res_d.params.index:
                ci_dose_low, ci_dose_high = res_d.conf_int().loc["dose"].tolist()
            else:
                ci_dose_low, ci_dose_high = np.nan, np.nan

            or_dose = safe_exp(coef_dose)
            or_ci_dose_low = safe_exp(ci_dose_low)
            or_ci_dose_high = safe_exp(ci_dose_high)

        except Exception as e:
            append_skip_row(
                rows=rows,
                variant=v,
                n_used=n_used_presence,
                n_carriers=n_car,
                status=f"error_dose_{type(e).__name__}",
                coef_present=coef_present,
                se_present=se_present,
                p_present=p_present,
                ci_present_low=ci_present_low,
                ci_present_high=ci_present_high,
                or_present=or_present,
                or_ci_present_low=or_ci_present_low,
                or_ci_present_high=or_ci_present_high,
            )
            continue

        rows.append({
            "variant": v,
            "coef_present": coef_present,
            "se_present": se_present,
            "p_present": p_present,
            "ci_present_low": ci_present_low,
            "ci_present_high": ci_present_high,
            "or_present": or_present,
            "or_ci_present_low": or_ci_present_low,
            "or_ci_present_high": or_ci_present_high,
            "coef_dose": coef_dose,
            "se_dose": se_dose,
            "p_dose": p_dose,
            "ci_dose_low": ci_dose_low,
            "ci_dose_high": ci_dose_high,
            "or_dose": or_dose,
            "or_ci_dose_low": or_ci_dose_low,
            "or_ci_dose_high": or_ci_dose_high,
            "n_used": n_used_presence,
            "n_carriers": n_car,
            "status": "ok",
        })

    res = pd.DataFrame(rows)

    pres_ok = res["p_present"].notna()
    dose_ok = res["p_dose"].notna()

    if pres_ok.any():
        res.loc[pres_ok, "fdr_present"] = multipletests(
            res.loc[pres_ok, "p_present"],
            method="fdr_bh"
        )[1]

    if dose_ok.any():
        res.loc[dose_ok, "fdr_dose"] = multipletests(
            res.loc[dose_ok, "p_dose"],
            method="fdr_bh"
        )[1]

    res.to_csv(args.results_csv, index=False)
    print(f"[write] {args.results_csv}")
    print(f"Variants attempted: {len(res)}")
    print(f"Presence tests run: {int(pres_ok.sum())}")
    print(f"Dose tests run: {int(dose_ok.sum())}")
    print(f"Significant presence FDR<0.05: {int((res.get('fdr_present', pd.Series(dtype=float)) < 0.05).sum())}")
    print(f"Significant dose     FDR<0.05: {int((res.get('fdr_dose', pd.Series(dtype=float)) < 0.05).sum())}")


if __name__ == "__main__":
    main()
