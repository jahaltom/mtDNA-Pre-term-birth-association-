#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
import pandas as pd
from scipy import sparse

def logit(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return np.log(x / (1 - x))

def variant_id(pos, ref, alt):
    return f"chrM:{int(pos)}_{ref}>{alt}"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_parquet_dir", required=True)
    ap.add_argument("--calls_parquet_dir", required=True)
    ap.add_argument("--covariates_csv", required=True)
    ap.add_argument("--cov_sep", default=None)
    ap.add_argument("--sample_col", default="Sample_ID")
    ap.add_argument("--site_col", default="site")

    ap.add_argument("--low", type=float, default=0.01)
    ap.add_argument("--high", type=float, default=0.95)
    ap.add_argument("--min_dp", type=int, default=50)

    ap.add_argument("--min_n_used", type=int, default=5000)
    ap.add_argument("--min_carriers", type=int, default=10)
    ap.add_argument("--min_sites_with_carriers", type=int, default=1)
    ap.add_argument("--min_per_site_among_carrier_sites", type=int, default=1)
    ap.add_argument("--min_dose_sd", type=float, default=0.0)

    ap.add_argument("--dose_transform", choices=["raw_af","logit_af","rel_to_low_logit"], default="raw_af")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--write_sparse_npz", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # covariates
    if args.cov_sep is None:
        cov = pd.read_csv(args.covariates_csv, sep=None, engine="python", low_memory=False)
    else:
        cov = pd.read_csv(args.covariates_csv, sep=args.cov_sep, low_memory=False)
    cov[args.sample_col] = cov[args.sample_col].astype(str)
    cov[args.site_col] = cov[args.site_col].astype(str)

    sample_order = cov[args.sample_col].drop_duplicates().tolist()
    sample_index = pd.Index(sample_order, name="sample")

    # read depth parts
    dparts = sorted(glob.glob(os.path.join(args.depth_parquet_dir, "depth_part_*.parquet")))
    cparts = sorted(glob.glob(os.path.join(args.calls_parquet_dir, "calls_part_*.parquet")))
    if not dparts:
        raise SystemExit("No depth_part_*.parquet found")
    if not cparts:
        raise SystemExit("No calls_part_*.parquet found")

    depth = pd.concat([pd.read_parquet(p) for p in dparts], ignore_index=True)
    calls = pd.concat([pd.read_parquet(p) for p in cparts], ignore_index=True)

    depth["sample"] = depth["sample"].astype(str)
    depth["POS"] = pd.to_numeric(depth["POS"], errors="coerce")
    depth["DP"] = pd.to_numeric(depth["DP"], errors="coerce")

    calls["sample"] = calls["sample"].astype(str)
    calls["POS"] = pd.to_numeric(calls["POS"], errors="coerce")
    calls["AF"] = pd.to_numeric(calls["AF"], errors="coerce")
    calls = calls.dropna(subset=["POS","REF","ALT","AF"])

    # filter calls by AF threshold + HIGH
    calls = calls[(calls["AF"] >= args.low) & (calls["AF"] <= args.high)].copy()
    calls["variant"] = [variant_id(p, r, a) for p, r, a in zip(calls["POS"], calls["REF"], calls["ALT"])]

    # merge site into both
    depth = depth.merge(cov[[args.sample_col, args.site_col]], left_on="sample", right_on=args.sample_col, how="inner") \
                 .drop(columns=[args.sample_col]).rename(columns={args.site_col:"site"})
    calls = calls.merge(cov[[args.sample_col, args.site_col]], left_on="sample", right_on=args.sample_col, how="inner") \
                 .drop(columns=[args.sample_col]).rename(columns={args.site_col:"site"})

    depth["evaluable"] = depth["DP"] >= args.min_dp

    # evaluable counts by POS
    eval_by_pos = depth.groupby("POS")["evaluable"].sum().astype(int)
    sites_by_pos = depth[depth["evaluable"]].groupby("POS")["site"].unique()

    # QC per allele-specific variant
    qc_rows = []
    for v, g in calls.groupby("variant", sort=False):
        pos = int(g["POS"].iloc[0])
        n_used = int(eval_by_pos.get(pos, 0))
        n_carriers = int(g["sample"].nunique())
        prevalence = n_carriers / max(n_used, 1)

        car_by_site = g.groupby("site")["sample"].nunique()
        sites_in_model = sites_by_pos.get(pos, np.array([], dtype=object))
        car_by_site_full = pd.Series({s: int(car_by_site.get(s, 0)) for s in sites_in_model})
        sites_with_carriers = int((car_by_site_full > 0).sum()) if len(car_by_site_full) else 0

        if (car_by_site_full > 0).any():
            min_per_site_among_carrier_sites = int(car_by_site_full[car_by_site_full > 0].min())
        else:
            min_per_site_among_carrier_sites = 0

        af_vals = pd.to_numeric(g["AF"], errors="coerce").dropna().to_numpy(dtype=float)
        dose_sd = float(np.std(af_vals, ddof=1)) if len(af_vals) >= 2 else 0.0

        qc_rows.append({
            "variant": v,
            "POS": pos,
            "n_used": n_used,
            "n_carriers": n_carriers,
            "prevalence": prevalence,
            "sites_with_carriers": sites_with_carriers,
            "min_per_site_among_carrier_sites": min_per_site_among_carrier_sites,
            "dose_sd_carriers": dose_sd
        })

    qc = pd.DataFrame(qc_rows)
    qc_csv = args.out_prefix + ".variant_qc.csv"
    qc.to_csv(qc_csv, index=False)
    print(f"[write] {qc_csv}")
    print(f"Allele-specific variants observed: {len(qc):,}")

    if qc.empty:
        raise SystemExit("No variants found in calls parquet after filtering.")

    qc["pass_presence_qc"] = (
    (qc["n_used"] >= args.min_n_used) &
    (qc["n_carriers"] >= args.min_carriers) &
    (qc["sites_with_carriers"] >= args.min_sites_with_carriers) &
    (qc["min_per_site_among_carrier_sites"] >= args.min_per_site_among_carrier_sites)
)

    qc["pass_dose_qc"] = (
        qc["pass_presence_qc"] &
        (qc["dose_sd_carriers"] >= args.min_dose_sd)
    )

    passing_presence = qc.loc[qc["pass_presence_qc"], "variant"].tolist()
    passing_dose = qc.loc[qc["pass_dose_qc"], "variant"].tolist()

    print(f"Passing presence QC: {len(passing_presence):,}")
    print(f"Passing dose QC: {len(passing_dose):,}")

    if not passing_presence:
        raise SystemExit("No variants passed presence QC. Loosen thresholds and re-run.")

    qc_pass_presence = qc.loc[qc["pass_presence_qc"], ["variant","POS"]].copy()
    qc_pass_dose = qc.loc[qc["pass_dose_qc"], ["variant","POS"]].copy()

    pos_set_presence = sorted(qc_pass_presence["POS"].unique().tolist())

    # ---------------------------
    # Presence matrix inputs
    # ---------------------------
    depth_sub_presence = depth[depth["POS"].isin(pos_set_presence)].copy()
    depth_sub_presence = depth_sub_presence.merge(qc_pass_presence, on="POS", how="left")
    
    # base presence = 0 if evaluable else NaN
    depth_sub_presence["presence"] = np.where(depth_sub_presence["evaluable"].to_numpy(), 0.0, np.nan)
    
    # ---------------------------
    # Calls: collapse duplicate sample/variant rows first
    # ---------------------------
    calls2 = calls.groupby(["sample", "variant"], as_index=False).agg({
        "AF": "max",
        "POS": "first",
        "REF": "first",
        "ALT": "first",
        "site": "first"
    })
    
    # ---------------------------
    # Presence carriers
    # ---------------------------
    carriers_presence = calls2[calls2["variant"].isin(passing_presence)][["sample","variant","AF"]].copy()
    carriers_presence["presence_car"] = 1.0
    
    pres_long = depth_sub_presence[["sample","variant","presence"]].merge(
        carriers_presence[["sample","variant","presence_car"]],
        on=["sample","variant"],
        how="left"
    )
    pres_long["presence"] = np.where(pres_long["presence_car"].notna(), 1.0, pres_long["presence"])
    pres_long = pres_long[["sample","variant","presence"]]
    
    # ---------------------------
    # Dose carriers
    # ---------------------------
    carriers_dose = calls2[calls2["variant"].isin(passing_dose)][["sample","variant","AF"]].copy()
    
    af = carriers_dose["AF"].to_numpy(dtype=float)
    if args.dose_transform == "raw_af":
        dose_x = af
    elif args.dose_transform == "logit_af":
        dose_x = logit(af)
    else:
        dose_x = logit(af) - logit(args.low)
    
    dose_long = carriers_dose[["sample","variant"]].copy()
    dose_long["dose"] = dose_x

    pres = pres_long.pivot_table(index="sample", columns="variant", values="presence", aggfunc="max").reindex(sample_index)
    dose_m = dose_long.pivot_table(index="sample", columns="variant", values="dose", aggfunc="max").reindex(sample_index)

    # Keep sparse versions for optional NPZ export
    pres_sparse = pres.astype(pd.SparseDtype("float", np.nan))
    dose_sparse = dose_m.astype(pd.SparseDtype("float", np.nan))
    
    # Parquet cannot store pandas SparseDtype directly, so write dense float32
    pres_out = pres_sparse.sparse.to_dense().astype("float32")
    dose_out = dose_sparse.sparse.to_dense().astype("float32")
    
    pres_path = args.out_prefix + ".presence_matrix.parquet"
    dose_path = args.out_prefix + f".dose_matrix.{args.dose_transform}.parquet"
    pres_out.to_parquet(pres_path)
    dose_out.to_parquet(dose_path)
    print(f"[write] {pres_path} shape={pres_out.shape}")
    print(f"[write] {dose_path} shape={dose_out.shape}")
    print(f"[write] {pres_path} shape={pres.shape}")
    print(f"[write] {dose_path} shape={dose_m.shape}")

    if args.write_sparse_npz:
        pres_dense = pres_out
        eval_mask = (~pres_dense.isna()).astype(np.int8)
        car_mask = (pres_dense == 1).astype(np.int8)
        sparse.save_npz(args.out_prefix + ".presence_evaluable_mask.npz", sparse.csr_matrix(eval_mask.values))
        sparse.save_npz(args.out_prefix + ".presence_carrier_mask.npz", sparse.csr_matrix(car_mask.values))

        dose_dense = dose_out.fillna(0.0)
        sparse.save_npz(args.out_prefix + f".dose_values.{args.dose_transform}.npz", sparse.csr_matrix(dose_dense.values))

        pd.Series(pres_dense.index, name="sample").to_csv(args.out_prefix + ".samples.csv", index=False)
        pd.Series(pres_dense.columns, name="variant").to_csv(args.out_prefix + ".variants.csv", index=False)
        print(f"[write] sparse npz + labels with prefix {args.out_prefix}")

    print("Done.")

if __name__ == "__main__":
    main()
    
    
    
    
