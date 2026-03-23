#!/usr/bin/env python3
import os, glob, argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", required=True)
    ap.add_argument("--out_depth_parquet", required=True)
    ap.add_argument("--out_calls_parquet", required=True)
    ap.add_argument("--batch", type=int, default=200, help="samples per write batch")
    return ap.parse_args()

def sample_from_depth(path: str) -> str:
    # fffc...e0917.chrM.sites.depth.txt -> fffc...e0917
    b = os.path.basename(path)
    s = b.replace(".sites.depth.txt", "")
    s = s.replace(".chrM", "")
    return s

def read_depth(path: str, sample: str) -> pd.DataFrame:
    # robust whitespace parsing
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["CHROM","POS","DP"],
        engine="python"
    )
    df["sample"] = sample
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce").astype("Int64")
    df["DP"] = pd.to_numeric(df["DP"], errors="coerce").astype("Int64")
    return df[["sample","POS","DP"]].dropna(subset=["POS"])

def read_calls(path: str, sample: str) -> pd.DataFrame:
    # CHROM POS REF ALT AF DP
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["CHROM","POS","REF","ALT","AF","DP_VCF"],
        engine="python"
    )
    df["sample"] = sample
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce").astype("Int64")
    df["DP_VCF"] = pd.to_numeric(df["DP_VCF"], errors="coerce").astype("Int64")

    # split multiallelic ALT/AF
    df["ALT_list"] = df["ALT"].astype(str).str.split(",")
    df["AF_list"]  = df["AF"].astype(str).str.split(",")

    # validate ALT/AF list lengths match
    df["n_alt"] = df["ALT_list"].str.len()
    df["n_af"] = df["AF_list"].str.len()

    bad = df["n_alt"] != df["n_af"]
    n_bad = int(bad.sum())

    if n_bad > 0:
        print(f"[warn] {sample}: dropping {n_bad} malformed call rows with ALT/AF length mismatch from {path}")

    df = df.loc[~bad].copy()

    if df.empty:
        return pd.DataFrame(columns=["sample","POS","REF","ALT","AF","DP_VCF"])

    df = df.explode(["ALT_list","AF_list"], ignore_index=True)

    df["ALT"] = df["ALT_list"].astype(str)
    df["AF"] = pd.to_numeric(df["AF_list"], errors="coerce")

    out = df[["sample","POS","REF","ALT","AF","DP_VCF"]].dropna(subset=["POS","ALT","AF"])
    return out



def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_depth_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_calls_parquet), exist_ok=True)

    depth_files = sorted(glob.glob(os.path.join(args.txt_dir, "*.sites.depth.txt")))
    if not depth_files:
        raise SystemExit(f"No *.sites.depth.txt in {args.txt_dir}")

    # Write to partitioned "parts" (most robust)
    depth_part_dir = args.out_depth_parquet
    calls_part_dir = args.out_calls_parquet
    os.makedirs(depth_part_dir, exist_ok=True)
    os.makedirs(calls_part_dir, exist_ok=True)

    depth_buf = []
    calls_buf = []
    part = 0

    for i, dpath in enumerate(depth_files, 1):
        sample = sample_from_depth(dpath)
        hpath = os.path.join(args.txt_dir, f"{sample}.chrM.heteroplasmy.txt")
        # some users might have ".chrM." already in basename; try fallback:
        if not os.path.exists(hpath):
            # try matching original stem in depth file
            stem = os.path.basename(dpath).replace(".sites.depth.txt","")
            hpath2 = os.path.join(args.txt_dir, f"{stem}.heteroplasmy.txt")
            hpath = hpath2

        depth_buf.append(read_depth(dpath, sample))

        if os.path.exists(hpath) and os.path.getsize(hpath) > 0:
            calls_buf.append(read_calls(hpath, sample))

        if (i % args.batch) == 0:
            part += 1
            depth_df = pd.concat(depth_buf, ignore_index=True)
            calls_df = pd.concat(calls_buf, ignore_index=True) if calls_buf else pd.DataFrame(
                columns=["sample","POS","REF","ALT","AF","DP_VCF"]
            )

            depth_out = os.path.join(depth_part_dir, f"depth_part_{part:04d}.parquet")
            calls_out = os.path.join(calls_part_dir, f"calls_part_{part:04d}.parquet")

            pq.write_table(pa.Table.from_pandas(depth_df, preserve_index=False), depth_out, compression="zstd")
            pq.write_table(pa.Table.from_pandas(calls_df, preserve_index=False), calls_out, compression="zstd")

            print(f"[write] {depth_out} rows={len(depth_df):,}")
            print(f"[write] {calls_out} rows={len(calls_df):,}")

            depth_buf, calls_buf = [], []

    if depth_buf:
        part += 1
        depth_df = pd.concat(depth_buf, ignore_index=True)
        calls_df = pd.concat(calls_buf, ignore_index=True) if calls_buf else pd.DataFrame(
            columns=["sample","POS","REF","ALT","AF","DP_VCF"]
        )

        depth_out = os.path.join(depth_part_dir, f"depth_part_{part:04d}.parquet")
        calls_out = os.path.join(calls_part_dir, f"calls_part_{part:04d}.parquet")

        pq.write_table(pa.Table.from_pandas(depth_df, preserve_index=False), depth_out, compression="zstd")
        pq.write_table(pa.Table.from_pandas(calls_df, preserve_index=False), calls_out, compression="zstd")

        print(f"[write] {depth_out} rows={len(depth_df):,}")
        print(f"[write] {calls_out} rows={len(calls_df):,}")

    print(f"Done. Parts written to:\n  depth: {depth_part_dir}\n  calls: {calls_part_dir}")
    print(f"[summary] malformed multiallelic rows dropped: {total_bad_rows}")

if __name__ == "__main__":
    main()
    
    
    
