import os
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------
# Input files
# -----------------------
ga_brm_file = "ga_brm_posterior_probs.csv"
ga_tmb_file = "ga_glmmtmb_gaussian.csv"
ptb_brm_file = "ptb_brm_prior_sensitivity_haps.csv"
ptb_tmb_file = "ptb_glmmtmb.csv"
rate_file = "hap_site_ptb_table.csv"

# -----------------------
# Populations
# -----------------------
populations = [
    "African",
    "AMANHI-Pemba",
    "GAPPS-Zambia"
]
# populations = [
#     "AMANHI-Bangladesh",
#     "GAPPS-Bangladesh",
#     "BangladeshCombo",
#     "AMANHI-Pakistan",
#     "South_Asian"
# ]


GA = []
PTB = []

# -----------------------
# Helpers
# -----------------------
def direction_ga(x):
    if pd.isna(x):
        return np.nan
    return "longer GA" if x > 0 else "shorter GA"


def direction_ptb(or_value):
    if pd.isna(or_value):
        return np.nan
    return "higher PTB odds" if or_value > 1 else "lower PTB odds"


def prep_rate_table(rate_path, population_name):
    rate_df = pd.read_csv(rate_path).copy()

    rate_df = rate_df.rename(columns={
        "SubHap": "haplogroup",
        "MainHap": "haplogroup",
        "PhyloHap": "haplogroup",
        "site": "population"
    })

    if "n_ptb" not in rate_df.columns:
        rate_df["n_ptb"] = np.round(
            rate_df["n_total"] * rate_df["ptb_rate"]
        ).astype("Int64")

    if population_name == "African":
        rate_df = (
            rate_df
            .groupby("haplogroup", as_index=False)
            .agg(
                n_total=("n_total", "sum"),
                n_ptb=("n_ptb", "sum")
            )
        )
        rate_df["population"] = "African"
        rate_df["ptb_rate"] = rate_df["n_ptb"] / rate_df["n_total"]
        
    
    if population_name == "BangladeshCombo":
        rate_df = (
            rate_df
            .groupby("haplogroup", as_index=False)
            .agg(
                n_total=("n_total", "sum"),
                n_ptb=("n_ptb", "sum")
            )
        )
        rate_df["population"] = "BangladeshCombo"
        rate_df["ptb_rate"] = rate_df["n_ptb"] / rate_df["n_total"]
            
    if population_name == "South_Asian":
        rate_df = (
            rate_df
            .groupby("haplogroup", as_index=False)
            .agg(
                n_total=("n_total", "sum"),
                n_ptb=("n_ptb", "sum")
            )
        )
        rate_df["population"] = "South_Asian"
        rate_df["ptb_rate"] = rate_df["n_ptb"] / rate_df["n_total"]
        
                
                

    if "n_term" not in rate_df.columns:
        rate_df["n_term"] = rate_df["n_total"] - rate_df["n_ptb"]

    rate_df["ptb_percent"] = rate_df["ptb_rate"] * 100

    return rate_df[
        [
            "haplogroup",
            "population",
            "n_total",
            "n_ptb",
            "n_term",
            "ptb_rate",
            "ptb_percent"
        ]
    ].copy()


def empty_rate_table():
    return pd.DataFrame(columns=[
        "haplogroup",
        "population",
        "n_total",
        "n_ptb",
        "n_term",
        "ptb_rate",
        "ptb_percent"
    ])


def add_ptb_descriptives(final_ptb, rate_df):
    if rate_df.empty:
        for col in [
            "hap_n_total", "hap_n_ptb", "hap_n_term",
            "hap_ptb_rate", "hap_ptb_percent",
            "ref_n_total", "ref_n_ptb", "ref_n_term",
            "ref_ptb_rate", "ref_ptb_percent"
        ]:
            final_ptb[col] = np.nan
        final_ptb["descriptive_ptb"] = ""
        return final_ptb

    hap_rates = rate_df.rename(columns={
        "n_total": "hap_n_total",
        "n_ptb": "hap_n_ptb",
        "n_term": "hap_n_term",
        "ptb_rate": "hap_ptb_rate",
        "ptb_percent": "hap_ptb_percent"
    })

    ref_rates = rate_df.rename(columns={
        "haplogroup": "reference_haplogroup",
        "n_total": "ref_n_total",
        "n_ptb": "ref_n_ptb",
        "n_term": "ref_n_term",
        "ptb_rate": "ref_ptb_rate",
        "ptb_percent": "ref_ptb_percent"
    })

    final_ptb = final_ptb.merge(
        hap_rates,
        on=["haplogroup", "population"],
        how="left"
    )

    final_ptb = final_ptb.merge(
        ref_rates,
        on=["reference_haplogroup", "population"],
        how="left"
    )

    final_ptb["descriptive_ptb"] = np.where(
        final_ptb["hap_n_total"].notna() & final_ptb["ref_n_total"].notna(),
        final_ptb["haplogroup"] + ": " +
        final_ptb["hap_ptb_percent"].round(1).astype(str) + "% (n=" +
        final_ptb["hap_n_total"].astype("Int64").astype(str) + "); " +
        final_ptb["reference_haplogroup"] + ": " +
        final_ptb["ref_ptb_percent"].round(1).astype(str) + "% (n=" +
        final_ptb["ref_n_total"].astype("Int64").astype(str) + ")",
        ""
    )

    return final_ptb


# -----------------------
# Main loop
# -----------------------
for p in populations:

    model_root = Path(
        p + "/mtDNA-Pre-term-birth-association-/Final_Model/model_outputs"
    )

    if not model_root.exists():
        print(f"Skipping missing path: {model_root}")
        continue

    for file in os.listdir(model_root):
        run_path = model_root / file

        if not run_path.is_dir():
            continue

        # rate file is in SAME folder as model output files
        rate_path = run_path / rate_file

        if rate_path.exists():
            rate_df = prep_rate_table(rate_path, p)
        else:
            print(f"Warning: missing rate table: {rate_path}")
            rate_df = empty_rate_table()

        parts = file.split("_")
        reference_haplogroup = parts[0]
        covariates = "_".join(parts[1:])

        GA_EQUATION = f"GA ~ Haplogroup + {covariates}"
        PTB_EQUATION = f"PTB ~ Haplogroup + {covariates}"

        print("\nPopulation:", p)
        print("Run folder:", file)
        print("Reference:", reference_haplogroup)
        print("GA equation:", GA_EQUATION)
        print("PTB equation:", PTB_EQUATION)

        # -----------------------
        # GA table
        # -----------------------
        ga_brm_path = run_path / ga_brm_file
        ga_tmb_path = run_path / ga_tmb_file

        if ga_brm_path.exists() and ga_tmb_path.exists():

            ga_brm = pd.read_csv(ga_brm_path).copy()

            ga_brm_final = pd.DataFrame({
                "haplogroup": ga_brm["label"],
                "population": p,
                "reference_haplogroup": reference_haplogroup,
                "model_equation": GA_EQUATION,
                "model": "brms",
                "effect_days": ga_brm["beta_days"],
                "ci_low_days": ga_brm["lo_days"],
                "ci_high_days": ga_brm["hi_days"],
                "direction": ga_brm["beta_days"].apply(direction_ga),
                "Pr_longer_GA": ga_brm["Pr_beta_gt0"],
                "Pr_GA_gt_1_day_longer": ga_brm["Pr_days_gt_1"],
                "Pr_GA_gt_1_day_shorter": ga_brm["Pr_days_lt_m1"],
                "p_or_p_two": ga_brm["p_two"],
                "padj": ga_brm["padj_signprob"]
            })

            ga_tmb = pd.read_csv(ga_tmb_path).copy()

            ga_tmb_final = pd.DataFrame({
                "haplogroup": ga_tmb["label"],
                "population": p,
                "reference_haplogroup": reference_haplogroup,
                "model_equation": GA_EQUATION,
                "model": "glmmTMB",
                "effect_days": ga_tmb["estimate"],
                "ci_low_days": ga_tmb["conf.low"],
                "ci_high_days": ga_tmb["conf.high"],
                "direction": ga_tmb["estimate"].apply(direction_ga),
                "Pr_longer_GA": np.nan,
                "Pr_GA_gt_1_day_longer": np.nan,
                "Pr_GA_gt_1_day_shorter": np.nan,
                "p_or_p_two": ga_tmb["p.value"],
                "padj": ga_tmb["padj"],
                "AIC": ga_tmb["AIC"] if "AIC" in ga_tmb.columns else np.nan,
                "BIC": ga_tmb["BIC"] if "BIC" in ga_tmb.columns else np.nan,
                "logLik": ga_tmb["logLik"] if "logLik" in ga_tmb.columns else np.nan
            })

            final_ga = pd.concat(
                [ga_brm_final, ga_tmb_final],
                ignore_index=True
            )

            final_ga = final_ga.sort_values(["haplogroup", "model"])

            final_ga[final_ga.select_dtypes(include="number").columns] = (
                final_ga.select_dtypes(include="number").round(4)
            )

            GA.append(final_ga)

        else:
            print(f"Skipping GA files for: {run_path}")

        # -----------------------
        # PTB table
        # -----------------------
        ptb_brm_path = run_path / ptb_brm_file
        ptb_tmb_path = run_path / ptb_tmb_file

        if ptb_brm_path.exists() and ptb_tmb_path.exists():

            ptb_brm = pd.read_csv(ptb_brm_path).copy()

            ptb_brm_final = pd.DataFrame({
                "haplogroup": ptb_brm["label"],
                "population": p,
                "reference_haplogroup": reference_haplogroup,
                "model_equation": PTB_EQUATION,
                "model": "brms",
                "prior_setting": ptb_brm["prior_setting"],
                "OR": ptb_brm["OR"],
                "OR_low": ptb_brm["OR_lo"],
                "OR_high": ptb_brm["OR_hi"],
                "direction": ptb_brm["OR"].apply(direction_ptb),
                "Pr_higher_PTB_odds": ptb_brm["Pr_OR_gt_1"],
                "p_or_p_two": ptb_brm["p_two"],
                "padj": ptb_brm["padj"]
            })

            ptb_tmb = pd.read_csv(ptb_tmb_path).copy()

            ptb_tmb_final = pd.DataFrame({
                "haplogroup": ptb_tmb["label"],
                "population": p,
                "reference_haplogroup": reference_haplogroup,
                "model_equation": PTB_EQUATION,
                "model": "glmmTMB",
                "prior_setting": np.nan,
                "OR": ptb_tmb["OR"],
                "OR_low": ptb_tmb["OR_low"],
                "OR_high": ptb_tmb["OR_hi"],
                "direction": ptb_tmb["OR"].apply(direction_ptb),
                "Pr_higher_PTB_odds": np.nan,
                "p_or_p_two": ptb_tmb["p.value"],
                "padj": ptb_tmb["padj"],
                "AIC": ptb_tmb["AIC"] if "AIC" in ptb_tmb.columns else np.nan,
                "BIC": ptb_tmb["BIC"] if "BIC" in ptb_tmb.columns else np.nan,
                "logLik": ptb_tmb["logLik"] if "logLik" in ptb_tmb.columns else np.nan
            })

            final_ptb = pd.concat(
                [ptb_brm_final, ptb_tmb_final],
                ignore_index=True
            )

            final_ptb = add_ptb_descriptives(final_ptb, rate_df)

            final_ptb = final_ptb.sort_values(
                ["haplogroup", "model", "prior_setting"],
                na_position="last"
            )

            final_ptb[final_ptb.select_dtypes(include="number").columns] = (
                final_ptb.select_dtypes(include="number").round(4)
            )

            PTB.append(final_ptb)

        else:
            print(f"Skipping PTB files for: {run_path}")


# -----------------------
# Save final outputs
# -----------------------
if PTB:
    final_ptb = pd.concat(PTB, ignore_index=True)
    final_ptb.to_csv("Final_PTB_summary_table.csv", index=False)
    print("\nSaved: Final_PTB_summary_table.csv")
else:
    print("\nNo PTB tables found.")

if GA:
    final_ga = pd.concat(GA, ignore_index=True)
    final_ga.to_csv("Final_GA_summary_table.csv", index=False)
    print("Saved: Final_GA_summary_table.csv")
else:
    print("No GA tables found.")
