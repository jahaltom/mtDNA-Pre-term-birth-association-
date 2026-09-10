import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "Final_GA_summary_table.csv"

BASE_MODEL = (
    "GA ~ Haplogroup + BABY_SEX_PW_AGE_MAT_HEIGHT_PC1_PC2_PC3"
)

MODEL_TYPE = "glmmTMB_gaussian"

PADJ_CUTOFF = 0.05
EFFECT_CHANGE_CUTOFF = 10.0   # percent


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# IDENTIFY BASE MODEL
# ============================================================

base = df[
    (df["model"] == MODEL_TYPE) &
    (df["model_equation"] == BASE_MODEL)
].copy()


# ============================================================
# KEEP ACTUAL HAPLOGROUP EFFECTS
# ============================================================

base_haps = base[
    base["hap_n_total"].notna()
].copy()


# ============================================================
# KEEP SIGNIFICANT BASE-MODEL HAPLOGROUPS
# ============================================================

sig_base = base_haps[
    base_haps["padj"].notna() &
    (base_haps["padj"] < PADJ_CUTOFF)
].copy()


if sig_base.empty:
    print("No haplogroup effects have padj <", PADJ_CUTOFF)
    raise SystemExit


# ============================================================
# COMPARE AGAINST ALL OTHER MODEL EQUATIONS
# ============================================================

results = []

for _, b in sig_base.iterrows():

    population = b["population"]
    hap = b["haplogroup"]
    ref = b["reference_haplogroup"]
    IDENT = b["IDENT"]

    base_effect = b["effect_days"]
    base_padj = b["padj"]
    base_aic = b["AIC"]
    base_bic = b["BIC"]

    comparisons = df[
        (df["model"] == MODEL_TYPE) &
        (df["population"] == population) &
        (df["haplogroup"] == hap) &
        (df["reference_haplogroup"] == ref) &
        (df["model_equation"] != BASE_MODEL) &
        (df["IDENT"] == IDENT)
    ].copy()

    for _, c in comparisons.iterrows():

        new_effect = c["effect_days"]

        # Percent change in GA effect
        if pd.isna(base_effect) or pd.isna(new_effect) or base_effect == 0:
            pct_effect_change = np.nan
        else:
            pct_effect_change = (
                abs(new_effect - base_effect) / abs(base_effect)
            ) * 100

        # Absolute change in days
        if pd.isna(base_effect) or pd.isna(new_effect):
            absolute_day_change = np.nan
        else:
            absolute_day_change = abs(new_effect - base_effect)

        # Direction of effect movement
        if pd.isna(new_effect) or pd.isna(base_effect):
            effect_direction = np.nan
        elif new_effect > base_effect:
            effect_direction = "Effect increased"
        elif new_effect < base_effect:
            effect_direction = "Effect decreased"
        else:
            effect_direction = "No change"

        # AIC/BIC differences
        delta_aic = (
            c["AIC"] - base_aic
            if pd.notna(c["AIC"]) and pd.notna(base_aic)
            else np.nan
        )

        delta_bic = (
            c["BIC"] - base_bic
            if pd.notna(c["BIC"]) and pd.notna(base_bic)
            else np.nan
        )

        results.append({

            "population": population,
            "haplogroup": hap,
            "reference_haplogroup": ref,
            "IDENT": IDENT,

            "base_model_equation": BASE_MODEL,
            "comparison_model_equation": c["model_equation"],

            "base_effect_days": base_effect,
            "comparison_effect_days": new_effect,

            "absolute_day_change": absolute_day_change,
            "effect_change_percent": pct_effect_change,

            "effect_change_direction": effect_direction,

            "effect_change_ge_10pct":
                pct_effect_change >= EFFECT_CHANGE_CUTOFF
                if pd.notna(pct_effect_change)
                else False,

            "base_padj": base_padj,
            "comparison_padj": c["padj"],

            "base_significant":
                base_padj < PADJ_CUTOFF
                if pd.notna(base_padj)
                else False,

            "comparison_significant":
                c["padj"] < PADJ_CUTOFF
                if pd.notna(c["padj"])
                else False,

            "base_AIC": base_aic,
            "comparison_AIC": c["AIC"],
            "delta_AIC": delta_aic,

            "base_BIC": base_bic,
            "comparison_BIC": c["BIC"],
            "delta_BIC": delta_bic,
        })


# ============================================================
# CREATE OUTPUT TABLE
# ============================================================

results = pd.DataFrame(results)

if results.empty:
    print("\nNo alternative models matched the significant base haplogroups.")
    raise SystemExit


# ============================================================
# ROUND
# ============================================================

round_cols = [
    "base_effect_days",
    "comparison_effect_days",
    "absolute_day_change",
    "effect_change_percent",
    "base_padj",
    "comparison_padj",
    "base_AIC",
    "comparison_AIC",
    "delta_AIC",
    "base_BIC",
    "comparison_BIC",
    "delta_BIC"
]

for col in round_cols:
    if col in results.columns:
        results[col] = results[col].round(4)


# ============================================================
# SORT
# ============================================================

results = results.sort_values(
    ["population", "haplogroup", "effect_change_percent"],
    ascending=[True, True, False]
)


# ============================================================
# SAVE EVERYTHING
# ============================================================

results.to_csv(
    "GA_base_model_effect_change_all.csv",
    index=False
)


# ============================================================
# SAVE ONLY >=10% EFFECT CHANGES
# ============================================================

flagged = results[
    results["effect_change_ge_10pct"]
].copy()

flagged.to_csv(
    "GA_base_model_effect_change_ge10pct.csv",
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    "\nSignificant haplogroup effects in base model:",
    len(sig_base)
)

print(
    "Alternative model comparisons:",
    len(results)
)

print(
    "Comparisons with >=10% effect change:",
    len(flagged)
)


if len(flagged) > 0:

    print("\n>=10% effect-change cases:\n")

    print(
        flagged[
            [
                "population",
                "haplogroup",
                "reference_haplogroup",
                "IDENT",
                "comparison_model_equation",
                "base_effect_days",
                "comparison_effect_days",
                "absolute_day_change",
                "effect_change_percent",
                "comparison_padj",
                "delta_AIC",
                "delta_BIC"
            ]
        ].to_string(index=False)
    )


print("\nSaved:")
print("  GA_base_model_effect_change_all.csv")
print("  GA_base_model_effect_change_ge10pct.csv")
