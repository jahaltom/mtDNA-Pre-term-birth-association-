import os

path = r"C:\Users\haltomj\OneDrive - Children's Hospital of Philadelphia\Documents"
os.chdir(path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


########################################################
# INPUT
########################################################

INPUT_FILE = "Pemba.Sub.csv"
REFERENCE = "L2a"

OUTPUT_DIR = Path("Publication_Tables")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)


########################################################
# SELECT REFERENCE MODEL
########################################################

dat = df[
    (df["reference_haplogroup"] == REFERENCE) &
    (~df["haplogroup"].isin(["Intercept", "(Intercept)"]))
].copy()


########################################################
# IDENTIFY HAPLOGROUPS VS COVARIATES
########################################################

dat["Type"] = np.where(
    dat["hap_n_total"].notna(),
    "Haplogroup",
    "Covariate"
)


########################################################
# POPULATION NAME
########################################################

population = dat["population"].dropna().iloc[0]


########################################################
# FORMAT FUNCTIONS
########################################################

def fmt_num(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_p(x):
    if pd.isna(x):
        return ""
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def fmt_ci(low, high):
    if pd.isna(low) or pd.isna(high):
        return ""
    return f"{low:.2f}–{high:.2f}"


########################################################
# FREQUENTIST RESULTS
########################################################

freq = dat[
    dat["model"] == "glmmTMB"
].copy()

freq_table = pd.DataFrame({
    "Population": freq["population"],
    "Variable": freq["haplogroup"],

    "Freq_OR": freq["OR"].apply(
        lambda x: fmt_num(x, 2)
    ),

    "Freq_CI": freq.apply(
        lambda x: fmt_ci(
            x["OR_low"],
            x["OR_high"]
        ),
        axis=1
    ),

    "Freq_p": freq["p_or_p_two"].apply(fmt_p),

    "Freq_padj": freq["padj"].apply(fmt_p)
})


########################################################
# BAYESIAN RESULTS
########################################################

bayes = dat[
    dat["model"] == "brms"
].copy()

bayes_table = pd.DataFrame({
    "Population": bayes["population"],
    "Variable": bayes["haplogroup"],

    "Bayes_OR": bayes["OR"].apply(
        lambda x: fmt_num(x, 2)
    ),

    "Bayes_CrI": bayes.apply(
        lambda x: fmt_ci(
            x["OR_low"],
            x["OR_high"]
        ),
        axis=1
    ),

    "Pr_OR_gt_1": bayes[
        "Pr_higher_PTB_odds"
    ].apply(
        lambda x: ""
        if pd.isna(x)
        else f"{x:.3f}"
    )
})


########################################################
# VARIABLE INFORMATION
########################################################

variables = (
    dat[
        [
            "population",
            "haplogroup",
            "Type"
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "population": "Population",
            "haplogroup": "Variable"
        }
    )
)


########################################################
# DESCRIPTIVE INFORMATION FOR HAPLOGROUPS
########################################################

descriptive = (
    dat[
        dat["Type"] == "Haplogroup"
    ][
        [
            "population",
            "haplogroup",
            "hap_n_total",
            "hap_n_ptb",
            "hap_ptb_percent"
        ]
    ]
    .drop_duplicates()
    .copy()
)

descriptive["n"] = descriptive[
    "hap_n_total"
].apply(
    lambda x: ""
    if pd.isna(x)
    else str(int(x))
)

descriptive["PTB"] = descriptive.apply(
    lambda x:
        ""
        if pd.isna(x["hap_n_ptb"])
        else (
            f'{int(x["hap_n_ptb"])} '
            f'({x["hap_ptb_percent"]:.2f})'
        ),
    axis=1
)

descriptive = descriptive.rename(
    columns={
        "population": "Population",
        "haplogroup": "Variable"
    }
)

descriptive = descriptive[
    [
        "Population",
        "Variable",
        "n",
        "PTB"
    ]
]


########################################################
# MERGE EVERYTHING
########################################################

results = (
    variables
    .merge(
        freq_table,
        on=["Population", "Variable"],
        how="left"
    )
    .merge(
        bayes_table,
        on=["Population", "Variable"],
        how="left"
    )
    .merge(
        descriptive,
        on=["Population", "Variable"],
        how="left"
    )
)


########################################################
# ADD REFERENCE HAPLOGROUP
########################################################

ref_info = df[
    (df["reference_haplogroup"] == REFERENCE) &
    (df["ref_n_total"].notna())
].iloc[0]

reference_row = pd.DataFrame({
    "Population": [ref_info["population"]],
    "Variable": [REFERENCE],
    "Type": ["Haplogroup"],

    "Freq_OR": ["1 (ref)"],
    "Freq_CI": [""],
    "Freq_p": [""],
    "Freq_padj": [""],

    "Bayes_OR": ["1 (ref)"],
    "Bayes_CrI": [""],
    "Pr_OR_gt_1": [""],

    "n": [
        str(int(ref_info["ref_n_total"]))
    ],

    "PTB": [
        (
            f'{int(ref_info["ref_n_ptb"])} '
            f'({ref_info["ref_ptb_percent"]:.2f})'
        )
    ]
})


########################################################
# REMOVE REFERENCE IF DUPLICATED
########################################################

results = results[
    results["Variable"] != REFERENCE
].copy()


########################################################
# COMBINE
########################################################

final_table = pd.concat(
    [
        reference_row,
        results
    ],
    ignore_index=True
)


########################################################
# CLEAN COVARIATE NAMES
########################################################

pretty_names = {
    "BABY_SEX1": "Infant sex",
    "PW_AGE": "Maternal age",
    "MAT_HEIGHT": "Maternal height",
    "PC1": "nDNA PC1",
    "PC2": "nDNA PC2",
    "PC3": "nDNA PC3"
}

final_table["Variable"] = (
    final_table["Variable"]
    .replace(pretty_names)
)


########################################################
# BLANK DESCRIPTIVES FOR COVARIATES
########################################################

final_table.loc[
    final_table["Type"] == "Covariate",
    ["n", "PTB"]
] = ""


########################################################
# ORDER ROWS
########################################################

hap_rows = final_table[
    final_table["Type"] == "Haplogroup"
].copy()

cov_rows = final_table[
    final_table["Type"] == "Covariate"
].copy()

hap_order = [
    REFERENCE
] + sorted(
    [
        x for x in hap_rows["Variable"].unique()
        if x != REFERENCE
    ]
)

hap_rows["Variable"] = pd.Categorical(
    hap_rows["Variable"],
    categories=hap_order,
    ordered=True
)

hap_rows = hap_rows.sort_values(
    "Variable"
)

final_table = pd.concat(
    [
        hap_rows,
        cov_rows
    ],
    ignore_index=True
)


########################################################
# SELECT DISPLAY COLUMNS
########################################################

display_df = final_table[
    [
        "Variable",
        "n",
        "PTB",
        "Freq_OR",
        "Freq_CI",
        "Freq_p",
        "Freq_padj",
        "Bayes_OR",
        "Bayes_CrI",
        "Pr_OR_gt_1"
    ]
].copy()

display_df.columns = [
    "Variable",
    "n",
    "PTB, n (%)",
    "OR",
    "95% CI",
    "p",
    "p-adj",
    "OR",
    "95% CrI",
    "P(OR > 1)"
]


########################################################
# INSERT SECTION HEADER FOR COVARIATES
########################################################

n_haps = len(hap_rows)

section_row = pd.DataFrame(
    [[
        "Covariates",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    ]],
    columns=display_df.columns
)

display_df = pd.concat(
    [
        display_df.iloc[:n_haps],
        section_row,
        display_df.iloc[n_haps:]
    ],
    ignore_index=True
)


########################################################
# CREATE MATPLOTLIB TABLE
########################################################

n_rows = len(display_df)

fig_height = max(
    5,
    0.42 * n_rows + 2.8
)

fig, ax = plt.subplots(
    figsize=(14, fig_height)
)

ax.axis("off")


########################################################
# TITLE
########################################################

fig.text(
    0.5,
    0.965,
    "Association of maternal mtDNA haplogroup with preterm birth",
    ha="center",
    va="top",
    fontsize=14,
    fontweight="bold"
)

fig.text(
    0.5,
    0.93,
    f"{population}; reference haplogroup = {REFERENCE}",
    ha="center",
    va="top",
    fontsize=10
)


########################################################
# SPANNER HEADERS
########################################################

fig.text(
    0.48,
    0.885,
    "Frequentist (glmmTMB)",
    ha="center",
    fontsize=11,
    fontweight="bold"
)

fig.text(
    0.80,
    0.885,
    "Bayesian (brms)",
    ha="center",
    fontsize=11,
    fontweight="bold"
)


########################################################
# TABLE
########################################################

col_widths = [
    0.16,
    0.06,
    0.09,
    0.07,
    0.10,
    0.06,
    0.06,
    0.07,
    0.10,
    0.09
]

table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    cellLoc="center",
    colLoc="center",
    colWidths=col_widths,
    loc="center"
)


########################################################
# BASE FORMATTING
########################################################

table.auto_set_font_size(False)
table.set_fontsize(9)

table.scale(
    1,
    1.45
)


########################################################
# FORMAT CELLS
########################################################

for (row, col), cell in table.get_celld().items():

    cell.set_linewidth(0)

    # Header row
    if row == 0:
        cell.set_text_props(
            fontweight="bold"
        )
        cell.set_linewidth(0.8)

    # Left-align variable names
    if col == 0:
        cell.get_text().set_ha("left")


########################################################
# COVARIATE SECTION ROW
########################################################

cov_section_index = n_haps + 1

for col in range(len(display_df.columns)):

    cell = table[
        cov_section_index,
        col
    ]

    cell.set_text_props(
        fontweight="bold"
    )

    cell.set_linewidth(0.8)

    if col != 0:
        cell.get_text().set_text("")


########################################################
# BOLD REFERENCE ROW
########################################################

reference_index = 1

for col in range(len(display_df.columns)):

    table[
        reference_index,
        col
    ].set_text_props(
        fontweight="bold"
    )


########################################################
# OPTIONAL: BOLD SIGNIFICANT FREQUENTIST RESULTS
########################################################

for i, row in final_table.iterrows():

    # +1 because matplotlib table includes header row
    table_row = i + 1

    # Account for inserted Covariates section row
    if i >= n_haps:
        table_row += 1

    pval = row.get("Freq_p", "")

    if isinstance(pval, str):

        if pval == "<0.001":
            significant = True
        else:
            try:
                significant = float(pval) < 0.05
            except:
                significant = False

        if significant:

            for col in [3, 4, 5]:

                table[
                    table_row,
                    col
                ].set_text_props(
                    fontweight="bold"
                )

########################################################
# BOLD SIGNIFICANT / STRONG RESULTS
########################################################

for i, row in final_table.iterrows():

    # +1 because matplotlib table has a header row
    table_row = i + 1

    # Account for inserted Covariates section row
    if i >= n_haps:
        table_row += 1


    ####################################################
    # FREQUENTIST: BOLD p-adj IF SIGNIFICANT
    ####################################################

    padj = row.get("Freq_padj", "")

    freq_sig = False

    if isinstance(padj, str):

        if padj == "<0.001":
            freq_sig = True

        elif padj != "":
            try:
                freq_sig = float(padj) < 0.05
            except:
                freq_sig = False


    if freq_sig:

        # Bold adjusted p-value only
        table[
            table_row,
            6
        ].set_text_props(
            fontweight="bold"
        )


    ####################################################
    # BAYESIAN SIGNIFICANCE / STRONG EVIDENCE
    ####################################################

    pr = row.get("Pr_OR_gt_1", "")

    bayes_prob_sig = False

    if isinstance(pr, str) and pr != "":
        try:
            pr_value = float(pr)

            bayes_prob_sig = (
                pr_value > 0.95
                or
                pr_value < 0.05
            )

        except:
            bayes_prob_sig = False


    ####################################################
    # CHECK WHETHER CrI EXCLUDES 1
    ####################################################

    cri_excludes_1 = False

    cri = row.get("Bayes_CrI", "")

    if isinstance(cri, str) and cri != "":

        try:
            # Handles strings like:
            # 1.16–4.22
            # or 1.16-4.22

            cri_clean = cri.replace("–", "-")

            low, high = cri_clean.split("-")

            low = float(low.strip())
            high = float(high.strip())

            cri_excludes_1 = (
                high < 1
                or
                low > 1
            )

        except:
            cri_excludes_1 = False


    ####################################################
    # BOLD BAYESIAN OR + CrI + PROBABILITY
    ####################################################

    bayes_sig = (
        bayes_prob_sig
        or
        cri_excludes_1
    )

    if bayes_sig:

        # Bayesian OR
        table[
            table_row,
            7
        ].set_text_props(
            fontweight="bold"
        )

        # Bayesian 95% CrI
        table[
            table_row,
            8
        ].set_text_props(
            fontweight="bold"
        )

        # P(OR > 1)
        table[
            table_row,
            9
        ].set_text_props(
            fontweight="bold"
        )
########################################################
# FOOTNOTE
########################################################

fig.text(
    0.05,
    0.025,
    (
        "OR, odds ratio; CI, confidence interval; "
        "CrI, credible interval; PTB, preterm birth. "
        f"Reference haplogroup: {REFERENCE}."
    ),
    ha="left",
    fontsize=8
)


########################################################
# SAVE
########################################################

plt.tight_layout(
    rect=[
        0.02,
        0.05,
        0.98,
        0.87
    ]
)

png_file = (
    OUTPUT_DIR /
    f"Pemba_{REFERENCE}_publication_table.png"
)

pdf_file = (
    OUTPUT_DIR /
    f"Pemba_{REFERENCE}_publication_table.pdf"
)

plt.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.show()


########################################################
# SAVE CLEAN DATA TABLE
########################################################

csv_file = (
    OUTPUT_DIR /
    f"Pemba_{REFERENCE}_publication_table.csv"
)

final_table.to_csv(
    csv_file,
    index=False
)

print(f"Saved: {png_file}")
print(f"Saved: {pdf_file}")
print(f"Saved: {csv_file}")
