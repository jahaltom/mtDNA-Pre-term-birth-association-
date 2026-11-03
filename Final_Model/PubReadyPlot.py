import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

path = r"C:\Users\haltomj\OneDrive - Children's Hospital of Philadelphia\Documents"
os.chdir(path)


# ============================================================
# USER SETTING
# ============================================================
#
# Choose:
#   "MainHap"
#   "SubHap"
#
# ============================================================

ANALYSIS = "SubHap"


# ============================================================
# ANALYSIS-SPECIFIC CONFIGURATION
# ============================================================

if ANALYSIS == "MainHap":

    INFILE = Path("Final_PTB_summary_table.Main.csv")

    OUT_PNG = Path("Figure_Selected_MainHap_PTB.png")
    OUT_PDF = Path("Figure_Selected_MainHap_PTB.pdf")
    OUT_CSV = Path("Figure_Selected_MainHap_PTB_source.csv")

    ORDER = [
        # L2 vs L3
        ("L2", "African",       "L3"),
        ("L2", "GAPPS-Zambia",  "L3"),
        ("L2", "AMANHI-Pemba",  "L3"),
        ("L2", "IPOP",          "L3"),
        ("L2", "ZAPPS1",        "L3"),
        ("L2", "ZAPPS2",        "L3"),

        # L2 vs L0
        ("L2", "African",       "L0"),
        ("L2", "GAPPS-Zambia",  "L0"),
        ("L2", "AMANHI-Pemba",  "L0"),
        ("L2", "IPOP",          "L0"),
        ("L2", "ZAPPS1",        "L0"),
        ("L2", "ZAPPS2",        "L0"),

        # M
        ("M", "AMANHI-Pemba", "L3"),
        ("M", "AMANHI-Pemba", "L0"),
    ]

    GROUP_LABELS = [
        ("L2 vs L3", 0, 5),
        ("L2 vs L0", 6, 11),
        ("M vs L3", 12, 12),
        ("M vs L0", 13, 13),
    ]

    FIGURE_TITLE = (
        "L2 and M PTB associations across African populations\n"
        "glmmTMB models; L3 and L0 references"
    )

    # Extra visual gap before small/single-row groups
    EXTRA_GROUP_GAP = 0.70


elif ANALYSIS == "SubHap":

    INFILE = Path("Final_PTB_summary_table.Sub.csv")

    OUT_PNG = Path("Figure_Selected_SubHap_PTB.png")
    OUT_PDF = Path("Figure_Selected_SubHap_PTB.pdf")
    OUT_CSV = Path("Figure_Selected_SubHap_PTB_source.csv")

    ORDER = [
        # L0a vs L2a
        ("L0a", "African",       "L2a"),
        ("L0a", "GAPPS-Zambia",  "L2a"),
        ("L0a", "AMANHI-Pemba",  "L2a"),
        ("L0a", "IPOP",          "L2a"),
        ("L0a", "ZAPPS1",        "L2a"),
        ("L0a", "ZAPPS2",        "L2a"),

        # L3e vs L2a
        ("L3e", "African",       "L2a"),
        ("L3e", "GAPPS-Zambia",  "L2a"),
        ("L3e", "AMANHI-Pemba",  "L2a"),
        ("L3e", "IPOP",          "L2a"),
        ("L3e", "ZAPPS1",        "L2a"),
        ("L3e", "ZAPPS2",        "L2a"),
        
        ("L3k", "AMANHI-Pemba",  "L2a"),
    ]

    GROUP_LABELS = [
        ("L0a vs L2a", 0, 5),
        ("L3e vs L2a", 6, 11),
        ("L3k vs L2a", 12, 12),
    ]

    FIGURE_TITLE = (
        "L0a and L3e PTB associations across African populations\n"
        "glmmTMB models; L2a reference"
    )

    EXTRA_GROUP_GAP = 0.55


else:
    raise ValueError(
        "ANALYSIS must be either 'MainHap' or 'SubHap'"
    )


# ============================================================
# TYPOGRAPHY
# ============================================================

BASE_SIZE = 12
TITLE_SIZE = 15
HEADER_SIZE = 12
TICK_SIZE = 12
PVALUE_SIZE = 11

plt.rcParams.update({
    "font.size": BASE_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": BASE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
})


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INFILE)


# ============================================================
# SELECT PRIMARY PC-ADJUSTED MODEL
# ============================================================
#
# Keep PC1/PC2/PC3 model.
# Exclude mtcn sensitivity model.
# ============================================================

model_text = df["model_equation"].astype(str)

primary_mask = (
    model_text.str.contains(
        "PC1_PC2_PC3",
        na=False
    )
    &
    ~model_text.str.contains(
        "mtcn",
        case=False,
        na=False
    )
)

df_primary = df[
    primary_mask
].copy()


# ============================================================
# ORDER TABLE
# ============================================================

keep = pd.DataFrame(
    ORDER,
    columns=[
        "haplogroup",
        "population",
        "reference_haplogroup"
    ]
)

keep["row_order"] = np.arange(
    len(keep)
)


# ============================================================
# FREQUENTIST RESULTS
# ============================================================

freq = df_primary[
    df_primary["model"].eq("glmmTMB")
].copy()

freq = freq.merge(
    keep,
    on=[
        "haplogroup",
        "population",
        "reference_haplogroup"
    ],
    how="inner"
)

freq = (
    freq
    .sort_values("row_order")
    .drop_duplicates(
        subset=[
            "haplogroup",
            "population",
            "reference_haplogroup"
        ],
        keep="first"
    )
)


# ============================================================
# FREQUENTIST P-VALUE LABEL
# ============================================================

def freq_label(row):

    p = pd.to_numeric(
        row.get(
            "p_or_p_two",
            np.nan
        ),
        errors="coerce"
    )

    padj = pd.to_numeric(
        row.get(
            "padj",
            np.nan
        ),
        errors="coerce"
    )

    # Prefer BH-adjusted p if significant
    if pd.notna(padj) and padj < 0.05:
        return f"padj={padj:.4f}"

    # Otherwise show nominal p if < .05
    if pd.notna(p) and p < 0.05:
        return f"p={p:.4f}"

    return ""


freq["freq_label"] = freq.apply(
    freq_label,
    axis=1
)


# ============================================================
# BAYESIAN RESULTS
# ============================================================

bayes = df_primary[
    df_primary["model"].eq("brms")
].copy()

bayes = bayes.merge(
    keep,
    on=[
        "haplogroup",
        "population",
        "reference_haplogroup"
    ],
    how="inner"
)


# ============================================================
# SELECT ONE BAYESIAN ROW PER COMPARISON
# ============================================================

if "prior_setting" in bayes.columns:

    bayes_final = bayes[
        bayes[
            "prior_setting"
        ].astype(str).eq(
            "Normal(0,1.0)"
        )
    ].copy()

    # Fallback if exact prior label isn't present
    if bayes_final.empty:
        bayes_final = bayes.copy()

else:
    bayes_final = bayes.copy()


bayes_final = (
    bayes_final
    .sort_values("row_order")
    .drop_duplicates(
        subset=[
            "haplogroup",
            "population",
            "reference_haplogroup"
        ],
        keep="first"
    )
)


# ============================================================
# POSTERIOR SUPPORT
# ============================================================

bayes_final[
    "Pr_higher_PTB_odds"
] = pd.to_numeric(
    bayes_final[
        "Pr_higher_PTB_odds"
    ],
    errors="coerce"
)

bayes_final["OR"] = pd.to_numeric(
    bayes_final["OR"],
    errors="coerce"
)


# Probability supporting estimated direction
bayes_final[
    "posterior_support"
] = np.where(
    bayes_final["OR"] >= 1,
    bayes_final["Pr_higher_PTB_odds"],
    1 - bayes_final["Pr_higher_PTB_odds"]
)


# ============================================================
# PREP FREQUENTIST TABLE
# ============================================================

cols_freq = [
    "haplogroup",
    "population",
    "reference_haplogroup",
    "row_order",
    "OR",
    "OR_low",
    "OR_high",
    "p_or_p_two",
    "padj",
    "freq_label"
]

freq_final = freq[
    cols_freq
].copy()

freq_final = freq_final.rename(
    columns={
        "OR": "freq_OR",
        "OR_low": "freq_OR_low",
        "OR_high": "freq_OR_high"
    }
)


# ============================================================
# PREP BAYESIAN TABLE
# ============================================================

cols_bayes = [
    "haplogroup",
    "population",
    "reference_haplogroup",
    "OR",
    "OR_low",
    "OR_high",
    "posterior_support"
]

if "descriptive_ptb" in bayes_final.columns:
    cols_bayes.append(
        "descriptive_ptb"
    )

bayes_final = bayes_final[
    cols_bayes
].copy()

bayes_final = bayes_final.rename(
    columns={
        "OR": "bayes_OR",
        "OR_low": "bayes_OR_low",
        "OR_high": "bayes_OR_high"
    }
)


# ============================================================
# JOIN FREQUENTIST + BAYESIAN
# ============================================================

plot_df = freq_final.merge(
    bayes_final,
    on=[
        "haplogroup",
        "population",
        "reference_haplogroup"
    ],
    how="left"
)

plot_df = (
    plot_df
    .sort_values("row_order")
    .reset_index(drop=True)
)


# ============================================================
# DESCRIPTIVE PTB
# ============================================================

if "descriptive_ptb" not in plot_df.columns:
    plot_df["descriptive_ptb"] = ""

plot_df[
    "descriptive_ptb"
] = plot_df[
    "descriptive_ptb"
].fillna("")


# ============================================================
# BAYESIAN DISPLAY COLUMNS
# ============================================================

plot_df[
    "Bayesian OR (95% CrI)"
] = plot_df.apply(
    lambda r:
        ""
        if pd.isna(
            r["bayes_OR"]
        )
        else (
            f'{r["bayes_OR"]:.2f} '
            f'({r["bayes_OR_low"]:.2f}–'
            f'{r["bayes_OR_high"]:.2f})'
        ),
    axis=1
)


def support_label(row):

    if pd.isna(
        row["posterior_support"]
    ):
        return ""

    support = (
        row["posterior_support"]
        * 100
    )

    direction = (
        "↑ PTB"
        if row["bayes_OR"] >= 1
        else "↓ PTB"
    )

    return (
        f"{direction} "
        f"{support:.1f}%"
    )


plot_df[
    "Posterior support"
] = plot_df.apply(
    support_label,
    axis=1
)

plot_df[
    "Descriptive PTB"
] = plot_df[
    "descriptive_ptb"
]


# ============================================================
# FRIENDLY POPULATION LABELS
# ============================================================

site_names = {
    "African": "African pooled",
    "GAPPS-Zambia": "GAPPS-Zambia pooled",
    "AMANHI-Pemba": "AMANHI-Pemba",
    "IPOP": "IPOP",
    "ZAPPS1": "ZAPPS1",
    "ZAPPS2": "ZAPPS2",
}

plot_df[
    "site_label"
] = (
    plot_df["population"]
    .map(site_names)
    .fillna(
        plot_df["population"]
    )
)


# ============================================================
# SAVE FIGURE SOURCE DATA
# ============================================================

source_cols = [
    "haplogroup",
    "population",
    "reference_haplogroup",

    "freq_OR",
    "freq_OR_low",
    "freq_OR_high",
    "p_or_p_two",
    "padj",

    "bayes_OR",
    "bayes_OR_low",
    "bayes_OR_high",

    "posterior_support",
    "Descriptive PTB"
]

plot_df[
    source_cols
].to_csv(
    OUT_CSV,
    index=False
)


# ============================================================
# AUTOMATIC Y POSITIONS
# ============================================================
#
# Adds extra space before every NEW comparison group.
#
# Therefore this works automatically for:
#
# MainHap:
#   L2 vs L3
#   L2 vs L0
#   M vs L3
#   M vs L0
#
# SubHap:
#   L0a vs L2a
#   L3e vs L2a
#
# ============================================================

n = len(plot_df)

y = np.zeros(n)

current_y = 0.0

# Determine first index of each group after first group
group_starts = {
    start
    for _, start, _ in GROUP_LABELS[1:]
}

for i in range(n):

    if i in group_starts:
        current_y += EXTRA_GROUP_GAP

    y[i] = current_y
    current_y += 1.0


# ============================================================
# AUTOMATIC GROUP SEPARATORS
# ============================================================

separators = []

for g in range(
    len(GROUP_LABELS) - 1
):

    _, _, end_index = (
        GROUP_LABELS[g]
    )

    _, next_start, _ = (
        GROUP_LABELS[g + 1]
    )

    sep = (
        y[end_index]
        +
        y[next_start]
    ) / 2

    separators.append(
        sep
    )


# ============================================================
# FIGURE
# ============================================================

fig_h = max(
    6.5,
    (y[-1] + 2) * 0.62
)

fig = plt.figure(
    figsize=(
        22,
        fig_h
    )
)


# Forest panel
ax = fig.add_axes([
    0.14,
    0.12,
    0.49,
    0.78
])


# Right table
ax_text = fig.add_axes([
    0.64,
    0.12,
    0.35,
    0.78
])

ax_text.axis(
    "off"
)


# ============================================================
# FOREST DATA
# ============================================================

x = plot_df[
    "freq_OR"
].astype(float).to_numpy()

xerr_low = (
    x
    -
    plot_df[
        "freq_OR_low"
    ].astype(float).to_numpy()
)

xerr_high = (
    plot_df[
        "freq_OR_high"
    ].astype(float).to_numpy()
    -
    x
)


ax.errorbar(
    x,
    y,
    xerr=[
        xerr_low,
        xerr_high
    ],
    fmt="o",
    capsize=3,
    linestyle="none",
    markersize=6
)


# Reference line
ax.axvline(
    1,
    linestyle="--",
    linewidth=1
)


# ============================================================
# LOG X AXIS
# ============================================================

ax.set_xscale("log")

min_ci = plot_df["freq_OR_low"].min()
max_ci = plot_df["freq_OR_high"].max()


if ANALYSIS == "MainHap":

    xmin = max(
        0.35,
        min_ci * 0.75
    )

    xmax = max(
        40,
        max_ci * 1.20
    )

    candidate_ticks = [
        0.5,
        1,
        2,
        4,
        8,
        16,
        32
    ]


elif ANALYSIS == "SubHap":

    xmin = max(
        0.02,
        min_ci * 0.75
    )

    xmax = max(
        10,
        max_ci * 1.15
    )

    candidate_ticks = [
        0.07,
        0.125,
        0.25,
        0.5,
        1,
        2,
        4,
        8
    ]


ax.set_xlim(
    xmin,
    xmax
)

xticks = [
    t for t in candidate_ticks
    if xmin <= t <= xmax
]

ax.set_xticks(
    xticks
)

ax.set_xticklabels(
    [f"{t:g}" for t in xticks]
)

ax.set_xlabel(
    "Odds ratio for PTB (log scale)"
)

# ============================================================
# Y AXIS
# ============================================================

ax.set_yticks(
    y
)

ax.set_yticklabels(
    plot_df[
        "site_label"
    ]
)

ax.invert_yaxis()


# ============================================================
# P / PADJ LABELS
# ============================================================
#
# Fixed horizontal position so labels cannot leave panel.
#
# For a single-row group, move label slightly toward row
# center if necessary.
# ============================================================

single_row_indices = {
    start
    for _, start, end in GROUP_LABELS
    if start == end
}


for i, row in plot_df.iterrows():

    if not row["freq_label"]:
        continue

    y_text = y[i]

    # Small vertical nudge for isolated single-row blocks
    if i in single_row_indices:
        y_text -= 0.16

    ax.text(
        0.80,
        y_text,
        row["freq_label"],
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=PVALUE_SIZE,
        clip_on=True
    )


# ============================================================
# GROUP SEPARATORS
# ============================================================

for sep in separators:

    ax.axhline(
        sep,
        linewidth=0.8,
        alpha=0.45
    )


# ============================================================
# GROUP LABELS
# ============================================================

for (
    label,
    start,
    end
) in GROUP_LABELS:

    center = (
        y[start]
        +
        y[end]
    ) / 2

    ax.text(
        -0.22,
        center,
        label,
        transform=ax.get_yaxis_transform(),
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
        clip_on=False
    )


# ============================================================
# FOREST FORMATTING
# ============================================================

ax.grid(
    axis="x",
    alpha=0.25
)

ax.set_xlabel(
    "Odds ratio for PTB"
)

ax.set_title(
    FIGURE_TITLE,
    weight="bold",
    pad=10
)


# ============================================================
# RIGHT-SIDE TABLE
# ============================================================

ax_text.set_xlim(
    0,
    1
)

ax_text.set_ylim(
    ax.get_ylim()
)


x_or = 0.00
x_support = 0.35
x_desc = 0.61


# Headers
ax_text.text(
    x_or,
    y[0] - 0.65,
    "Bayesian OR (95% CrI)",
    weight="bold",
    fontsize=HEADER_SIZE,
    va="center"
)

ax_text.text(
    x_support,
    y[0] - 0.65,
    "Posterior support",
    weight="bold",
    fontsize=HEADER_SIZE,
    va="center"
)

ax_text.text(
    x_desc,
    y[0] - 0.65,
    "Descriptive PTB",
    weight="bold",
    fontsize=HEADER_SIZE,
    va="center"
)


ax_text.hlines(
    y[0] - 0.35,
    0,
    1,
    linewidth=1
)


# Rows
for i, row in plot_df.iterrows():

    ax_text.text(
        x_or,
        y[i],
        row[
            "Bayesian OR (95% CrI)"
        ],
        fontsize=TICK_SIZE,
        va="center"
    )

    support_value = row["posterior_support"]
    
    ax_text.text(
        x_support,
        y[i],
        row["Posterior support"],
        fontsize=TICK_SIZE,
        va="center",
        fontweight=(
            "bold"
            if pd.notna(support_value) and support_value >= 0.95
            else "normal"
        )
    )

    ax_text.text(
        x_desc,
        y[i],
        row[
            "Descriptive PTB"
        ],
        fontsize=TICK_SIZE,
        va="center"
    )


# Same separator lines
for sep in separators:

    ax_text.hlines(
        sep,
        0,
        1,
        linewidth=0.8,
        alpha=0.45
    )


# ============================================================
# OVERALL TITLE
# ============================================================

fig.suptitle(
    "Maternal mtDNA haplogroup associations with preterm birth",
    fontsize=17,
    fontweight="bold",
    y=0.985
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PRINT SOURCE TABLE
# ============================================================

print(
    plot_df[
        [
            "haplogroup",
            "population",
            "reference_haplogroup",
            "freq_OR",
            "freq_OR_low",
            "freq_OR_high",
            "p_or_p_two",
            "padj",
            "bayes_OR",
            "bayes_OR_low",
            "bayes_OR_high",
            "Posterior support",
            "Descriptive PTB"
        ]
    ].to_string(
        index=False
    )
)

print("\nAnalysis:", ANALYSIS)

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_CSV)
