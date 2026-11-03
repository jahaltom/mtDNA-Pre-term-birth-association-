import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


########################################################
# INPUT
########################################################

path = r"C:\Users\haltomj\OneDrive - Children's Hospital of Philadelphia\Documents"
os.chdir(path)

INPUT_FILE = "Final_PTB_summary_table.Sub.csv"
REFERENCE = "L2a"

OUTPUT_DIR = Path("Publication_Tables")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)


########################################################
# HELPER FUNCTIONS
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


def safe_filename(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


########################################################
# PRETTY VARIABLE NAMES
########################################################

pretty_names = {
    "BABY_SEX1": "Infant sex",
    "BABY_SEX2": "Infant sex",
    "PW_AGE": "Maternal age",
    "MAT_HEIGHT": "Maternal height",
    "mtcn": "mtDNA copy number",
}

pretty_names.update({
    f"PC{i}": f"nDNA PC{i}"
    for i in range(1, 11)
})


########################################################
# POPULATIONS
########################################################

populations = (
    df["population"]
    .dropna()
    .unique()
)

print("\nPopulations found:")
for p in populations:
    print(" -", p)


########################################################
# LOOP THROUGH POPULATIONS
########################################################

for population in populations:

    ####################################################
    # RESTRICT TO POPULATION + REFERENCE
    ####################################################

    pop_df = df[
        (df["population"] == population) &
        (df["reference_haplogroup"] == REFERENCE) &
        (~df["haplogroup"].isin(["Intercept", "(Intercept)"]))
    ].copy()

    if pop_df.empty:
        print(
            f"\nSkipping {population}: "
            f"no {REFERENCE}-reference model."
        )
        continue


    ####################################################
    # MODEL EQUATIONS
    ####################################################

    model_equations = (
        pop_df["model_equation"]
        .dropna()
        .unique()
    )


    for model_equation in model_equations:

        print("\n" + "=" * 80)
        print("Population:", population)
        print("Reference:", REFERENCE)
        print("Model:", model_equation)
        print("=" * 80)


        ################################################
        # RESTRICT TO ONE MODEL EQUATION
        ################################################

        dat = pop_df[
            pop_df["model_equation"] == model_equation
        ].copy()

        if dat.empty:
            continue


        ################################################
        # IDENTIFY HAPLOGROUPS VS COVARIATES
        ################################################

        dat["Type"] = np.where(
            dat["hap_n_total"].notna(),
            "Haplogroup",
            "Covariate"
        )


        ################################################
        # FREQUENTIST RESULTS
        ################################################

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

            "Freq_p": freq[
                "p_or_p_two"
            ].apply(fmt_p),

            "Freq_padj": freq[
                "padj"
            ].apply(fmt_p)
        })


        ################################################
        # BAYESIAN RESULTS
        ################################################

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


        ################################################
        # VARIABLE INFORMATION
        ################################################

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


        ################################################
        # DESCRIPTIVE INFORMATION
        ################################################

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


        ################################################
        # MERGE RESULTS
        ################################################

        results = (
            variables

            .merge(
                freq_table,
                on=[
                    "Population",
                    "Variable"
                ],
                how="left",
                validate="one_to_one"
            )

            .merge(
                bayes_table,
                on=[
                    "Population",
                    "Variable"
                ],
                how="left",
                validate="one_to_one"
            )

            .merge(
                descriptive,
                on=[
                    "Population",
                    "Variable"
                ],
                how="left"
            )
        )


        ################################################
        # REFERENCE INFORMATION
        ################################################

        ref_rows = df[
            (df["population"] == population) &
            (df["reference_haplogroup"] == REFERENCE) &
            (df["model_equation"] == model_equation) &
            (df["ref_n_total"].notna())
        ].copy()

        if ref_rows.empty:
            print(
                "Skipping model: no reference counts found."
            )
            continue

        ref_info = ref_rows.iloc[0]


        ################################################
        # REFERENCE ROW
        ################################################

        reference_row = pd.DataFrame({
            "Population": [population],
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
                str(
                    int(
                        ref_info["ref_n_total"]
                    )
                )
            ],

            "PTB": [
                (
                    f'{int(ref_info["ref_n_ptb"])} '
                    f'({ref_info["ref_ptb_percent"]:.2f})'
                )
            ]
        })


        ################################################
        # REMOVE REFERENCE IF PRESENT
        ################################################

        results = results[
            results["Variable"] != REFERENCE
        ].copy()


        ################################################
        # COMBINE
        ################################################

        final_table = pd.concat(
            [
                reference_row,
                results
            ],
            ignore_index=True
        )


        ################################################
        # PRETTY VARIABLE NAMES
        ################################################

        final_table["Variable"] = (
            final_table["Variable"]
            .replace(pretty_names)
        )


        ################################################
        # BLANK DESCRIPTIVES FOR COVARIATES
        ################################################

        final_table.loc[
            final_table["Type"] == "Covariate",
            ["n", "PTB"]
        ] = ""


        ################################################
        # ORDER ROWS
        ################################################

        hap_rows = final_table[
            final_table["Type"] == "Haplogroup"
        ].copy()

        cov_rows = final_table[
            final_table["Type"] == "Covariate"
        ].copy()


        hap_order = (
            [REFERENCE] +
            sorted([
                x
                for x in hap_rows["Variable"].unique()
                if x != REFERENCE
            ])
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


        ################################################
        # DISPLAY TABLE
        ################################################

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


        ################################################
        # COVARIATE SECTION HEADER
        ################################################

        n_haps = len(hap_rows)

        if len(cov_rows) > 0:

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


        ################################################
        # FIGURE
        ################################################

        n_rows = len(display_df)

        fig_height = max(
            5,
            0.42 * n_rows + 3
        )

        fig, ax = plt.subplots(
            figsize=(14, fig_height)
        )

        ax.axis("off")


        ################################################
        # TITLE
        ################################################

        fig.text(
            0.5,
            0.97,
            "Association of maternal mtDNA haplogroup with preterm birth",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold"
        )

        fig.text(
            0.5,
            0.94,
            f"{population}; reference haplogroup = {REFERENCE}",
            ha="center",
            va="top",
            fontsize=10
        )


        ################################################
        # MODEL LABEL
        ################################################

        short_model = model_equation

        if "mtcn" in str(model_equation):
            model_label = "Model including mtDNA copy number"
        else:
            model_label = "Primary model"

        fig.text(
            0.5,
            0.915,
            model_label,
            ha="center",
            fontsize=9,
            fontstyle="italic"
        )


        ################################################
        # SPANNER HEADERS
        ################################################

        fig.text(
            0.48,
            0.875,
            "Frequentist (glmmTMB)",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )

        fig.text(
            0.80,
            0.875,
            "Bayesian (brms)",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )


        ################################################
        # CREATE TABLE
        ################################################

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

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)


        ################################################
        # BASIC CELL FORMATTING
        ################################################

        for (row, col), cell in table.get_celld().items():

            cell.set_linewidth(0)

            if row == 0:
                cell.set_text_props(
                    fontweight="bold"
                )
                cell.set_linewidth(0.8)

            if col == 0:
                cell.get_text().set_ha("left")


        ################################################
        # COVARIATE SECTION
        ################################################

        if len(cov_rows) > 0:

            cov_section_index = n_haps + 1

            for col in range(
                len(display_df.columns)
            ):

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


        ################################################
        # BOLD REFERENCE ROW
        ################################################

        reference_index = 1

        for col in range(
            len(display_df.columns)
        ):

            table[
                reference_index,
                col
            ].set_text_props(
                fontweight="bold"
            )


        ################################################
        # SIGNIFICANCE FORMATTING
        ################################################

        for i, row in final_table.iterrows():

            table_row = i + 1

            if (
                len(cov_rows) > 0
                and i >= n_haps
            ):
                table_row += 1


            ############################################
            # FREQUENTIST NOMINAL p
            ############################################

            pval = row.get(
                "Freq_p",
                ""
            )

            nominal_sig = False

            if isinstance(pval, str):

                if pval == "<0.001":
                    nominal_sig = True

                elif pval != "":
                    try:
                        nominal_sig = (
                            float(pval) < 0.05
                        )
                    except ValueError:
                        pass


            if nominal_sig:

                for col in [3, 4, 5]:

                    table[
                        table_row,
                        col
                    ].set_text_props(
                        fontweight="bold"
                    )


            ############################################
            # FREQUENTIST ADJUSTED p
            ############################################

            padj = row.get(
                "Freq_padj",
                ""
            )

            adjusted_sig = False

            if isinstance(padj, str):

                if padj == "<0.001":
                    adjusted_sig = True

                elif padj != "":
                    try:
                        adjusted_sig = (
                            float(padj) < 0.05
                        )
                    except ValueError:
                        pass


            if adjusted_sig:

                table[
                    table_row,
                    6
                ].set_text_props(
                    fontweight="bold"
                )


            ############################################
            # BAYESIAN PROBABILITY
            ############################################

            pr = row.get(
                "Pr_OR_gt_1",
                ""
            )

            bayes_prob_sig = False

            if (
                isinstance(pr, str)
                and pr != ""
            ):

                try:
                    pr_value = float(pr)

                    bayes_prob_sig = (
                        pr_value > 0.95
                        or
                        pr_value < 0.05
                    )

                except ValueError:
                    pass


            ############################################
            # BAYESIAN CrI
            ############################################

            cri_excludes_1 = False

            cri = row.get(
                "Bayes_CrI",
                ""
            )

            if (
                isinstance(cri, str)
                and cri != ""
            ):

                try:

                    cri_clean = cri.replace(
                        "–",
                        "-"
                    )

                    low, high = cri_clean.split("-")

                    low = float(low.strip())
                    high = float(high.strip())

                    cri_excludes_1 = (
                        high < 1
                        or
                        low > 1
                    )

                except:
                    pass


            if (
                bayes_prob_sig
                or
                cri_excludes_1
            ):

                for col in [7, 8, 9]:

                    table[
                        table_row,
                        col
                    ].set_text_props(
                        fontweight="bold"
                    )


        ################################################
        # FOOTNOTE
        ################################################

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


        ################################################
        # LAYOUT
        ################################################

        plt.tight_layout(
            rect=[
                0.02,
                0.05,
                0.98,
                0.86
            ]
        )


        ################################################
        # OUTPUT FILENAMES
        ################################################

        safe_population = safe_filename(
            population
        )

        if "mtcn" in str(model_equation):
            model_suffix = "mtcn"
        else:
            model_suffix = "primary"


        base_name = (
            f"{safe_population}_"
            f"{REFERENCE}_"
            f"{model_suffix}_"
            f"publication_table"
        )


        png_file = (
            OUTPUT_DIR /
            f"{base_name}.png"
        )

        pdf_file = (
            OUTPUT_DIR /
            f"{base_name}.pdf"
        )

        csv_file = (
            OUTPUT_DIR /
            f"{base_name}.csv"
        )


        ################################################
        # SAVE
        ################################################

        plt.savefig(
            png_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            pdf_file,
            bbox_inches="tight"
        )

        plt.close(fig)


        final_table.to_csv(
            csv_file,
            index=False
        )


        print("Saved:", png_file)
        print("Saved:", pdf_file)
        print("Saved:", csv_file)


########################################################
# DONE
########################################################

print("\n========================================")
print("ALL TABLES COMPLETE")
print("========================================")
print("Output directory:", OUTPUT_DIR)
