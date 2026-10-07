#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

METADATA_FILE = "momi_mothers_merged.csv"

# Change if your 1000 Genomes annotation file has another name.
PANEL_FILE = "1KGP.metadata.tsv"

# First/global QC PCA from the original eligible maternal cohort
INITIAL_PCA_FILE = "pca/joint_global.eigenvec"
INITIAL_EIGENVAL_FILE = "pca/joint_global.eigenval"

# Second/global PCA after ancestry outliers are removed
FINAL_PCA_FILE = "pca/joint_global_final.eigenvec"
FINAL_EIGENVAL_FILE = "pca/joint_global_final.eigenval"

OUTDIR = Path("pca/global_qc")
OUTDIR.mkdir(parents=True, exist_ok=True)

INITIAL_KEEP_FILE = "qc/maternal_ptb_globalPCA.keep"

ULTIMATE_METADATA_FILE = "momi_mothers_ultimate.csv"


# PCs used for Mahalanobis ancestry comparison.
#
# Keep this relatively small for broad/global ancestry QC.
DISTANCE_PCS = [
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5"
]

# Empirical population-specific ancestry QC threshold
OUTLIER_QUANTILE = 0.99


# ============================================================
# STUDY SITE DEFINITIONS
# ============================================================

AFR_SITES = [
    "AMANHI-Pemba",
    "IPOP",
    "ZAPPS1",
    "ZAPPS2"
]

SAS_SITES = [
    "AMANHI-Bangladesh",
    "GAPPS-Bangladesh",
    "AMANHI-Pakistan"
]


# ============================================================
# 1000 GENOMES POPULATIONS
#
# AFR/SAS subsets are used for INITIAL outlier QC.
#
# FINAL nearest-reference assignment uses every available
# 1000G population in the panel.
# ============================================================

AFR_POPS = [
    "ACB",
    "ASW",
    "ESN",
    "GWD",
    "LWK",
    "MSL",
    "YRI"
]

SAS_POPS = [
    "BEB",
    "GIH",
    "ITU",
    "PJL",
    "STU"
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Global 1000G PCA ancestry QC and final "
        "maternal metadata annotation."
    )
)

parser.add_argument(
    "--stage",
    required=True,
    choices=[
        "initial",
        "final"
    ],
    help=(
        "initial = identify/remove global ancestry outliers; "
        "final = annotate cleaned second PCA and create "
        "ultimate metadata"
    )
)

args = parser.parse_args()


# ============================================================
# HELPER: CLEAN STRING
# ============================================================

def clean_string_series(x):

    return (
        x.astype("string")
        .str.strip()
    )


# ============================================================
# HELPER: READ PCA
# ============================================================

def read_pca(pca_file):

    pca = pd.read_csv(
        pca_file,
        sep=r"\s+"
    )

    if "#FID" in pca.columns:

        pca = pca.rename(
            columns={
                "#FID": "FID"
            }
        )

    if "#IID" in pca.columns:

        pca = pca.rename(
            columns={
                "#IID": "IID"
            }
        )

    if "IID" not in pca.columns:

        raise ValueError(
            f"IID column not found in {pca_file}. "
            f"Columns are: {pca.columns.tolist()}"
        )

    pca["IID"] = clean_string_series(
        pca["IID"]
    )

    return pca


# ============================================================
# HELPER: READ EIGENVALUES
# ============================================================

def read_eigenvalues(eigenval_file):

    eigenvalues = np.loadtxt(
        eigenval_file
    )

    variance_pct = (
        eigenvalues /
        eigenvalues.sum()
    ) * 100

    return (
        eigenvalues,
        variance_pct
    )


# ============================================================
# HELPER: READ STUDY METADATA
# ============================================================

def read_metadata():

    meta = pd.read_csv(
        METADATA_FILE,
        dtype=str,
        low_memory=False
    )

    if "id" not in meta.columns:

        raise ValueError(
            "Column 'id' is not present in "
            f"{METADATA_FILE}"
        )

    if "merge_site" not in meta.columns:

        raise ValueError(
            "Column 'merge_site' is not present in "
            f"{METADATA_FILE}"
        )

    meta["id"] = clean_string_series(
        meta["id"]
    )

    meta["merge_site"] = clean_string_series(
        meta["merge_site"]
    )

    # --------------------------------------------------------
    # Expected broad ancestry from known study design
    # --------------------------------------------------------

    meta["Expected_superpop"] = pd.NA

    meta.loc[
        meta["merge_site"].isin(
            AFR_SITES
        ),
        "Expected_superpop"
    ] = "AFR"

    meta.loc[
        meta["merge_site"].isin(
            SAS_SITES
        ),
        "Expected_superpop"
    ] = "SAS"

    return meta


# ============================================================
# HELPER: READ 1000G PANEL
# ============================================================

# ============================================================
# HELPER: READ 1000 GENOMES METADATA
#
# Expected input columns:
#
# Sample name
# Population code
# Population name
# Superpopulation code
# Superpopulation name
#
# Example:
# HG00315   FIN   Finnish   EUR   European Ancestry
#
# Some samples are also present in other projects such as SGDP,
# producing comma-separated annotations such as:
#
# FIN,FinnishSGDP
#
# For this analysis we use the FIRST annotation, which
# corresponds to the 1000 Genomes population.
# ============================================================

def read_1000g_panel():

    panel = pd.read_csv(
        PANEL_FILE,
        sep="\t",
        dtype=str,
        low_memory=False
    )

    print("\n========================================")
    print("1000 GENOMES METADATA")
    print("========================================")

    print(
        "Rows in metadata:",
        len(panel)
    )

    print(
        "\nColumns:"
    )

    print(
        panel.columns.tolist()
    )


    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "Sample name",
        "Population code",
        "Population name",
        "Superpopulation code",
        "Superpopulation name"
    ]

    missing = [
        col
        for col in required
        if col not in panel.columns
    ]

    if missing:

        raise ValueError(
            "1KGP metadata is missing required columns: "
            + ", ".join(missing)
        )


    # --------------------------------------------------------
    # Keep relevant columns
    # --------------------------------------------------------

    panel = panel[
        [
            "Sample name",
            "Population code",
            "Population name",
            "Superpopulation code",
            "Superpopulation name"
        ]
    ].copy()


    # --------------------------------------------------------
    # Rename to standard names used by the PCA script
    # --------------------------------------------------------

    panel = panel.rename(
        columns={
            "Sample name":
                "IID",

            "Population code":
                "Population",

            "Population name":
                "Population_name",

            "Superpopulation code":
                "Superpopulation",

            "Superpopulation name":
                "Superpopulation_name"
        }
    )


    # --------------------------------------------------------
    # Clean whitespace
    # --------------------------------------------------------

    for col in [
        "IID",
        "Population",
        "Population_name",
        "Superpopulation",
        "Superpopulation_name"
    ]:

        panel[col] = (
            panel[col]
            .astype("string")
            .str.strip()
        )


    # --------------------------------------------------------
    # Some metadata rows contain annotations from more than
    # one project.
    #
    # Example:
    #
    # Population:
    # FIN,FinnishSGDP
    #
    # Superpopulation name:
    # European Ancestry,West Eurasia (SGDP)
    #
    # Since this PCA uses the 1000 Genomes reference panel,
    # retain the FIRST annotation.
    # --------------------------------------------------------

    for col in [
        "Population",
        "Population_name",
        "Superpopulation",
        "Superpopulation_name"
    ]:

        panel[col] = (
            panel[col]
            .str.split(",")
            .str[0]
            .str.strip()
        )


    # --------------------------------------------------------
    # Remove missing sample IDs
    # --------------------------------------------------------

    panel = panel[
        panel["IID"].notna()
    ].copy()


    # --------------------------------------------------------
    # Verify unique sample IDs
    # --------------------------------------------------------

    duplicate_ids = panel[
        panel.duplicated(
            subset="IID",
            keep=False
        )
    ].copy()


    print(
        "\nDuplicate Sample name rows:",
        len(duplicate_ids)
    )


    if len(duplicate_ids) > 0:

        duplicate_ids.to_csv(
            OUTDIR /
            "duplicate_1KGP_metadata_IDs.csv",
            index=False
        )

        raise ValueError(
            "1KGP metadata contains duplicate Sample name "
            "values. See duplicate_1KGP_metadata_IDs.csv."
        )


    # --------------------------------------------------------
    # Basic QC
    # --------------------------------------------------------

    print(
        "\n1000G superpopulations:"
    )

    print(
        panel[
            "Superpopulation"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


    print(
        "\n1000G populations:"
    )

    print(
        panel[
            "Population"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


    print(
        "\nPopulation labels:"
    )

    print(
        panel[
            [
                "Population",
                "Population_name",
                "Superpopulation",
                "Superpopulation_name"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Superpopulation",
                "Population"
            ]
        )
        .to_string(
            index=False
        )
    )


    return panel


# ============================================================
# HELPER: LABEL JOINT PCA
# ============================================================

def label_joint_pca(
    pca,
    meta,
    panel
):

    # --------------------------------------------------------
    # Add 1000G labels
    # --------------------------------------------------------

    pca = pca.merge(
        panel,
        on="IID",
        how="left",
        validate="many_to_one"
    )


    # --------------------------------------------------------
    # Add study site labels
    # --------------------------------------------------------

    study_labels = (
        meta[
            [
                "id",
                "merge_site",
                "Expected_superpop"
            ]
        ]
        .rename(
            columns={
                "id": "IID"
            }
        )
    )


    if study_labels["IID"].duplicated().any():

        duplicates = (
            study_labels[
                study_labels["IID"].duplicated(
                    keep=False
                )
            ]
        )

        raise ValueError(
            "Study metadata contains duplicate IDs. "
            f"Found {len(duplicates)} duplicate rows."
        )


    pca = pca.merge(
        study_labels,
        on="IID",
        how="left",
        validate="many_to_one"
    )


    # --------------------------------------------------------
    # Identify source
    # --------------------------------------------------------

    pca["Source"] = "Unknown"


    pca.loc[
        pca["Superpopulation"].notna(),
        "Source"
    ] = "1000G"


    pca.loc[
        pca["merge_site"].notna(),
        "Source"
    ] = "Study"


    kg = (
        pca[
            pca["Source"].eq(
                "1000G"
            )
        ]
        .copy()
    )


    study = (
        pca[
            pca["Source"].eq(
                "Study"
            )
        ]
        .copy()
    )


    unknown = (
        pca[
            pca["Source"].eq(
                "Unknown"
            )
        ]
        .copy()
    )


    return (
        pca,
        kg,
        study,
        unknown
    )


# ============================================================
# HELPER: MAHALANOBIS DISTANCE
# ============================================================

def mahalanobis_distance(
    x,
    centroid,
    inv_cov
):

    delta = (
        x -
        centroid
    )

    distance_squared = (
        delta.T
        @ inv_cov
        @ delta
    )

    # Numerical protection against values like -1e-15
    distance_squared = max(
        float(distance_squared),
        0.0
    )

    return float(
        np.sqrt(
            distance_squared
        )
    )


# ============================================================
# HELPER: BUILD REFERENCE MODEL
# ============================================================

def make_reference_model(
    X
):

    centroid = (
        X.mean(
            axis=0
        )
    )


    covariance = np.cov(
        X,
        rowvar=False
    )


    # --------------------------------------------------------
    # Small covariance regularization
    # --------------------------------------------------------

    mean_variance = (
        np.trace(
            covariance
        )
        /
        covariance.shape[0]
    )


    if not np.isfinite(
        mean_variance
    ):

        mean_variance = 1.0


    epsilon = (
        max(
            mean_variance,
            1e-12
        )
        *
        1e-6
    )


    covariance = (
        covariance
        +
        np.eye(
            covariance.shape[0]
        )
        * epsilon
    )


    inv_cov = np.linalg.pinv(
        covariance
    )


    return {
        "centroid":
            centroid,

        "covariance":
            covariance,

        "inv_cov":
            inv_cov,

        "n":
            len(X)
    }


# ============================================================
# HELPER: BUILD ALL 1000G POPULATION REFERENCE MODELS
# ============================================================

def build_reference_stats(
    kg,
    populations=None
):

    reference_stats = {}


    for population, g in (
        kg.groupby(
            "Population"
        )
    ):

        if populations is not None:

            if population not in populations:

                continue


        X = (
            g[
                DISTANCE_PCS
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )


        minimum_n = (
            len(
                DISTANCE_PCS
            )
            +
            3
        )


        if len(X) < minimum_n:

            print(
                "WARNING: skipping population",
                population,
                "because N =",
                len(X)
            )

            continue


        reference_stats[
            population
        ] = (
            make_reference_model(
                X
            )
        )


    return reference_stats


# ============================================================
# HELPER: LEAVE-ONE-OUT REFERENCE DISTANCES
#
# Every 1000G individual is compared with a model built from
# all OTHER individuals from the same 1000G population.
#
# This gives an empirical within-population distance
# distribution without letting a reference individual help
# define its own centroid.
# ============================================================

def calculate_loo_reference_distances(
    kg,
    populations=None
):

    distance_lookup = {}


    for population, g in (
        kg.groupby(
            "Population"
        )
    ):

        if populations is not None:

            if population not in populations:

                continue


        X = (
            g[
                DISTANCE_PCS
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .dropna()
            .to_numpy()
        )


        minimum_n = (
            len(
                DISTANCE_PCS
            )
            +
            4
        )


        if len(X) < minimum_n:

            continue


        distances = []


        for i in range(
            len(X)
        ):

            training = np.delete(
                X,
                i,
                axis=0
            )


            model = (
                make_reference_model(
                    training
                )
            )


            distance = (
                mahalanobis_distance(
                    X[i],
                    model["centroid"],
                    model["inv_cov"]
                )
            )


            distances.append(
                distance
            )


        distance_lookup[
            population
        ] = np.array(
            distances
        )


    return distance_lookup


# ============================================================
# HELPER: REFERENCE THRESHOLDS
# ============================================================

def calculate_reference_thresholds(
    reference_distances
):

    thresholds = {}


    for pop, distances in (
        reference_distances.items()
    ):

        thresholds[pop] = float(
            np.quantile(
                distances,
                OUTLIER_QUANTILE
            )
        )


    return thresholds


# ============================================================
# HELPER: EMPIRICAL PERCENTILE
# ============================================================

def empirical_percentile(
    distance,
    reference_distances
):

    if len(
        reference_distances
    ) == 0:

        return np.nan


    return float(
        100
        *
        np.mean(
            reference_distances
            <= distance
        )
    )


# ============================================================
# HELPER: CALCULATE DISTANCES TO CANDIDATE POPULATIONS
# ============================================================

def population_distances(
    row,
    candidate_pops,
    reference_stats
):

    x = (
        row[
            DISTANCE_PCS
        ]
        .astype(float)
        .to_numpy()
    )


    distances = {}


    for pop in candidate_pops:

        if pop not in reference_stats:

            continue


        stats = (
            reference_stats[
                pop
            ]
        )


        distances[pop] = (
            mahalanobis_distance(
                x,
                stats["centroid"],
                stats["inv_cov"]
            )
        )


    return distances


# ============================================================
# HELPER: GLOBAL PCA PLOT
# ============================================================

def plot_pca(
    kg,
    study,
    variance_pct,
    pc_x,
    pc_y,
    outfile,
    title,
    flagged_ids=None
):

    pc_x_index = (
        int(
            pc_x.replace(
                "PC",
                ""
            )
        )
        - 1
    )


    pc_y_index = (
        int(
            pc_y.replace(
                "PC",
                ""
            )
        )
        - 1
    )


    fig, ax = plt.subplots(
        figsize=(
            12,
            8
        )
    )


    # --------------------------------------------------------
    # 1000G background
    # --------------------------------------------------------

    for superpop, g in (
        kg.groupby(
            "Superpopulation"
        )
    ):

        ax.scatter(
            g[pc_x],
            g[pc_y],
            s=18,
            alpha=0.35,
            label=(
                f"1000G {superpop}"
            )
        )


    # --------------------------------------------------------
    # Study samples, by site
    # --------------------------------------------------------

    for site, g in (
        study.groupby(
            "merge_site"
        )
    ):

        ax.scatter(
            g[pc_x],
            g[pc_y],
            s=28,
            alpha=0.70,
            marker="x",
            label=site
        )


    # --------------------------------------------------------
    # Optional ancestry-outlier outline
    # --------------------------------------------------------

    if flagged_ids is not None:

        flagged_set = set(
            flagged_ids
        )


        bad = study[
            study["IID"].isin(
                flagged_set
            )
        ]


        if len(
            bad
        ) > 0:

            ax.scatter(
                bad[pc_x],
                bad[pc_y],
                s=110,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label=(
                    "Flagged ancestry outlier"
                )
            )


    ax.set_xlabel(
        f"{pc_x} "
        f"({variance_pct[pc_x_index]:.2f}%)"
    )


    ax.set_ylabel(
        f"{pc_y} "
        f"({variance_pct[pc_y_index]:.2f}%)"
    )


    ax.set_title(
        title
    )


    ax.legend(
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left",
        fontsize=8
    )


    plt.tight_layout()


    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved plot:",
        outfile
    )


# ============================================================
# HELPER: PRINT PCA SUMMARY
# ============================================================

def print_pca_summary(
    pca,
    kg,
    study,
    unknown
):

    print(
        "\n========================================"
    )

    print(
        "JOINT PCA SUMMARY"
    )

    print(
        "========================================"
    )


    print(
        "Total PCA samples:",
        len(pca)
    )

    print(
        "Study:",
        len(study)
    )

    print(
        "1000G:",
        len(kg)
    )

    print(
        "Unknown:",
        len(unknown)
    )


    print(
        "\nStudy samples by site:"
    )

    print(
        study[
            "merge_site"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\n1000G superpopulations:"
    )

    print(
        kg[
            "Superpopulation"
        ]
        .value_counts()
        .sort_index()
    )


# ============================================================
# LOAD COMMON INPUTS
# ============================================================

meta = read_metadata()

panel = read_1000g_panel()


# ============================================================
# ============================================================
#
# INITIAL STAGE
#
# ============================================================
# ============================================================

if args.stage == "initial":

    print(
        "\n########################################"
    )

    print(
        "# INITIAL GLOBAL ANCESTRY QC"
    )

    print(
        "########################################"
    )


    pca = read_pca(
        INITIAL_PCA_FILE
    )


    (
        eigenvalues,
        variance_pct
    ) = read_eigenvalues(
        INITIAL_EIGENVAL_FILE
    )


    (
        pca,
        kg,
        study,
        unknown
    ) = label_joint_pca(
        pca,
        meta,
        panel
    )


    print_pca_summary(
        pca,
        kg,
        study,
        unknown
    )


    # --------------------------------------------------------
    # Save fully labeled PCA
    # --------------------------------------------------------

    pca.to_csv(
        OUTDIR /
        "initial_joint_global_labeled.csv",
        index=False
    )


    # --------------------------------------------------------
    # BEFORE plots
    # --------------------------------------------------------

    plot_pairs = [
        (
            "PC1",
            "PC2"
        ),
        (
            "PC1",
            "PC3"
        ),
        (
            "PC2",
            "PC3"
        ),
        (
            "PC1",
            "PC4"
        ),
        (
            "PC2",
            "PC4"
        )
    ]


    for pc_x, pc_y in plot_pairs:

        plot_pca(
            kg=kg,
            study=study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=(
                OUTDIR /
                f"BEFORE_{pc_x}_{pc_y}.png"
            ),
            title=(
                "Before global ancestry QC: "
                f"{pc_x} vs {pc_y}"
            )
        )


    # --------------------------------------------------------
    # Build AFR + SAS reference models
    # --------------------------------------------------------

    plausible_pops = (
        AFR_POPS +
        SAS_POPS
    )


    reference_stats = (
        build_reference_stats(
            kg,
            populations=plausible_pops
        )
    )


    reference_distances = (
        calculate_loo_reference_distances(
            kg,
            populations=plausible_pops
        )
    )


    thresholds = (
        calculate_reference_thresholds(
            reference_distances
        )
    )


    # --------------------------------------------------------
    # Print thresholds
    # --------------------------------------------------------

    threshold_table = []


    for pop in sorted(
        thresholds
    ):

        threshold_table.append({
            "Population":
                pop,

            "N_reference":
                reference_stats[
                    pop
                ]["n"],

            "Threshold99":
                thresholds[
                    pop
                ]
        })


    threshold_df = pd.DataFrame(
        threshold_table
    )


    threshold_df.to_csv(
        OUTDIR /
        "initial_1000G_population_thresholds.csv",
        index=False
    )


    print(
        "\n========================================"
    )

    print(
        "1000G 99TH-PERCENTILE THRESHOLDS"
    )

    print(
        "========================================"
    )

    print(
        threshold_df.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Evaluate each study sample
    # --------------------------------------------------------

    results = []


    for _, row in (
        study.iterrows()
    ):

        expected = (
            row[
                "Expected_superpop"
            ]
        )


        if expected == "AFR":

            candidate_pops = (
                AFR_POPS
            )


        elif expected == "SAS":

            candidate_pops = (
                SAS_POPS
            )


        else:

            print(
                "WARNING: no expected "
                "superpopulation for",
                row["IID"],
                row["merge_site"]
            )

            continue


        distances = (
            population_distances(
                row,
                candidate_pops,
                reference_stats
            )
        )


        if len(
            distances
        ) == 0:

            raise ValueError(
                "No available 1000G "
                "reference populations for "
                f"{expected}"
            )


        ordered = sorted(
            distances.items(),
            key=lambda x:
                x[1]
        )


        nearest_pop = (
            ordered[0][0]
        )

        nearest_distance = (
            ordered[0][1]
        )


        if len(
            ordered
        ) > 1:

            second_pop = (
                ordered[1][0]
            )

            second_distance = (
                ordered[1][1]
            )

        else:

            second_pop = pd.NA

            second_distance = np.nan


        threshold = (
            thresholds[
                nearest_pop
            ]
        )


        percentile = (
            empirical_percentile(
                nearest_distance,
                reference_distances[
                    nearest_pop
                ]
            )
        )


        distance_ratio = (
            nearest_distance /
            threshold
        )


        global_outlier = (
            nearest_distance >
            threshold
        )


        result = {
            "IID":
                row["IID"],

            "merge_site":
                row[
                    "merge_site"
                ],

            "Expected_superpop":
                expected,

            "Nearest_Plausible_1KGP_Pop":
                nearest_pop,

            "Nearest_Plausible_Mahalanobis":
                nearest_distance,

            "Second_Plausible_1KGP_Pop":
                second_pop,

            "Second_Plausible_Mahalanobis":
                second_distance,

            "Nearest_vs_Second_Delta":
                (
                    second_distance -
                    nearest_distance
                ),

            "Nearest_Pop_Threshold99":
                threshold,

            "Nearest_Pop_Empirical_Percentile":
                percentile,

            "Distance_over_Threshold":
                distance_ratio,

            "Global_PCA_outlier":
                global_outlier
        }


        for pop, distance in (
            distances.items()
        ):

            result[
                f"Mahalanobis_{pop}"
            ] = distance


        results.append(
            result
        )


    qc = pd.DataFrame(
        results
    )


    qc = qc.sort_values(
        "Distance_over_Threshold",
        ascending=False
    )


    qc.to_csv(
        OUTDIR /
        "initial_global_PCA_ancestry_QC.csv",
        index=False
    )


    # --------------------------------------------------------
    # Flagged samples
    # --------------------------------------------------------

    flagged = (
        qc[
            qc[
                "Global_PCA_outlier"
            ]
        ]
        .copy()
    )


    flagged.to_csv(
        OUTDIR /
        "initial_global_PCA_outliers.csv",
        index=False
    )


    print(
        "\n========================================"
    )

    print(
        "INITIAL GLOBAL ANCESTRY OUTLIERS"
    )

    print(
        "========================================"
    )


    print(
        "Study samples tested:",
        len(qc)
    )

    print(
        "Outliers flagged:",
        len(flagged)
    )


    print(
        "\nOutliers by site:"
    )

    print(
        flagged[
            "merge_site"
        ]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # BEFORE plots with flagged samples outlined
    # --------------------------------------------------------

    for pc_x, pc_y in plot_pairs:

        plot_pca(
            kg=kg,
            study=study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=(
                OUTDIR /
                f"BEFORE_FLAGGED_{pc_x}_{pc_y}.png"
            ),
            title=(
                "Initial PCA with ancestry "
                f"outliers flagged: {pc_x} vs {pc_y}"
            ),
            flagged_ids=(
                flagged[
                    "IID"
                ].tolist()
            )
        )


    # --------------------------------------------------------
    # Remove flagged study samples
    # --------------------------------------------------------

    flagged_ids = set(
        flagged[
            "IID"
        ]
    )


    clean_study = (
        study[
            ~study[
                "IID"
            ].isin(
                flagged_ids
            )
        ]
        .copy()
    )


    # --------------------------------------------------------
    # AFTER plots
    #
    # NOTE:
    # These use the SAME initial PCA coordinates.
    #
    # They show the visual effect of removing flagged samples
    # before the second PCA is computed.
    # --------------------------------------------------------

    for pc_x, pc_y in plot_pairs:

        plot_pca(
            kg=kg,
            study=clean_study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=(
                OUTDIR /
                f"AFTER_REMOVAL_{pc_x}_{pc_y}.png"
            ),
            title=(
                "After removing global ancestry "
                f"outliers: {pc_x} vs {pc_y}"
            )
        )


    # --------------------------------------------------------
    # Write clean PLINK keep file
    # --------------------------------------------------------

    clean_keep = pd.DataFrame({
        "FID":
            ["0"] *
            len(
                clean_study
            ),

        "IID":
            clean_study[
                "IID"
            ]
    })


    clean_keep.to_csv(
        INITIAL_KEEP_FILE,
        sep="\t",
        index=False,
        header=False
    )


    # --------------------------------------------------------
    # Save retained study samples
    # --------------------------------------------------------

    clean_study.to_csv(
        OUTDIR /
        "initial_globalPCA_retained_study.csv",
        index=False
    )


    # --------------------------------------------------------
    # Final INITIAL summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "INITIAL GLOBAL PCA QC COMPLETE"
    )

    print(
        "========================================"
    )


    print(
        "Study samples before ancestry QC:",
        len(study)
    )

    print(
        "Global ancestry outliers removed:",
        len(flagged)
    )

    print(
        "Study samples retained:",
        len(clean_study)
    )


    print(
        "\nPLINK keep file:"
    )

    print(
        INITIAL_KEEP_FILE
    )


    print(
        "\nNEXT STEP:"
    )

    print(
        "Create qc/maternal_ptb_globalPCA "
        "with this keep file, then rerun "
        "the joint 1000G PCA to generate:"
    )

    print(
        FINAL_PCA_FILE
    )

    print(
        FINAL_EIGENVAL_FILE
    )


# ============================================================
# ============================================================
#
# FINAL STAGE
#
# ============================================================
# ============================================================

elif args.stage == "final":

    print(
        "\n########################################"
    )

    print(
        "# FINAL CLEAN GLOBAL PCA ANNOTATION"
    )

    print(
        "########################################"
    )


    if not Path(
        FINAL_PCA_FILE
    ).exists():

        raise FileNotFoundError(
            "Final PCA file does not exist: "
            f"{FINAL_PCA_FILE}\n"
            "Run the second joint PCA first."
        )


    if not Path(
        FINAL_EIGENVAL_FILE
    ).exists():

        raise FileNotFoundError(
            "Final eigenvalue file does not exist: "
            f"{FINAL_EIGENVAL_FILE}"
        )


    pca = read_pca(
        FINAL_PCA_FILE
    )


    (
        eigenvalues,
        variance_pct
    ) = read_eigenvalues(
        FINAL_EIGENVAL_FILE
    )


    (
        pca,
        kg,
        study,
        unknown
    ) = label_joint_pca(
        pca,
        meta,
        panel
    )


    print_pca_summary(
        pca,
        kg,
        study,
        unknown
    )


    pca.to_csv(
        OUTDIR /
        "final_joint_global_labeled.csv",
        index=False
    )


    # --------------------------------------------------------
    # Final cleaned PCA plots
    # --------------------------------------------------------

    plot_pairs = [
        (
            "PC1",
            "PC2"
        ),
        (
            "PC1",
            "PC3"
        ),
        (
            "PC2",
            "PC3"
        ),
        (
            "PC1",
            "PC4"
        ),
        (
            "PC2",
            "PC4"
        )
    ]


    for pc_x, pc_y in plot_pairs:

        plot_pca(
            kg=kg,
            study=study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=(
                OUTDIR /
                f"FINAL_{pc_x}_{pc_y}.png"
            ),
            title=(
                "Final cleaned study cohort + "
                f"1000G: {pc_x} vs {pc_y}"
            )
        )


    # ========================================================
    # FINAL 1000G REFERENCE MODELS
    #
    # For annotation we use ALL populations available in the
    # 1000G panel, not just AFR/SAS.
    #
    # This allows Nearest_1KGP_SuperPop to be an independent
    # QC check rather than forcing AFR/SAS based on study site.
    # ========================================================

    all_1000g_pops = sorted(
        kg[
            "Population"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    reference_stats = (
        build_reference_stats(
            kg,
            populations=all_1000g_pops
        )
    )


    reference_distances = (
        calculate_loo_reference_distances(
            kg,
            populations=all_1000g_pops
        )
    )


    thresholds = (
        calculate_reference_thresholds(
            reference_distances
        )
    )


    # --------------------------------------------------------
    # Population -> superpopulation lookup
    # --------------------------------------------------------

    pop_superpop = (
        kg[
            [
                "Population",
                "Superpopulation"
            ]
        ]
        .drop_duplicates()
    )


    duplicate_pop_labels = (
        pop_superpop[
            pop_superpop[
                "Population"
            ].duplicated(
                keep=False
            )
        ]
    )


    if len(
        duplicate_pop_labels
    ) > 0:

        raise ValueError(
            "A 1000G population maps to multiple "
            "superpopulation labels."
        )


    pop_to_superpop = dict(
        zip(
            pop_superpop[
                "Population"
            ],
            pop_superpop[
                "Superpopulation"
            ]
        )
    )


    # ========================================================
    # FINAL NEAREST 1000G REFERENCE ANNOTATION
    # ========================================================

    results = []


    for _, row in (
        study.iterrows()
    ):

        distances = (
            population_distances(
                row,
                all_1000g_pops,
                reference_stats
            )
        )


        if len(
            distances
        ) == 0:

            raise ValueError(
                "No 1000G population distances "
                f"could be calculated for {row['IID']}"
            )


        ordered = sorted(
            distances.items(),
            key=lambda x:
                x[1]
        )


        nearest_pop = (
            ordered[0][0]
        )

        nearest_distance = (
            ordered[0][1]
        )


        second_pop = (
            ordered[1][0]
            if len(
                ordered
            ) > 1
            else pd.NA
        )


        second_distance = (
            ordered[1][1]
            if len(
                ordered
            ) > 1
            else np.nan
        )


        nearest_superpop = (
            pop_to_superpop.get(
                nearest_pop,
                pd.NA
            )
        )


        second_superpop = (
            pop_to_superpop.get(
                second_pop,
                pd.NA
            )
            if pd.notna(
                second_pop
            )
            else pd.NA
        )


        threshold = (
            thresholds.get(
                nearest_pop,
                np.nan
            )
        )


        percentile = (
            empirical_percentile(
                nearest_distance,
                reference_distances[
                    nearest_pop
                ]
            )
            if nearest_pop
            in reference_distances
            else np.nan
        )


        distance_ratio = (
            nearest_distance /
            threshold
            if np.isfinite(
                threshold
            )
            and threshold > 0
            else np.nan
        )


        expected_superpop = (
            row[
                "Expected_superpop"
            ]
        )


        superpop_concordant = (
            expected_superpop
            ==
            nearest_superpop
        )


        result = {
            "IID":
                row["IID"],

            "merge_site":
                row[
                    "merge_site"
                ],

            "Expected_superpop":
                expected_superpop,

            "Nearest_1KGP_Pop":
                nearest_pop,

            "Nearest_1KGP_SuperPop":
                nearest_superpop,

            "Nearest_1KGP_Mahalanobis":
                nearest_distance,

            "Second_1KGP_Pop":
                second_pop,

            "Second_1KGP_SuperPop":
                second_superpop,

            "Second_1KGP_Mahalanobis":
                second_distance,

            "Nearest_vs_Second_Delta":
                (
                    second_distance -
                    nearest_distance
                ),

            "Nearest_vs_Second_Ratio":
                (
                    second_distance /
                    nearest_distance
                    if nearest_distance > 0
                    else np.nan
                ),

            "Nearest_1KGP_Threshold99":
                threshold,

            "Nearest_1KGP_Empirical_Percentile":
                percentile,

            "Nearest_1KGP_DistanceRatio":
                distance_ratio,

            "SuperPop_concordant":
                superpop_concordant
        }


        # ----------------------------------------------------
        # Save distance to every reference population
        # ----------------------------------------------------

        for pop, distance in (
            distances.items()
        ):

            result[
                f"Mahalanobis_{pop}"
            ] = distance


        results.append(
            result
        )


    ancestry = pd.DataFrame(
        results
    )


    ancestry.to_csv(
        OUTDIR /
        "final_1KGP_nearest_reference.csv",
        index=False
    )


    # ========================================================
    # ALSO CALCULATE NEAREST PLAUSIBLE AFR/SAS REFERENCE
    #
    # This is useful because an unrestricted nearest-reference
    # assignment is an independent QC check, while this column
    # answers:
    #
    # "Within the expected broad ancestry group, which 1000G
    # reference population is closest?"
    # ========================================================

    plausible_results = []


    for _, row in (
        study.iterrows()
    ):

        expected = (
            row[
                "Expected_superpop"
            ]
        )


        if expected == "AFR":

            candidate_pops = (
                AFR_POPS
            )

        elif expected == "SAS":

            candidate_pops = (
                SAS_POPS
            )

        else:

            continue


        distances = (
            population_distances(
                row,
                candidate_pops,
                reference_stats
            )
        )


        if len(
            distances
        ) == 0:

            continue


        ordered = sorted(
            distances.items(),
            key=lambda x:
                x[1]
        )


        nearest_pop = (
            ordered[0][0]
        )


        nearest_distance = (
            ordered[0][1]
        )


        threshold = (
            thresholds.get(
                nearest_pop,
                np.nan
            )
        )


        plausible_results.append({
            "IID":
                row["IID"],

            "Nearest_Plausible_1KGP_Pop":
                nearest_pop,

            "Nearest_Plausible_1KGP_Mahalanobis":
                nearest_distance,

            "Nearest_Plausible_1KGP_Threshold99":
                threshold,

            "Nearest_Plausible_1KGP_DistanceRatio":
                (
                    nearest_distance /
                    threshold
                    if np.isfinite(
                        threshold
                    )
                    and threshold > 0
                    else np.nan
                )
        })


    plausible = pd.DataFrame(
        plausible_results
    )


    ancestry = ancestry.merge(
        plausible,
        on="IID",
        how="left",
        validate="one_to_one"
    )


    # ========================================================
    # FINAL GLOBAL PC TABLE
    # ========================================================

    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            21
        )
        if f"PC{i}"
        in study.columns
    ]


    global_pcs = (
        study[
            [
                "IID"
            ]
            +
            pc_columns
        ]
        .copy()
    )


    global_pcs = global_pcs.rename(
        columns={
            pc:
            f"Global_{pc}"
            for pc
            in pc_columns
        }
    )


    # ========================================================
    # CREATE ULTIMATE METADATA
    # ========================================================

    final_ids = set(
        study[
            "IID"
        ]
    )


    ultimate = (
        meta[
            meta[
                "id"
            ].isin(
                final_ids
            )
        ]
        .copy()
    )


    print(
        "\n========================================"
    )

    print(
        "ULTIMATE METADATA MERGE"
    )

    print(
        "========================================"
    )


    print(
        "Filtered metadata rows:",
        len(
            ultimate
        )
    )

    print(
        "Final study PCA samples:",
        len(
            study
        )
    )


    # --------------------------------------------------------
    # Merge final global PCs
    # --------------------------------------------------------

    ultimate = ultimate.merge(
        global_pcs,
        left_on="id",
        right_on="IID",
        how="left",
        validate="one_to_one"
    )


    ultimate = ultimate.drop(
        columns=[
            "IID"
        ]
    )


    # --------------------------------------------------------
    # Merge final 1KGP reference annotations
    # --------------------------------------------------------

    ultimate = ultimate.merge(
        ancestry,
        left_on="id",
        right_on="IID",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_PCA"
        )
    )


    ultimate = ultimate.drop(
        columns=[
            "IID"
        ]
    )


    # --------------------------------------------------------
    # Remove duplicate merge_site generated by ancestry table
    # --------------------------------------------------------

    if "merge_site_PCA" in ultimate.columns:

        ultimate = ultimate.drop(
            columns=[
                "merge_site_PCA"
            ]
        )


    # --------------------------------------------------------
    # QC: no final study mother should be missing PCA
    # --------------------------------------------------------

    if "Global_PC1" in ultimate.columns:

        missing_pc = (
            ultimate[
                "Global_PC1"
            ]
            .isna()
            .sum()
        )

    else:

        missing_pc = len(
            ultimate
        )


    if missing_pc > 0:

        raise ValueError(
            f"{missing_pc} final metadata rows "
            "are missing final global PCA values."
        )


    # --------------------------------------------------------
    # QC: every final study mother should have annotation
    # --------------------------------------------------------

    missing_annotation = (
        ultimate[
            "Nearest_1KGP_Pop"
        ]
        .isna()
        .sum()
    )


    if missing_annotation > 0:

        raise ValueError(
            f"{missing_annotation} final metadata rows "
            "are missing 1KGP reference annotation."
        )


    # --------------------------------------------------------
    # Save ultimate metadata
    # --------------------------------------------------------

    ultimate.to_csv(
        ULTIMATE_METADATA_FILE,
        index=False
    )


    # --------------------------------------------------------
    # Final summaries
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "FINAL 1KGP REFERENCE SUMMARY"
    )

    print(
        "========================================"
    )


    print(
        "\nNearest superpopulation:"
    )

    print(
        ultimate[
            "Nearest_1KGP_SuperPop"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


    print(
        "\nNearest 1000G population:"
    )

    print(
        ultimate[
            "Nearest_1KGP_Pop"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


    print(
        "\nExpected vs nearest 1000G superpopulation:"
    )

    print(
        pd.crosstab(
            ultimate[
                "Expected_superpop"
            ],
            ultimate[
                "Nearest_1KGP_SuperPop"
            ],
            dropna=False
        )
    )


    print(
        "\nSuperpopulation concordance:"
    )

    print(
        ultimate[
            "SuperPop_concordant"
        ]
        .value_counts(
            dropna=False
        )
    )


    print(
        "\nFinal mothers by site:"
    )

    print(
        ultimate[
            "merge_site"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\n========================================"
    )

    print(
        "FINAL GLOBAL PCA PIPELINE COMPLETE"
    )

    print(
        "========================================"
    )


    print(
        "\nUltimate metadata:"
    )

    print(
        ULTIMATE_METADATA_FILE
    )


    print(
        "\nFinal nearest-reference table:"
    )

    print(
        OUTDIR /
        "final_1KGP_nearest_reference.csv"
    )
