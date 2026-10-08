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
PANEL_FILE = "1KGP.metadata.tsv"

INITIAL_PCA_FILE = "pca/joint_global.eigenvec"
INITIAL_EIGENVAL_FILE = "pca/joint_global.eigenval"

FINAL_PCA_FILE = "pca/joint_global_final.eigenvec"
FINAL_EIGENVAL_FILE = "pca/joint_global_final.eigenval"

OUTDIR = Path("pca/global_qc")
OUTDIR.mkdir(parents=True, exist_ok=True)

INITIAL_KEEP_FILE = "qc/maternal_ptb_globalPCA.keep"
ULTIMATE_METADATA_FILE = "momi_mothers_ultimate.csv"

DISTANCE_PCS = ["PC1", "PC2", "PC3", "PC4", "PC5"]

# Broad/global ancestry QC should be conservative
OUTLIER_QUANTILE = 0.999


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
# ============================================================

AFR_POPS = ["ACB", "ASW", "ESN", "GWD", "LWK", "MSL", "YRI"]
SAS_POPS = ["BEB", "GIH", "ITU", "PJL", "STU"]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Global 1000G PCA ancestry QC and final maternal metadata annotation."
)

parser.add_argument(
    "--stage",
    required=True,
    choices=["initial", "final"],
    help="initial = identify/remove broad ancestry outliers; final = annotate cleaned second PCA"
)

args = parser.parse_args()


# ============================================================
# HELPERS
# ============================================================
def build_pooled_population_model(kg):
    groups = {}
    covariance_sum = None
    total_df = 0

    for population, g in kg.groupby("Population"):
        X = g[DISTANCE_PCS].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()

        if len(X) < len(DISTANCE_PCS) + 3:
            print("WARNING: skipping population", population, "because N =", len(X))
            continue

        centroid = X.mean(axis=0)
        covariance = np.cov(X, rowvar=False)

        groups[population] = {
            "centroid": centroid,
            "n": len(X)
        }

        weighted_cov = covariance * (len(X) - 1)

        if covariance_sum is None:
            covariance_sum = weighted_cov
        else:
            covariance_sum += weighted_cov

        total_df += len(X) - 1

    pooled_covariance = covariance_sum / total_df

    mean_variance = np.trace(pooled_covariance) / pooled_covariance.shape[0]
    epsilon = max(mean_variance, 1e-12) * 1e-6
    pooled_covariance += np.eye(pooled_covariance.shape[0]) * epsilon

    pooled_inv_cov = np.linalg.pinv(pooled_covariance)

    return groups, pooled_covariance, pooled_inv_cov
def pooled_population_distances(row, candidate_pops, population_models, pooled_inv_cov):
    x = row[DISTANCE_PCS].astype(float).to_numpy()
    distances = {}

    for pop in candidate_pops:
        if pop not in population_models:
            continue

        distances[pop] = mahalanobis_distance(
            x,
            population_models[pop]["centroid"],
            pooled_inv_cov
        )

    return distances

def clean_string_series(x):
    return x.astype("string").str.strip()


def read_pca(pca_file):
    pca = pd.read_csv(pca_file, sep=r"\s+")

    if "#FID" in pca.columns:
        pca = pca.rename(columns={"#FID": "FID"})

    if "#IID" in pca.columns:
        pca = pca.rename(columns={"#IID": "IID"})

    if "IID" not in pca.columns:
        raise ValueError(f"IID column not found in {pca_file}. Columns: {pca.columns.tolist()}")

    pca["IID"] = clean_string_series(pca["IID"])

    return pca


def read_eigenvalues(eigenval_file):
    eigenvalues = np.loadtxt(eigenval_file)
    variance_pct = eigenvalues / eigenvalues.sum() * 100

    return eigenvalues, variance_pct


def read_metadata():
    meta = pd.read_csv(METADATA_FILE, dtype=str, low_memory=False)

    if "id" not in meta.columns:
        raise ValueError(f"Column 'id' is not present in {METADATA_FILE}")

    if "merge_site" not in meta.columns:
        raise ValueError(f"Column 'merge_site' is not present in {METADATA_FILE}")

    meta["id"] = clean_string_series(meta["id"])
    meta["merge_site"] = clean_string_series(meta["merge_site"])

    meta["Expected_superpop"] = pd.NA

    meta.loc[meta["merge_site"].isin(AFR_SITES), "Expected_superpop"] = "AFR"
    meta.loc[meta["merge_site"].isin(SAS_SITES), "Expected_superpop"] = "SAS"

    return meta


def read_1000g_panel():
    panel = pd.read_csv(PANEL_FILE, sep="\t", dtype=str, low_memory=False)

    print("\n========================================")
    print("1000 GENOMES METADATA")
    print("========================================")
    print("Rows in metadata:", len(panel))
    print("\nColumns:")
    print(panel.columns.tolist())

    required = [
        "Sample name",
        "Population code",
        "Population name",
        "Superpopulation code",
        "Superpopulation name"
    ]

    missing = [col for col in required if col not in panel.columns]

    if missing:
        raise ValueError("1KGP metadata is missing required columns: " + ", ".join(missing))

    panel = panel[required].copy()

    panel = panel.rename(columns={
        "Sample name": "IID",
        "Population code": "Population",
        "Population name": "Population_name",
        "Superpopulation code": "Superpopulation",
        "Superpopulation name": "Superpopulation_name"
    })

    for col in ["IID", "Population", "Population_name", "Superpopulation", "Superpopulation_name"]:
        panel[col] = panel[col].astype("string").str.strip()

    # Keep first annotation when metadata contains multiple project labels
    for col in ["Population", "Population_name", "Superpopulation", "Superpopulation_name"]:
        panel[col] = panel[col].str.split(",").str[0].str.strip()

    panel = panel[panel["IID"].notna()].copy()

    duplicate_ids = panel[panel.duplicated(subset="IID", keep=False)].copy()

    print("\nDuplicate Sample name rows:", len(duplicate_ids))

    if len(duplicate_ids) > 0:
        duplicate_ids.to_csv(OUTDIR / "duplicate_1KGP_metadata_IDs.csv", index=False)
        raise ValueError("1KGP metadata contains duplicate Sample name values.")

    print("\n1000G superpopulations:")
    print(panel["Superpopulation"].value_counts(dropna=False).sort_index())

    print("\n1000G populations:")
    print(panel["Population"].value_counts(dropna=False).sort_index())

    return panel


def label_joint_pca(pca, meta, panel):
    pca = pca.merge(panel, on="IID", how="left", validate="many_to_one")

    study_labels = meta[
        ["id", "merge_site", "Expected_superpop"]
    ].rename(columns={"id": "IID"})

    if study_labels["IID"].duplicated().any():
        duplicates = study_labels[study_labels["IID"].duplicated(keep=False)]
        raise ValueError(f"Study metadata contains duplicate IDs. Found {len(duplicates)} duplicate rows.")

    pca = pca.merge(study_labels, on="IID", how="left", validate="many_to_one")

    pca["Source"] = "Unknown"
    pca.loc[pca["Superpopulation"].notna(), "Source"] = "1000G"
    pca.loc[pca["merge_site"].notna(), "Source"] = "Study"

    kg = pca[pca["Source"].eq("1000G")].copy()
    study = pca[pca["Source"].eq("Study")].copy()
    unknown = pca[pca["Source"].eq("Unknown")].copy()

    return pca, kg, study, unknown


def mahalanobis_distance(x, centroid, inv_cov):
    delta = x - centroid
    distance_squared = delta.T @ inv_cov @ delta
    distance_squared = max(float(distance_squared), 0.0)

    return float(np.sqrt(distance_squared))


def make_reference_model(X):
    centroid = X.mean(axis=0)
    covariance = np.cov(X, rowvar=False)

    mean_variance = np.trace(covariance) / covariance.shape[0]

    if not np.isfinite(mean_variance):
        mean_variance = 1.0

    epsilon = max(mean_variance, 1e-12) * 1e-6
    covariance = covariance + np.eye(covariance.shape[0]) * epsilon
    inv_cov = np.linalg.pinv(covariance)

    return {
        "centroid": centroid,
        "covariance": covariance,
        "inv_cov": inv_cov,
        "n": len(X)
    }


# ============================================================
# INITIAL QC HELPERS: SUPERPOPULATION LEVEL
# ============================================================

def build_superpop_reference(kg, superpop):
    g = kg[kg["Superpopulation"].eq(superpop)].copy()

    X = (
        g[DISTANCE_PCS]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .to_numpy()
    )

    minimum_n = len(DISTANCE_PCS) + 10

    if len(X) < minimum_n:
        raise ValueError(f"Too few 1000G samples for superpopulation {superpop}: N={len(X)}")

    return make_reference_model(X)


def calculate_superpop_loo_distances(kg, superpop):
    g = (
        kg[kg["Superpopulation"].eq(superpop)]
        .copy()
        .reset_index(drop=True)
    )

    X = (
        g[DISTANCE_PCS]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .to_numpy()
    )

    distances = []

    for i in range(len(X)):
        training = np.delete(X, i, axis=0)
        model = make_reference_model(training)
        distance = mahalanobis_distance(X[i], model["centroid"], model["inv_cov"])
        distances.append(distance)

    return np.array(distances)


# ============================================================
# FINAL ANNOTATION HELPERS: POPULATION LEVEL
# ============================================================

def build_reference_stats(kg, populations=None):
    reference_stats = {}

    for population, g in kg.groupby("Population"):

        if populations is not None and population not in populations:
            continue

        X = (
            g[DISTANCE_PCS]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .to_numpy()
        )

        minimum_n = len(DISTANCE_PCS) + 3

        if len(X) < minimum_n:
            print("WARNING: skipping population", population, "because N =", len(X))
            continue

        reference_stats[population] = make_reference_model(X)

    return reference_stats


def calculate_loo_reference_distances(kg, populations=None):
    distance_lookup = {}

    for population, g in kg.groupby("Population"):

        if populations is not None and population not in populations:
            continue

        X = (
            g[DISTANCE_PCS]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .to_numpy()
        )

        minimum_n = len(DISTANCE_PCS) + 4

        if len(X) < minimum_n:
            continue

        distances = []

        for i in range(len(X)):
            training = np.delete(X, i, axis=0)
            model = make_reference_model(training)
            distance = mahalanobis_distance(X[i], model["centroid"], model["inv_cov"])
            distances.append(distance)

        distance_lookup[population] = np.array(distances)

    return distance_lookup


def calculate_reference_thresholds(reference_distances):
    return {
        pop: float(np.quantile(distances, OUTLIER_QUANTILE))
        for pop, distances in reference_distances.items()
    }


def empirical_percentile(distance, reference_distances):
    if len(reference_distances) == 0:
        return np.nan

    return float(100 * np.mean(reference_distances <= distance))


def population_distances(row, candidate_pops, reference_stats):
    x = row[DISTANCE_PCS].astype(float).to_numpy()

    distances = {}

    for pop in candidate_pops:

        if pop not in reference_stats:
            continue

        stats = reference_stats[pop]
        distances[pop] = mahalanobis_distance(x, stats["centroid"], stats["inv_cov"])

    return distances


# ============================================================
# PLOTTING
# ============================================================

def plot_pca(kg, study, variance_pct, pc_x, pc_y, outfile, title, flagged_ids=None):
    pc_x_index = int(pc_x.replace("PC", "")) - 1
    pc_y_index = int(pc_y.replace("PC", "")) - 1

    fig, ax = plt.subplots(figsize=(12, 8))

    for superpop, g in kg.groupby("Superpopulation"):
        ax.scatter(
            g[pc_x],
            g[pc_y],
            s=18,
            alpha=0.35,
            label=f"1000G {superpop}"
        )

    for site, g in study.groupby("merge_site"):
        ax.scatter(
            g[pc_x],
            g[pc_y],
            s=28,
            alpha=0.70,
            marker="x",
            label=site
        )

    if flagged_ids is not None:
        flagged_set = set(flagged_ids)
        bad = study[study["IID"].isin(flagged_set)]

        if len(bad) > 0:
            ax.scatter(
                bad[pc_x],
                bad[pc_y],
                s=110,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Flagged ancestry outlier"
            )

    ax.set_xlabel(f"{pc_x} ({variance_pct[pc_x_index]:.2f}%)")
    ax.set_ylabel(f"{pc_y} ({variance_pct[pc_y_index]:.2f}%)")
    ax.set_title(title)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved plot:", outfile)



def plot_population_pca(kg, study, variance_pct, pc_x, pc_y, outfile, title, superpop):
    kg_plot = kg[kg["Superpopulation"].eq(superpop)].copy()
    study_plot = study[study["Expected_superpop"].eq(superpop)].copy()

    pc_x_index = int(pc_x.replace("PC", "")) - 1
    pc_y_index = int(pc_y.replace("PC", "")) - 1

    fig, ax = plt.subplots(figsize=(12, 8))

    # --------------------------------------------------------
    # Plot study cohort FIRST (background)
    # --------------------------------------------------------
    for site, g in study_plot.groupby("merge_site"):
        ax.scatter(
            g[pc_x],
            g[pc_y],
            s=14,
            alpha=0.25,
            marker="x",
            label=site,
            zorder=1
        )

    # --------------------------------------------------------
    # Plot 1000G SECOND (foreground)
    # --------------------------------------------------------
    # Plot 1000G second (foreground) as hollow circles
    for i, (population, g) in enumerate(kg_plot.groupby("Population")):
        ax.scatter(
            g[pc_x],
            g[pc_y],
            s=28,
            facecolors="none",
            edgecolors=f"C{i}",
            linewidths=1.1,
            alpha=0.9,
            marker="o",
            label=f"1000G {population}",
            zorder=3
        )

  

    ax.set_xlabel(f"{pc_x} ({variance_pct[pc_x_index]:.2f}%)")
    ax.set_ylabel(f"{pc_y} ({variance_pct[pc_y_index]:.2f}%)")
    ax.set_title(title)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved plot:", outfile)
def print_pca_summary(pca, kg, study, unknown):
    print("\n========================================")
    print("JOINT PCA SUMMARY")
    print("========================================")

    print("Total PCA samples:", len(pca))
    print("Study:", len(study))
    print("1000G:", len(kg))
    print("Unknown:", len(unknown))

    print("\nStudy samples by site:")
    print(study["merge_site"].value_counts().sort_index())

    print("\n1000G superpopulations:")
    print(kg["Superpopulation"].value_counts().sort_index())


# ============================================================
# LOAD COMMON INPUTS
# ============================================================

meta = read_metadata()
panel = read_1000g_panel()


# ============================================================
# INITIAL STAGE
# ============================================================

if args.stage == "initial":

    print("\n########################################")
    print("# INITIAL GLOBAL ANCESTRY QC")
    print("########################################")

    pca = read_pca(INITIAL_PCA_FILE)
    eigenvalues, variance_pct = read_eigenvalues(INITIAL_EIGENVAL_FILE)

    pca, kg, study, unknown = label_joint_pca(pca, meta, panel)

    print_pca_summary(pca, kg, study, unknown)

    pca.to_csv(
        OUTDIR / "initial_joint_global_labeled.csv",
        index=False
    )

    plot_pairs = [
        ("PC1", "PC2"),
        ("PC1", "PC3"),
        ("PC2", "PC3"),
        ("PC1", "PC4"),
        ("PC2", "PC4")
    ]

    # --------------------------------------------------------
    # BEFORE plots
    # --------------------------------------------------------

    for pc_x, pc_y in plot_pairs:
        plot_pca(
            kg=kg,
            study=study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=OUTDIR / f"BEFORE_{pc_x}_{pc_y}.png",
            title=f"Before global ancestry QC: {pc_x} vs {pc_y}"
        )

    # --------------------------------------------------------
    # Broad AFR / SAS reference models
    # --------------------------------------------------------

    superpop_models = {
        "AFR": build_superpop_reference(kg, "AFR"),
        "SAS": build_superpop_reference(kg, "SAS")
    }

    superpop_reference_distances = {
        "AFR": calculate_superpop_loo_distances(kg, "AFR"),
        "SAS": calculate_superpop_loo_distances(kg, "SAS")
    }

    superpop_thresholds = {
        superpop: float(np.quantile(distances, OUTLIER_QUANTILE))
        for superpop, distances in superpop_reference_distances.items()
    }

    threshold_table = []

    for superpop in ["AFR", "SAS"]:
        threshold_table.append({
            "Superpopulation": superpop,
            "N_reference": superpop_models[superpop]["n"],
            "Quantile": OUTLIER_QUANTILE,
            "Threshold": superpop_thresholds[superpop]
        })

    threshold_df = pd.DataFrame(threshold_table)

    threshold_df.to_csv(
        OUTDIR / "initial_1000G_superpopulation_thresholds.csv",
        index=False
    )

    print("\n========================================")
    print("1000G SUPERPOPULATION QC THRESHOLDS")
    print("========================================")
    print(threshold_df.to_string(index=False))

    # --------------------------------------------------------
    # Evaluate every study sample
    # --------------------------------------------------------

    results = []

    for _, row in study.iterrows():

        expected = row["Expected_superpop"]

        if expected not in ["AFR", "SAS"]:
            print(
                "WARNING: no expected superpopulation for",
                row["IID"],
                row["merge_site"]
            )
            continue

        x = row[DISTANCE_PCS].astype(float).to_numpy()

        model = superpop_models[expected]
        distance = mahalanobis_distance(x, model["centroid"], model["inv_cov"])

        threshold = superpop_thresholds[expected]
        reference_distances = superpop_reference_distances[expected]
        percentile = empirical_percentile(distance, reference_distances)
        distance_ratio = distance / threshold
        global_outlier = distance > threshold

        results.append({
            "IID": row["IID"],
            "merge_site": row["merge_site"],
            "Expected_superpop": expected,
            "Superpop_Mahalanobis": distance,
            "Superpop_Threshold": threshold,
            "Superpop_Empirical_Percentile": percentile,
            "Distance_over_Threshold": distance_ratio,
            "Global_PCA_outlier": global_outlier
        })

    qc = pd.DataFrame(results)
    qc = qc.sort_values("Distance_over_Threshold", ascending=False)

    qc.to_csv(
        OUTDIR / "initial_global_PCA_ancestry_QC.csv",
        index=False
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n========================================")
    print("GLOBAL ANCESTRY QC DISTRIBUTION")
    print("========================================")

    print(
        qc.groupby("Expected_superpop")[
            ["Superpop_Mahalanobis", "Distance_over_Threshold"]
        ].describe()
    )

    print("\nFlagged by expected ancestry:")
    print(
        pd.crosstab(
            qc["Expected_superpop"],
            qc["Global_PCA_outlier"]
        )
    )

    print("\nFlagged by study site:")
    print(
        pd.crosstab(
            qc["merge_site"],
            qc["Global_PCA_outlier"]
        )
    )

    # --------------------------------------------------------
    # Flag outliers
    # --------------------------------------------------------

    flagged = qc[
        qc["Global_PCA_outlier"]
    ].copy()

    flagged.to_csv(
        OUTDIR / "initial_global_PCA_outliers.csv",
        index=False
    )

    print("\n========================================")
    print("INITIAL GLOBAL ANCESTRY OUTLIERS")
    print("========================================")

    print("Study samples tested:", len(qc))
    print("Outliers flagged:", len(flagged))

    print("\nOutliers by site:")
    print(flagged["merge_site"].value_counts().sort_index())

    # --------------------------------------------------------
    # BEFORE plots with flagged samples
    # --------------------------------------------------------

    for pc_x, pc_y in plot_pairs:
        plot_pca(
            kg=kg,
            study=study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=OUTDIR / f"BEFORE_FLAGGED_{pc_x}_{pc_y}.png",
            title=f"Initial PCA with ancestry outliers flagged: {pc_x} vs {pc_y}",
            flagged_ids=flagged["IID"].tolist()
        )

    # --------------------------------------------------------
    # Remove flagged study samples
    # --------------------------------------------------------

    flagged_ids = set(flagged["IID"])

    clean_study = study[
        ~study["IID"].isin(flagged_ids)
    ].copy()

    # --------------------------------------------------------
    # AFTER plots using same initial PCA coordinates
    # --------------------------------------------------------

    for pc_x, pc_y in plot_pairs:
        plot_pca(
            kg=kg,
            study=clean_study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=OUTDIR / f"AFTER_REMOVAL_{pc_x}_{pc_y}.png",
            title=f"After removing global ancestry outliers: {pc_x} vs {pc_y}"
        )

    # --------------------------------------------------------
    # PLINK keep file
    #
    # Use actual FID/IID pairs from original maternal .fam.
    # --------------------------------------------------------

    fam = pd.read_csv(
        "qc/maternal_ptb.fam",
        sep=r"\s+",
        header=None,
        dtype=str
    )

    fam.columns = [
        "FID",
        "IID",
        "PAT",
        "MAT",
        "SEX",
        "PHENO"
    ]

    fam["IID"] = fam["IID"].astype(str).str.strip()

    retained_ids = set(
        clean_study["IID"]
        .astype(str)
        .str.strip()
    )

    clean_keep = fam[
        fam["IID"].isin(retained_ids)
    ][
        ["FID", "IID"]
    ].copy()

    missing_from_fam = retained_ids - set(clean_keep["IID"])

    print("\nRetained study IDs:", len(retained_ids))
    print("Retained IDs found in maternal FAM:", len(clean_keep))
    print("Retained IDs missing from maternal FAM:", len(missing_from_fam))

    if len(missing_from_fam) > 0:
        pd.Series(
            sorted(missing_from_fam),
            name="IID"
        ).to_csv(
            OUTDIR / "retained_PCA_IDs_missing_from_maternal_fam.csv",
            index=False
        )

        raise ValueError(
            "Some retained PCA IDs are not present in qc/maternal_ptb.fam."
        )

    clean_keep.to_csv(
        INITIAL_KEEP_FILE,
        sep="\t",
        index=False,
        header=False
    )

    clean_study.to_csv(
        OUTDIR / "initial_globalPCA_retained_study.csv",
        index=False
    )

    print("\n========================================")
    print("INITIAL GLOBAL PCA QC COMPLETE")
    print("========================================")

    print("Study samples before ancestry QC:", len(study))
    print("Global ancestry outliers removed:", len(flagged))
    print("Study samples retained:", len(clean_study))

    print("\nPLINK keep file:")
    print(INITIAL_KEEP_FILE)

    print("\nNEXT STEP:")
    print(
        "Create qc/maternal_ptb_globalPCA with this keep file, "
        "then rerun the joint 1000G PCA to generate:"
    )
    print(FINAL_PCA_FILE)
    print(FINAL_EIGENVAL_FILE)


# ============================================================
# FINAL STAGE
# ============================================================

elif args.stage == "final":

    print("\n########################################")
    print("# FINAL CLEAN GLOBAL PCA ANNOTATION")
    print("########################################")

    if not Path(FINAL_PCA_FILE).exists():
        raise FileNotFoundError(
            f"Final PCA file does not exist: {FINAL_PCA_FILE}\n"
            "Run the second joint PCA first."
        )

    if not Path(FINAL_EIGENVAL_FILE).exists():
        raise FileNotFoundError(
            f"Final eigenvalue file does not exist: {FINAL_EIGENVAL_FILE}"
        )

    pca = read_pca(FINAL_PCA_FILE)
    eigenvalues, variance_pct = read_eigenvalues(FINAL_EIGENVAL_FILE)

    pca, kg, study, unknown = label_joint_pca(pca, meta, panel)

    print_pca_summary(pca, kg, study, unknown)

    pca.to_csv(
        OUTDIR / "final_joint_global_labeled.csv",
        index=False
    )

    plot_pairs = [
        ("PC1", "PC2"),
        ("PC1", "PC3"),
        ("PC2", "PC3"),
        ("PC1", "PC4"),
        ("PC2", "PC4")
    ]

    for pc_x, pc_y in plot_pairs:
        plot_pca(
            kg=kg,
            study=study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=OUTDIR / f"FINAL_{pc_x}_{pc_y}.png",
            title=f"Final cleaned study cohort + 1000G: {pc_x} vs {pc_y}"
        )
    # --------------------------------------------------------
    # Final population-level 1000G PCA plots
    # --------------------------------------------------------

    population_plot_pairs = [
        ("PC1", "PC2"),
        ("PC1", "PC3"),
        ("PC2", "PC3")
    ]

    for pc_x, pc_y in population_plot_pairs:

        plot_population_pca(
            kg=kg,
            study=study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=OUTDIR / f"FINAL_SAS_POPULATIONS_{pc_x}_{pc_y}.png",
            title=f"South Asia cohorts + 1000G populations: {pc_x} vs {pc_y}",
            superpop="SAS"
        )

        plot_population_pca(
            kg=kg,
            study=study,
            variance_pct=variance_pct,
            pc_x=pc_x,
            pc_y=pc_y,
            outfile=OUTDIR / f"FINAL_AFR_POPULATIONS_{pc_x}_{pc_y}.png",
            title=f"African cohorts + 1000G populations: {pc_x} vs {pc_y}",
            superpop="AFR"
        )
    # --------------------------------------------------------
    # Population-level 1000G reference models
    # --------------------------------------------------------

    all_1000g_pops = sorted(
        kg["Population"]
        .dropna()
        .unique()
        .tolist()
    )

    population_models, pooled_covariance, pooled_inv_cov = build_pooled_population_model(kg)
    
    print("\n========================================")
    print("POOLED 1KGP POPULATION MODEL")
    print("========================================")
    print("Populations:", len(population_models))
    print("Reference samples:", sum(x["n"] for x in population_models.values()))

    reference_distances = calculate_loo_reference_distances(
        kg,
        populations=all_1000g_pops
    )

    thresholds = calculate_reference_thresholds(
        reference_distances
    )

    # --------------------------------------------------------
    # Population -> superpopulation lookup
    # --------------------------------------------------------

    pop_superpop = (
        kg[
            ["Population", "Superpopulation"]
        ]
        .drop_duplicates()
    )

    duplicate_pop_labels = pop_superpop[
        pop_superpop["Population"].duplicated(keep=False)
    ]

    if len(duplicate_pop_labels) > 0:
        raise ValueError(
            "A 1000G population maps to multiple superpopulation labels."
        )

    pop_to_superpop = dict(
        zip(
            pop_superpop["Population"],
            pop_superpop["Superpopulation"]
        )
    )

    # Names too
    pop_names = (
        kg[
            ["Population", "Population_name"]
        ]
        .drop_duplicates()
    )

    pop_to_name = dict(
        zip(
            pop_names["Population"],
            pop_names["Population_name"]
        )
    )

    superpop_names = (
        kg[
            ["Superpopulation", "Superpopulation_name"]
        ]
        .drop_duplicates()
    )

    superpop_to_name = dict(
        zip(
            superpop_names["Superpopulation"],
            superpop_names["Superpopulation_name"]
        )
    )

    # --------------------------------------------------------
    # Final nearest 1000G reference
    # --------------------------------------------------------

    results = []

    for _, row in study.iterrows():

        distances = pooled_population_distances(
            row,
            all_1000g_pops,
            population_models,
            pooled_inv_cov
        )

        if len(distances) == 0:
            raise ValueError(
                f"No 1000G population distances could be calculated for {row['IID']}"
            )

        ordered = sorted(
            distances.items(),
            key=lambda x: x[1]
        )

        nearest_pop = ordered[0][0]
        nearest_distance = ordered[0][1]

        second_pop = ordered[1][0] if len(ordered) > 1 else pd.NA
        second_distance = ordered[1][1] if len(ordered) > 1 else np.nan

        nearest_superpop = pop_to_superpop.get(nearest_pop, pd.NA)

        second_superpop = (
            pop_to_superpop.get(second_pop, pd.NA)
            if pd.notna(second_pop)
            else pd.NA
        )

        threshold = thresholds.get(nearest_pop, np.nan)

        percentile = (
            empirical_percentile(
                nearest_distance,
                reference_distances[nearest_pop]
            )
            if nearest_pop in reference_distances
            else np.nan
        )

        distance_ratio = (
            nearest_distance / threshold
            if np.isfinite(threshold) and threshold > 0
            else np.nan
        )

        expected_superpop = row["Expected_superpop"]
        superpop_concordant = expected_superpop == nearest_superpop

        result = {
            "IID": row["IID"],
            "merge_site": row["merge_site"],
            "Expected_superpop": expected_superpop,

            "Nearest_1KGP_Pop": nearest_pop,
            "Nearest_1KGP_Pop_Name": pop_to_name.get(nearest_pop, pd.NA),

            "Nearest_1KGP_SuperPop": nearest_superpop,
            "Nearest_1KGP_SuperPop_Name": superpop_to_name.get(
                nearest_superpop,
                pd.NA
            ),

            "Nearest_1KGP_PooledMahalanobis": nearest_distance,

            "Second_1KGP_Pop": second_pop,
            "Second_1KGP_Pop_Name": pop_to_name.get(second_pop, pd.NA),

            "Second_1KGP_SuperPop": second_superpop,
            "Second_1KGP_SuperPop_Name": superpop_to_name.get(
                second_superpop,
                pd.NA
            ),

            "Second_1KGP_Mahalanobis": second_distance,

            "Nearest_vs_Second_Delta": second_distance - nearest_distance,

            "Nearest_vs_Second_Ratio": (
                second_distance / nearest_distance
                if nearest_distance > 0
                else np.nan
            ),

            "Nearest_1KGP_Threshold": threshold,
            "Nearest_1KGP_Empirical_Percentile": percentile,
            "Nearest_1KGP_DistanceRatio": distance_ratio,

            "SuperPop_concordant": superpop_concordant
        }

        for pop, distance in distances.items():
            result[f"PooledMahalanobis_{pop}"] = distance

        results.append(result)

    ancestry = pd.DataFrame(results)

    ancestry.to_csv(
        OUTDIR / "final_1KGP_nearest_reference.csv",
        index=False
    )

    # --------------------------------------------------------
    # Expected-group nearest reference too
    # --------------------------------------------------------

    plausible_results = []

    for _, row in study.iterrows():

        expected = row["Expected_superpop"]

        if expected == "AFR":
            candidate_pops = AFR_POPS

        elif expected == "SAS":
            candidate_pops = SAS_POPS

        else:
            continue

        distances = pooled_population_distances(
            row,
            candidate_pops,
            population_models,
            pooled_inv_cov
        )

        if len(distances) == 0:
            continue

        ordered = sorted(
            distances.items(),
            key=lambda x: x[1]
        )

        nearest_pop = ordered[0][0]
        nearest_distance = ordered[0][1]
        threshold = thresholds.get(nearest_pop, np.nan)

        plausible_results.append({
            "IID": row["IID"],
            "Nearest_Plausible_1KGP_Pop": nearest_pop,
            "Nearest_Plausible_1KGP_Pop_Name": pop_to_name.get(
                nearest_pop,
                pd.NA
            ),
            "Nearest_Plausible_1KGP_Mahalanobis": nearest_distance,
            "Nearest_Plausible_1KGP_Threshold": threshold,
            "Nearest_Plausible_1KGP_DistanceRatio": (
                nearest_distance / threshold
                if np.isfinite(threshold) and threshold > 0
                else np.nan
            )
        })

    plausible = pd.DataFrame(plausible_results)

    ancestry = ancestry.merge(
        plausible,
        on="IID",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Global PCs
    # --------------------------------------------------------

    pc_columns = [
        f"PC{i}"
        for i in range(1, 21)
        if f"PC{i}" in study.columns
    ]

    global_pcs = study[
        ["IID"] + pc_columns
    ].copy()

    global_pcs = global_pcs.rename(
        columns={
            pc: f"Global_{pc}"
            for pc in pc_columns
        }
    )

    # --------------------------------------------------------
    # Ultimate metadata
    # --------------------------------------------------------

    final_ids = set(study["IID"])

    ultimate = meta[
        meta["id"].isin(final_ids)
    ].copy()

    print("\n========================================")
    print("ULTIMATE METADATA MERGE")
    print("========================================")

    print("Filtered metadata rows:", len(ultimate))
    print("Final study PCA samples:", len(study))

    ultimate = ultimate.merge(
        global_pcs,
        left_on="id",
        right_on="IID",
        how="left",
        validate="one_to_one"
    )

    ultimate = ultimate.drop(columns=["IID"])

    ultimate = ultimate.merge(
        ancestry,
        left_on="id",
        right_on="IID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_PCA")
    )

    ultimate = ultimate.drop(columns=["IID"])

    if "merge_site_PCA" in ultimate.columns:
        ultimate = ultimate.drop(columns=["merge_site_PCA"])

    if "Global_PC1" in ultimate.columns:
        missing_pc = ultimate["Global_PC1"].isna().sum()
    else:
        missing_pc = len(ultimate)

    if missing_pc > 0:
        raise ValueError(
            f"{missing_pc} final metadata rows are missing final global PCA values."
        )

    missing_annotation = ultimate["Nearest_1KGP_Pop"].isna().sum()

    if missing_annotation > 0:
        raise ValueError(
            f"{missing_annotation} final metadata rows are missing 1KGP reference annotation."
        )

    ultimate.to_csv(
        ULTIMATE_METADATA_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summaries
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL 1KGP REFERENCE SUMMARY")
    print("========================================")

    print("\nNearest superpopulation:")
    print(
        ultimate["Nearest_1KGP_SuperPop"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nNearest 1000G population:")
    print(
        ultimate["Nearest_1KGP_Pop"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nExpected vs nearest 1000G superpopulation:")
    print(
        pd.crosstab(
            ultimate["Expected_superpop"],
            ultimate["Nearest_1KGP_SuperPop"],
            dropna=False
        )
    )

    print("\nSuperpopulation concordance:")
    print(
        ultimate["SuperPop_concordant"]
        .value_counts(dropna=False)
    )

    print("\nFinal mothers by site:")
    print(
        ultimate["merge_site"]
        .value_counts()
        .sort_index()
    )

    print("\n========================================")
    print("FINAL GLOBAL PCA PIPELINE COMPLETE")
    print("========================================")

    print("\nUltimate metadata:")
    print(ULTIMATE_METADATA_FILE)

    print("\nFinal nearest-reference table:")
    print(OUTDIR / "final_1KGP_nearest_reference.csv")
