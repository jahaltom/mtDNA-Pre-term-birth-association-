import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

MOMI_FILE = "momi_pregnancy_level.csv"
SAMPLES_FILE = "samples.tab"
MTCN_FILE = "momi_mtcn_per_sample.csv"
MISSINGNESS_FILE = "qc/cohort_raw_missing.smiss"
HAPLOGREP_FILE = "haplogrep3OUT_22175"
DICT= "momi_pregnancy_level_dictionary.csv"
OUTPUT_FILE = "momi_mothers_merged.csv"


# ============================================================
# HELPERS
# ============================================================

def clean(x):
    if pd.isna(x):
        return pd.NA

    x = str(x).strip()

    if x in ["", "NA", "N/A", "nan", "None", "."]:
        return pd.NA

    return x


def zambia_subcohort(x):
    if pd.isna(x):
        return pd.NA

    x = str(x).strip().upper()

    if x.startswith("17"):
        return "IPOP"

    if x.startswith("30"):
        return "ZAPPS1"

    if x.startswith("Z2"):
        return "ZAPPS2"

    return pd.NA

def collapse_id_biospecimen_pairs(group):
    pairs = []

    for _, row in group.iterrows():
        gid = row["gencove_id"]
        bio = row["biospecimen"]

        if pd.isna(gid):
            continue

        gid = str(gid).strip()

        if pd.isna(bio):
            bio = "NA"
        else:
            bio = str(bio).strip()

        pairs.append(
            f"{gid}|{bio}"
        )

    # preserve order, remove exact duplicates
    pairs = list(dict.fromkeys(pairs))

    if len(pairs) == 0:
        return pd.NA

    return ";".join(pairs)
def collapse_ids(x):
    vals = list(
        dict.fromkeys(
            x.dropna().astype(str)
        )
    )

    vals = [
        v.strip()
        for v in vals
        if v.strip() != ""
    ]

    if len(vals) == 0:
        return pd.NA

    return ";".join(vals)


def get_other_ids(row):

    if pd.isna(row["All_ids"]):
        return pd.NA

    ids = str(row["All_ids"]).split(";")

    other = [
        x
        for x in ids
        if x != str(row["gencove_id"])
    ]

    if len(other) == 0:
        return pd.NA

    return ";".join(other)


# ============================================================
# READ DATA
# ============================================================

momi = pd.read_csv(
    MOMI_FILE,
    dtype=str,
    low_memory=False
)

samples_raw = pd.read_csv(
    SAMPLES_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)

mtcn = pd.read_csv(
    MTCN_FILE,
    dtype=str,
    low_memory=False
)

missing = pd.read_csv(
    MISSINGNESS_FILE,
    sep=r"\s+",
    dtype={"IID": str}
)
haplogrep = pd.read_csv(
    HAPLOGREP_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)

# Update MOMI columns
dictFile = pd.read_csv(
    DICT,
    dtype=str,
    low_memory=False
)
rename_map = dict(zip(dictFile["var_base"], dictFile["original_name"].str.upper()))
momi = momi.rename(columns=rename_map)




print("\n========================================")
print("ORIGINAL DATA")
print("========================================")

print(
    "Original MOMI rows:",
    len(momi)
)

print(
    "Original samples.tab rows:",
    len(samples_raw)
)

print(
    "Original mtCN rows:",
    len(mtcn)
)

# ============================================================
# CLEAN RAW samples.tab
# ============================================================

for col in [
    "id",
    "site",
    "ORIG_ID",
    "M/C",
    "Subject_ID",
    "Sex",
    "Vial_ID"
]:
    samples_raw[col] = (
        samples_raw[col]
        .apply(clean)
    )


# ============================================================
# REMOVE SPOTBio
# ============================================================

before = len(samples_raw)

samples_raw = samples_raw[
    samples_raw["site"].ne(
        "SPOTBio"
    )
].copy()


print("\n========================================")
print("SPOTBio FILTER")
print("========================================")

print(
    "SPOTBio rows removed:",
    before - len(samples_raw)
)

# ============================================================
# VERIFY PEER DUPLICATE Subject_ID CLAIM USING samples.tab
# ============================================================

dup_subject_rows = samples_raw[
    samples_raw["Subject_ID"].notna()
].copy()

# Keep only Subject_IDs appearing more than once
dup_subject_rows = dup_subject_rows[
    dup_subject_rows.duplicated(
        subset=["site", "Subject_ID"],
        keep=False
    )
].copy()


peer_subject_summary = (
    dup_subject_rows
    .groupby(
        ["site", "Subject_ID"],
        dropna=False
    )
    .agg(
        N_records=("id", "size"),
        N_ORIG_ID=("ORIG_ID", "nunique"),
        N_vials=("Vial_ID", "nunique")
    )
    .reset_index()
)


print("\n========================================")
print("DUPLICATE Subject_ID CHECK IN samples.tab")
print("========================================")

print(
    "Duplicate Subject_ID groups:",
    len(peer_subject_summary)
)

print(
    "Groups with >1 ORIG_ID "
    "(likely multiple pregnancies):",
    (peer_subject_summary["N_ORIG_ID"] > 1).sum()
)

print(
    "Groups with same ORIG_ID "
    "(replicate/same-pregnancy cases):",
    (peer_subject_summary["N_ORIG_ID"] == 1).sum()
)
# ============================================================
# CLEAN MISSINGNESS
# ============================================================

missing["IID"] = (
    missing["IID"]
    .astype("string")
    .str.strip()
)

missing["F_MISS"] = pd.to_numeric(
    missing["F_MISS"],
    errors="coerce"
)

missing = missing[
    [
        "IID",
        "F_MISS"
    ]
].copy()


# In case the missingness file somehow contains duplicate IIDs,
# retain the lowest F_MISS.
missing = (
    missing
    .sort_values(
        "F_MISS",
        na_position="last"
    )
    .drop_duplicates(
        subset="IID",
        keep="first"
    )
)


# ============================================================
# CLEAN mtCN METADATA
# ============================================================

for col in [
    "gencove_id",
    "SITE_CODE",
    "site",
    "ORIG_ID",
    "MC",
    "Subject_ID",
    "biospecimen"
]:
    if col in mtcn.columns:
        mtcn[col] = mtcn[col].apply(clean)


mtcn["MC"] = (
    mtcn["MC"]
    .astype("string")
    .str.upper()
    .str.strip()
)


# ============================================================
# REMOVE INVALID mtCN RECORDS
# ============================================================

before = len(mtcn)

mtcn = mtcn[
    mtcn["SITE_CODE"].notna() &
    mtcn["ORIG_ID"].notna() &
    mtcn["MC"].isin(["M", "C"]) &
    mtcn["gencove_id"].notna()
].copy()

print("\n========================================")
print("mtCN KEY QC")
print("========================================")

print(
    "Rows with valid SITE_CODE + ORIG_ID + MC + gencove_id:",
    len(mtcn)
)

print(
    "Invalid rows removed:",
    before - len(mtcn)
)


# ============================================================
# ADD RAW COHORT GENOTYPE MISSINGNESS
#
# gencove_id in mtCN should correspond to PLINK IID.
# ============================================================

mtcn = mtcn.merge(
    missing,
    left_on="gencove_id",
    right_on="IID",
    how="left",
    validate="many_to_one"
)

mtcn = mtcn.drop(
    columns="IID"
)

# ============================================================
# DEFINE USABLE GENOTYPE RECORDS
# ============================================================

mtcn["In_samples_tab"] = (
    mtcn["gencove_id"]
    .isin(samples_raw["id"])
)

mtcn["In_raw_genotype"] = (
    mtcn["F_MISS"]
    .notna()
)

mtcn["Usable_genotype"] = (
    mtcn["In_samples_tab"] &
    mtcn["In_raw_genotype"]
)

print("\n========================================")
print("GENOTYPE AVAILABILITY")
print("========================================")

print(
    "mtCN records present in samples.tab:",
    mtcn["In_samples_tab"].sum()
)

print(
    "mtCN records present in raw genotype cohort:",
    mtcn["In_raw_genotype"].sum()
)

print(
    "mtCN records usable for replicate selection:",
    mtcn["Usable_genotype"].sum()
)

print("\n========================================")
print("RAW COHORT MISSINGNESS")
print("========================================")

print(
    "mtCN records with F_MISS:",
    mtcn["F_MISS"].notna().sum()
)

print(
    "mtCN records without F_MISS:",
    mtcn["F_MISS"].isna().sum()
)


# ============================================================
# BIOLOGICAL / PREGNANCY-ROLE KEY
#
# Per peer recommendation:
#
# SITE_CODE + ORIG_ID + MC
#
# This preserves different pregnancies because ORIG_ID differs.
# It also keeps mother and child separate because MC differs.
# ============================================================

key = [
    "SITE_CODE",
    "ORIG_ID",
    "MC"
]

# ============================================================
# VERIFY PEER'S DUPLICATE Subject_ID CLAIM
#
# Peer estimate:
# ~370 multiple pregnancies
# ~447 technical replicates
# ~9 different biospecimens
# ============================================================

peer_qc = mtcn.copy()

# Only rows with the identifiers needed for classification
peer_qc = peer_qc[
    peer_qc["Subject_ID"].notna() &
    peer_qc["ORIG_ID"].notna() &
    peer_qc["MC"].isin(["M", "C"]) &
    peer_qc["gencove_id"].notna()
].copy()


# ------------------------------------------------------------
# 1. MULTIPLE PREGNANCIES
#
# Same SITE_CODE + Subject_ID + MC,
# but more than one distinct ORIG_ID.
#
# This means the same biological person appears across
# multiple pregnancies.
# ------------------------------------------------------------

subject_summary = (
    peer_qc
    .groupby(
        ["SITE_CODE", "Subject_ID", "MC"]
    )
    .agg(
        N_records=("gencove_id", "size"),
        N_ORIG_ID=("ORIG_ID", "nunique"),
        N_biospecimens=("biospecimen", "nunique")
    )
    .reset_index()
)


multiple_pregnancy = subject_summary[
    subject_summary["N_ORIG_ID"] > 1
].copy()


# ------------------------------------------------------------
# 2. DUPLICATE RECORDS WITHIN SAME PREGNANCY/PERSON-ROLE
# ------------------------------------------------------------

same_pregnancy = (
    peer_qc
    .groupby(
        ["SITE_CODE", "ORIG_ID", "MC"]
    )
    .agg(
        N_records=("gencove_id", "size"),
        N_biospecimens=("biospecimen", "nunique")
    )
    .reset_index()
)

same_pregnancy = same_pregnancy[
    same_pregnancy["N_records"] > 1
].copy()


# ------------------------------------------------------------
# 3. TECHNICAL REPLICATES
#
# Multiple genotype records for the same:
# SITE_CODE + ORIG_ID + MC
#
# but only one biospecimen type.
# ------------------------------------------------------------

technical_replicates = same_pregnancy[
    same_pregnancy["N_biospecimens"] <= 1
].copy()


# ------------------------------------------------------------
# 4. DIFFERENT BIOSPECIMENS
#
# Multiple genotype records for the same biological key
# and more than one biospecimen type.
# ------------------------------------------------------------

different_biospecimens = same_pregnancy[
    same_pregnancy["N_biospecimens"] > 1
].copy()


print("\n========================================")
print("PEER DUPLICATE CLAIM CHECK")
print("========================================")

print(
    "Multiple-pregnancy Subject_ID groups:",
    len(multiple_pregnancy)
)

print(
    "Technical-replicate groups:",
    len(technical_replicates)
)

print(
    "Different-biospecimen groups:",
    len(different_biospecimens)
)


multiple_pregnancy.to_csv(
    "peer_check_multiple_pregnancies.tsv",
    sep="\t",
    index=False
)

technical_replicates.to_csv(
    "peer_check_technical_replicates.tsv",
    sep="\t",
    index=False
)

different_biospecimens.to_csv(
    "peer_check_different_biospecimens.tsv",
    sep="\t",
    index=False
)
print("\n========================================")
print("PEER DUPLICATE CLAIM CHECK - MOTHERS ONLY")
print("========================================")

technical_replicates_m = technical_replicates[
    technical_replicates["MC"].eq("M")
]

different_biospecimens_m = different_biospecimens[
    different_biospecimens["MC"].eq("M")
]

print(
    "Technical-replicate mother groups:",
    len(technical_replicates_m)
)

print(
    "Different-biospecimen mother groups:",
    len(different_biospecimens_m)
)
print("\nDifferent-biospecimen groups by M/C:")
print(
    different_biospecimens["MC"]
    .value_counts(dropna=False)
)
# ============================================================
# IDENTIFY MULTIPLE BIOSPECIMEN GROUPS
# ============================================================

biospecimen_counts = (
    mtcn
    .groupby(key)["biospecimen"]
    .nunique(dropna=True)
    .reset_index(
        name="N_biospecimens"
    )
)

multi_biospecimen_keys = biospecimen_counts[
    biospecimen_counts["N_biospecimens"] > 1
].copy()


multi_biospecimen_rows = mtcn.merge(
    multi_biospecimen_keys[key],
    on=key,
    how="inner"
)


print("\n========================================")
print("MULTIPLE BIOSPECIMEN GROUPS")
print("========================================")

print(
    "Groups with >1 biospecimen type:",
    len(multi_biospecimen_keys)
)

print(
    "Rows involved:",
    len(multi_biospecimen_rows)
)


if len(multi_biospecimen_rows) > 0:

    print("\nBiospecimen combinations:")

    bio_combo = (
        multi_biospecimen_rows
        .groupby(key)["biospecimen"]
        .apply(collapse_ids)
        .value_counts()
    )

    print(
        bio_combo
    )


multi_biospecimen_rows.to_csv(
    "multiple_biospecimen_groups.tsv",
    sep="\t",
    index=False
)


# ============================================================
# IDENTIFY ALL DUPLICATE BIOLOGICAL KEYS
#
# These may include:
#   - technical replicates
#   - different biospecimens
#
# Both are resolved by choosing the genotype record with the
# lowest raw-cohort F_MISS.
# ============================================================

duplicate_counts = (
    mtcn
    .groupby(key)
    .size()
    .reset_index(
        name="N_records"
    )
)

duplicate_keys = duplicate_counts[
    duplicate_counts["N_records"] > 1
].copy()


print("\n========================================")
print("DUPLICATE SITE_CODE + ORIG_ID + MC")
print("========================================")

print(
    "Duplicate groups:",
    len(duplicate_keys)
)

print("\nDistribution:")

print(
    duplicate_keys[
        "N_records"
    ]
    .value_counts()
    .sort_index()
)


duplicate_keys.to_csv(
    "duplicate_sitecode_ORIGID_MC.tsv",
    sep="\t",
    index=False
)


# ============================================================
# PRESERVE ALL IDs AND BIOSPECIMENS BEFORE COLLAPSING
# ============================================================

all_records = (
    mtcn
    .groupby(key)
    .apply(
        lambda g: pd.Series({
            "All_ids":
                collapse_ids(
                    g["gencove_id"]
                ),

            "All_biospecimens":
                collapse_ids(
                    g["biospecimen"]
                ),

            "All_id_biospecimen_pairs":
                collapse_id_biospecimen_pairs(
                    g
                ),

            "N_records":
                g["gencove_id"].size,

            "N_biospecimens":
                g["biospecimen"]
                .dropna()
                .astype(str)
                .nunique()
        })
    )
    .reset_index()
)
# ============================================================
# SELECT BEST RECORD
#
# Lowest raw cohort F_MISS wins.
#
# This selection handles both:
#
# 1. technical replicate gencove_ids
# 2. rare multiple-biospecimen cases
#
# Missing F_MISS sorts last.
# ============================================================

# ============================================================
# SELECT BEST USABLE RECORD
#
# Only IDs that:
#   1. exist in samples.tab
#   2. exist in the raw genotype cohort
#
# are eligible.
#
# Lowest raw-cohort F_MISS wins.
# ============================================================

selection_pool = mtcn[
    mtcn["Usable_genotype"]
].copy()

selection_pool = selection_pool.sort_values(
    key + ["F_MISS", "gencove_id"],
    ascending=True,
    na_position="last"
)

mtcn_best = (
    selection_pool
    .drop_duplicates(
        subset=key,
        keep="first"
    )
    .copy()
)

mtcn_best = mtcn_best.merge(
    all_records,
    on=key,
    how="left",
    validate="one_to_one"
)

# ============================================================
# BIOLOGICAL KEYS WITH NO USABLE GENOTYPE
# ============================================================

all_keys = (
    mtcn[key]
    .drop_duplicates()
)

usable_keys = (
    selection_pool[key]
    .drop_duplicates()
)

no_usable_genotype = (
    all_keys
    .merge(
        usable_keys,
        on=key,
        how="left",
        indicator=True
    )
)

no_usable_genotype = no_usable_genotype[
    no_usable_genotype["_merge"].eq("left_only")
].drop(columns="_merge")

print(
    "SITE_CODE + ORIG_ID + MC groups with no usable genotype:",
    len(no_usable_genotype)
)

no_usable_genotype.to_csv(
    "groups_without_usable_genotype.tsv",
    sep="\t",
    index=False
)

# ============================================================
# QC: GROUPS WITHOUT A USABLE GENOTYPE
# ============================================================

print("\n========================================")
print("NO USABLE GENOTYPE BY M/C")
print("========================================")

print(
    no_usable_genotype["MC"]
    .value_counts(dropna=False)
)


print("\n========================================")
print("NO USABLE GENOTYPE BY SITE_CODE")
print("========================================")

print(
    no_usable_genotype["SITE_CODE"]
    .value_counts(dropna=False)
    .sort_index()
)


print("\n========================================")
print("NO USABLE GENOTYPE BY SITE_CODE + M/C")
print("========================================")

no_usable_site_mc = (
    no_usable_genotype
    .groupby(
        ["SITE_CODE", "MC"],
        dropna=False
    )
    .size()
    .reset_index(name="N")
)

print(no_usable_site_mc)


no_usable_site_mc.to_csv(
    "no_usable_genotype_by_site_MC.tsv",
    sep="\t",
    index=False
)

# ============================================================
# RECORD DISCARDED / ALTERNATE IDs
# ============================================================

mtcn_best["Other_ids"] = (
    mtcn_best.apply(
        get_other_ids,
        axis=1
    )
)


mtcn_best[
    "Multiple_biospecimens"
] = (
    mtcn_best[
        "N_biospecimens"
    ] > 1
)


mtcn_best[
    "Technical_or_duplicate_records"
] = (
    mtcn_best[
        "N_records"
    ] > 1
)


print("\n========================================")
print("FINAL mtCN SAMPLE SELECTION")
print("========================================")

print(
    "Rows before selection:",
    len(mtcn)
)

print(
    "Rows after selection:",
    len(mtcn_best)
)

print(
    "Rows removed:",
    len(mtcn) - len(mtcn_best)
)

print(
    "Selected records with alternate IDs:",
    mtcn_best["Other_ids"]
    .notna()
    .sum()
)

print(
    "Selected records from multi-biospecimen groups:",
    mtcn_best["Multiple_biospecimens"]
    .sum()
)


# ============================================================
# SAVE SELECTED mtCN RECORDS
# ============================================================

mtcn_best.to_csv(
    "mtcn_selected_samples.tsv",
    sep="\t",
    index=False
)




# ============================================================
# EXTRACT SELECTED IDS FROM samples.tab
#
# gencove_id from mtCN == id in samples.tab
# ============================================================

selected_meta_columns = [
    "gencove_id",
    "SITE_CODE",
    "biospecimen",
    "n_seq",
    "mtcn",
    "mt_depth",
    "auto_depth",
    "F_MISS",
    "All_ids",
    "Other_ids",
    "All_biospecimens",
    "All_id_biospecimen_pairs", 
    "N_records",
    "N_biospecimens",
    "Multiple_biospecimens",
    "Technical_or_duplicate_records"
]


selected_meta_columns = [
    col
    for col in selected_meta_columns
    if col in mtcn_best.columns
]


selected_meta = mtcn_best[
    selected_meta_columns
].copy()


samples = samples_raw.merge(
    selected_meta,
    left_on="id",
    right_on="gencove_id",
    how="inner",
    validate="one_to_one"
)


print("\n========================================")
print("SELECTED IDs FOUND IN samples.tab")
print("========================================")

print(
    "Selected records:",
    len(mtcn_best)
)

print(
    "Selected records found in samples.tab:",
    len(samples)
)


selected_not_found = mtcn_best[
    ~mtcn_best["gencove_id"].isin(
        samples_raw["id"]
    )
].copy()

print(
    "Selected mtCN IDs NOT found in samples.tab:",
    len(selected_not_found)
)

if len(selected_not_found) > 0:
    selected_not_found.to_csv(
        "selected_mtcn_ids_not_in_samples.tsv",
        sep="\t",
        index=False
    )

    raise ValueError(
        "Selected mtCN IDs unexpectedly missing from samples.tab."
    )

# ============================================================
# IMPORTANT
#
# THERE IS NO SECOND MISSINGNESS-BASED DEDUPLICATION BELOW.
#
# samples now already contains one selected genotype record per:
#
# SITE_CODE + ORIG_ID + MC
# ============================================================


# ============================================================
# NORMALIZE M/C
# ============================================================

samples["MC"] = (
    samples["M/C"]
    .astype("string")
    .str.upper()
    .str.strip()
)


# ============================================================
# CREATE MERGE KEYS FOR samples.tab
# ============================================================

samples["merge_site"] = (
    samples["site"]
)

samples["merge_ORIG_ID"] = (
    samples["ORIG_ID"]
)


# ============================================================
# AMANHI-BANGLADESH
#
# Remove leading zeroes from samples.tab ORIG_ID
# for MOMI matching.
# ============================================================

bd = samples[
    "site"
].eq(
    "AMANHI-Bangladesh"
)


samples.loc[
    bd,
    "merge_ORIG_ID"
] = (
    samples.loc[
        bd,
        "ORIG_ID"
    ]
    .str.replace(
        r"^0+",
        "",
        regex=True
    )
)


# ============================================================
# GAPPS-ZAMBIA
#
# samples.tab ORIG_ID may be blank.
#
# Derive:
#
# 17-025-0017-M
#     ->
# 17-025-0017
# ============================================================

zambia = samples[
    "site"
].eq(
    "GAPPS-Zambia"
)


samples.loc[
    zambia,
    "merge_ORIG_ID"
] = (
    samples.loc[
        zambia,
        "Subject_ID"
    ]
    .str.replace(
        r"-(M|C)$",
        "",
        regex=True
    )
)


# ============================================================
# ZAMBIA SUBCOHORT
#
# 17 -> IPOP
# 30 -> ZAPPS1
# Z2 -> ZAPPS2
# ============================================================

samples[
    "Zambia_subcohort"
] = pd.NA


samples.loc[
    zambia,
    "Zambia_subcohort"
] = (
    samples.loc[
        zambia,
        "merge_ORIG_ID"
    ]
    .apply(
        zambia_subcohort
    )
)


samples.loc[
    zambia &
    samples[
        "Zambia_subcohort"
    ].notna(),
    "merge_site"
] = (
    samples.loc[
        zambia &
        samples[
            "Zambia_subcohort"
        ].notna(),
        "Zambia_subcohort"
    ]
)


# ============================================================
# REMOVE INVALID MERGE KEYS
# ============================================================

invalid_samples = samples[
    samples["merge_site"].isna() |
    samples["merge_ORIG_ID"].isna() |
    ~samples["MC"].isin(
        ["M", "C"]
    )
].copy()


invalid_samples.to_csv(
    "samples_invalid_merge_keys.tsv",
    sep="\t",
    index=False
)


samples = samples[
    samples["merge_site"].notna() &
    samples["merge_ORIG_ID"].notna() &
    samples["MC"].isin(
        ["M", "C"]
    )
].copy()


print("\n========================================")
print("SAMPLE MERGE KEY QC")
print("========================================")

print(
    "Valid selected sample records:",
    len(samples)
)

print(
    "Invalid merge-key records removed:",
    len(invalid_samples)
)


# ============================================================
# VERIFY SELECTED samples.tab IS UNIQUE
#
# It should now be one record per:
#
# merge_site + merge_ORIG_ID + MC
# ============================================================

sample_key = [
    "merge_site",
    "merge_ORIG_ID",
    "MC"
]


sample_duplicates = samples[
    samples.duplicated(
        subset=sample_key,
        keep=False
    )
].copy()


print(
    "Duplicate selected sample keys after mtCN selection:",
    len(sample_duplicates)
)


sample_duplicates.to_csv(
    "unexpected_selected_sample_duplicates.tsv",
    sep="\t",
    index=False
)


# ============================================================
# SPLIT MOTHERS AND CHILDREN
# ============================================================

mothers = samples[
    samples["MC"].eq("M")
].copy()

children = samples[
    samples["MC"].eq("C")
].copy()


print("\n========================================")
print("MOTHERS / CHILDREN")
print("========================================")

print(
    "Mothers:",
    len(mothers)
)

print(
    "Children:",
    len(children)
)


# ============================================================
# BUILD CHILD LOOKUP
#
# Pair on:
#
# merge_site + merge_ORIG_ID
# ============================================================

child_lookup = (
    children[
        [
            "merge_site",
            "merge_ORIG_ID",
            "id",
            "Other_ids",
            "biospecimen",
            "F_MISS"
        ]
    ]
    .rename(
        columns={
            "id":
                "Child_id",

            "Other_ids":
                "Child_Other_ids",

            "biospecimen":
                "Child_biospecimen",

            "F_MISS":
                "Child_F_MISS"
        }
    )
)


# This should normally already be one child per pregnancy,
# but aggregate defensively.
child_lookup = (
    child_lookup
    .groupby(
        [
            "merge_site",
            "merge_ORIG_ID"
        ],
        as_index=False
    )
    .agg({
        "Child_id":
            collapse_ids,

        "Child_Other_ids":
            collapse_ids,

        "Child_biospecimen":
            collapse_ids,

        "Child_F_MISS":
            "min"
    })
)


# ============================================================
# ADD CHILD INFORMATION TO MOTHERS
# ============================================================

mothers = mothers.merge(
    child_lookup,
    on=[
        "merge_site",
        "merge_ORIG_ID"
    ],
    how="left",
    validate="one_to_one"
)


print("\n========================================")
print("MOTHER-CHILD PAIRING")
print("========================================")

print(
    "Mothers with Child_id:",
    mothers[
        "Child_id"
    ].notna().sum()
)

print(
    "Mothers without Child_id:",
    mothers[
        "Child_id"
    ].isna().sum()
)


# ============================================================
# CLEAN MOMI
# ============================================================

for col in [
    "site",
    "ORIG_ID"
]:
    momi[col] = (
        momi[col]
        .apply(clean)
    )


# ============================================================
# REMOVE THSTI FROM MOMI
# ============================================================

before = len(momi)

momi = momi[
    momi["site"]
    .str.upper()
    .ne("THSTI")
].copy()


print("\n========================================")
print("THSTI FILTER")
print("========================================")

print(
    "THSTI rows removed:",
    before - len(momi)
)

print(
    "MOMI rows remaining:",
    len(momi)
)


# ============================================================
# NORMALIZE MOMI MERGE KEYS
# ============================================================

momi[
    "merge_site"
] = momi["site"]

momi[
    "merge_ORIG_ID"
] = momi["ORIG_ID"]


# ============================================================
# ZAMBIA SUBCOHORT IN MOMI
# ============================================================

momi[
    "Zambia_subcohort"
] = (
    momi[
        "merge_ORIG_ID"
    ]
    .apply(
        zambia_subcohort
    )
)


mz = momi[
    "Zambia_subcohort"
].notna()


momi.loc[
    mz,
    "merge_site"
] = (
    momi.loc[
        mz,
        "Zambia_subcohort"
    ]
)


# ============================================================
# VERIFY MOMI KEY UNIQUENESS
# ============================================================

momi_dup = momi[
    momi.duplicated(
        subset=[
            "merge_site",
            "merge_ORIG_ID"
        ],
        keep=False
    )
].copy()


print("\n========================================")
print("MOMI KEY QC")
print("========================================")

print(
    "Duplicate MOMI site + ORIG_ID rows:",
    len(momi_dup)
)


momi_dup.to_csv(
    "momi_duplicate_keys.tsv",
    sep="\t",
    index=False
)


# ============================================================
# STOP IF MOMI IS NOT UNIQUE
# ============================================================

if len(momi_dup) > 0:

    raise ValueError(
        "MOMI contains duplicate merge_site + merge_ORIG_ID "
        "keys. See momi_duplicate_keys.tsv."
    )


# ============================================================
# VERIFY MOTHERS ARE UNIQUE
# ============================================================

mother_dup = mothers[
    mothers.duplicated(
        subset=[
            "merge_site",
            "merge_ORIG_ID"
        ],
        keep=False
    )
].copy()


print(
    "Duplicate mother merge keys:",
    len(mother_dup)
)


mother_dup.to_csv(
    "mother_duplicate_keys.tsv",
    sep="\t",
    index=False
)


if len(mother_dup) > 0:

    raise ValueError(
        "Mothers are not unique by merge_site + merge_ORIG_ID. "
        "See mother_duplicate_keys.tsv."
    )


# ============================================================
# MERGE MOMI WITH MOTHERS
# ============================================================

merged = momi.merge(
    mothers,
    on=[
        "merge_site",
        "merge_ORIG_ID"
    ],
    how="outer",
    suffixes=(
        "_momi",
        "_sample"
    ),
    indicator=True,
    validate="one_to_one"
)


# ============================================================
# MERGE STATUS
# ============================================================

print("\n========================================")
print("MERGE STATUS")
print("========================================")

print(
    merged[
        "_merge"
    ]
    .value_counts()
)


matched = merged[
    merged["_merge"].eq(
        "both"
    )
].copy()


momi_only = merged[
    merged["_merge"].eq(
        "left_only"
    )
].copy()


mother_only = merged[
    merged["_merge"].eq(
        "right_only"
    )
].copy()


print(
    "\nMatched pregnancies/mothers:",
    len(matched)
)

print(
    "Unmatched MOMI:",
    len(momi_only)
)

print(
    "Unmatched mothers:",
    len(mother_only)
)


# ============================================================
# IMPORTANT SANITY CHECK
#
# matched + MOMI-only should equal all retained MOMI rows.
# ============================================================

print("\nMOMI reconciliation:")

print(
    len(matched),
    "+",
    len(momi_only),
    "=",
    len(matched) + len(momi_only)
)

print(
    "Expected MOMI total:",
    len(momi)
)


# ============================================================
# MATCHED BY SITE
# ============================================================

print("\n========================================")
print("MATCHED BY SITE")
print("========================================")

print(
    matched[
        "merge_site"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# CHILD PAIRING SUMMARY
# ============================================================

print("\n========================================")
print("CHILD PAIRING IN MATCHED DATA")
print("========================================")

print(
    "Matched mothers with Child_id:",
    matched[
        "Child_id"
    ]
    .notna()
    .sum()
)

print(
    "Matched mothers without Child_id:",
    matched[
        "Child_id"
    ]
    .isna()
    .sum()
)


# ============================================================
# SELECTED REPLICATE SUMMARY
# ============================================================

print("\n========================================")
print("REPLICATE / BIOSPECIMEN SELECTION")
print("========================================")

print(
    "Matched mothers with alternate genotype IDs:",
    matched[
        "Other_ids"
    ]
    .notna()
    .sum()
)

print(
    "Matched children with alternate genotype IDs:",
    matched[
        "Child_Other_ids"
    ]
    .notna()
    .sum()
)

print(
    "Matched mothers selected from multiple-biospecimen groups:",
    matched[
        "Multiple_biospecimens"
    ]
    .fillna(False)
    .sum()
)

# ============================================================
# MERGE HAPLOGREP3 RESULTS
#
# Haplogrep3 SampleID corresponds to maternal samples.tab id.
# Keep only:
#   1. Haplogrep3 Quality >= 0.9
#   2. Live births: PREG_OUTCOME == 2
# ============================================================

haplogrep["SampleID"] = (
    haplogrep["SampleID"]
    .astype("string")
    .str.strip()
)

haplogrep["Quality"] = pd.to_numeric(
    haplogrep["Quality"],
    errors="coerce"
)


# Make sure Haplogrep has only one row per sample
haplogrep_dup = haplogrep[
    haplogrep.duplicated(
        subset="SampleID",
        keep=False
    )
].copy()

print("\n========================================")
print("HAPLOGREP3 QC")
print("========================================")

print(
    "Haplogrep3 rows:",
    len(haplogrep)
)

print(
    "Duplicate Haplogrep3 SampleID rows:",
    len(haplogrep_dup)
)

if len(haplogrep_dup) > 0:
    raise ValueError(
        "Haplogrep3 contains duplicate SampleID values."
    )


# ============================================================
# MERGE HAPLOGREP3 ON MATERNAL GENOTYPE ID
# ============================================================

before = len(matched)

matched = matched.merge(
    haplogrep,
    left_on="id",
    right_on="SampleID",
    how="left",
    validate="one_to_one"
)

print(
    "Matched maternal records before Haplogrep/live-birth filtering:",
    before
)

print(
    "Maternal records with Haplogrep3 result:",
    matched["SampleID"].notna().sum()
)


# ============================================================
# LIVE BIRTH FILTER
# ============================================================

matched["PREG_OUTCOME"] = pd.to_numeric(
    matched["PREG_OUTCOME"],
    errors="coerce"
)

before_live = len(matched)

matched = matched[
    matched["PREG_OUTCOME"].eq(2)
].copy()

print(
    "Removed non-live births:",
    before_live - len(matched)
)


# ============================================================
# HAPLOGREP QUALITY FILTER
# ============================================================

before_haplo = len(matched)

matched = matched[
    matched["Quality"].ge(0.9)
].copy()

print(
    "Removed Haplogrep3 Quality < 0.9 or missing:",
    before_haplo - len(matched)
)


print("\n========================================")
print("FINAL PRE-1KGP MATERNAL COHORT")
print("========================================")

print(
    "Final live-birth + Haplogrep3 Quality >= 0.9 mothers:",
    len(matched)
)

print(
    "By site:"
)

print(
    matched["merge_site"]
    .value_counts()
    .sort_index()
)
# ============================================================
# SAVE FINAL OUTPUTS
# ============================================================

matched.drop(
    columns="_merge"
).to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CREATE PLINK KEEP FILE FOR FINAL MATERNAL COHORT
#
# gencove_id = selected maternal genotype UUID = PLINK IID
# ============================================================

import os

os.makedirs(
    "qc",
    exist_ok=True
)


maternal_ids = (
    matched["gencove_id"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
)


maternal_keep = pd.DataFrame({
    "FID": "0",
    "IID": maternal_ids
})


maternal_keep.to_csv(
    "qc/maternal_ptb.keep",
    sep="\t",
    index=False,
    header=False
)


print("\n========================================")
print("MATERNAL PLINK KEEP FILE")
print("========================================")

print(
    "Final matched maternal records:",
    len(matched)
)

print(
    "Unique maternal genotype IDs written:",
    len(maternal_keep)
)

print(
    "Keep file:",
    "qc/maternal_ptb.keep"
)


# Sanity check: every matched mother should have one genotype ID
if len(maternal_keep) != len(matched):

    raise ValueError(
        "Number of unique maternal genotype IDs does not match "
        "number of matched maternal records."
    )


momi_only.to_csv(
    "momi_unmatched.csv",
    index=False
)


mother_only.to_csv(
    "mothers_unmatched.csv",
    index=False
)


# Also save mother/child selected metadata before MOMI merge
mothers.to_csv(
    "selected_mothers.csv",
    index=False
)

children.to_csv(
    "selected_children.csv",
    index=False
)

# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print("\n========================================")
print("OUTPUTS")
print("========================================")

print(
    "Final matched master metadata:",
    OUTPUT_FILE
)

print(
    "Selected mtCN records:",
    "mtcn_selected_samples.tsv"
)

print(
    "Selected mothers:",
    "selected_mothers.csv"
)

print(
    "Selected children:",
    "selected_children.csv"
)

print(
    "Multi-biospecimen QC:",
    "multiple_biospecimen_groups.tsv"
)

print(
    "Duplicate-key QC:",
    "duplicate_sitecode_ORIGID_MC.tsv"
)

print(
    "Selected mtCN IDs missing from samples.tab:",
    "selected_mtcn_ids_not_in_samples.tsv"
)

print(
    "Unmatched MOMI:",
    "momi_unmatched.csv"
)

print(
    "Unmatched mothers:",
    "mothers_unmatched.csv"
)
print(
    "Maternal PLINK keep file:",
    "qc/maternal_ptb.keep"
)
