# Maternal PTB Cohort Preparation and Ancestry QC Pipeline

This workflow prepares the maternal pregnancy cohort for downstream mtDNA haplogroup / preterm birth analyses and nuclear ancestry QC.

The pipeline is divided into three major components:

1. Maternal cohort construction and genotype sample selection
2. Joint study + 1000 Genomes genotype QC/PCA
3. Final ancestry annotation and metadata generation

---

## PLINK2 Setup

Install the current PLINK2 release:

```bash
mkdir -p ~/bin/plink2_latest
cd ~/bin/plink2_latest

wget -O plink2.zip \
  https://s3.amazonaws.com/plink2-assets/alpha7/plink2_linux_avx2_20261001.zip

unzip plink2.zip
chmod +x plink2

./plink2 --version

export PATH="$HOME/bin/plink2_latest:$PATH"

which plink2
plink2 --version
```


## RAW COHORT MISSINGNESS
```
mkdir -p qc

/home/haltomj/bin/plink2_latest/plink2 \
  --bfile plink2 \
  --missing sample-only \
  --out qc/cohort_raw_missing
 ```

---

## 1. Maternal Cohort Construction

### Script purpose

The maternal cohort preparation script reconciles pregnancy metadata, sequencing/sample metadata, mtDNA copy-number metadata, raw nuclear-genotype missingness, and Haplogrep3 results to produce one high-quality maternal genotype record per pregnancy.

The main outputs are:

- `momi_mothers_merged.csv`
- `qc/maternal_ptb.keep`

`momi_mothers_merged.csv` contains the maternal pregnancy metadata used by the downstream ancestry pipeline.

`qc/maternal_ptb.keep` contains the selected maternal PLINK sample IDs used to generate the maternal nuclear-genotype cohort.

---

### Input files

The script uses the following files:

| File | Purpose |
|---|---|
| `momi_pregnancy_level.csv` | Pregnancy-level MOMI clinical and outcome metadata |
| `momi_pregnancy_level_dictionary.csv` | Maps MOMI variable names to the desired original variable names |
| `samples.tab` | Sample metadata including genotype ID, site, pregnancy ID, mother/child status, Subject_ID, sex, and vial |
| `momi_mtcn_per_sample.csv` | mtDNA copy-number/sample metadata including Gencove genotype IDs and biospecimen information |
| `qc/cohort_raw_missing.smiss` | Raw nuclear-genotype sample missingness from PLINK |
| `haplogrep3OUT_22175` | Haplogrep3 maternal mtDNA haplogroup calls and quality scores |

---

### Cohort preparation steps

### 1. Standardize metadata

MOMI variables are renamed using `momi_pregnancy_level_dictionary.csv`.

Sample identifiers and key metadata fields are cleaned and standardized. Blank strings and values such as `NA`, `N/A`, `nan`, `None`, and `.` are treated as missing.

`SPOTBio` samples are removed from `samples.tab`, and `THSTI` records are removed from the MOMI pregnancy metadata.

---

### 2. Define the biological sample key

The primary biological key used for genotype sample selection is:

```text
SITE_CODE + ORIG_ID + MC



# Includes:
# - selection of one best genotype record per biological sample/replicate group
# - one-to-one matching of selected maternal samples to MOMI pregnancy metadata
# - restriction to live births (PREG_OUTCOME == 2)
# - restriction to high-confidence Haplogrep3 calls (Quality >= 0.90)


# ============================================================
```
python MetadataUltimate.py
```



# ============================================================
# 3. CREATE INITIAL MATERNAL PLINK COHORT
# ============================================================
```
/home/haltomj/bin/plink2_latest/plink2 \
  --bfile plink2 \
  --keep qc/maternal_ptb.keep \
  --make-bed \
  --out qc/maternal_ptb
#Those should match.
wc -l qc/maternal_ptb.keep
wc -l qc/maternal_ptb.fam
```

# ============================================================
# 4. INITIAL STUDY + 1000G PCA
#
# Uses:
# qc/maternal_ptb
#
# Produces:
# pca/joint_global.eigenvec
# pca/joint_global.eigenval
# ============================================================

```
snakemake -j 22 -s snakefile --use-conda  --config run=initial --latency-wait 60 --cluster "sbatch -t 05:00:00 -c 8 -N 1"
```

# ============================================================
# 5. GLOBAL ANCESTRY QC
#
# Uses first PCA.
# Flags broad ancestry mismatches.
#
# Produces:
# qc/maternal_ptb_globalPCA.keep
# ============================================================
```
python globalPCA_pipeline.py --stage initial
```
# ============================================================
# 6. CREATE ANCESTRY-CLEAN MATERNAL COHORT
# ============================================================
```
/home/haltomj/bin/plink2_latest/plink2 \
  --bfile qc/maternal_ptb \
  --keep qc/maternal_ptb_globalPCA.keep \
  --make-bed \
  --out qc/maternal_ptb_globalPCA

wc -l qc/maternal_ptb_globalPCA.fam
```

# ============================================================
# 7. FINAL CLEAN STUDY + 1000G PCA
#
# Uses:
# qc/maternal_ptb_globalPCA
#
# Produces:
# pca/joint_global_final.eigenvec
# pca/joint_global_final.eigenval
# ============================================================

```
snakemake -j 22 -s snakefile --use-conda  --config run=final --latency-wait 60 --cluster "sbatch -t 05:00:00 -c 8 -N 1"
```


# ============================================================
# 8. FINAL 1000G ANNOTATION + ULTIMATE METADATA
# ============================================================
```
python globalPCA_pipeline.py --stage final
```

