

# 01_mtDNA_call_and_sites.sh: Stage 1–3 (Variant Calling + Site Aggregation + Depth Extraction)

## Overview

This script performs the **core preprocessing and variant calling pipeline** for mitochondrial DNA (mtDNA) heteroplasmy analysis using BAM files aligned to `chrM`.

It consists of **three stages**:

1. Per-sample variant calling (GATK Mutect2, mitochondria mode)
2. Union site list construction across all samples
3. Per-sample depth extraction at union variant sites

The output supports downstream heteroplasmy matrix construction and regression analysis.

---

## Requirements

### Software
- samtools (≥1.10)
- bcftools
- gatk (≥4.x)
- GNU parallel


---

## Input Data

- BAM files:
  ${ROOT}/bam/*chrM.bam

- Reference genome:
  ${ROOT}/NC_012920.1.fasta

---



## Stage 1: Variant Calling

Steps:
- Reheader BAM (chrM only)
- Sort + index
- Run Mutect2 (--mitochondria-mode)
- Filter variants
- Keep SNPs only
- Export AF + DP

Outputs:
- *.raw.vcf
- *.filtered.vcf
- *.filtered.snps.vcf
- *.heteroplasmy.txt

---

## Stage 2: Union Site List

Steps:
- Extract SNP positions
- Merge + deduplicate
- Convert to BED

Outputs:
- sites.pos
- sites.bed

---

## Stage 3: Depth Extraction

Steps:
- samtools depth at union sites

Outputs:
- *.sites.depth.txt

---

## Key Parameters

- Jobs: 70
- Min AF: 0.01
- Min alt reads: 3
- MQ ≥ 30

---

## Notes

- SNPs only (no indels)
- Designed for low-frequency heteroplasmy
- Assumes chrM BAM input

---



## Outputs Summary

- heteroplasmy.txt → AF + DP
- sites.depth.txt → depth
- sites.pos / sites.bed → union sites

---



# 02_txt_to_two_parquets.py: (TXT → Parquet Conversion)

## Overview

This script converts per-sample text outputs from the mtDNA heteroplasmy pipeline into efficient, analysis-ready **Parquet datasets**.

It processes:
- Depth files (`*.sites.depth.txt`)
- Heteroplasmy call files (`*.heteroplasmy.txt`)

Outputs are written as **partitioned Parquet files** for scalable downstream analysis.

---

## Purpose

- Standardize data format across samples
- Enable fast I/O and large-scale processing
- Prepare inputs for matrix construction and regression modeling

---

## Requirements

### Python Packages
- pandas
- pyarrow


---

## Input Files

From previous pipeline stage:

- Depth:
  ```
  *.sites.depth.txt
  ```
- Calls:
  ```
  *.heteroplasmy.txt
  ```

---

## Output

Two partitioned Parquet datasets:

```
parquet_depth/
  depth_part_0001.parquet
  depth_part_0002.parquet
  ...

parquet_calls/
  calls_part_0001.parquet
  calls_part_0002.parquet
  ...
```

---

## Data Schemas

### Depth Table
| Column | Description |
|--------|------------|
| sample | Sample ID |
| POS    | mtDNA position |
| DP     | Depth |

---

### Calls Table
| Column | Description |
|--------|------------|
| sample | Sample ID |
| POS    | mtDNA position |
| REF    | Reference allele |
| ALT    | Alternate allele |
| AF     | Allele fraction |
| DP_VCF | Depth from VCF |

---

## Key Features

- Handles **multiallelic variants** via explode
- Robust parsing of whitespace-delimited files
- Batch processing (default: 200 samples)
- Writes compressed Parquet (`zstd`)
- Memory-efficient buffering

---

## Usage

```
python 02_txt_to_two_parquets.py \
  --txt_dir /scr1/users/haltomj/PTB/heteroplasmy/out \
  --out_depth_parquet /scr1/users/haltomj/PTB/heteroplasmy/parquet_depth \
  --out_calls_parquet /scr1/users/haltomj/PTB/heteroplasmy/parquet_calls \
  --batch 200
  
```

---

## Parameters

| Parameter | Description |
|----------|------------|
| `--txt_dir` | Directory containing input txt files |
| `--out_depth_parquet` | Output directory for depth parquet |
| `--out_calls_parquet` | Output directory for calls parquet |
| `--batch` | Samples per batch write |

---

## Notes

- Files are written as **partitions**, not a single Parquet file
- Multiallelic variants are split into separate rows
- Missing or empty call files are handled gracefully
- Designed for large cohorts (10k+ samples)

---



---

## Next Steps

- Merge depth + calls into a matrix
- Encode presence/dose
- Perform association testing

---




# 03_qc_build_matrices_from_two_parquets.py: (QC + Matrix Construction)

## Overview

This script performs **variant-level quality control (QC)** and constructs analysis-ready matrices from Parquet datasets:

- Presence matrix (binary carrier status)
- Dose matrix (heteroplasmy levels)

It integrates:
- Depth data
- Variant calls
- Covariates (including site)

---

## Purpose

- Apply rigorous QC filters to mtDNA variants
- Ensure cross-site robustness
- Generate matrices for regression / ML models

---

## Inputs

### Parquet Data (from Stage 2)
- `parquet_depth/`
- `parquet_calls/`

### Covariates File
- CSV/TSV containing:
  - Sample IDs
  - Site information
  - Age, BMI

---

## Outputs

### Main Outputs
- `*.variant_qc.csv` → QC metrics per variant
- `*.presence_matrix.parquet` → binary matrix
- `*.dose_matrix.<transform>.parquet` → continuous matrix

### Optional Outputs (if enabled)
- Sparse NPZ matrices:
  - evaluable mask
  - carrier mask
  - dose values
- Sample and variant labels

---

## QC Metrics

Each variant is evaluated using:

| Metric | Description |
|------|------------|
| n_used | Samples with sufficient depth |
| n_carriers | Number of carriers |
| prevalence | Carrier frequency |
| sites_with_carriers | Sites with ≥1 carrier |
| min_per_site_among_carrier_sites | Minimum carriers per site |
| dose_sd_carriers | AF variability |

---

## QC Filters

Variants must pass:

- Minimum evaluable samples (`--min_n_used`)
- Minimum carriers (`--min_carriers`)
- Multi-site support (`--min_sites_with_carriers`)
- Per-site carrier threshold
- Dose variability threshold

---

## Matrix Definitions

### Presence Matrix
- 1 = carrier
- 0 = non-carrier (if evaluable)
- NaN = not evaluable (low depth)

---

### Dose Matrix
Depends on `--dose_transform`:

| Option | Description |
|-------|------------|
| raw_af | Raw allele fraction |
| logit_af | Logit-transformed AF |
| rel_to_low_logit | Relative to threshold |

---

## Usage

```
python 03_qc_build_matrices_from_two_parquets.py \
  --depth_parquet_dir /scr1/users/haltomj/PTB/heteroplasmy/parquet_depth \
  --calls_parquet_dir /scr1/users/haltomj/PTB/heteroplasmy/parquet_calls \
  --covariates_csv /scr1/users/haltomj/PTB/heteroplasmy/covariates.csv \
  --cov_sep $'\t' \
  --sample_col Sample_ID \
  --site_col site \
  --low 0.03 \
  --high 0.95 \
  --min_dp 50 \
  --min_n_used 5000 \
  --min_carriers 20 \
  --min_sites_with_carriers 2 \
  --min_per_site_among_carrier_sites 2 \
  --min_dose_sd 0.05 \
  --dose_transform raw_af \
  --out_prefix /scr1/users/haltomj/PTB/heteroplasmy/matrices/mtDNA \
  --write_sparse_npz
```

---

## Key Features

- Site-aware QC (critical for multi-cohort data)
- Handles allele-specific variants
- Supports multiple dose encodings
- Efficient large-scale processing
- Optional sparse matrix export

---

## Notes

- Depth threshold (`min_dp`) defines evaluable samples
- Presence matrix distinguishes missing vs true absence
- Sparse export recommended for large datasets

---

## Next Steps

- Regression modeling (logit / linear)
- ML approaches (RF, NN)
- Variant-level association testing

---



