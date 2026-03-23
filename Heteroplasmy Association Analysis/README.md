

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



