

# mtDNA Heteroplasmy Pipeline – Stage 1–3 (Variant Calling + Site Aggregation + Depth Extraction): 01_mtDNA_call_and_sites.sh

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


