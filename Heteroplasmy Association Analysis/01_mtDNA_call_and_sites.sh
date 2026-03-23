#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 24:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL

set -euo pipefail
shopt -s nullglob

# =========================================================
# User settings
# =========================================================
ROOT=/scr1/users/haltomj/PTB/heteroplasmy
BAMDIR="${ROOT}/bam"
REF="${ROOT}/NC_012920.1.fasta"

VCFDIR="${ROOT}/vcf"
OUTDIR="${ROOT}/out"
TMPDIR="${ROOT}/tmp"

JOBS=70

# =========================================================
# Create directories
# =========================================================
mkdir -p "${VCFDIR}" "${OUTDIR}" "${TMPDIR}"



# =========================================================
# Stage 1: Per-sample variant calling
# =========================================================
find "${BAMDIR}" -maxdepth 1 -type f -name '*chrM.bam' -print \
| parallel --jobs "${JOBS}" --halt soon,fail=1 '
  set -euo pipefail

  bam="{}"
  base="{/.}"

  ROOT=/scr1/users/haltomj/PTB/heteroplasmy
  REF="${ROOT}/NC_012920.1.fasta"
  VCFDIR="${ROOT}/vcf"
  OUTDIR="${ROOT}/out"
  TMPDIR="${ROOT}/tmp"

  mkdir -p "${VCFDIR}" "${OUTDIR}" "${TMPDIR}"

  workbam="${TMPDIR}/${base}.reheader.bam"
  sortbam="${TMPDIR}/${base}.reheader.sorted.bam"

  rawvcf="${VCFDIR}/${base}.raw.vcf"
  filtvcf="${VCFDIR}/${base}.filtered.vcf"
  hettxt="${OUTDIR}/${base}.heteroplasmy.txt"

  echo "[${base}] Preparing BAM"

  # Keep chrM header line plus standard SAM headers and all alignments.
  # This assumes the input BAM is already chrM-focused.
  samtools view -h "${bam}" \
   | grep -E "^@HD|^@PG|^@RG|^@CO|^@SQ.*SN:chrM|^[^@]" \
   | samtools view -F 3844 -bo "${workbam}" -

  samtools sort -o "${sortbam}" "${workbam}"
  samtools index "${sortbam}"

  echo "[${base}] Running Mutect2"
  gatk Mutect2 \
    -R "${REF}" \
    -I "${sortbam}" \
    --mitochondria-mode \
    -O "${rawvcf}"

  echo "[${base}] Running FilterMutectCalls"
  gatk FilterMutectCalls \
    -R "${REF}" \
    -V "${rawvcf}" \
    -O "${filtvcf}" \
    --max-events-in-region 2 \
    --min-allele-fraction 0.01 \
    --unique-alt-read-count 3 \
    --min-median-mapping-quality 30

  echo "[${base}] Exporting heteroplasmy table"
  
  snpsvcf="${VCFDIR}/${base}.filtered.snps.vcf"

  bcftools view -v snps "${filtvcf}" | \
  bcftools view -i 'strlen(REF)=1 && strlen(ALT)=1' -o "${snpsvcf}"

  bcftools query -f "%CHROM\t%POS\t%REF\t%ALT\t[%AF]\t[%DP]\n" \
    "${snpsvcf}" > "${hettxt}"

  echo "[${base}] Done stage 1"
'

echo "Stage 1 complete."
echo "Starting stage 2: build union site list"

# =========================================================
# Stage 2: Build union site list ONCE
# =========================================================
filtered_vcfs=( "${VCFDIR}"/*.filtered.vcf )
if [[ ${#filtered_vcfs[@]} -eq 0 ]]; then
  echo "ERROR: No filtered VCFs found in ${VCFDIR}" >&2
  exit 1
fi

SITES_POS="${ROOT}/sites.pos"
SITES_BED="${ROOT}/sites.bed"

: > "${SITES_POS}"

for vcf in "${filtered_vcfs[@]}"; do
    bcftools view -v snps "${vcf}" | \
    bcftools view -i 'strlen(REF)=1 && strlen(ALT)=1' | \
    bcftools query -f "%CHROM\t%POS\n"
done \
| sort -u \
> "${SITES_POS}"

if [[ ! -s "${SITES_POS}" ]]; then
  echo "ERROR: sites.pos was created but is empty. No variant positions found." >&2
  exit 1
fi

awk 'BEGIN{OFS="\t"} {print $1, $2-1, $2}' "${SITES_POS}" > "${SITES_BED}"
sort -k1,1 -k2,2n "${SITES_BED}" -o "${SITES_BED}"

if [[ ! -s "${SITES_BED}" ]]; then
  echo "ERROR: sites.bed was created but is empty." >&2
  exit 1
fi

echo "Stage 2 complete."
echo "Union sites written to:"
echo "  ${SITES_POS}"
echo "  ${SITES_BED}"

echo "Starting stage 3: per-sample depth extraction"

# =========================================================
# Stage 3: Per-sample depth extraction at union sites
# =========================================================
find "${BAMDIR}" -maxdepth 1 -type f -name '*chrM.bam' -print \
| parallel --jobs "${JOBS}" --halt soon,fail=1 '
  set -euo pipefail

  bam="{}"
  base="{/.}"

  ROOT=/scr1/users/haltomj/PTB/heteroplasmy
  OUTDIR="${ROOT}/out"
  TMPDIR="${ROOT}/tmp"
  SITES_BED="${ROOT}/sites.bed"

  sortbam="${TMPDIR}/${base}.reheader.sorted.bam"
  depthtxt="${OUTDIR}/${base}.sites.depth.txt"

  if [[ ! -f "${SITES_BED}" || ! -s "${SITES_BED}" ]]; then
    echo "ERROR: Missing or empty sites.bed: ${SITES_BED}" >&2
    exit 1
  fi

  if [[ ! -f "${sortbam}" ]]; then
    echo "ERROR: Expected sorted BAM not found: ${sortbam}" >&2
    exit 1
  fi

  echo "[${base}] Extracting depth"
  samtools depth -a -d 0 -Q 20 -q 20 -b "${SITES_BED}" "${sortbam}" > "${depthtxt}"

  echo "[${base}] Done stage 3"
'

echo "Stage 3 complete."
echo "All steps finished successfully."
