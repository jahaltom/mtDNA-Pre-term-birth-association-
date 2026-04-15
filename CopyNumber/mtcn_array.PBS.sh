#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 6:00:00
#SBATCH --job-name=mtcn_array
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@domain.com
#SBATCH -o logs/mtcn_%A_%a.out
#SBATCH -e logs/mtcn_%A_%a.err

set -euo pipefail
shopt -s nullglob

# =========================================================
# Usage
# sbatch --array=1-N mtcn_array.sh bam_list.txt ref.fa outdir
# =========================================================
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <bam_list.txt> <reference.fasta> <outdir>" >&2
    exit 1
fi

BAMLIST="$1"
REF="$2"
OUTDIR="$3"

# Optional: activate conda if needed
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mtcn-parallel || true
fi

PICARD_CMD="picard"

# =========================================================
# Checks
# =========================================================
[[ -f "$BAMLIST" ]] || { echo "ERROR: missing BAM list: $BAMLIST" >&2; exit 1; }
[[ -f "$REF" ]] || { echo "ERROR: missing reference FASTA: $REF" >&2; exit 1; }
[[ -f "${REF}.fai" ]] || { echo "ERROR: missing FASTA index: ${REF}.fai" >&2; exit 1; }

DICT="${REF%.*}.dict"
[[ -f "$DICT" ]] || { echo "ERROR: missing sequence dictionary: $DICT" >&2; exit 1; }

command -v samtools >/dev/null 2>&1 || { echo "ERROR: samtools not in PATH" >&2; exit 1; }
command -v "$PICARD_CMD" >/dev/null 2>&1 || { echo "ERROR: picard not in PATH" >&2; exit 1; }

mkdir -p "$OUTDIR"
mkdir -p "$OUTDIR/rows"
mkdir -p "$OUTDIR/tmp"
mkdir -p logs

# =========================================================
# Get BAM for this array task
# =========================================================
TASK_ID="${SLURM_ARRAY_TASK_ID:?ERROR: SLURM_ARRAY_TASK_ID not set}"

bam=$(sed -n "${TASK_ID}p" "$BAMLIST")

if [[ -z "${bam:-}" ]]; then
    echo "ERROR: no BAM found for task ID $TASK_ID in $BAMLIST" >&2
    exit 1
fi

if [[ ! -f "$bam" ]]; then
    echo "ERROR: BAM path does not exist: $bam" >&2
    exit 1
fi

sample=$(basename "$bam")
sample="${sample%.bam}"

echo "[$(date)] Processing sample: $sample" >&2
echo "BAM: $bam" >&2

tmp_root=$(mktemp -d "$OUTDIR/tmp/${sample}.XXXXXX")
trap 'rm -rf "$tmp_root"' EXIT

# =========================================================
# Validate BAM
# =========================================================
if ! samtools quickcheck "$bam" 2>/dev/null; then
    echo "ERROR: invalid BAM: $bam" >&2
    exit 1
fi

# =========================================================
# Ensure BAM index exists
# =========================================================
if [[ ! -f "${bam}.bai" && ! -f "${bam%.bam}.bai" ]]; then
    echo "BAM index missing; creating." >&2
    samtools index -@ 4 "$bam"
fi

# =========================================================
# Detect mitochondrial contig
# =========================================================
idxstats="${tmp_root}/${sample}.idxstats.txt"
samtools idxstats "$bam" > "$idxstats"

mt_contig=""
for c in chrM MT M; do
    if awk -v target="$c" '$1==target {found=1} END{exit(found?0:1)}' "$idxstats"; then
        mt_contig="$c"
        break
    fi
done

if [[ -z "$mt_contig" ]]; then
    echo "ERROR: could not detect mt contig in $bam (tried chrM, MT, M)" >&2
    exit 1
fi

echo "Detected mt contig: $mt_contig" >&2

# =========================================================
# Mean mtDNA coverage
# =========================================================
mean_mt_cov=$(
    samtools depth -aa -d 0 -r "$mt_contig" "$bam" \
    | awk '
        {sum += $3; n++}
        END {
            if (n == 0) print "NA";
            else printf "%.6f\n", sum / n
        }
    '
)

if [[ "$mean_mt_cov" == "NA" || -z "$mean_mt_cov" ]]; then
    echo "ERROR: failed to compute mean mt coverage for $bam" >&2
    exit 1
fi

echo "Mean mt coverage: $mean_mt_cov" >&2

# =========================================================
# Build autosomal interval_list (autosomes 1-22 only)
# =========================================================
autosomal_intervals="${tmp_root}/${sample}.autosomal.interval_list"

grep '^@' "$DICT" > "$autosomal_intervals"

chr_style_count=$(awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2])$/ {count++} END{print count+0}' "${REF}.fai")
nonchr_style_count=$(awk '$1 ~ /^([1-9]|1[0-9]|2[0-2])$/ {count++} END{print count+0}' "${REF}.fai")

if [[ "$chr_style_count" -gt 0 ]]; then
    awk '
        BEGIN{OFS="\t"}
        $1 ~ /^chr([1-9]|1[0-9]|2[0-2])$/ {
            print $1, 1, $2, "+", $1
        }
    ' "${REF}.fai" >> "$autosomal_intervals"
elif [[ "$nonchr_style_count" -gt 0 ]]; then
    awk '
        BEGIN{OFS="\t"}
        $1 ~ /^([1-9]|1[0-9]|2[0-2])$/ {
            print $1, 1, $2, "+", $1
        }
    ' "${REF}.fai" >> "$autosomal_intervals"
else
    echo "ERROR: could not identify autosomes 1-22 in ${REF}.fai" >&2
    exit 1
fi

if [[ $(grep -vc '^@' "$autosomal_intervals") -eq 0 ]]; then
    echo "ERROR: autosomal interval list is empty" >&2
    exit 1
fi

# =========================================================
# Picard CollectWgsMetrics
# =========================================================
wgs_metrics_txt="${tmp_root}/${sample}.wgs_metrics.txt"

"$PICARD_CMD" CollectWgsMetrics \
    I="$bam" \
    O="$wgs_metrics_txt" \
    R="$REF" \
    INTERVALS="$autosomal_intervals" \
    MINIMUM_MAPPING_QUALITY=20 \
    MINIMUM_BASE_QUALITY=20 \
    COVERAGE_CAP=100000 \
    STOP_AFTER=0 \
    VALIDATION_STRINGENCY=SILENT >/dev/null

median_autosomal_cov=$(
    awk '
        BEGIN{FS="\t"}
        /^GENOME_TERRITORY/ {
            for (i=1; i<=NF; i++) {
                if ($i == "MEDIAN_COVERAGE") col=i
            }
            getline
            if (col > 0) print $col
            exit
        }
    ' "$wgs_metrics_txt"
)

if [[ -z "${median_autosomal_cov:-}" ]]; then
    echo "ERROR: failed to extract MEDIAN_COVERAGE from Picard output" >&2
    echo "Inspect: $wgs_metrics_txt" >&2
    exit 1
fi

echo "Median autosomal coverage: $median_autosomal_cov" >&2

# =========================================================
# mtDNA copy number
# =========================================================
mtcn=$(
    awk -v m="$mean_mt_cov" -v n="$median_autosomal_cov" '
        BEGIN {
            if (n == 0 || n == "NA" || n == "") print "NA";
            else printf "%.6f\n", (2.0 * m) / n
        }
    '
)

echo "mtCN: $mtcn" >&2

# =========================================================
# Write per-sample row
# =========================================================
rowfile="$OUTDIR/rows/${sample}.tsv"
echo -e "sample\tbam\tmt_contig\tmean_mt_cov\tmedian_autosomal_cov\tmtcn" > "$rowfile"
echo -e "${sample}\t${bam}\t${mt_contig}\t${mean_mt_cov}\t${median_autosomal_cov}\t${mtcn}" >> "$rowfile"

echo "[$(date)] Finished sample: $sample" >&2
