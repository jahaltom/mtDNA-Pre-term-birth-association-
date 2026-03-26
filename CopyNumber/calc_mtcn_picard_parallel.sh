#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --mem=48G
#SBATCH --job-name=mtcn_picard_parallel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@domain.com

set -euo pipefail
shopt -s nullglob

# =========================================================
# Usage
# =========================================================
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <bam_dir> <reference.fasta> <output.tsv> [jobs]" >&2
    exit 1
fi

BAMDIR="$1"
REF="$2"
OUTTSV="$3"
JOBS="${4:-4}"

PICARD_CMD="picard"

# =========================================================
# Checks
# =========================================================
if ! command -v samtools >/dev/null 2>&1; then
    echo "ERROR: samtools not found in PATH" >&2
    exit 1
fi

if ! command -v "$PICARD_CMD" >/dev/null 2>&1; then
    echo "ERROR: picard not found in PATH" >&2
    exit 1
fi

if ! command -v parallel >/dev/null 2>&1; then
    echo "ERROR: GNU parallel not found in PATH" >&2
    exit 1
fi

if [[ ! -d "$BAMDIR" ]]; then
    echo "ERROR: BAM directory not found: $BAMDIR" >&2
    exit 1
fi

if [[ ! -f "$REF" ]]; then
    echo "ERROR: reference fasta not found: $REF" >&2
    exit 1
fi

if [[ ! -f "${REF}.fai" ]]; then
    echo "ERROR: reference fasta index not found: ${REF}.fai" >&2
    echo "Run: samtools faidx $REF" >&2
    exit 1
fi

DICT="${REF%.*}.dict"
if [[ ! -f "$DICT" ]]; then
    echo "ERROR: reference dict not found: $DICT" >&2
    echo "Run: gatk CreateSequenceDictionary -R $REF" >&2
    exit 1
fi

# =========================================================
# Temp dir
# =========================================================
tmp_root=$(mktemp -d)
trap 'rm -rf "$tmp_root"' EXIT

# =========================================================
# Gather BAMs
# Skip temp/intermediate BAMs
# =========================================================
mapfile -t bams < <(
    find "$BAMDIR" -maxdepth 1 -type f -name "*.bam" \
    ! -name "*.tmp.*.bam" \
    | sort
)

if [[ ${#bams[@]} -eq 0 ]]; then
    echo "ERROR: no BAM files found in $BAMDIR" >&2
    exit 1
fi

# =========================================================
# Export variables/functions for GNU parallel
# =========================================================
export REF
export DICT
export PICARD_CMD
export tmp_root

process_bam() {
    local bam="$1"
    local sample
    sample=$(basename "$bam")
    sample="${sample%.bam}"

    echo "Processing $sample ..." >&2

    # -----------------------------------
    # Validate BAM
    # -----------------------------------
    if ! samtools quickcheck "$bam" 2>/dev/null; then
        echo "WARNING: skipping invalid BAM: $bam" >&2
        return 0
    fi

    # -----------------------------------
    # Ensure BAM index exists
    # -----------------------------------
    if [[ ! -f "${bam}.bai" && ! -f "${bam%.bam}.bai" ]]; then
        echo "  BAM index missing; creating for $sample" >&2
        samtools index -@ 2 "$bam"
    fi

    # -----------------------------------
    # Detect mitochondrial contig
    # -----------------------------------
    local idxstats="${tmp_root}/${sample}.idxstats.txt"
    samtools idxstats "$bam" > "$idxstats"

    local mt_contig=""
    local c
    for c in chrM MT M; do
        if awk -v target="$c" '$1==target {found=1} END{exit(found?0:1)}' "$idxstats"; then
            mt_contig="$c"
            break
        fi
    done

    if [[ -z "$mt_contig" ]]; then
        echo "WARNING: could not detect mt contig in $bam (tried chrM, MT, M); skipping" >&2
        return 0
    fi

    # -----------------------------------
    # Mean mtDNA coverage
    # -----------------------------------
    local mean_mt_cov
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
        echo "WARNING: failed to compute mean mtDNA coverage for $bam; skipping" >&2
        return 0
    fi

    # -----------------------------------
    # Build autosomal interval_list for Picard
    # autosomes 1-22 only
    # -----------------------------------
    local autosomal_intervals="${tmp_root}/${sample}.autosomal.interval_list"

    grep '^@' "$DICT" > "$autosomal_intervals"

    local chr_style_count
    local nonchr_style_count

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
        echo "WARNING: could not identify autosomes 1-22 in ${REF}.fai; skipping $bam" >&2
        return 0
    fi

    if [[ $(grep -vc '^@' "$autosomal_intervals") -eq 0 ]]; then
        echo "WARNING: autosomal interval list is empty for $bam; skipping" >&2
        return 0
    fi

    # -----------------------------------
    # Picard CollectWgsMetrics
    # -----------------------------------
    local wgs_metrics_txt="${tmp_root}/${sample}.wgs_metrics.txt"

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

    local median_autosomal_cov
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
        echo "WARNING: failed to extract MEDIAN_COVERAGE from Picard output for $bam; skipping" >&2
        return 0
    fi

    # -----------------------------------
    # mtDNA copy number
    # -----------------------------------
    local mtcn
    mtcn=$(
        awk -v m="$mean_mt_cov" -v n="$median_autosomal_cov" '
            BEGIN {
                if (n == 0 || n == "NA" || n == "") print "NA";
                else printf "%.6f\n", (2.0 * m) / n
            }
        '
    )

    # -----------------------------------
    # Write one per-sample output row
    # -----------------------------------
    local rowfile="${tmp_root}/${sample}.row.tsv"
    echo -e "${sample}\t${bam}\t${mt_contig}\t${mean_mt_cov}\t${median_autosomal_cov}\t${mtcn}" > "$rowfile"
}

export -f process_bam

# =========================================================
# Run in parallel across BAMs
# =========================================================
parallel -j "$JOBS" process_bam ::: "${bams[@]}"

# =========================================================
# Merge outputs safely
# =========================================================
echo -e "sample\tbam\tmt_contig\tmean_mt_cov\tmedian_autosomal_cov\tmtcn" > "$OUTTSV"

row_count=$(find "$tmp_root" -maxdepth 1 -type f -name "*.row.tsv" | wc -l)

if [[ "$row_count" -eq 0 ]]; then
    echo "WARNING: no successful BAMs processed; output contains header only" >&2
else
    find "$tmp_root" -maxdepth 1 -type f -name "*.row.tsv" | sort | xargs cat >> "$OUTTSV"
fi

echo "Done. Results written to $OUTTSV" >&2
