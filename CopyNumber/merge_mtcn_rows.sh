#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <rows_dir> <output.tsv>" >&2
    exit 1
fi

ROWSDIR="$1"
OUTTSV="$2"

[[ -d "$ROWSDIR" ]] || { echo "ERROR: missing rows dir: $ROWSDIR" >&2; exit 1; }

echo -e "sample\tbam\tmt_contig\tmean_mt_cov\tmedian_autosomal_cov\tmtcn" > "$OUTTSV"

found=0
for f in "$ROWSDIR"/*.tsv; do
    [[ -e "$f" ]] || continue
    tail -n +2 "$f" >> "$OUTTSV"
    found=1
done

if [[ "$found" -eq 0 ]]; then
    echo "WARNING: no row TSV files found in $ROWSDIR" >&2
fi

echo "Wrote merged output to $OUTTSV" >&2
