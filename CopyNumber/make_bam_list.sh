#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <bam_dir> <bam_list.txt>" >&2
    exit 1
fi

BAMDIR="$1"
OUTLIST="$2"

find "$BAMDIR" -maxdepth 1 -type f -name "*.bam" ! -name "*.tmp.*.bam" | sort > "$OUTLIST"

n=$(wc -l < "$OUTLIST")
echo "Wrote $n BAM paths to $OUTLIST" >&2

if [[ "$n" -eq 0 ]]; then
    echo "ERROR: no BAMs found in $BAMDIR" >&2
    exit 1
fi
