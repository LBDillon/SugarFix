#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_CSV="${1:-$PIPELINE_ROOT/data/af3_structural_validation.csv}"
OUT_DIR="${2:-$PIPELINE_ROOT/data/af3_results/analysis}"

if [ ! -f "$DEFAULT_CSV" ]; then
  echo "ERROR: $DEFAULT_CSV not found."
  echo "Run collect_af3_metrics.sh first."
  exit 1
fi

python3 "$SCRIPT_DIR/supplementary_af3_validation.py" \
  --csv "$DEFAULT_CSV" \
  --out-dir "$OUT_DIR"
