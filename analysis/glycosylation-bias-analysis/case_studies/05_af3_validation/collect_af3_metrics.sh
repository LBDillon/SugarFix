#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

AF3_DIR="${1:-$HOME/Downloads/AF3_Structures}"
RMSD_CSV="${2:-$PIPELINE_ROOT/data/rmsd_values_template.csv}"
OUTPUT_CSV="${3:-$PIPELINE_ROOT/data/af3_structural_validation.csv}"

python3 "$SCRIPT_DIR/collect_af3_metrics.py" \
  --af3-dir "$AF3_DIR" \
  --rmsd-csv "$RMSD_CSV" \
  --output-csv "$OUTPUT_CSV"
