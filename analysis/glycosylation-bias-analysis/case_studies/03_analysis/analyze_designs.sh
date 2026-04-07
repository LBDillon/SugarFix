#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PDB_ID="${1:-}"
PREP_DIR="${2:-$PIPELINE_ROOT/data/prep/$PDB_ID}"
OUTPUT_DIR="${3:-$PIPELINE_ROOT/data/outputs/output_${PDB_ID}}"
PDB_PATH="$PREP_DIR/structure/${PDB_ID}_protein.pdb"

if [ -z "$PDB_ID" ]; then
  echo "Usage: $0 <PDB_ID> [PREP_DIR] [OUTPUT_DIR]"
  exit 1
fi

if [ -n "${PROTEINMPNN_DIR:-}" ] && [ -f "${PROTEINMPNN_DIR}/protein_mpnn_utils.py" ]; then
  MPNN_DIR="$PROTEINMPNN_DIR"
else
  MPNN_DIR=""
  for candidate in \
    "$PIPELINE_ROOT/ProteinMPNN" \
    "$PIPELINE_ROOT/../ProteinMPNN" \
    "$PIPELINE_ROOT/../../ProteinMPNN" \
    "$PIPELINE_ROOT/../../../ProteinMPNN"
  do
    if [ -f "$candidate/protein_mpnn_utils.py" ]; then
      MPNN_DIR="$candidate"
      break
    fi
  done
fi

if [ -z "$MPNN_DIR" ]; then
  echo "ERROR: Could not locate ProteinMPNN directory."
  echo "Set PROTEINMPNN_DIR to a folder containing protein_mpnn_utils.py."
  exit 1
fi

export PROTEINMPNN_DIR="$MPNN_DIR"
export PYTHONPATH="$MPNN_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 "$SCRIPT_DIR/analyze_case_designs.py" \
  --pdb-id "$PDB_ID" \
  --pdb-path "$PDB_PATH" \
  --designs-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "Analysis outputs for $PDB_ID refreshed in: $OUTPUT_DIR"
