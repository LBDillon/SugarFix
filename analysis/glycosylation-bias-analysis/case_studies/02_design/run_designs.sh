#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PDB_ID="${1:-}"
NUM_SEQS="${2:-64}"
PREP_DIR="${3:-$PIPELINE_ROOT/data/prep/$PDB_ID}"
OUTPUT_DIR="${4:-$PIPELINE_ROOT/data/outputs/output_${PDB_ID}}"
PDB_PATH="$PREP_DIR/structure/${PDB_ID}_protein.pdb"

if [ -z "$PDB_ID" ]; then
  echo "Usage: $0 <PDB_ID> [NUM_SEQS] [PREP_DIR] [OUTPUT_DIR]"
  exit 1
fi

if [ -n "${PROTEINMPNN_PATH:-}" ] && [ -f "${PROTEINMPNN_PATH}" ]; then
  MPNN_RUNNER="$PROTEINMPNN_PATH"
else
  MPNN_RUNNER=""
  for candidate in \
    "$PIPELINE_ROOT/ProteinMPNN/protein_mpnn_run.py" \
    "$PIPELINE_ROOT/../ProteinMPNN/protein_mpnn_run.py" \
    "$PIPELINE_ROOT/../../ProteinMPNN/protein_mpnn_run.py" \
    "$PIPELINE_ROOT/../../../ProteinMPNN/protein_mpnn_run.py"
  do
    if [ -f "$candidate" ]; then
      MPNN_RUNNER="$candidate"
      break
    fi
  done
fi

if [ -z "$MPNN_RUNNER" ]; then
  echo "ERROR: Could not locate protein_mpnn_run.py."
  echo "Set PROTEINMPNN_PATH to the full path of protein_mpnn_run.py."
  exit 1
fi

MPNN_DIR="$(dirname "$MPNN_RUNNER")"
export PROTEINMPNN_DIR="$MPNN_DIR"
export PYTHONPATH="$MPNN_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 "$SCRIPT_DIR/run_case_designs.py" \
  --pdb-id "$PDB_ID" \
  --pdb-path "$PDB_PATH" \
  --num-seqs "$NUM_SEQS" \
  --sampling-temp 0.1 \
  --seed 42 \
  --proteinmpnn-path "$MPNN_RUNNER" \
  --output-dir "$OUTPUT_DIR"

echo "Design outputs for $PDB_ID written to: $OUTPUT_DIR"
