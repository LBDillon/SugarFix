#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PDB_ID="${1:-}"
PREP_DIR="${2:-$PIPELINE_ROOT/data/prep/$PDB_ID}"

if [ -z "$PDB_ID" ]; then
  echo "Usage: $0 <PDB_ID> [PREP_DIR]"
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

python3 "$SCRIPT_DIR/prepare_structure.py" \
  --pdb_id "$PDB_ID" \
  --output_dir "$PREP_DIR"

python3 "$SCRIPT_DIR/identify_sequons.py" \
  --pdb_dir "$PREP_DIR"

echo "Prepared structure for $PDB_ID at: $PREP_DIR/structure/${PDB_ID}_protein.pdb"
