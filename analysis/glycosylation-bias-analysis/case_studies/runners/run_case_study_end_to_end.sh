#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PDB_ID="${1:-}"
NUM_SEQS="${2:-64}"

if [ -z "$PDB_ID" ]; then
  echo "Usage: $0 <PDB_ID> [NUM_SEQS]"
  exit 1
fi

bash "$PIPELINE_ROOT/01_preparation/prepare_structure.sh" "$PDB_ID"
bash "$PIPELINE_ROOT/02_design/run_designs.sh" "$PDB_ID" "$NUM_SEQS"
bash "$PIPELINE_ROOT/03_analysis/analyze_designs.sh" "$PDB_ID"
bash "$PIPELINE_ROOT/04_af3_generation/generate_af3_jsons.sh" "$PIPELINE_ROOT/data/outputs/output_${PDB_ID}"
