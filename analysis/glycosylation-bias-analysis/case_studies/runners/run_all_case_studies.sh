#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_SEQS="${1:-64}"
shift || true

if [ "$#" -gt 0 ]; then
  PDB_IDS=("$@")
else
  PDB_IDS=("1GQV" "1ATJ" "1RUZ" "5EQG" "1C1Z")
fi

echo "Running case-study pipeline for: ${PDB_IDS[*]}"
echo "Designs per condition: $NUM_SEQS"
echo

for PDB_ID in "${PDB_IDS[@]}"; do
  echo "===================================================================="
  echo "CASE STUDY: $PDB_ID"
  echo "===================================================================="
  bash "$SCRIPT_DIR/run_case_study_end_to_end.sh" "$PDB_ID" "$NUM_SEQS"
  echo
done

echo "Completed case-study runs for: ${PDB_IDS[*]}"
