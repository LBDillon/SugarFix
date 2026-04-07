#!/usr/bin/env bash
set -euo pipefail

# Batch runner for the case study pipeline.
#
# Takes a candidates CSV (from screen_uniprot_candidates.py) and runs
# the full pipeline for each recommended protein.
#
# Usage:
#   ./run_batch_pipeline.sh candidates.csv [NUM_SEQS]
#
# Or for a single PDB:
#   ./run_batch_pipeline.sh --pdb 6B9O [NUM_SEQS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_SEQS="${2:-64}"
LOG_DIR="$PIPELINE_ROOT/data/batch_logs"
mkdir -p "$LOG_DIR"

SUMMARY_CSV="$PIPELINE_ROOT/data/batch_summary.csv"

run_single_protein() {
    local PDB_ID="$1"
    local LOG="$LOG_DIR/${PDB_ID}.log"
    local PREP_DIR="$PIPELINE_ROOT/data/prep/$PDB_ID"
    local OUTPUT_DIR="$PIPELINE_ROOT/data/outputs/output_${PDB_ID}"

    echo "========================================"
    echo "Processing: $PDB_ID"
    echo "========================================"

    # Check if already processed
    if [ -f "$OUTPUT_DIR/all_conditions_retention.csv" ]; then
        echo "  Already processed (output exists). Skipping."
        echo "  To reprocess, delete $OUTPUT_DIR first."
        echo "$PDB_ID,skipped,already_processed" >> "$SUMMARY_CSV"
        return 0
    fi

    # Step 1: Prepare structure
    echo "  Step 1: Preparing structure..."
    if ! bash "$PIPELINE_ROOT/01_preparation/prepare_structure.sh" "$PDB_ID" "$PREP_DIR" >> "$LOG" 2>&1; then
        echo "  FAILED at Step 1 (prepare_structure). See $LOG"
        echo "$PDB_ID,failed,prepare_structure" >> "$SUMMARY_CSV"
        return 1
    fi

    # Skip structures with missing residues (they cause X's in designs)
    STRUCTURE_JSON="$PREP_DIR/structure/structure_info.json"
    if [ -f "$STRUCTURE_JSON" ]; then
        TOTAL_MISSING=$(python3 -c "
import json
info = json.load(open('$STRUCTURE_JSON'))
print(info.get('total_missing_residues', 0))
" 2>/dev/null || echo "0")
        if [ "$TOTAL_MISSING" -gt 0 ]; then
            echo "  SKIPPED: $TOTAL_MISSING missing residues detected. Designs would contain X's."
            echo "$PDB_ID,skipped,missing_residues_$TOTAL_MISSING" >> "$SUMMARY_CSV"
            return 0
        fi
    fi

    # Step 2: Run designs
    echo "  Step 2: Running ProteinMPNN designs ($NUM_SEQS sequences)..."
    if ! bash "$PIPELINE_ROOT/02_design/run_designs.sh" "$PDB_ID" "$NUM_SEQS" "$PREP_DIR" "$OUTPUT_DIR" >> "$LOG" 2>&1; then
        echo "  FAILED at Step 2 (run_designs). See $LOG"
        echo "$PDB_ID,failed,run_designs" >> "$SUMMARY_CSV"
        return 1
    fi

    # Skip proteins with zero sequons (nothing to analyze)
    if grep -q "Total sequons found: 0" "$LOG" 2>/dev/null; then
        echo "  SKIPPED: No N-X-S/T sequons found in wild-type structure."
        echo "$PDB_ID,skipped,no_sequons" >> "$SUMMARY_CSV"
        return 0
    fi

    # Step 3: Analyze designs
    echo "  Step 3: Analyzing designs..."
    if ! bash "$PIPELINE_ROOT/03_analysis/analyze_designs.sh" "$PDB_ID" "$PREP_DIR" "$OUTPUT_DIR" >> "$LOG" 2>&1; then
        echo "  FAILED at Step 3 (analyze_designs). See $LOG"
        echo "$PDB_ID,failed,analyze_designs" >> "$SUMMARY_CSV"
        return 1
    fi

    # Step 4: Generate AF3 JSONs
    echo "  Step 4: Generating AF3 JSONs..."
    if ! bash "$PIPELINE_ROOT/04_af3_generation/generate_af3_jsons.sh" "$OUTPUT_DIR" >> "$LOG" 2>&1; then
        echo "  WARNING: AF3 JSON generation had issues. See $LOG"
        # Don't fail the whole pipeline for this
    fi

    echo "  SUCCESS: $PDB_ID complete."
    echo "$PDB_ID,success,complete" >> "$SUMMARY_CSV"
    return 0
}

# Parse arguments
if [ "${1:-}" = "--pdb" ]; then
    # Single PDB mode
    PDB_ID="${2:-}"
    NUM_SEQS="${3:-64}"
    if [ -z "$PDB_ID" ]; then
        echo "Usage: $0 --pdb <PDB_ID> [NUM_SEQS]"
        exit 1
    fi
    echo "pdb_id,status,detail" > "$SUMMARY_CSV"
    run_single_protein "$PDB_ID"
    exit $?
fi

# Batch mode from CSV
CSV_FILE="${1:-}"
if [ -z "$CSV_FILE" ] || [ ! -f "$CSV_FILE" ]; then
    echo "Usage: $0 <candidates.csv> [NUM_SEQS]"
    echo "   or: $0 --pdb <PDB_ID> [NUM_SEQS]"
    exit 1
fi

echo "============================================="
echo "BATCH PIPELINE RUNNER"
echo "============================================="
echo "Candidates CSV: $CSV_FILE"
echo "Sequences per design: $NUM_SEQS"
echo ""

# Initialize summary
echo "pdb_id,status,detail" > "$SUMMARY_CSV"

# Extract recommended PDB IDs from CSV
PDBS=$(python3 -c "
import csv
seen = set()
with open('$CSV_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('recommended', '').lower() in ('true', '1', 'yes'):
            pdb = row.get('best_pdb', '').strip()
            if pdb and len(pdb) == 4 and pdb not in seen:
                seen.add(pdb)
                print(pdb)
")

if [ -z "$PDBS" ]; then
    echo "No recommended proteins found in CSV."
    exit 0
fi

N_TOTAL=$(echo "$PDBS" | wc -l | tr -d ' ')
echo "Found $N_TOTAL recommended proteins to process."
echo ""

N_SUCCESS=0
N_FAILED=0
N_SKIPPED=0
N_CURRENT=0

for PDB_ID in $PDBS; do
    N_CURRENT=$((N_CURRENT + 1))
    echo "[$N_CURRENT/$N_TOTAL] $PDB_ID"

    if run_single_protein "$PDB_ID"; then
        N_SUCCESS=$((N_SUCCESS + 1))
    else
        N_FAILED=$((N_FAILED + 1))
    fi
    echo ""
done

echo "============================================="
echo "BATCH COMPLETE"
echo "============================================="
echo "  Total:   $N_TOTAL"
echo "  Success: $N_SUCCESS"
echo "  Failed:  $N_FAILED"
echo "  Summary: $SUMMARY_CSV"
echo "  Logs:    $LOG_DIR/"
