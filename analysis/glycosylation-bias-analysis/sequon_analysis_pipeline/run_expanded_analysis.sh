#!/bin/bash
# Expanded Analysis: Run pipeline on all available proteins
# This will take several hours - run in background with nohup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROTEINMPNN_PATH="${PROJECT_DIR}/ProteinMPNN/protein_mpnn_run.py"
NUM_DESIGNS=32

# Output directories
GLY_RESULTS="${SCRIPT_DIR}/results_full_gly"
NON_GLY_RESULTS="${SCRIPT_DIR}/results_full_non_gly"
LOG_FILE="${SCRIPT_DIR}/expanded_analysis.log"

mkdir -p "${GLY_RESULTS}"
mkdir -p "${NON_GLY_RESULTS}"

echo "=============================================="
echo "EXPANDED ANALYSIS - ALL PROTEINS"
echo "=============================================="
echo "Started at: $(date)"
echo "Log file: ${LOG_FILE}"
echo ""

# Function to process a single protein
process_protein() {
    local pdb_id="$1"
    local output_dir="$2"
    local pdb_upper=$(echo "$pdb_id" | tr '[:lower:]' '[:upper:]')

    # Skip if already processed
    if [ -f "${output_dir}/designs/unconstrained/seqs/${pdb_upper}_protein.fa" ] || \
       [ -f "${output_dir}/designs/unconstrained/seqs/${pdb_upper}.fa" ]; then
        echo "[SKIP] ${pdb_id} - already processed"
        return 0
    fi

    echo "[START] ${pdb_id} at $(date)"

    python "${SCRIPT_DIR}/run_pipeline.py" \
        --pdb_id "${pdb_id}" \
        --output_dir "${output_dir}" \
        --proteinmpnn_path "${PROTEINMPNN_PATH}" \
        --num_designs "${NUM_DESIGNS}" 2>&1 || {
            echo "[FAIL] ${pdb_id}"
            return 1
        }

    echo "[DONE] ${pdb_id} at $(date)"
    return 0
}

# Count totals
total_gly=$(ls "${SCRIPT_DIR}/PDBs_gly"/*.pdb 2>/dev/null | wc -l | tr -d ' ')
total_non_gly=$(ls "${SCRIPT_DIR}/PDBs_non_gly"/*.pdb 2>/dev/null | wc -l | tr -d ' ')

echo "Glycosylated proteins: ${total_gly}"
echo "Non-glycosylated proteins: ${total_non_gly}"
echo ""

# Process glycosylated proteins
echo "=============================================="
echo "PROCESSING GLYCOSYLATED PROTEINS"
echo "=============================================="

count=0
success=0
for pdb_file in "${SCRIPT_DIR}/PDBs_gly"/*.pdb; do
    if [ -f "$pdb_file" ]; then
        pdb_id=$(basename "$pdb_file" .pdb)
        count=$((count + 1))
        echo "[$count/$total_gly] Processing: $pdb_id"

        if process_protein "$pdb_id" "${GLY_RESULTS}/${pdb_id}"; then
            success=$((success + 1))
        fi
    fi
done

echo ""
echo "Glycosylated: $success/$total_gly successful"
echo ""

# Process non-glycosylated proteins
echo "=============================================="
echo "PROCESSING NON-GLYCOSYLATED PROTEINS"
echo "=============================================="

count=0
success_ng=0
for pdb_file in "${SCRIPT_DIR}/PDBs_non_gly"/*.pdb; do
    if [ -f "$pdb_file" ]; then
        pdb_id=$(basename "$pdb_file" .pdb)
        count=$((count + 1))
        echo "[$count/$total_non_gly] Processing: $pdb_id"

        if process_protein "$pdb_id" "${NON_GLY_RESULTS}/${pdb_id}"; then
            success_ng=$((success_ng + 1))
        fi
    fi
done

echo ""
echo "Non-glycosylated: $success_ng/$total_non_gly successful"
echo ""

# Run aggregate baseline analysis
echo "=============================================="
echo "RUNNING AGGREGATE ANALYSIS"
echo "=============================================="

echo "Analyzing glycosylated proteins..."
python "${SCRIPT_DIR}/scripts/00_baseline_aa_retention.py" \
    --pdb_folder "${GLY_RESULTS}" \
    --output_dir "${GLY_RESULTS}" \
    --condition unconstrained 2>&1 || echo "Warning: Glycosylated baseline failed"

echo ""
echo "Analyzing non-glycosylated proteins..."
python "${SCRIPT_DIR}/scripts/00_baseline_aa_retention.py" \
    --pdb_folder "${NON_GLY_RESULTS}" \
    --output_dir "${NON_GLY_RESULTS}" \
    --condition unconstrained 2>&1 || echo "Warning: Non-glycosylated baseline failed"

echo ""
echo "Comparing datasets..."
mkdir -p "${SCRIPT_DIR}/aggregate_comparison"
python "${SCRIPT_DIR}/scripts/00_baseline_aa_retention.py" \
    --compare_folders "${GLY_RESULTS}" "${NON_GLY_RESULTS}" \
    --folder_labels "Glycosylated" "Non-Glycosylated" \
    --output_dir "${SCRIPT_DIR}/aggregate_comparison" \
    --condition unconstrained 2>&1 || echo "Warning: Comparison failed"

echo ""
echo "=============================================="
echo "EXPANDED ANALYSIS COMPLETE"
echo "=============================================="
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  Glycosylated: ${GLY_RESULTS}/"
echo "  Non-glycosylated: ${NON_GLY_RESULTS}/"
echo "  Comparison: ${SCRIPT_DIR}/aggregate_comparison/"
