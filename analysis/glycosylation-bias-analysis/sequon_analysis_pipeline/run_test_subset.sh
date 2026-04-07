#!/bin/bash
# Test script: Run pipeline on 5 proteins from each group (<=3 chains)
# Then compare baseline AA retention between glycosylated and non-glycosylated
#
# Selected proteins (all with <=3 chains):
# Glycosylated: 1i9e (1), 1bte (2), 4bfg (1), 3mw4 (3), 5b5k (1)
# Non-glycosylated: 1914 (1), 1alb (1), 1cxv (2), 1dmy (2), 1a1f (3)

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROTEINMPNN_PATH="${PROJECT_DIR}/ProteinMPNN/protein_mpnn_run.py"
NUM_DESIGNS=32  # Number of designs per condition

# Output directories
GLY_RESULTS="${SCRIPT_DIR}/test_results_gly"
NON_GLY_RESULTS="${SCRIPT_DIR}/test_results_non_gly"
COMPARISON_DIR="${SCRIPT_DIR}/test_comparison"

# Selected proteins (<=3 chains each)
GLY_PDBS=("1i9e" "1bte" "4bfg" "3mw4" "5b5k")
NON_GLY_PDBS=("1914" "1alb" "1cxv" "1dmy" "1a1f")

echo "=============================================="
echo "SEQUON ANALYSIS PIPELINE - TEST SUBSET"
echo "=============================================="
echo ""
echo "Glycosylated proteins: ${GLY_PDBS[*]}"
echo "Non-glycosylated proteins: ${NON_GLY_PDBS[*]}"
echo "ProteinMPNN path: ${PROTEINMPNN_PATH}"
echo "Designs per condition: ${NUM_DESIGNS}"
echo ""

# Create output directories
mkdir -p "${GLY_RESULTS}"
mkdir -p "${NON_GLY_RESULTS}"

# Process glycosylated proteins
echo "=============================================="
echo "PROCESSING GLYCOSYLATED PROTEINS"
echo "=============================================="
for pdb in "${GLY_PDBS[@]}"; do
    pdb_upper=$(echo "$pdb" | tr '[:lower:]' '[:upper:]')
    output_dir="${GLY_RESULTS}/${pdb}"

    # Skip if already processed successfully (check for FASTA file)
    if [ -f "${output_dir}/designs/unconstrained/seqs/${pdb_upper}.fa" ]; then
        echo "Skipping ${pdb} - already processed"
        continue
    fi

    echo ""
    echo "Processing: ${pdb}"

    # Run pipeline (downloads from RCSB and processes)
    python "${SCRIPT_DIR}/run_pipeline.py" \
        --pdb_id "${pdb}" \
        --output_dir "${output_dir}" \
        --proteinmpnn_path "${PROTEINMPNN_PATH}" \
        --num_designs "${NUM_DESIGNS}" || {
            echo "Warning: Pipeline failed for ${pdb}"
            continue
        }

    echo "Completed: ${pdb}"
done

# Process non-glycosylated proteins
echo ""
echo "=============================================="
echo "PROCESSING NON-GLYCOSYLATED PROTEINS"
echo "=============================================="
for pdb in "${NON_GLY_PDBS[@]}"; do
    pdb_upper=$(echo "$pdb" | tr '[:lower:]' '[:upper:]')
    output_dir="${NON_GLY_RESULTS}/${pdb}"

    # Skip if already processed successfully
    if [ -f "${output_dir}/designs/unconstrained/seqs/${pdb_upper}.fa" ]; then
        echo "Skipping ${pdb} - already processed"
        continue
    fi

    echo ""
    echo "Processing: ${pdb}"

    # Run pipeline
    python "${SCRIPT_DIR}/run_pipeline.py" \
        --pdb_id "${pdb}" \
        --output_dir "${output_dir}" \
        --proteinmpnn_path "${PROTEINMPNN_PATH}" \
        --num_designs "${NUM_DESIGNS}" || {
            echo "Warning: Pipeline failed for ${pdb}"
            continue
        }

    echo "Completed: ${pdb}"
done

# Run baseline AA retention analysis
echo ""
echo "=============================================="
echo "BASELINE AMINO ACID RETENTION ANALYSIS"
echo "=============================================="

# Analyze glycosylated
echo ""
echo "Analyzing glycosylated proteins..."
python "${SCRIPT_DIR}/scripts/00_baseline_aa_retention.py" \
    --pdb_folder "${GLY_RESULTS}" \
    --output_dir "${GLY_RESULTS}" \
    --condition unconstrained || echo "Warning: Glycosylated baseline analysis failed"

# Analyze non-glycosylated
echo ""
echo "Analyzing non-glycosylated proteins..."
python "${SCRIPT_DIR}/scripts/00_baseline_aa_retention.py" \
    --pdb_folder "${NON_GLY_RESULTS}" \
    --output_dir "${NON_GLY_RESULTS}" \
    --condition unconstrained || echo "Warning: Non-glycosylated baseline analysis failed"

# Compare the two datasets
echo ""
echo "=============================================="
echo "COMPARING GLYCOSYLATED vs NON-GLYCOSYLATED"
echo "=============================================="
mkdir -p "${COMPARISON_DIR}"
python "${SCRIPT_DIR}/scripts/00_baseline_aa_retention.py" \
    --compare_folders "${GLY_RESULTS}" "${NON_GLY_RESULTS}" \
    --folder_labels "Glycosylated" "Non-Glycosylated" \
    --output_dir "${COMPARISON_DIR}" \
    --condition unconstrained || echo "Warning: Comparison analysis failed"

echo ""
echo "=============================================="
echo "TEST COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  Glycosylated:     ${GLY_RESULTS}/"
echo "  Non-Glycosylated: ${NON_GLY_RESULTS}/"
echo "  Comparison:       ${COMPARISON_DIR}/"
echo ""
echo "Key outputs:"
echo "  - ${GLY_RESULTS}/analysis/baseline/aa_retention_summary.csv"
echo "  - ${NON_GLY_RESULTS}/analysis/baseline/aa_retention_summary.csv"
echo "  - ${COMPARISON_DIR}/dataset_comparison.csv"
echo "  - ${COMPARISON_DIR}/comparison_figures/retention_comparison_barplot.png"
echo "  - ${COMPARISON_DIR}/comparison_figures/asparagine_comparison.png"
