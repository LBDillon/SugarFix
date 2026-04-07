#!/bin/bash
# Influenza HA Case Study
# Analyzes how ProteinMPNN treats a heavily glycosylated multi-chain protein

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROTEINMPNN_PATH="${PROJECT_DIR}/ProteinMPNN/protein_mpnn_run.py"
NUM_DESIGNS=32

# HA PDB options (uncomment one):
# 1RUZ: H3N2 HA (commonly used)
# 3HMG: H1N1 HA
# 4HMG: H1N1 HA with glycans modeled
HA_PDB_ID="1RUZ"

OUTPUT_DIR="${SCRIPT_DIR}/results_HA_case_study"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "INFLUENZA HA CASE STUDY"
echo "=============================================="
echo "PDB: ${HA_PDB_ID}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Step 1: Run the standard pipeline
echo "Running standard pipeline..."
python "${SCRIPT_DIR}/run_pipeline.py" \
    --pdb_id "${HA_PDB_ID}" \
    --output_dir "${OUTPUT_DIR}/${HA_PDB_ID}" \
    --proteinmpnn_path "${PROTEINMPNN_PATH}" \
    --num_designs "${NUM_DESIGNS}" 2>&1 || {
        echo "Pipeline failed - check if structure has issues"
        exit 1
    }

# Step 2: Re-run retention analysis (in case FASTA naming issue)
echo ""
echo "Re-running retention analysis..."
python "${SCRIPT_DIR}/scripts/04_analyze_retention.py" \
    --pdb_dir "${OUTPUT_DIR}/${HA_PDB_ID}" 2>&1

# Step 3: De novo sequon analysis
echo ""
echo "Running de novo sequon analysis..."
python "${SCRIPT_DIR}/scripts/05_analyze_denovo.py" \
    --pdb_dir "${OUTPUT_DIR}/${HA_PDB_ID}" 2>&1 || echo "De novo analysis had issues"

# Step 4: Summarize results
echo ""
echo "=============================================="
echo "HA CASE STUDY RESULTS"
echo "=============================================="

echo ""
echo "Structure info:"
cat "${OUTPUT_DIR}/${HA_PDB_ID}/structure/chain_summary.csv" 2>/dev/null || echo "Chain summary not available"

echo ""
echo "Sequons identified:"
cat "${OUTPUT_DIR}/${HA_PDB_ID}/sequons/sequons.csv" 2>/dev/null || echo "Sequons not available"

echo ""
echo "Retention summary:"
cat "${OUTPUT_DIR}/${HA_PDB_ID}/analysis/retention/retention_summary.csv" 2>/dev/null || echo "Retention summary not available"

echo ""
echo "=============================================="
echo "KEY FILES"
echo "=============================================="
echo "Structure: ${OUTPUT_DIR}/${HA_PDB_ID}/structure/"
echo "Sequons: ${OUTPUT_DIR}/${HA_PDB_ID}/sequons/sequons.csv"
echo "Designs: ${OUTPUT_DIR}/${HA_PDB_ID}/designs/"
echo "Retention: ${OUTPUT_DIR}/${HA_PDB_ID}/analysis/retention/"
echo ""
echo "Figures:"
echo "  - ${OUTPUT_DIR}/${HA_PDB_ID}/sequons/figures/sequon_map.png"
echo "  - ${OUTPUT_DIR}/${HA_PDB_ID}/analysis/retention/figures/retention_heatmap.png"
