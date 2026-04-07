#!/bin/bash
# Glycosylation bias analysis setup
set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ ! -d "ProteinMPNN" ]; then
    echo "Cloning ProteinMPNN..."
    git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git
else
    echo "ProteinMPNN already present."
fi

echo ""
echo "Setup complete."
echo ""
echo "To reproduce the analysis with your own proteins:"
echo "  1. Edit data/candidates_template.csv with your protein list"
echo "  2. Run: bash case_studies/runners/run_batch_pipeline.sh"
echo ""
echo "See README.md for the full pipeline walkthrough."
