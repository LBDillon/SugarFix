#!/bin/bash
# SugarFix setup — install dependencies and clone ProteinMPNN
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
echo "Setup complete. Open the notebook:"
echo "  jupyter lab sugarfix_walkthrough.ipynb"
