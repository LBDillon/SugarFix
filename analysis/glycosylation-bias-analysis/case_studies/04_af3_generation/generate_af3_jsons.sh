#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEARCH_DIR="${1:-$PIPELINE_ROOT/data/outputs}"

python3 "$SCRIPT_DIR/generate_af3_jsons.py" --search-dir "$SEARCH_DIR"

echo "AF3 JSONs were generated into each output folder:"
echo "  <output_dir>/top_designs_for_AF3"
