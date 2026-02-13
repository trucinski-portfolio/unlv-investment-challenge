#!/bin/bash
# =============================================================================
# ASTRYX INVESTING - Daily Automation Pipeline
# Chains: Fetch -> Validate -> Process -> Generate Recommendations
#
# Usage:
#   ./run_daily.sh               # Run full pipeline for today
#   ./run_daily.sh 2026-02-10    # Run for a specific date
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
DATE="${1:-$(date +%Y-%m-%d)}"

echo "=============================================="
echo " ASTRYX INVESTING - Daily Pipeline"
echo " Date: ${DATE}"
echo " Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# Step 1: Fetch daily fundamentals
echo ""
echo "[1/4] Fetching fundamentals..."
python3 "${SRC_DIR}/data_fetcher.py" --date "${DATE}"

# Step 2: Validate the new CSV
echo ""
echo "[2/4] Validating data..."
python3 "${SRC_DIR}/validator.py" --date "${DATE}"

# Step 3: Process trends (requires historical data)
echo ""
echo "[3/4] Processing trends..."
python3 "${SRC_DIR}/processor.py" --date "${DATE}"

# Step 4: Generate recommendations
echo ""
echo "[4/4] Generating recommendations..."
python3 "${SRC_DIR}/models.py" --date "${DATE}"

echo ""
echo "=============================================="
echo " Pipeline complete!"
echo " Check output/ for recommendations"
echo "=============================================="
