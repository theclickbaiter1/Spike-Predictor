#!/usr/bin/env bash
# run_daily.sh — Run spike detector prediction + send Telegram alert.
# Used by GitHub Actions and can also be run manually.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== Spike Detector Daily Run ==="
echo "Date: $(date)"

# Install deps (GitHub Actions needs this every run)
pip install -q -r requirements.txt

# Run prediction
python spike_detector.py

# Send Telegram notification
python notify.py

echo "=== Done ==="
