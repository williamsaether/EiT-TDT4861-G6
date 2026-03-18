#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python3 || command -v python)"

echo "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/app/requirements.txt"

echo "Starting unified app at http://localhost:8000 ..."
echo "Driving demo: http://localhost:8000/driving/ or ../examples"
echo "Tuning dashboard: http://localhost:8000/tuning/"
(cd "$SCRIPT_DIR" && "$PYTHON_BIN" -m app.app)
