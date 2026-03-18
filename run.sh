#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/web_demo/requirements.txt"

echo "Starting web demo at http://localhost:8000 ..."
python "$SCRIPT_DIR/web_demo/app.py"
