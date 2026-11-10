#!/bin/bash
echo "============================================================"
echo "      DermaScan AI - Skin Disease Detection Studio"
echo "============================================================"

# Check if .venv exists; if not, create and install automatically
if [ ! -d ".venv" ]; then
    echo "[1/3] First-time setup detected on Linux!"
    echo "[2/3] Creating virtual environment (.venv)..."
    python3 -m venv .venv
    echo "[3/3] Installing dependencies from requirements.txt..."
    source .venv/bin/activate
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    echo "Setup complete!"
    echo "============================================================"
fi

echo "Starting DermaScan AI PyTorch Server..."
source .venv/bin/activate

# Open default browser on Linux (if supported)
if command -v xdg-open > /dev/null; then
    xdg-open "http://localhost:8000" &
fi

python3 server.py
