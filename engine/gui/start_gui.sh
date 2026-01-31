#!/bin/bash
# Hivemind Bughouse GUI Launcher
# Starts the GUI server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Install dependencies if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask dependencies..."
    pip3 install -r requirements.txt
fi

PORT=${1:-8080}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           🐝 Hivemind Bughouse GUI 🐝                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Open your browser to: http://localhost:$PORT               ║"
echo "║                                                              ║"
echo "║  All controls are in the browser - just select models,      ║"
echo "║  configure parameters, and click Start!                     ║"
echo "║                                                              ║"
echo "║  Press Ctrl+C to stop the server                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python3 server.py --port "$PORT"
