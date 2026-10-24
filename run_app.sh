#!/usr/bin/env bash
# ==============================================================================
# run_app.sh - Binzy Smart Waste Segregation Assistant (macOS / Linux Launcher)
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Set up local cache directories inside project folder
export HF_HOME="$DIR/.cache/hf"
export TORCH_HOME="$DIR/.cache/torch"
export OMP_NUM_THREADS=1

# Check for virtual environment
if [ ! -d ".venv" ] || [ ! -f ".venv/bin/python" ]; then
    echo "========================================================================"
    echo " Virtual environment (.venv) not found!"
    echo " Setting up dedicated Python virtual environment..."
    echo "========================================================================"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

PYTHON=".venv/bin/python"

# Parse CLI arguments
DEVICE="auto"
LAN_FLAG=""
PORT="8000"
PREFLIGHT_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --mps)
            DEVICE="mps"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --lan)
            LAN_FLAG="--lan"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --preflight)
            PREFLIGHT_ONLY=1
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_app.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu               Force CPU fallback (disables Apple MPS)"
            echo "  --mps               Force Apple Silicon Metal Performance Shaders"
            echo "  --device <dev>      Select device: auto, mps, cpu, cuda"
            echo "  --lan               Listen on all network interfaces (for phone demo over Wi-Fi)"
            echo "  --port <port>       Port to listen on (default: 8000)"
            echo "  --preflight         Run prerequisite artifact check without starting server"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    "$PYTHON" preflight.py
    exit $?
fi

# Run preflight check
echo "========================================================================"
echo " ♻️  Starting Binzy on macOS ($(uname -m))..."
echo "========================================================================"
"$PYTHON" preflight.py || true

# Set device environment if overridden
if [ "$DEVICE" != "auto" ]; then
    export BINZY_DEVICE="$DEVICE"
    echo " Execution device explicitly set to: $DEVICE"
fi

HOST="127.0.0.1"
if [ -n "$LAN_FLAG" ]; then
    HOST="0.0.0.0"
    IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<your-ip>")
    echo ""
    echo " 🌐 Accessible on local network at: http://${IP}:${PORT}"
fi

echo " 🚀 Web application starting at:   http://localhost:${PORT}"
echo "    Press Ctrl+C to stop."
echo "========================================================================"

exec "$PYTHON" server.py $LAN_FLAG --port "$PORT" ${DEVICE:+--device "$DEVICE"}
