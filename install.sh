#!/bin/bash
# install.sh — Install Python Ragger to this machine
#
# Usage: ./install.sh
#
# Copies the project to /Users/Shared/Ragger (macOS) or
# /opt/ragger (Linux), creates a venv, and installs the
# wrapper script to /usr/local/bin/ragger.

set -euo pipefail

RED='\033[0;31m'
NC='\033[0m'

missing=()
command -v python3 &>/dev/null || missing+=("python3")
command -v rsync &>/dev/null   || missing+=("rsync")

if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${RED}[!] Missing required tools:${NC} ${missing[*]}"
    exit 1
fi

# Check python3 has venv module (some distros strip it out)
if ! python3 -c "import venv" 2>/dev/null; then
    echo -e "${RED}[!] Python venv module not available.${NC}"
    echo "    Linux (apt): sudo apt install python3-venv"
    exit 1
fi

SRC="$(cd "$(dirname "$0")" && pwd)"
OS="$(uname -s)"

case "$OS" in
    Darwin) DEST="/usr/local/lib/ragger" ;;
    Linux)  DEST="/opt/ragger" ;;
    *)      echo "[!] Unsupported OS: $OS" >&2; exit 1 ;;
esac

echo "[+] Installing from $SRC to $DEST"
mkdir -p "$DEST"

rsync -a --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='.idea' \
    --exclude='.venv' \
    --exclude='.DS_Store' \
    --exclude='.gitignore' \
    --exclude='memories.db' \
    "$SRC/" "$DEST/"

# Create venv if missing
if [ ! -d "$DEST/.venv" ]; then
    echo "[+] Creating venv..."
    python3 -m venv "$DEST/.venv"
fi

# Always install/update dependencies
if [ -f "$DEST/requirements.txt" ]; then
    echo "[+] Installing/updating dependencies..."
    "$DEST/.venv/bin/pip" install -q --upgrade -r "$DEST/requirements.txt"
fi

# Install wrapper script
WRAPPER="/usr/local/bin/ragger"
echo "[+] Installing wrapper to $WRAPPER (requires sudo)"
sudo tee "$WRAPPER" > /dev/null << SCRIPT
#!/bin/bash
export PYTHONPATH="$DEST"
exec "$DEST/.venv/bin/python" -m ragger_memory.cli "\$@"
SCRIPT
sudo chmod 0755 "$WRAPPER"

# Restart daemon if running
if [ "$OS" = "Darwin" ]; then
    PLIST="com.diskerror.ragger"
    if sudo launchctl list "$PLIST" &>/dev/null; then
        echo "[+] Restarting daemon..."
        sudo launchctl kickstart -k "system/$PLIST"
    else
        echo "[*] Daemon not loaded — skipping restart"
    fi
elif [ "$OS" = "Linux" ]; then
    if systemctl is-active --quiet ragger; then
        echo "[+] Restarting daemon..."
        sudo systemctl restart ragger
    else
        echo "[*] Service not active — skipping restart"
    fi
fi

echo ""
echo "✓ Installed: $WRAPPER → $DEST"
