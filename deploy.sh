#!/bin/bash
# deploy.sh — Deploy Python Ragger to /Users/Shared/Ragger
#
# Copies the project (minus dev artifacts) to the shared deployment location.
# The /usr/local/bin/ragger wrapper points here.
#
# Usage: bash deploy.sh

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="/Users/Shared/Ragger"

echo "[+] Deploying from $SRC to $DEST"

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
  --exclude='migrate_collections.py' \
  "$SRC/" "$DEST/"

# Ensure venv exists in deployment
if [ ! -d "$DEST/.venv" ]; then
    echo "[+] Creating venv in $DEST"
    python3 -m venv "$DEST/.venv"
    "$DEST/.venv/bin/pip" install -q -r "$DEST/requirements.txt"
fi

echo "[+] Deployed successfully"
echo "    Binary: /usr/local/bin/ragger → $DEST"
