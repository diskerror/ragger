#!/bin/bash
# Wait for external drive to mount, then start Ragger Memory server.
# Used by launchd plist to handle boot timing.

VOLUME="/Volumes/WDBlack2"
RAGGER_DIR="$VOLUME/PyCharmProjects/Ragger"
PYTHON="$RAGGER_DIR/.venv/bin/python3"
RAGGER="$RAGGER_DIR/ragger.py"
MAX_WAIT=120  # seconds

elapsed=0
while [ ! -d "$RAGGER_DIR" ]; do
    sleep 5
    elapsed=$((elapsed + 5))
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo "Timed out waiting for $VOLUME after ${MAX_WAIT}s" >&2
        exit 1
    fi
done

export HOME="$VOLUME"
exec "$PYTHON" "$RAGGER" --server
