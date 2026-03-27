#!/bin/bash
# install.sh — Install Python Ragger to this machine
#
# Usage: sudo ./install.sh
#
# Idempotent: creates system user/group/directories on first run,
# updates executable and reconciles users on subsequent runs.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "[!] This script must be run as root: sudo ./install.sh" >&2
    exit 1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

cd "$(dirname "$0")"
SRC="$(pwd)"

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Darwin)
        DEST="/usr/local/lib/ragger"
        RAGGER_USER="_ragger"
        RAGGER_GROUP="ragger"
        ;;
    Linux)
        DEST="/usr/local/lib/ragger"
        RAGGER_USER="ragger"
        RAGGER_GROUP="ragger"
        ;;
    *)
        echo "[!] Unsupported OS: $OS" >&2
        exit 1
        ;;
esac

# --- Paths (single source of truth) ---
LOG_DIR="/var/log/ragger"
DATA_DIR="/var/ragger"
CONF_FILE="/etc/ragger.ini"

# Detect the actual user who ran sudo (not root)
if [ -n "${SUDO_USER:-}" ]; then
    ACTUAL_USER="$SUDO_USER"
    ACTUAL_HOME=$(eval echo ~"$SUDO_USER")
else
    ACTUAL_USER="$USER"
    ACTUAL_HOME="$HOME"
fi

# Check dependencies
missing=()
command -v python3 &>/dev/null || missing+=("python3")
command -v rsync &>/dev/null   || missing+=("rsync")

if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${RED}[!] Missing required tools:${NC} ${missing[*]}"
    exit 1
fi

if ! python3 -c "import venv" 2>/dev/null; then
    echo -e "${RED}[!] Python venv module not available.${NC}"
    echo "    Linux (apt): sudo apt install python3-venv"
    exit 1
fi

# ============================================================
# PHASE 1: System setup (group, user, directories, permissions)
# ============================================================

# --- Create group if missing ---
if [ "$OS" = "Darwin" ]; then
    if ! dscl . -read /Groups/$RAGGER_GROUP &>/dev/null; then
        info "Creating group: $RAGGER_GROUP"
        LAST_GID=$(dscl . -list /Groups PrimaryGroupID | awk '{print $2}' | sort -n | tail -1)
        NEXT_GID=$((LAST_GID + 1))
        if [ "$NEXT_GID" -lt 400 ]; then NEXT_GID=400; fi
        dscl . -create /Groups/$RAGGER_GROUP
        dscl . -create /Groups/$RAGGER_GROUP PrimaryGroupID "$NEXT_GID"
    fi
else
    if ! getent group $RAGGER_GROUP &>/dev/null; then
        info "Creating group: $RAGGER_GROUP"
        groupadd --system $RAGGER_GROUP
    fi
fi

# --- Create system user if missing ---
if [ "$OS" = "Darwin" ]; then
    if ! dscl . -read /Users/$RAGGER_USER &>/dev/null; then
        info "Creating user: $RAGGER_USER"
        LAST_UID=$(dscl . -list /Users UniqueID | awk '{print $2}' | sort -n | tail -1)
        NEXT_UID=$((LAST_UID + 1))
        if [ "$NEXT_UID" -lt 400 ]; then NEXT_UID=400; fi
        GID=$(dscl . -read /Groups/$RAGGER_GROUP PrimaryGroupID | awk '{print $2}')
        dscl . -create /Users/$RAGGER_USER
        dscl . -create /Users/$RAGGER_USER UniqueID "$NEXT_UID"
        dscl . -create /Users/$RAGGER_USER PrimaryGroupID "$GID"
        dscl . -create /Users/$RAGGER_USER UserShell /usr/bin/false
        dscl . -create /Users/$RAGGER_USER NFSHomeDirectory /var/empty
        dscl . -create /Users/$RAGGER_USER IsHidden 1
    fi
else
    if ! id $RAGGER_USER &>/dev/null; then
        info "Creating user: $RAGGER_USER"
        useradd --system --no-create-home --shell /usr/bin/false \
                --gid $RAGGER_GROUP --home-dir /var/empty $RAGGER_USER
    fi
fi

# --- Create directories and fix ownership/permissions ---
for dir in "$DATA_DIR" "$LOG_DIR"; do
    if [ ! -d "$dir" ]; then
        info "Creating $dir"
        mkdir -p "$dir"
    fi
    chown -R "$RAGGER_USER:$RAGGER_GROUP" "$dir"
    chmod 0750 "$dir"
    find "$dir" -type d -exec chmod 0750 {} +
done

# Fix file permissions: DB files group-writable (for MCP users), logs 0660
info "Fixing file permissions"
find "$DATA_DIR" -name "*.db" -o -name "*.db-shm" -o -name "*.db-wal" 2>/dev/null | while read f; do
    chmod 0660 "$f"
done
find "$LOG_DIR" -type f 2>/dev/null | while read f; do
    chmod 0660 "$f"
done

# ============================================================
# PHASE 2: Mode selection
# ============================================================

# Read current mode from config if it exists
CURRENT_MODE=""
if [ -f "$CONF_FILE" ]; then
    CURRENT_MODE=$(grep -E "^single_user" "$CONF_FILE" | head -1 | sed 's/.*= *//' | tr -d ' ')
fi

echo ""
echo "Ragger can run in two modes:"
echo ""
echo "  1) Single-user  — One user, everything in ~/.ragger/"
echo "                     No shared database or group permissions needed."
echo ""
echo "  2) Multi-user   — Multiple users share a common database for the"
echo "                     user table. Each user keeps their own memories"
echo "                     in ~/.ragger/. Requires ragger group membership."
echo ""

if [ -n "$CURRENT_MODE" ]; then
    if [ "$CURRENT_MODE" = "true" ]; then
        echo "  Current mode: single-user"
        DEFAULT_CHOICE="1"
    else
        echo "  Current mode: multi-user"
        DEFAULT_CHOICE="2"
    fi
    echo ""
    read -p "Choose mode [1/2] (default: $DEFAULT_CHOICE = keep current): " MODE_CHOICE
    MODE_CHOICE="${MODE_CHOICE:-$DEFAULT_CHOICE}"
else
    read -p "Choose mode [1/2] (default: 1): " MODE_CHOICE
    MODE_CHOICE="${MODE_CHOICE:-1}"
fi

case "$MODE_CHOICE" in
    2) SINGLE_USER="false" ;;
    *) SINGLE_USER="true" ;;
esac
echo ""

# ============================================================
# PHASE 3: Config and daemon files
# ============================================================

if [ ! -f "$CONF_FILE" ]; then
    info "Creating $CONF_FILE"
    if [ -f example-system.ini ]; then
        cp example-system.ini "$CONF_FILE"
        if command -v sed &>/dev/null; then
            sed -i.bak "s/^single_user.*/single_user = $SINGLE_USER/" "$CONF_FILE"
            rm -f "$CONF_FILE.bak"
        fi
    else
        cat > "$CONF_FILE" << EOF
[server]
host = 127.0.0.1
port = 8432
single_user = $SINGLE_USER

[logging]
log_dir = $LOG_DIR

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384

[search]
default_limit = 5
default_min_score = 0.4
bm25_weight = 0.3
vector_weight = 0.7

[auth]
token_rotation_minutes = 1440
EOF
    fi
    chmod 0644 "$CONF_FILE"
else
    # Config exists — update mode if changed
    if command -v sed &>/dev/null; then
        if [ "$CURRENT_MODE" != "$SINGLE_USER" ]; then
            sed -i.bak "s/^single_user.*/single_user = $SINGLE_USER/" "$CONF_FILE"
            rm -f "$CONF_FILE.bak"
            info "Updated $CONF_FILE: single_user = $SINGLE_USER"
        fi
    fi
fi

# --- Install default SOUL.md if missing ---
SOUL_DEST="$DATA_DIR/SOUL.md"
SOUL_SRC="$SRC/SOUL.md"
if [ ! -f "$SOUL_DEST" ] && [ -f "$SOUL_SRC" ]; then
    info "Installing default SOUL.md to $DATA_DIR"
    cp "$SOUL_SRC" "$SOUL_DEST"
    chown "$RAGGER_USER:$RAGGER_GROUP" "$SOUL_DEST"
    chmod 0644 "$SOUL_DEST"
fi

# --- Install/update daemon config (always rewrite to stay current) ---
if [ "$OS" = "Darwin" ]; then
    PLIST="/Library/LaunchDaemons/com.diskerror.ragger.plist"
    info "Writing LaunchDaemon plist"
    cat > "$PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.diskerror.ragger</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/ragger</string>
        <string>serve</string>
    </array>
    <key>UserName</key>
    <string>$RAGGER_USER</string>
    <key>GroupName</key>
    <string>$RAGGER_GROUP</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/stderr.log</string>
</dict>
</plist>
EOF
    chmod 0644 "$PLIST"
elif [ "$OS" = "Linux" ]; then
    UNIT="/etc/systemd/system/ragger.service"
    info "Writing systemd unit file"
    cat > "$UNIT" << EOF
[Unit]
Description=Ragger Memory Server
After=network.target

[Service]
Type=simple
User=$RAGGER_USER
Group=$RAGGER_GROUP
ExecStart=/usr/local/bin/ragger serve
Restart=on-failure
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    chmod 0644 "$UNIT"
    systemctl daemon-reload
fi

# ============================================================
# PHASE 4: Install executable (always overwritten)
# ============================================================

info "Installing to $DEST"
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

# --- Create venv ---
if [ ! -d "$DEST/.venv" ]; then
    info "Creating venv"
    python3 -m venv "$DEST/.venv"
fi

# --- Install dependencies ---
if [ -f "$DEST/requirements.txt" ]; then
    info "Installing dependencies"
    "$DEST/.venv/bin/pip" install -q --upgrade -r "$DEST/requirements.txt"
fi

# --- Install wrapper script ---
info "Installing /usr/local/bin/ragger"
cat > /usr/local/bin/ragger << EOF
#!/bin/bash
export PYTHONPATH="$DEST"
exec "$DEST/.venv/bin/python" -m ragger_memory.cli "\$@"
EOF
chmod 0755 /usr/local/bin/ragger

echo ""
info "Installed: /usr/local/bin/ragger → $DEST"
/usr/local/bin/ragger --version 2>/dev/null || true

# ============================================================
# PHASE 5: User reconciliation
# ============================================================

echo ""
info "User reconciliation"
echo ""

# Build list of real system users (skip service accounts)
NOLOGIN_SHELLS="/usr/bin/false /bin/false /sbin/nologin /usr/sbin/nologin"
SKIP_USERS="root nobody nfsnobody"
SKIP_HOMES="/ /var /var/empty /dev/null /nonexistent"

get_system_users() {
    # Output: username:homedir for each real user
    if [ "$OS" = "Darwin" ]; then
        # macOS: use dscl
        dscl . -list /Users NFSHomeDirectory | while read uname homedir; do
            # Skip service accounts
            echo "$SKIP_USERS" | grep -qw "$uname" && continue
            [ "${uname#_}" != "$uname" ] && continue
            echo "$SKIP_HOMES" | grep -qw "$homedir" && continue
            [ ! -d "$homedir" ] && continue
            shell=$(dscl . -read /Users/"$uname" UserShell 2>/dev/null | awk '{print $2}')
            echo "$NOLOGIN_SHELLS" | grep -qw "$shell" && continue
            echo "$uname:$homedir"
        done
    else
        # Linux: use getent
        getent passwd | while IFS=: read uname x uid gid gecos homedir shell; do
            echo "$SKIP_USERS" | grep -qw "$uname" && continue
            [ "${uname#_}" != "$uname" ] && continue
            echo "$SKIP_HOMES" | grep -qw "$homedir" && continue
            [ ! -d "$homedir" ] && continue
            echo "$NOLOGIN_SHELLS" | grep -qw "$shell" && continue
            echo "$uname:$homedir"
        done
    fi
}

# Get ragger-registered users from the DB (if it exists)
get_ragger_users() {
    local db="$DATA_DIR/memories.db"
    if [ -f "$db" ] && command -v sqlite3 &>/dev/null; then
        sqlite3 "$db" "SELECT username FROM users;" 2>/dev/null || true
    fi
}

# Helper: install client configs for a user
install_client_configs() {
    local username="$1"
    local homedir="$2"

    # OpenClaw plugin
    local openclaw_dir="$homedir/.openclaw"
    local plugin_src="$SRC/openclaw-plugin"
    if [ -d "$openclaw_dir" ] && [ -d "$plugin_src" ]; then
        local plugin_dest="$openclaw_dir/extensions/memory-ragger"
        mkdir -p "$plugin_dest"
        cp "$plugin_src/openclaw.plugin.json" "$plugin_dest/"
        cp "$plugin_src/index.ts" "$plugin_dest/"
        chown -R "$username" "$plugin_dest"
        echo "      Installed OpenClaw plugin"
    fi

    # Claude Desktop MCP config
    local claude_config="$homedir/Library/Application Support/Claude/claude_desktop_config.json"
    if [ -d "$(dirname "$claude_config")" ]; then
        if [ ! -f "$claude_config" ] || ! grep -q "ragger" "$claude_config" 2>/dev/null; then
            echo "      Claude Desktop detected — add ragger MCP config manually:"
            echo "        See docs/openclaw.md for MCP transport configuration"
        fi
    fi
}

SYSTEM_USERS=$(get_system_users)
RAGGER_USERS=$(get_ragger_users)

if [ -z "$SYSTEM_USERS" ]; then
    warn "No eligible system users found"
else
    echo "  Comparing system users against ragger database..."
    echo ""

    USERS_CHANGED=false

    echo "$SYSTEM_USERS" | while IFS=: read uname homedir; do
        IN_RAGGER=false
        if echo "$RAGGER_USERS" | grep -qw "$uname"; then
            IN_RAGGER=true
        fi

        if [ "$IN_RAGGER" = "true" ]; then
            # User exists in ragger — offer to reconfigure or remove
            read -p "  $uname (registered) — [L]eave / [C]onfigure clients / [R]emove? [L/c/r] " CHOICE </dev/tty
            CHOICE="${CHOICE:-L}"
            case "$CHOICE" in
                [Cc])
                    echo "    Updating client configs for $uname..."
                    install_client_configs "$uname" "$homedir"
                    ;;
                [Rr])
                    echo "    Removing $uname..."
                    /usr/local/bin/ragger remove-user "$uname" --keep-data 2>&1 | sed 's/^/      /'
                    USERS_CHANGED=true
                    ;;
                *)
                    echo "    Left as is."
                    ;;
            esac
        else
            # User not in ragger — offer to add
            read -p "  $uname (not registered) — [L]eave / [A]dd? [L/a] " CHOICE </dev/tty
            CHOICE="${CHOICE:-L}"
            case "$CHOICE" in
                [Aa])
                    echo "    Adding $uname..."
                    /usr/local/bin/ragger add-user "$uname" 2>&1 | sed 's/^/      /'
                    install_client_configs "$uname" "$homedir"
                    USERS_CHANGED=true
                    ;;
                *)
                    echo "    Skipped."
                    ;;
            esac
        fi
    done

    # Check for ragger users no longer on the system
    if [ -n "$RAGGER_USERS" ]; then
        echo "$RAGGER_USERS" | while read ruser; do
            [ -z "$ruser" ] && continue
            if ! echo "$SYSTEM_USERS" | grep -q "^${ruser}:"; then
                read -p "  $ruser (in ragger but not on system) — [L]eave / [R]emove? [L/r] " CHOICE </dev/tty
                CHOICE="${CHOICE:-L}"
                case "$CHOICE" in
                    [Rr])
                        echo "    Removing stale user $ruser..."
                        /usr/local/bin/ragger remove-user "$ruser" 2>&1 | sed 's/^/      /'
                        ;;
                    *)
                        echo "    Left as is."
                        ;;
                esac
            fi
        done
    fi
fi

# ============================================================
# PHASE 6: Final summary
# ============================================================

echo ""
echo -e "${GREEN}[+]${NC} To (re)start the daemon, run:"
if [ "$OS" = "Darwin" ]; then
    echo "    sudo launchctl bootout system/com.diskerror.ragger 2>/dev/null || true"
    echo "    sudo launchctl bootstrap system $PLIST"
elif [ "$OS" = "Linux" ]; then
    echo "    sudo systemctl restart ragger"
fi

echo ""
if [ "$SINGLE_USER" = "true" ]; then
    echo -e "${GREEN}[+]${NC} Installed in single-user mode."
    echo "    You're ready to go — run 'ragger store \"test\"' to verify."
    echo ""
    echo "    To switch to multi-user mode later:"
    echo "        1. Edit $CONF_FILE → set single_user = false"
    echo "        2. sudo ragger add-user <username>  (for each user)"
    echo "        3. Restart the daemon"
else
    echo -e "${GREEN}[+]${NC} Installed in multi-user mode."
    echo "    Users must log out and back in for group membership to take effect."
fi
