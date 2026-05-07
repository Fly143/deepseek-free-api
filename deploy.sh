#!/bin/bash
# DeepSeek Free API One-click Deployment Script
# Supports Termux (Android) / Linux (Ubuntu/Debian/CentOS)
# Usage: bash deploy.sh [--bg] [--stop] [--status]

set -euo pipefail

INSTALL_DIR="${HOME}/ds-free-api"
PORT="${PROXY_PORT:-8000}"
LOG_FILE="${HOME}/dsapi.log"
VENV_DIR=".venv"

# ── Colors ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; exit 1; }

# ── Help ──
show_help() {
    echo "Usage: bash deploy.sh [options]"
    echo ""
    echo "Options:"
    echo "  (no args)    Start in foreground (Ctrl+C to stop)"
    echo "  --bg         Start in background"
    echo "  --stop       Stop background process"
    echo "  --status     Check running status"
    echo "  --help       Show this help"
    echo ""
    echo "Environment variables:"
    echo "  PROXY_PORT  Port number (default 8000)"
}

# ── Stop ──
do_stop() {
    if pgrep -f "python.*proxy.py" >/dev/null 2>&1; then
        pkill -f "python.*proxy.py" 2>/dev/null || true
        sleep 1
        info "Stopped"
    else
        warn "No running proxy process found"
    fi
}

# ── Status ──
do_status() {
    if pgrep -f "python.*proxy.py" >/dev/null 2>&1; then
        local pid=$(pgrep -f "python.*proxy.py" | head -1)
        info "Running (PID: $pid)"
        if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
            info "Health check passed"
        else
            warn "Health check failed"
        fi
    else
        warn "Not running"
    fi
}

# ── Parse arguments ──
ACTION="start"
for arg in "$@"; do
    case "$arg" in
        --bg)     ACTION="bg" ;;
        --stop)   do_stop; exit 0 ;;
        --status) do_status; exit 0 ;;
        --help|-h) show_help; exit 0 ;;
        *) warn "Unknown option: $arg" ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DeepSeek Free API Proxy Deploy     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"

# ── 1. Check uv ──
if ! command -v uv &>/dev/null; then
    error "uv is required. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
info "uv $(uv --version 2>&1 | head -1)"

# ── 2. Check Node.js (required for PoW solver) ──
if ! command -v node &>/dev/null; then
    warn "Node.js not found, attempting to install..."
    if command -v pkg &>/dev/null; then
        # Termux
        pkg install -y nodejs 2>/dev/null || error "Node.js installation failed, run: pkg install nodejs"
    elif command -v apt &>/dev/null; then
        sudo apt update -qq && sudo apt install -y nodejs 2>/dev/null || error "Node.js installation failed"
    elif command -v yum &>/dev/null; then
        sudo yum install -y nodejs 2>/dev/null || error "Node.js installation failed"
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y nodejs 2>/dev/null || error "Node.js installation failed"
    else
        error "Please install Node.js manually: https://nodejs.org/"
    fi
fi
info "Node.js $(node --version 2>&1)"

# ── 3. Check curl (needed for health check) ──
if ! command -v curl &>/dev/null; then
    warn "curl not found, health check will be unavailable"
fi

# ── 4. Determine working directory ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# If script is inside INSTALL_DIR or has proxy.py alongside, use directly
if [ "$SCRIPT_DIR" = "$INSTALL_DIR" ] || [ -f "$SCRIPT_DIR/proxy.py" ]; then
    WORK_DIR="$SCRIPT_DIR"
else
    # Otherwise install to ~/ds-free-api
    mkdir -p "$INSTALL_DIR"
    WORK_DIR="$INSTALL_DIR"

    # Extract tarball if target directory lacks proxy.py
    if [ ! -f "$INSTALL_DIR/proxy.py" ]; then
        TARBALL=""
        for f in "$SCRIPT_DIR/ds-free-api.tar.gz" "./ds-free-api.tar.gz" "../ds-free-api.tar.gz"; do
            [ -f "$f" ] && TARBALL="$f" && break
        done
        if [ -n "$TARBALL" ]; then
            info "Extracting from $TARBALL..."
            tar -xzf "$TARBALL" -C "$INSTALL_DIR"
        else
            error "Deployment package not found. Place ds-free-api.tar.gz next to this script."
        fi
    fi
fi

cd "$WORK_DIR"
info "Working directory: $WORK_DIR"

# ── 5. Create uv virtual environment and install dependencies ──
info "Creating uv virtual environment..."
uv venv -p python3 "$VENV_DIR" 2>/dev/null || uv venv "$VENV_DIR" 2>/dev/null || error "Failed to create uv virtual environment"

PYTHON_BIN="$WORK_DIR/$VENV_DIR/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$WORK_DIR/$VENV_DIR/bin/python"
fi
info "Virtual environment created at $WORK_DIR/$VENV_DIR"

info "Installing dependencies..."
uv pip install -r requirements.txt --quiet 2>/dev/null || uv pip install -r requirements.txt || error "Dependency installation failed"
info "Dependencies installed"

# ── 6. Check required files ──
for f in proxy.py tool_call.py pow_native.py pow_solver.js; do
    [ -f "$f" ] || error "Missing $f, please check the deployment package"
done
info "File check passed"

# ── 7. Stop old processes ──
if pgrep -f "python.*proxy.py" >/dev/null 2>&1; then
    warn "Stopping old proxy process..."
    pkill -f "python.*proxy.py" 2>/dev/null || true
    sleep 1
fi

# ── 8. Start ──
export PROXY_PORT="$PORT"

if [ "$ACTION" = "bg" ]; then
    nohup "$PYTHON_BIN" proxy.py > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    sleep 2

    if command -v curl &>/dev/null && curl -s "http://localhost:$PORT/health" | grep -q "ok" 2>/dev/null; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✅ Deployment successful!           ║${NC}"
        echo -e "${GREEN}╠══════════════════════════════════════╣${NC}"
        echo -e "${GREEN}║  Admin UI:  http://localhost:$PORT/admin${NC}"
        echo -e "${GREEN}║  API URL:   http://localhost:$PORT/v1   ${NC}"
        echo -e "${GREEN}║  Process PID: $BG_PID${NC}"
        echo -e "${GREEN}║  Log file:  $LOG_FILE${NC}"
        echo -e "${GREEN}╠══════════════════════════════════════╣${NC}"
        echo -e "${GREEN}║  Stop:    bash deploy.sh --stop      ${NC}"
        echo -e "${GREEN}║  Status:  bash deploy.sh --status    ${NC}"
        echo -e "${GREEN}║  Logs:    tail -f $LOG_FILE${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
    else
        warn "Starting... If unreachable, check the log: $LOG_FILE"
    fi
else
    echo ""
    info "Ready!"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Open your browser and visit: ${BLUE}http://localhost:$PORT/admin${NC}"
    echo -e "  2. Log in with your phone number or email"
    echo -e "  3. Configure your client with API URL: ${BLUE}http://localhost:$PORT/v1${NC}"
    echo ""
    echo -e "Run in background: ${YELLOW}bash deploy.sh --bg${NC}"
    echo ""
    "$PYTHON_BIN" proxy.py
fi
