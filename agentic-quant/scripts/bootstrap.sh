#!/usr/bin/env bash
# ==============================================================================
# scripts/bootstrap.sh
# Full bootstrap script cho AGENTIC-QUANT project
# Chay: bash scripts/bootstrap.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# --- Màu sac cho terminal ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

fail() { log_error "$1"; exit 1; }

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  AGENTIC-QUANT — Bootstrap Script${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# -----------------------------------------------------------
# 1. Kiem tra Python >= 3.11
# -----------------------------------------------------------
log_info "1. Checking Python version..."

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python not found. Please install Python 3.11+."
fi

PY_VER=$($PYTHON --version 2>&1 | cut -d' ' -f2)
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    fail "Python 3.11+ required, found $PY_VER"
fi

log_ok "Python $PY_VER found at $(command -v "$PYTHON")"

# -----------------------------------------------------------
# 2. Kiem tra Poetry
# -----------------------------------------------------------
log_info "2. Checking Poetry..."

if ! command -v poetry &>/dev/null; then
    log_warn "Poetry not found. Installing via pip..."
    $PYTHON -m pip install --quiet poetry
fi

POETRY_VER=$(poetry --version 2>&1)
log_ok "Poetry: $POETRY_VER"

# -----------------------------------------------------------
# 3. Poetry install (production only)
# -----------------------------------------------------------
log_info "3. Installing Python dependencies (--no-dev)..."

poetry install --no-dev --no-interaction 2>&1 | while IFS= read -r line; do
    echo "       $line"
done

log_ok "Python dependencies installed"

# -----------------------------------------------------------
# 4. UI dependencies (npm ci)
# -----------------------------------------------------------
log_info "4. Installing UI dependencies (npm ci)..."

if [ ! -d "ui/node_modules" ]; then
    (cd ui && npm ci --no-audit --no-fund 2>&1) | while IFS= read -r line; do
        echo "       $line"
    done
    log_ok "UI dependencies installed"
else
    log_ok "UI dependencies already installed (ui/node_modules exists)"
fi

# -----------------------------------------------------------
# 5. Database migration
# -----------------------------------------------------------
log_info "5. Running database migrations..."

if $PYTHON -m core.scripts.migrate 2>&1; then
    log_ok "Migrations completed"
else
    log_warn "Migrations failed — check database configuration"
fi

# -----------------------------------------------------------
# 6. Redis check (optional — only warning)
# -----------------------------------------------------------
log_info "6. Checking Redis connectivity..."

if command -v redis-cli &>/dev/null; then
    if redis-cli ping 2>&1 | grep -q "PONG"; then
        log_ok "Redis is reachable"
    else
        log_warn "Redis is not running — system will use in-memory fallback"
    fi
else
    log_warn "redis-cli not found — Redis connectivity cannot be verified"
fi

# -----------------------------------------------------------
# 7. Download model weights
# -----------------------------------------------------------
log_info "7. Downloading model weights..."

if [ -f "scripts/download_weights.py" ]; then
    $PYTHON scripts/download_weights.py 2>&1 | while IFS= read -r line; do
        echo "       $line"
    done
    log_ok "Model weights downloaded/verified"
else
    log_warn "scripts/download_weights.py not found — skipping"
fi

# -----------------------------------------------------------
# 8. Create .env from .env.example if not exists
# -----------------------------------------------------------
log_info "8. Setting up .env file..."

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        log_ok ".env created from .env.example"
        log_warn "Please edit .env with your actual API keys and configuration"
    else
        log_warn ".env.example not found, creating minimal .env"
        cat > .env <<-EOF
# AGENTIC-QUANT — Environment
ANTHROPIC_API_KEY=
REDIS_URL=redis://127.0.0.1:6379/0
VECTOR_DB_URL=http://127.0.0.1:6333
DEFAULT_SYMBOL=XAUUSD
LOG_LEVEL=INFO
AGENTIQ_BACKEND=1
EOF
        log_ok "Minimal .env created"
    fi
else
    log_ok ".env already exists"
fi

# -----------------------------------------------------------
# Summary
# -----------------------------------------------------------
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Bootstrap completed successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Next steps:"
echo "    - Edit .env with your API keys"
echo "    - Start Redis: redis-server"
echo "    - Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
echo "    - Run app:    $PYTHON core/main.py"
echo "    - Run tests:  poetry run pytest"
echo ""
