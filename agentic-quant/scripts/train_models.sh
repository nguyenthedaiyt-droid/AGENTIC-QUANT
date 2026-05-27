#!/usr/bin/env bash
# ==============================================================================
# scripts/train_models.sh
# Training pipeline — train LSTM, build dataset, train models, evaluate
# Usage: bash scripts/train_models.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# --- Màu sac ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

fail() { log_error "$1"; exit 1; }

# -----------------------------------------------------------
# Kiem tra Python
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Tim Poetry environment
# -----------------------------------------------------------
log_info "Detecting Poetry virtual environment..."

POETRY_ENV=$($PYTHON -m poetry env info --path 2>/dev/null || true)
if [ -n "$POETRY_ENV" ]; then
    log_ok "Poetry venv found at: $POETRY_ENV"
    source "$POETRY_ENV/bin/activate" 2>/dev/null || true
else
    log_warn "Poetry venv not detected. Using system Python."
fi

# -----------------------------------------------------------
# Pipeline
# -----------------------------------------------------------
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  AGENTIC-QUANT — Training Pipeline${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

PIPELINE_PASS=true

# ----- Step 1: Train LSTM -----
echo -e "${CYAN}[1/5]${NC} Training LSTM model..."
if $PYTHON -m core.ml.train_lstm 2>&1; then
    log_ok "LSTM training completed"
else
    log_error "LSTM training failed"
    PIPELINE_PASS=false
fi
echo ""

# ----- Step 2: Build XGBoost dataset -----
echo -e "${CYAN}[2/5]${NC} Building XGBoost dataset..."
if $PYTHON -m core.ml.build_xgb_dataset 2>&1; then
    log_ok "XGBoost dataset built"
else
    log_error "XGBoost dataset build failed"
    PIPELINE_PASS=false
fi
echo ""

# ----- Step 3: Train Model A -----
echo -e "${CYAN}[3/5]${NC} Training Model A..."
if $PYTHON -m core.ml.train_model_a 2>&1; then
    log_ok "Model A training completed"
else
    log_error "Model A training failed"
    PIPELINE_PASS=false
fi
echo ""

# ----- Step 4: Train Model B -----
echo -e "${CYAN}[4/5]${NC} Training Model B..."
if $PYTHON -m core.ml.train_model_b 2>&1; then
    log_ok "Model B training completed"
else
    log_error "Model B training failed"
    PIPELINE_PASS=false
fi
echo ""

# ----- Step 5: Evaluate and optionally deploy -----
echo -e "${CYAN}[5/5]${NC} Evaluating models..."
EVAL_CMD="$PYTHON -m core.ml.evaluate_models"

if [ "$PIPELINE_PASS" = true ]; then
    EVAL_CMD="$EVAL_CMD --deploy-if-pass"
fi

if $EVAL_CMD 2>&1; then
    log_ok "Model evaluation passed"
    if [ "$PIPELINE_PASS" = true ]; then
        log_ok "All models passed — deploying to production"
    fi
else
    log_error "Model evaluation failed"
    PIPELINE_PASS=false
fi
echo ""

# -----------------------------------------------------------
# Summary
# -----------------------------------------------------------
echo -e "${CYAN}============================================${NC}"
if [ "$PIPELINE_PASS" = true ]; then
    echo -e "${GREEN}  Pipeline completed SUCCESSFULLY${NC}"
    echo -e "${GREEN}  Models deployed and ready${NC}"
else
    echo -e "${RED}  Pipeline completed with ERRORS${NC}"
    echo -e "${RED}  Check logs above for details${NC}"
fi
echo -e "${CYAN}============================================${NC}"
echo "  Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ "$PIPELINE_PASS" = false ]; then
    exit 1
fi
