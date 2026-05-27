#!/usr/bin/env bash
# ==============================================================================
# scripts/run_backtest.sh
# Backtest runner — chay backtest voi symbol, start date, end date
# Usage: bash scripts/run_backtest.sh XAUUSD 2024-01-01 2024-12-31
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
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

fail() { log_error "$1"; exit 1; }

# -----------------------------------------------------------
# Kiem tra tham so
# -----------------------------------------------------------
if [ $# -lt 3 ]; then
    echo "Usage: bash $0 <SYMBOL> <START_DATE> <END_DATE>"
    echo ""
    echo "Examples:"
    echo "  bash $0 XAUUSD 2024-01-01 2024-12-31"
    echo "  bash $0 EURUSD 2023-06-01 2024-06-01"
    echo "  bash $0 BTCUSD 2024-01-01 2024-03-31"
    exit 1
fi

SYMBOL="$1"
START_DATE="$2"
END_DATE="$3"

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
# Tao output directory neu chua co
# -----------------------------------------------------------
OUTPUT_DIR="data/backtest_results"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}.json"

# -----------------------------------------------------------
# Hien thong tin
# -----------------------------------------------------------
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  AGENTIC-QUANT — Backtest Runner${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
echo "  Symbol:      $SYMBOL"
echo "  Start Date:  $START_DATE"
echo "  End Date:    $END_DATE"
echo "  Output:      $OUTPUT_FILE"
echo ""

# -----------------------------------------------------------
# Chay backtest
# -----------------------------------------------------------
log_info "Starting backtest..."

$PYTHON -m core.backtesting.run \
    --symbol "$SYMBOL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --output "$OUTPUT_FILE"

BACKTEST_EXIT=$?

echo ""

if [ $BACKTEST_EXIT -eq 0 ]; then
    log_ok "Backtest completed successfully"
    echo "  Results saved to: $OUTPUT_FILE"
else
    log_error "Backtest failed with exit code $BACKTEST_EXIT"
    exit $BACKTEST_EXIT
fi
