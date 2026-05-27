# =============================================================================
# AGENTIC-QUANT — Hybrid Agentic Quantitative Trading Framework
# =============================================================================
# Author: Trading Team
# Version: 0.1.0
# License: Proprietary
# =============================================================================

## Mục lụ

- [Tong quan](#tong-quan)
- [Cai dat](#cai-dat)
- [Cau truc du an](#cau-truc-du-an)
- [Phat trien](#phat-trien)
- [Luu y quan trong](#luu-y-quan-trong)

---

## Tong quan

**AGENTIC-QUANT** la mot Hybrid Agentic Quantitative Trading Framework, hoat dong nhu mot AI Co-pilot cho trader con nguoi, phan tich thi truong theo trường phái ICT/SMC (Inner Circle Trader / Smart Money Concepts) ket hop Machine Learning.

### Bon truc cot du bao chinh

1. **MTF Alignment** — Phan tich cau truc thi truong tu D1/H4/H1 den M1/M5/M15
2. **Macro News Integration** — Lich kinh te (CPI, FOMC, NFP) va co che phong ve khi ra tin
3. **Liquidity Prediction** — Du doan vuong hut thanh khoan (Buyside vs. Sellside): $P_{BSL}$ vs $P_{SSL}$
4. **Zone Hold Prediction** — Xac suat vuong FVG/OB giu duoc gia: $P_{hold}$

### Kiến truc he thong

```
+------------------+
|  Tauri Desktop   |  (Rust shell, system tray, window management)
+--------+---------+
         |
+--------+---------+
|   React UI      |  (TradingView Lightweight Charts, Ghost Zones, Heatmap)
+--------+---------+
         | WebSocket (ws://localhost:47290)
+--------+---------+
| Python Backend  |  asyncio-based, multi-module
+--------+---------+
  |       |       |
Module1  Module3  Module7
(INGEST) (AI)     (IPC)
  |       |       |
  +---+---+---+
      |
+-----+-----+-----+
Redis  SQLite  Qdrant
(Cache)(Persist)(Vector)
```

---

## Cai dat

### Yeu cau he thong

- Python 3.11+
- Node.js 20+
- Rust 1.75+ (neu build desktop)
- Git

### Buoc 1: Clone repository

```bash
git clone <repository-url>
cd agentic-quant
```

### Buoc 2: Python Backend

```bash
# Tao virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# Cai dat dependencies
make install
# Hoac thu cong:
poetry install
```

### Buoc 3: Frontend

```bash
cd ui
npm install
```

### Buoc 4: Desktop (Tauri) — tuy chon

```bash
# Cai dat Rust (neu chua co)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Khoi tao Tauri
cd desktop
cargo build
```

### Buoc 5: External services

```bash
# Redis (macOS)
brew install redis
redis-server

# Redis (Windows - WSL2)
wsl -d Ubuntu
sudo apt install redis-server
redis-server

# Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/vectordb:/qdrant/storage \
    qdrant/qdrant
```

### Buoc 6: Khoi dong he thong

```bash
# Backend (terminal 1)
make run

# Frontend dev server (terminal 2)
cd ui && npm run dev
```

---

## Cau truc du an

```
agentic-quant/
├── core/                          # Python backend
│   ├── config/                    # Cau hinh he thong
│   ├── ingestion/                 # Module 1: Tick receiver, OHLCV aggregator, Volumetrics
│   ├── macro/                     # Module 2: Calendar scraper, News vectorizer
│   ├── ai_engine/                 # Module 3: Feature engineering, LSTM, XGBoost, Debate
│   │   ├── feature_engineering/   # SMC/ICT symbolic features
│   │   ├── neural/                # LSTM Autoencoder, XGBoost inference
│   │   └── multi_agent/           # Bull/Bear/Critic agents
│   ├── memory/                    # Module 4: Redis + SQLite + VectorDB
│   │   ├── short_term/            # Redis cache managers
│   │   └── long_term/             # SQLite + VectorDB adapters
│   ├── backtesting/               # Module 5: Event-driven backtest
│   ├── ipc/                      # Module 7: WebSocket server, IPC
│   └── utils/                    # Common utilities
├── ui/                           # React frontend
│   ├── src/
│   │   ├── components/            # UI components
│   │   ├── hooks/                 # Custom React hooks
│   │   ├── store/                 # Redux slices + Zustand
│   │   ├── types/                 # TypeScript interfaces
│   │   └── workers/              # Web Workers
│   └── public/
├── desktop/                       # Tauri desktop shell
│   └── src-tauri/
│       └── src/
├── models/                        # ML model artifacts
│   ├── model_a/                  # XGBoost Model A weights
│   ├── model_b/                  # XGBoost Model B weights
│   └── lstm/                    # LSTM Autoencoder weights
├── data/                         # Data storage
│   ├── historical_ticks/         # Parquet tick data
│   ├── archive/                  # Archived predictions
│   └── agentic_quant.db          # SQLite database
├── vectordb/                    # Vector database storage (Qdrant/ChromaDB)
├── config/                      # YAML configuration files
│   ├── system.yaml              # He thong, ports, thresholds
│   ├── model_params.yaml        # ML hyperparameters
│   ├── killzones.yaml          # 6 trading sessions
│   └── news_weights.yaml       # News impact coefficients
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests (pytest)
│   ├── integration/           # Integration tests
│   └── backtest_scenarios/   # Backtest scenarios
├── scripts/                   # Build and utility scripts
│   ├── build_lstm_dataset.py  # LSTM training data builder
│   ├── build_xgb_dataset.py  # XGBoost training data builder
│   ├── train_lstm.py         # LSTM training pipeline
│   ├── train_model_a.py      # XGBoost Model A training
│   ├── train_model_b.py      # XGBoost Model B training
│   ├── export_ticks_mt5.py   # Export ticks from MT5
│   └── migrations/           # SQLite migrations
├── .venv/                    # Python virtual environment
├── pyproject.toml            # Poetry dependency management
├── requirements.txt          # Locked requirements
├── pytest.ini               # Pytest configuration
├── Makefile                # Development tasks
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .editorconfig           # Editor configuration
└── README.md              # This file
```

---

## Phat trien

### Lenh thuong dung

```bash
# Python
make install          # Cai dat moi truong
make test            # Chay tests
make lint            # Chay linter
make format          # Format code
make run             # Khoi dong backend
make clean           # Don dep build artifacts

# Frontend
cd ui
npm run dev          # Dev server
npm run build        # Production build
npm run test         # Unit tests

# Desktop
cd desktop
cargo build         # Build Tauri app
cargo run           # Run Tauri app
```

### Branch strategy

```
main              ← production-only, protected
develop          ← integration branch
feature/*        ← new features
fix/*            ← bug fixes
chore/*          ← maintenance tasks
```

### Quy tac commit

```
feat:     New feature
fix:      Bug fix
refactor: Code refactoring
docs:     Documentation
test:     Adding tests
chore:    Maintenance
```

---

## Lưu ý quan trọng

### Môi truong

Tất ca cac API keys (Anthropic, etc.) phải duoc cai dat qua bien moi truong, **không hardcode** trong code. Xem `.env.example` de biet danh sach.

### Database

- **Redis:** chi bind localhost (`127.0.0.1`), khong expose ra network
- **SQLite:** dung che do WAL, auto_vacuum
- **Vector DB:** chi su dung trong local network

### Logging

- **Dev:** DEBUG level, console output
- **Staging:** INFO level, file output
- **Production:** WARNING level, structured JSON

### Hieu nang

| Hanh dong | Muc tieu |
|-----------|----------|
| Tick → USV | < 5ms |
| USV → LSTM | < 20ms |
| LSTM → XGBoost | < 10ms |
| Total Python | < 40ms |
| **E2E (Python → UI)** | **< 50ms** |

---

## Lien he

De biet them chi tiet, xem cac tai lieu trong thu muc `document/`:
- `prompt.md` — Yeu cau dat ta ky thuat
- `Overview_document.md` — Tom tat kien truc
- `Extensive_documentation.md` — Tai lieu ky thuat chi tiet
- `todo.md` — Ke hoach phat trien
