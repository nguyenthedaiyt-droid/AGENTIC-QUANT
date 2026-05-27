# Deployment Guide — AGENTIC-QUANT

> Hướng dẫn cài đặt và triển khai AGENTIC-QUANT trên Windows và macOS.

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt trên Windows](#2-cài-đặt-trên-windows)
3. [Cài đặt trên macOS](#3-cài-đặt-trên-macos)
4. [Cấu hình MT5 EA](#4-cấu-hình-mt5-ea)
5. [Cấu hình TradingView Webhook](#5-cấu-hình-tradingview-webhook)
6. [Kiểm tra hệ thống](#6-kiểm-tra-hệ-thống)

---

## 1. Yêu cầu hệ thống

| Thành phần       | Phiên bản tối thiểu | Ghi chú                                       |
|------------------|---------------------|-----------------------------------------------|
| Python           | 3.11                | Bắt buộc (pyproject.toml: `>=3.11`)            |
| Poetry           | 1.8+                | Quản lý dependency Python                      |
| Node.js          | 18 LTS              | Cho UI frontend (Tauri/Vite)                   |
| npm              | 10+                 | Đi kèm Node.js                                 |
| Rust             | 1.75+               | Cho Tauri desktop app                          |
| Redis            | 7.x                 | Cache và state (có fallback in-memory)         |
| Qdrant           | 1.12+               | Vector database (có fallback SQLite)           |
| MetaTrader 5     | Build 4000+         | Cho MT5 EA connector (Windows)                 |

### Dung lượng đĩa

- Code + Dependencies: ~2 GB
- Model weights: ~500 MB – 2 GB (tuỳ model)
- Backtest data: ~100 MB – 10 GB (tuỳ khung thời gian)

---

## 2. Cài đặt trên Windows

### 2.1. Cài đặt prerequisites

```powershell
# 1. Python 3.11+ — Download từ python.org, nhớ check "Add to PATH"
# 2. Node.js LTS — Download từ nodejs.org
# 3. Git — Download từ git-scm.com
# 4. Rust — https://rustup.rs (dùng rustup-init.exe)
# 5. Redis — Download từ https://github.com/microsoftarchive/redis/releases
#    hoặc dùng WSL: wsl --install -d Ubuntu && sudo apt install redis
# 6. Qdrant — Docker Desktop hoặc:
#    docker run -d -p 6333:6333 qdrant/qdrant
```

### 2.2. Clone và bootstrap

```powershell
# Clone repo
git clone https://github.com/nguyenthedaiyt-droid/AGENTIC-QUANT.git
cd AGENTIC-QUANT

# Bootstrap (tự động check Python, cài dependencies, migrate DB)
.\scripts\bootstrap.ps1
# hoặc dùng bash trong WSL:
bash scripts/bootstrap.sh
```

### 2.3. Cấu hình .env

```powershell
# Edit file .env
notepad .env
```

Các biến bắt buộc:

| Biến               | Mô tả                       | Ví dụ                          |
|--------------------|-----------------------------|--------------------------------|
| `ANTHROPIC_API_KEY`| API key cho Claude (debate) | `sk-ant-xxxxxxxxxxxx`          |
| `REDIS_URL`        | Redis connection string     | `redis://127.0.0.1:6379/0`     |
| `VECTOR_DB_URL`    | Qdrant endpoint             | `http://127.0.0.1:6333`        |
| `DEFAULT_SYMBOL`   | Symbol mặc định             | `XAUUSD`                       |

### 2.4. Build Tauri desktop app

```powershell
cd desktop
npm install        # Cài Tauri CLI (nếu chưa có)
cargo tauri build  # Build .msi installer
# Output: desktop/src-tauri/target/release/bundle/msi/AGENTIC-QUANT_0.1.0_x64.msi
```

### 2.5. Chạy hệ thống

```powershell
# Terminal 1: Redis
redis-server

# Terminal 2: Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# Terminal 3: Backend
poetry run python core/main.py

# Terminal 4: Desktop app (hoặc dùng file .msi da build)
cd desktop
cargo tauri dev
```

---

## 3. Cài đặt trên macOS

### 3.1. Cài đặt prerequisites

```bash
# 1. Homebrew (nếu chưa có)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Python 3.11+
brew install python@3.11

# 3. Poetry
brew install poetry

# 4. Node.js
brew install node

# 5. Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 6. Redis
brew install redis
brew services start redis

# 7. Qdrant
brew install qdrant/tap/qdrant
brew services start qdrant
```

### 3.2. Clone và bootstrap

```bash
git clone https://github.com/nguyenthedaiyt-droid/AGENTIC-QUANT.git
cd AGENTIC-QUANT
bash scripts/bootstrap.sh
```

### 3.3. Build Tauri desktop app

```bash
cd desktop
npm install
cargo tauri build
# Output: desktop/src-tauri/target/release/bundle/dmg/AGENTIC-QUANT_0.1.0_x64.dmg
```

### 3.4. Chạy hệ thống

```bash
# Terminal 1: Redis (daemon)
brew services start redis

# Terminal 2: Qdrant (daemon)
brew services start qdrant

# Terminal 3: Backend
poetry run python core/main.py

# Terminal 4: Desktop
cd desktop && cargo tauri dev
```

---

## 4. Cấu hình MT5 EA

### 4.1. Yêu cầu

- MetaTrader 5 (Build 4000+)
- Tài khoản demo/live đã đăng nhập
- Automated trading được bật (Tools > Options > Expert Advisors > Allow Automated Trading)

### 4.2. Cài đặt EA

1. Copy file EA (`AGENTIC_QUANT_EA.ex5` hoặc `.mq5`) vào:
   ```
   %APPDATA%\MetaQuotes\Terminal\{INSTANCE_ID}\MQL5\Experts\
   ```
   (Windows) hoặc:
   ```
   ~/Library/Application Support/MetaQuotes/Terminal/{INSTANCE_ID}/MQL5/Experts/
   ```
   (macOS — wine)

2. Restart MT5 hoặc click chuột phải vào Navigator > Refresh

3. Kéo EA vào chart (khuyên dùng XAUUSD, H1 timeframe)

### 4.3. Cấu hình tham số EA

| Tham số            | Giá trị mặc định | Mô tả                               |
|---------------------|------------------|--------------------------------------|
| `ServerHost`        | `127.0.0.1`     | Địa chỉ IPC/ZMQ server               |
| `ServerPort`        | `5555`          | Port ZMQ                             |
| `Symbol`            | `XAUUSD`        | Symbol giao dịch                     |
| `LotSize`           | `0.01`          | Khối lượng giao dịch                 |
| `MaxSpread`         | `30`            | Spread tối da (points)               |
| `MagicNumber`       | `20240527`      | Magic number định danh EA            |
| `RiskPercent`       | `1.0`           | % rủi ro trên mỗi lệnh              |
| `MaxPositions`      | `3`             | Số lệnh tối da đồng thời            |

### 4.4. Kiểm tra kết nối

```bash
# Test kết nối từ Python den MT5
python test_mt5_connect.py

# Output mong đợi:
#   ✅ MT5 terminal found
#   ✅ Login successful
#   ✅ Symbol XAUUSD available
#   Current bid: 2350.42  ask: 2350.75
```

---

## 5. Cấu hình TradingView Webhook

### 5.1. Tạo alert trên TradingView

1. Mở chart bất kỳ (khuyên dùng XAUUSD, 1H)
2. Click chuột phải > **Add Alert**
3. Cấu hình:

| Trường           | Giá trị                                        |
|------------------|-------------------------------------------------|
| Condition        | Indicator/strategy của bạn                      |
| Expiration       | `1 day` hoặc `Until cancelled`                  |
| Alert name       | `AGENTIC-QUANT alert`                           |
| Webhook URL      | `http://<SERVER_IP>:9999/api/v1/webhook/tv`    |
| Webhook message  | (xem JSON bên dưới)                             |

### 5.2. Webhook payload mẫu

```json
{
    "symbol": "XAUUSD",
    "action": "buy",
    "confidence": 0.82,
    "timeframe": "1H",
    "reason": "MACD crossover + RSI oversold bounce",
    "price": 2350.42,
    "timestamp": "2024-05-27T14:30:00Z"
}
```

### 5.3. Xác thực webhook (tuỳ chọn)

Nếu cấu hình `TV_WEBHOOK_SECRET` trong `.env`:

```json
{
    "secret": "your_webhook_secret_key",
    "symbol": "XAUUSD",
    "action": "sell",
    "confidence": 0.75,
    "timeframe": "1H",
    "price": 2365.10,
    "timestamp": "2024-05-27T15:00:00Z"
}
```

### 5.4. Kiểm tra webhook

```bash
# Test webhook bằng curl
curl -X POST http://127.0.0.1:9999/api/v1/webhook/tv \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "XAUUSD",
    "action": "buy",
    "confidence": 0.82,
    "timeframe": "1H",
    "price": 2350.42,
    "timestamp": "2024-05-27T14:30:00Z"
  }'

# Output mong đợi:
#   {"status":"ok","signal_id":"sig_abc123"}
```

---

## 6. Kiểm tra hệ thống

### 6.1. Health check endpoints

```bash
# Backend health
curl http://127.0.0.1:9999/api/v1/health

# Redis
redis-cli ping

# Qdrant
curl http://127.0.0.1:6333/healthz

# MT5 connection
python test_mt5_connect.py
```

### 6.2. Quick sanity test

```bash
# Chạy test suite
poetry run pytest tests/ -v

# Chạy backtest nhanh
bash scripts/run_backtest.sh XAUUSD 2024-01-01 2024-01-07

# Chạy training pipeline (cần GPU khuyen dùng)
bash scripts/train_models.sh
```

### 6.3. Troubleshooting thường gặp

| Vấn đề                        | Nguyên nhân                       | Giải pháp                               |
|-------------------------------|-----------------------------------|-----------------------------------------|
| `ModuleNotFoundError`          | Poetry env chưa active            | `poetry shell` hoặc `poetry run`         |
| `redis.exceptions.ConnectionError` | Redis chưa chạy               | `redis-server` hoặc `brew services start redis` |
| `qt.qpa.plugin: Could not load` | Qt missing (Tauri)               | `pip install PyQt6` hoặc cài Qt SDK     |
| MT5 không connect              | MT5 chưa bật automated trading   | Tools > Options > EA > Allow Automated  |
| Webhook 404                    | Server chưa chạy                  | `poetry run python core/main.py`        |
| `cargo: command not found`     | Rust chưa cài                     | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` |

---

> **Liên hệ**: Mở issue trên GitHub hoặc liên he team dev để được hỗ trợ.
> **License**: MIT — xem file `LICENSE` để biết thêm chi tiết.
