# AGENTIC-QUANT — KẾ HOẠCH PHÁT TRIỂN TOÀN DỰ ÁN
## Development TODO Plan: From Zero to Production

---

> **Quy ước đọc tài liệu:**
> - `[ ]` = Chưa bắt đầu | `[~]` = Đang làm | `[x]` = Hoàn thành
> - **🔴 Critical Path** = Blocking các task phía sau
> - **🟡 Parallel** = Có thể làm song song với task khác
> - **🟢 Standalone** = Độc lập, làm bất cứ lúc nào
> - **(P0/P1/P2)** = Độ ưu tiên: P0 = phải có, P1 = nên có, P2 = nice-to-have

---

## PHASE 0 — PROJECT BOOTSTRAP & INFRASTRUCTURE
### Mục tiêu: Môi trường phát triển sạch, repo chuẩn, CI sẵn sàng
### Ước tính: 2–3 ngày

---

### 0.1 Khởi tạo Repository và Cấu trúc Dự án 🔴

- [x] **0.1.1** Tạo Git repository với branch strategy:
  - `main` (production-only, protected)
  - `develop` (integration branch)
  - `feature/*`, `fix/*`, `chore/*` (working branches)
- [x] **0.1.2** Tạo cấu trúc thư mục đầy đủ theo spec V1.0:
  ```
  agentic-quant/
  ├── core/ (ingestion, macro, ai_engine, memory, backtesting, ipc)
  ├── ui/ (React + TradingView)
  ├── desktop/ (Tauri)
  ├── models/ (ML artifacts)
  ├── data/ (historical data, sqlite)
  ├── vectordb/
  ├── config/
  ├── tests/ (unit, integration, backtest_scenarios)
  └── scripts/
  ```
- [x] **0.1.3** Tạo `.gitignore` chuẩn cho Python + Node + Rust
- [ ] **0.1.4** Tạo `.editorconfig` thống nhất coding style
- [x] **0.1.5** Viết `README.md` với hướng dẫn setup cơ bản

### 0.2 Môi Trường Python Backend 🔴

- [x] **0.2.1** Tạo `pyproject.toml` với Poetry (dependency management):
  - Runtime deps: `numpy`, `pandas`, `pyzmq`, `websockets`, `aiohttp`
  - ML deps: `xgboost`, `torch` (cho LSTM), `scikit-learn`
  - DB deps: `redis`, `sqlite3` (stdlib), `chromadb` hoặc `qdrant-client`
  - Util deps: `pyyaml`, `msgpack`, `loguru`, `prometheus-client`
- [ ] **0.2.2** Tạo `requirements.txt` lock file từ pyproject
- [x] **0.2.3** Cấu hình `pytest` với `pytest.ini`:
  - Test discovery pattern
  - Coverage threshold: 70% minimum
- [ ] **0.2.4** Cài đặt pre-commit hooks:
  - `black` (formatter)
  - `ruff` (linter, thay thế flake8)
  - `mypy` (type checking, strict mode)
  - `pytest` (chạy unit tests trước mỗi commit)
- [x] **0.2.5** Tạo `Makefile` với các target phổ biến:
  - `make install`, `make test`, `make lint`, `make run`

### 0.3 Môi Trường Frontend (React + TypeScript) 🟡

- [ ] **0.3.1** Khởi tạo Vite project trong `ui/`:
  - Template: React + TypeScript + SWC
- [ ] **0.3.2** Cài đặt dependencies:
  - `lightweight-charts` (TradingView chart library)
  - `@reduxjs/toolkit`, `react-redux`
  - `zustand` (local component state — nhẹ hơn Redux cho một số case)
  - `tailwindcss` (styling)
  - `vitest` + `@testing-library/react` (testing)
- [ ] **0.3.3** Cấu hình ESLint + Prettier cho TypeScript
- [ ] **0.3.4** Cấu hình path aliases trong `tsconfig.json`:
  - `@components`, `@hooks`, `@store`, `@workers`, `@types`
- [ ] **0.3.5** Tạo `src/types/` — định nghĩa toàn bộ TypeScript interfaces khớp với Python message schemas

### 0.4 Môi Trường Desktop (Tauri + Rust) 🟡

- [ ] **0.4.1** Cài đặt Rust toolchain (`rustup`) + Tauri CLI
- [ ] **0.4.2** Khởi tạo Tauri project trong `desktop/`:
  - `tauri init` với frontend pointing đến `ui/`
- [ ] **0.4.3** Cấu hình `tauri.conf.json`:
  - App identifier, window settings, allowlist permissions
  - Bundle: icon, updater endpoint placeholder
- [ ] **0.4.4** Verify build pipeline: `cargo build` thành công

### 0.5 Config Files và Schemas 🟢

- [x] **0.5.1** Viết `config/system.yaml` với tất cả tham số hệ thống:
  - ZeroMQ ports, WebSocket port (47290), Redis connection
  - Tick thresholds, ATR periods, lookback windows
- [x] **0.5.2** Viết `config/model_params.yaml`:
  - XGBoost hyperparameters (Model A & B)
  - LSTM architecture params
  - Ensemble weights
- [x] **0.5.3** Viết `config/killzones.yaml` đầy đủ 6 phiên
- [x] **0.5.4** Viết `config/news_weights.yaml`:
  - Alpha coefficient (α = 0.4), guardrail_dampening_factor
- [x] **0.5.5** Tạo Pydantic models cho config validation:
  `core/config/schemas.py` — validate config khi load

### 0.6 Logging & Observability Infrastructure 🟢

- [x] **0.6.1** Thiết lập `loguru` với structured JSON logging:
  - Log levels: DEBUG (dev), INFO (staging), WARNING (prod)
  - Log rotation: 100MB per file, giữ 7 ngày
  - Separate log files: `system.log`, `model.log`, `ipc.log`
- [x] **0.6.2** Tạo Prometheus metrics endpoint (`/metrics` tại port 9090):
  - Counters: tick_received_total, bar_closed_total, prediction_made_total
  - Histograms: inference_latency_ms, ipc_latency_ms
  - Gauges: redis_memory_bytes, itq_queue_depth
- [x] **0.6.3** Viết `core/utils/timing.py` — decorator `@measure_latency`

---

## PHASE 1 — MODULE 1: DATA INGESTION ENGINE
### Mục tiêu: Nhận tick, tổng hợp OHLCV đa khung, tính volumetrics
### Ước tính: 7–10 ngày
### Dependency: Phase 0 hoàn thành

---

### 1.1 Cơ Sở Hạ Tầng Event Bus 🔴

- [x] **1.1.1** Thiết kế `core/events/bus.py` — Central Event Bus:
  - Pattern: asyncio-based pub/sub
  - Methods: `publish(event_type, payload)`, `subscribe(event_type, handler)`
  - Đảm bảo non-blocking với asyncio queues
- [x] **1.1.2** Định nghĩa `core/events/types.py` — tất cả event types:
  - `BAR_CLOSE`, `TICK_RECEIVED`, `NEWS_ALERT`, `REGIME_CHANGE`
  - `OUTCOME_CONFIRMED`, `MODEL_DEGRADED`, `FEED_FAILURE`
- [x] **1.1.3** Unit tests: publish/subscribe, backpressure handling
- [ ] **1.1.4** Benchmark: đo throughput (target > 10,000 events/sec)

### 1.2 Tick Receiver — ZeroMQ 🔴

- [x] **1.2.1** Viết `core/ingestion/tick_receiver.py`:
  - ZeroMQ PULL socket trên port 5556
  - Deserialize TickFrame từ binary
  - Validate spread: `(ask - bid) > threshold` → ABNORMAL_SPREAD
  - Phân loại buy/sell aggressor (so sánh last với bid/ask)
- [x] **1.2.2** Viết MT5 Connector (`core/ingestion/mt5_connector.py`):
  - Ket noi truc tiep qua official `MetaTrader5` Python API (khong can EA)
  - Push tick qua ZeroMQ PUSH socket (TickReceiver nhan PULL)
  - Thumbnail MT5 `MqlTick` → `TickFrame` binary
  - Gom `MT5TickSimulator` cho backtest/integration test (khong can MT5)
- [ ] **1.2.3** Viết integration test: giả lập MT5 EA → verify nhận tick đúng
- [x] **1.2.4** Xử lý reconnect với exponential backoff (1s, 2s, 4s, ... max 60s)

### 1.3 Tick Receiver — TradingView Webhook 🟡

- [x] **1.3.1** Viết `core/ingestion/tv_webhook.py`:
  - HTTP server (aiohttp) lắng nghe POST `/webhook/tv` tại port 8080
  - HMAC-SHA256 signature validation từ header `X-TV-Signature`
  - Parse TVAlert JSON → chuẩn hóa thành OHLCVRecord
- [ ] **1.3.2** Unit test: valid webhook, invalid signature, malformed JSON
- [x] **1.3.3** Rate limiting: tối đa 100 requests/phút (bảo vệ DoS)

### 1.4 OHLCV Aggregator 🔴

- [x] **1.4.1** Viết `core/ingestion/ohlcv_aggregator.py`:
  - Tạo `ActiveBars` dict: `{timeframe: {bucket_time: BarState}}`
  - Hàm `update_bar(tick)`: cập nhật OHLCV cho tất cả 6 TF
  - Logic phát hiện đóng cửa nến: `current_time >= bucket + tf_seconds`
  - Phát `BAR_CLOSE(tf, ohlcv_record)` event khi nến đóng
- [x] **1.4.2** Implement Cascade Closure: M1 đóng → kiểm tra M5, M15, H1, H4, D1
- [x] **1.4.3** Unit tests với tick sequences giả lập:
  - Verify OHLCV chính xác cho từng TF
  - Verify thứ tự cascade closure đúng
- [x] **1.4.4** Edge case: tick đến muộn (out-of-order) — xử lý gracefully

### 1.5 MTF Synchronizer & Leakage Guard 🔴

- [x] **1.5.1** Định nghĩa `UnifiedStateVector` dataclass:
  - `snapshot_time`, `bars` (dict 6 TF), `tick_context`, `volumetrics`, `leakage_guard`
- [x] **1.5.2** Implement `LeakageGuard`:
  - `forward_locked: Set[str]` — tập TF bị khóa trong backtest mode
  - Hàm `apply_guard(usv, current_timestamp)`: xác định TF nào cần lock
  - Khi TF bị lock: `bar_forming.close = None`
- [x] **1.5.3** Hàm `build_usv(tick)`: thu thập toàn bộ BarState → đóng gói USV
- [x] **1.5.4** Unit test Leakage Guard:
  - Verify không có look-ahead ở bất kỳ TF nào trong backtest mode
  - Property-based test với `hypothesis` library

### 1.6 Volumetrics Engine 🔴

- [x] **1.6.1** Viết `core/ingestion/volumetrics_engine.py`:
  - Tick Delta: phân loại buy/sell aggressor theo quy tắc bid/ask
  - CVD tích lũy theo nến: `CVD_t = Σ δ_i`
  - CVD chuẩn hóa: `CVD_norm = CVD / V_total`
  - CVD rolling windows: 5, 10, 20, 50 ticks
- [ ] **1.6.2** Tính Order Book Imbalance (OBI) từ DOM data (nếu có)
- [x] **1.6.3** Tính Institutional Intensity Index (III):
  - `III_t = (CVD_t / V_avg_30) × (|ΔP_t| / ATR_14)`
  - Cần ATR running calculator (Wilder's smoothing method)
- [x] **1.6.4** Tính Divergence Score: `DIV_CVD = sign(ΔP) × sign(CVD)`
- [x] **1.6.5** Unit test: verify CVD tích lũy đúng, III trong range hợp lý

### 1.7 Backtest Data Loader 🟡

- [x] **1.7.1** Viết `core/ingestion/historical_tick_loader.py`:
  - Đọc file Parquet từ `data/historical_ticks/{symbol}/{year}/`
  - Iterator theo timestamp tăng dần
  - Giả lập tick như real-time (replay mode)
- [x] **1.7.2** Cấu trúc Parquet schema: `timestamp_us, bid, ask, last, volume, flags`
- [x] **1.7.3** Script download/convert dữ liệu từ MT5 history: `scripts/export_ticks_mt5.py`
- [x] **1.7.4** Validate: verify không có tick nào bị thiếu trong giờ market open

### 1.8 Edge Cases & Fail-safes 🟡

- [x] **1.8.1** D.1: Mất kết nối ZeroMQ → `STALENESS_ALERT` flag, reconnect backoff
- [x] **1.8.2** D.2: Phát hiện News Spike (`|Δprice| > 3×ATR_5min`) → không update CVD, set `SPIKE_REGIME`
- [x] **1.8.3** D.3: Timeframe Desync → nội suy tuyến tính (≤5 nến thiếu) hoặc mark UNRELIABLE
- [x] **1.8.4** D.4: ITQ Queue Overflow → Tick Sampling với K động
- [x] **1.8.5** Integration test toàn bộ Module 1: từ tick → USV hoàn chỉnh

---

## PHASE 2 — MODULE 2: MACRO CALENDAR ENGINE
### Mục tiêu: Thu thập lịch kinh tế, countdown, vector hóa tin tức
### Ước tính: 5–7 ngày
### Dependency: Phase 0, Event Bus (1.1)

---

### 2.1 Calendar Scraper 🔴

- [ ] **2.1.1** Viết `core/macro/calendar_scraper.py`:
  - HTTP client (aiohttp) với retry logic (3 lần, backoff 2s/4s/8s)
  - Parser cho ForexFactory JSON endpoint
  - Parser dự phòng cho Investing.com (HTML scraping với BeautifulSoup)
  - Chuẩn hóa thành `RawNewsEvent` dataclass
- [ ] **2.1.2** Lọc sự kiện theo `config/system.yaml`:
  - Chỉ lấy currencies liên quan đến symbol giao dịch (USD, XAU)
  - Chỉ lấy impact = Medium hoặc High
- [ ] **2.1.3** Lưu vào SQLite `economic_calendar` table (tạo table này)
- [ ] **2.1.4** Chu kỳ refresh: mỗi 6h (cron-like asyncio task)
- [ ] **2.1.5** 30 phút trước event: tăng tần suất lên mỗi 5 phút
- [ ] **2.1.6** Unit test với mock HTTP responses; integration test với real endpoints

### 2.2 News Impact Vectorizer 🔴

- [ ] **2.2.1** Viết `core/macro/news_vectorizer.py`:
  - `impact_base_score`: Low=0.2, Medium=0.5, High=1.0
  - Tra cứu `M_bar_ec` từ bảng lịch sử `news_historical_volatility`
  - Tính `I_news = base × (1 + α × M_bar_ec / ATR_D1)`
- [ ] **2.2.2** Tính Surprise Factor S khi có `actual`:
  - `S = (actual - forecast) / σ_surprise_ec`
  - `σ_surprise_ec` từ SQLite, cập nhật online sau mỗi event
- [ ] **2.2.3** Online update algorithm cho `σ_surprise_ec`:
  - Welford's online algorithm cho running variance
- [ ] **2.2.4** Unit test: verify I_news trong range [0, ~3.0], verify S calculation

### 2.3 Volatility Countdown Timer 🔴

- [ ] **2.3.1** Viết `core/macro/volatility_countdown.py`:
  - Asyncio task chạy mỗi 1 giây
  - Tính `seconds_to_next` cho mỗi event
  - Phân loại: NORMAL / PRE_NEWS / NEWS_WINDOW / POST_NEWS
  - Phát event tương ứng lên Event Bus
- [ ] **2.3.2** Điều kiện kích hoạt `active_guardrail = True`:
  - `seconds_to_next <= 900` (15 phút) VÀ `impact = High`
- [ ] **2.3.3** `NEWS_WINDOW` detection: `actual` vừa xuất hiện trong scrape
- [ ] **2.3.4** Unit test: mock time progression → verify state transitions đúng
- [ ] **2.3.5** Performance: verify vòng lặp 1 giây không tốn > 10ms CPU

### 2.4 Post-News Regime Classifier 🟡

- [ ] **2.4.1** Viết `core/macro/regime_classifier.py`:
  - `POST_NEWS_CLASSIFICATION` sau 5 phút từ khi tin ra
  - So sánh `directional_move = (P_5min - P0) / ATR_H1` với `surprise_direction`
  - Phân loại: IMPULSIVE_FOLLOW_THROUGH / REVERSAL_AFTER_SPIKE / CHOPPY_CONSOLIDATION
- [ ] **2.4.2** Lưu `NewsOutcome` vào SQLite `news_outcomes` table
- [ ] **2.4.3** Update `news_historical_volatility` table sau mỗi outcome
- [ ] **2.4.4** Unit test với mock price data và surprise scenarios

### 2.5 Edge Cases 🟡

- [ ] **2.5.1** D.1: Scraping failure → fallback to SQLite cache, set `CALENDAR_STALE`
- [ ] **2.5.2** D.2: Event reschedule > 30 phút → reset guardrail, recalculate
- [ ] **2.5.3** D.3: Event cancellation → xóa khỏi queue, reset nếu đang PRE_NEWS
- [ ] **2.5.4** D.4: Cluster events (2+ High events trong 30 phút) → gộp, amplify 1.25×

---

## PHASE 3 — MODULE 4: MEMORY & PERSISTENCE ENGINE
### Mục tiêu: Redis + SQLite + Vector DB sẵn sàng trước khi AI Engine cần
### Ước tính: 6–8 ngày
### Dependency: Phase 0
### ⚠️ Build trước Module 3 vì Module 3 phụ thuộc vào Memory

---

### 3.1 Redis Connection Manager 🔴

- [ ] **3.1.1** Viết `core/memory/short_term/redis_cache_manager.py`:
  - Connection pool với `redis-py` async client
  - Retry logic khi connection fail
  - Helper methods: `set_zone`, `get_zone`, `set_ai_output`, `get_debate`
  - Serialization: JSON cho dicts, binary MessagePack cho vectors
- [ ] **3.1.2** Implement tất cả 7 key namespaces từ spec V4.5:
  - zone, ai:output, macro:state, macro:events, debate, features, latent, metrics
- [ ] **3.1.3** Unit test với Redis mock (`fakeredis` library):
  - Verify TTL được set đúng
  - Verify serialization/deserialization round-trip

### 3.2 Active Zone Registry 🔴

- [ ] **3.2.1** Viết `core/memory/short_term/active_zone_registry.py`:
  - `upsert_zone(zone)`: INSERT or UPDATE zone trong Redis
  - `get_zones_near_price(price, symbol, window_pips)`: trả về zones trong range
  - `update_zone_status(zone_id, new_status)`: UNMITIGATED → PARTIALLY... → MITIGATED
  - `update_zone_p_hold(zone_id, p_hold)`: cập nhật xác suất từ Model B
- [ ] **3.2.2** Sorted Set `zone_rank:{symbol}:{zone_type}` theo `p_hold × w_zone`
- [ ] **3.2.3** Method `get_top_zones(symbol, zone_type, k=5)`: lấy top-K zones
- [ ] **3.2.4** Unit test: upsert, update, get operations

### 3.3 SQLite Database Manager 🔴

- [ ] **3.3.1** Viết `core/memory/long_term/sqlite_history_store.py`:
  - Connection với tất cả PRAGMAs từ spec V4.5 (WAL, cache, mmap, v.v.)
  - Schema migration system: check `PRAGMA user_version` → run migration scripts
- [ ] **3.3.2** Tạo `scripts/migrations/` với:
  - `migration_001_initial_schema.sql`: tạo tất cả 5 tables
  - `migration_002_add_indexes.sql`: tạo tất cả indexes
- [ ] **3.3.3** Implement CRUD cho từng bảng:
  - `insert_prediction(prediction)`, `update_prediction_outcome(id, outcome)`
  - `insert_news_outcome(outcome)`, `update_news_sigma(event_id, new_sigma)`
  - `insert_model_performance(metrics)`, `insert_system_metrics(metrics)`
  - `insert_zone_history(zone)`, `update_zone_history_result(zone_id, result)`
- [ ] **3.3.4** Async write queue: buffer writes, flush theo batch (mỗi 100ms)
- [ ] **3.3.5** Unit test với in-memory SQLite: verify schema, constraints, indexes

### 3.4 Outcome Determination Engine 🔴

- [ ] **3.4.1** Viết background worker chạy mỗi khi nhận `BAR_CLOSE(M1)`:
  - Query predictions có `outcome_determined = 0`
  - Check BSL hit: `high_since_prediction >= predicted_bsl_level`
  - Check SSL hit: `low_since_prediction <= predicted_ssl_level`
  - Check timeout: `elapsed > MAX_HORIZON (4h)`
  - Check Model B hold: zone touched AND held
- [ ] **3.4.2** Update predictions trong SQLite khi xác nhận
- [ ] **3.4.3** Phát `OUTCOME_CONFIRMED` event → Module 5
- [ ] **3.4.4** Unit test: mock price series → verify outcome determination logic

### 3.5 Vector DB Adapter 🟡

- [ ] **3.5.1** Viết `core/memory/long_term/vectordb_adapter.py`:
  - Abstraction layer hỗ trợ cả ChromaDB và Qdrant (config-driven)
  - Methods: `insert_debate`, `search_similar_debates`, `insert_zone_embedding`, `search_similar_zones`
  - Tự động tạo collections nếu chưa tồn tại
- [ ] **3.5.2** Tạo Collection `debate_archive`:
  - HNSW index: m=16, ef_construction=200
  - Payload indexes: symbol, macro_regime, actual_outcome
- [ ] **3.5.3** Tạo Collection `zone_embeddings`:
  - HNSW index: m=8, ef_construction=100
- [ ] **3.5.4** Re-ranking algorithm: `0.7×cosine_sim + 0.3×recency_weight`
- [ ] **3.5.5** Integration test: insert → search → verify top-3 results

### 3.6 Debate Archival Worker 🟡

- [ ] **3.6.1** Background worker: chạy mỗi 5 phút, archive debates từ Redis sang Vector DB:
  - Lọc debates có `archived = False` VÀ `TTL < 600s` (còn < 10 phút sống)
  - Tính `e_USV` projection (xem Phase 5.3)
  - Insert vào Qdrant `debate_archive`
  - Set `archived = True` trong Redis
- [ ] **3.6.2** Retry logic: nếu Qdrant unavailable → queue trong SQLite

### 3.7 RAG Retriever 🟡

- [ ] **3.7.1** Viết `core/memory/long_term/rag_retriever.py`:
  - `retrieve_precedents(e_usv, symbol, macro_regime, k=3)`:
    - Pre-filter bằng payload filter (symbol, actual_outcome IS NOT NULL)
    - ANN search với cosine_sim >= 0.80
    - Nếu < 3 kết quả: giảm threshold xuống 0.75
    - Re-ranking và return top-3
- [ ] **3.7.2** Cache kết quả RAG trong Redis 5 phút: `rag:{symbol}:{timestamp_rounded}`
- [ ] **3.7.3** Unit test: mock vector DB → verify retrieval logic

---

## PHASE 4 — MODULE 3, LAYER 1: SYMBOLIC FEATURE ENGINEERING
### Mục tiêu: Toàn bộ SMC/ICT features từ USV
### Ước tính: 8–12 ngày
### Dependency: Phase 1 (USV), Phase 3 (Zone Registry)

---

### 4.1 Swing Point Detector 🔴

- [ ] **4.1.1** Viết `core/ai_engine/feature_engineering/smc_detector.py`:
  - Implement SkipEQHigh/SkipEQLow (bỏ qua equal highs/lows)
  - Detect STH/STL với lookback k theo TF (M1/M5: k=3, M15/H1: k=5, H4/D1: k=10)
  - Phân loại: HH, LH (cho Highs) / LL, HL (cho Lows)
- [ ] **4.1.2** Implement ST → IT promotion (FindIT):
  - Ba STH liên tiếp: H2.price > H3.price AND H2.price > H1.price
  - Tương tự cho STL → ITL
- [ ] **4.1.3** Implement IT → LT promotion (FindLT)
- [ ] **4.1.4** Phát hiện Claimed Pivots: high crosses Pivot High → claimed
- [ ] **4.1.5** Unit test với synthetically generated price series:
  - Verify correct pivot classification
  - Verify no look-ahead bias

### 4.2 BSL/SSL Registry Integration 🔴

- [ ] **4.2.1** Sau khi detect Pivots → upsert vào Zone Registry (Redis)
- [ ] **4.2.2** Tính `F_liq` vector (24 chiều) từ active BSL/SSL:
  - Khoảng cách, tuổi, volume tích lũy, số lượng, tỷ lệ claimed
- [ ] **4.2.3** Method `get_f_liq_vector(symbol, current_price)` → `np.ndarray[24]`

### 4.3 ICT Structure Mapper 🔴

- [ ] **4.3.1** Viết `core/ai_engine/feature_engineering/ict_structure_mapper.py`:
  - `detect_premium_discount(bars)`:
    - Tìm major Swing High và Swing Low
    - Tính Equilibrium, Fibonacci 0.382, 0.618
    - Phân vùng Premium/Discount
  - `detect_mss(bars, tf)`: Market Structure Shift (Bullish & Bearish)
  - `detect_bos(bars, tf)`: Break of Structure
- [ ] **4.3.2** Trọng số: MSS quan trọng hơn BOS (MSS weight = 2×BOS weight)
- [ ] **4.3.3** Output: `StructureMap = {mss_events, bos_events, premium_zones, discount_zones, fib_levels}`
- [ ] **4.3.4** Unit test với price scenarios được thiết kế sẵn

### 4.4 FVG / OB Scanner 🔴

- [ ] **4.4.1** Viết `core/ai_engine/feature_engineering/fvg_ob_scanner.py`:
  - `scan_fvg(bars, tf)`: Phát hiện FVG Bull/Bear cho 3-candle pattern
  - Bộ lọc displacement strength: `|body[n-1]| > σ_body × D_level`
  - Cơ chế gộp VI+FVG (mergeVI) nếu config = True
  - `check_mitigated(fvg, current_price, bars)`: 5 loại mitigation
  - `detect_ifvg(fvg, bars)`: chuyển FVG → iFVG khi close < fvg.open
- [ ] **4.4.2** `scan_ob(bars, tf)`: Phát hiện Order Block theo BOS/MSS trigger
  - Tìm last bearish candle trước BOS (Bullish OB)
  - Tính `strength = |high[bos] - low[bos]| / ATR`
- [ ] **4.4.3** Phân loại Premium/Discount cho mỗi FVG/OB → tính `w_zone`
- [ ] **4.4.4** Persist FVG/OB vào Zone Registry (Redis) và Zone History (SQLite)
- [ ] **4.4.5** Unit test: verify FVG detection với candlestick fixtures

### 4.5 Equal Levels Detector 🟡

- [ ] **4.5.1** Phát hiện Equal Highs/Lows với tolerance parameter:
  - `|P1.price - P2.price| < ATR_current × EQ_Tolerance`
  - Verify không có nến nào vượt qua đường nối hai điểm EQ (testEQ function)
- [ ] **4.5.2** Track EQ claimed status
- [ ] **4.5.3** Tích hợp EQ count vào `F_struct` vector

### 4.6 Liquidity Pool Indexer 🔴

- [ ] **4.6.1** Viết `core/ai_engine/feature_engineering/liquidity_pool_indexer.py`:
  - Tổng hợp toàn bộ BSL/SSL từ tất cả TF
  - Tính `V_acc_zone`: tổng CVD trong vùng ±ε quanh mức liquidity
  - Tính `III_zone`: tổng III tích lũy trong vùng FVG/OB
- [ ] **4.6.2** Method `build_symbolic_feature_map(usv)` → `SymbolicFeatureMap`
- [ ] **4.6.3** Method `build_f_struct_vector(feature_map)` → `np.ndarray[64]`
- [ ] **4.6.4** Method `build_f_agg_vector(feature_map, current_price)` → `np.ndarray[16]`

### 4.7 Displacement Engine (từ Pine Script) 🟢

- [ ] **4.7.1** Implement `f_displacement(bars, length, factor)`:
  - `σ_body = std(|open - close|, window=length)`
  - `D_strength = |body[n-1]| / (σ_body × factor)`
  - Kết hợp FVG requirement: `D_strength > 1.0 AND FVG[n]`
- [ ] **4.7.2** Tính `D_strength_vector` (5 chiều) như định nghĩa trong V4.5

---

## PHASE 5 — MODULE 3, LAYER 2: NEURAL COMPRESSION (LSTM AUTOENCODER)
### Mục tiêu: LSTM Autoencoder tạo latent vector z ∈ ℝ^512
### Ước tính: 10–14 ngày
### Dependency: Phase 4 (features), Phase 3 (Redis cache)

---

### 5.1 Hierarchical LSTM Autoencoder Architecture 🔴

- [ ] **5.1.1** Viết `core/ai_engine/neural/hierarchical_lstm_ae.py`:
  - **Tick-Level Encoder**: BiLSTM(hidden=128, layers=2), input=8-dim per tick, seq=512
  - **Bar-Level Encoder**: LSTM(hidden=256, layers=3), input=12-dim per bar
  - **Multi-TF Encoder**: 6 Bar-Level LSTMs riêng (lookback khác nhau per TF)
  - **Cross-TF Attention**: Multi-Head Attention (8 heads, d_model=512)
  - **Decoder**: LSTM tái tạo chuỗi M1 từ z (chỉ dùng khi train)
- [ ] **5.1.2** Implement VAE variant (optional): thêm KL divergence loss
- [ ] **5.1.3** Verify model architecture với dummy data (forward pass không crash)
- [ ] **5.1.4** Tính số parameters: target < 50M parameters để inference < 50ms

### 5.2 Dataset Builder cho LSTM Training 🔴

- [ ] **5.2.1** Viết `scripts/build_lstm_dataset.py`:
  - Chạy Module 1 pipeline trên toàn bộ tick lịch sử
  - Lưu (tick_sequence[512], bar_sequence_m1[100], bar_sequences_mtf[...]) cho mỗi bar close
  - Format: numpy memmap hoặc HDF5 cho dataset lớn
- [ ] **5.2.2** Normalization: MinMax hoặc RobustScaler per feature, lưu scaler params vào `models/lstm/feature_scaler.pkl`
- [ ] **5.2.3** Train/Val/Test split theo thời gian (không random):
  - Train: 2022–2023 (80%)
  - Val: Q1 2024 (10%)
  - Test: Q2-Q3 2024 (10%)

### 5.3 LSTM Training Pipeline 🔴

- [ ] **5.3.1** Viết `scripts/train_lstm.py`:
  - PyTorch Lightning hoặc pure PyTorch training loop
  - Loss: `MSE(reconstructed, actual) + β×KL_divergence`
  - Optimizer: AdamW, lr=1e-3, weight_decay=1e-5
  - LR Scheduler: CosineAnnealingWarmRestarts
  - Early stopping: patience=10 epochs
  - Gradient clipping: max_norm=1.0
- [ ] **5.3.2** Checkpointing: save best model theo val loss
- [ ] **5.3.3** TensorBoard/W&B logging cho training metrics
- [ ] **5.3.4** GPU training support (CUDA device detection)

### 5.4 LSTM Inference Engine 🔴

- [ ] **5.4.1** Viết `core/ai_engine/neural/lstm_inference.py`:
  - Load weights từ `models/lstm/weights/`
  - Method `encode(tick_sequence, bar_sequences_mtf)` → `z ∈ ℝ^512`
  - Warm-up: chạy dummy inference khi khởi động để JIT compile
  - Đo latency: target < 20ms per inference trên CPU
- [ ] **5.4.2** Cache z trong Redis: `latent:{symbol}:{bar_close_ts}` TTL=300s
- [ ] **5.4.3** Benchmark: verify latency target met

### 5.5 USV Projection Matrix (cho Vector DB embedding) 🟡

- [ ] **5.5.1** Viết `core/ai_engine/neural/usv_projector.py`:
  - Projection: `e_USV = LayerNorm(tanh(W_proj × [z; f_smc; f_macro] + b))`
  - Input dim: 512 + 64 + 12 = 588 → Output dim: 256
  - `W_proj ∈ ℝ^{256×608}` được học bằng Metric Learning
- [ ] **5.5.2** Metric Learning training: Triplet Loss với online mining
- [ ] **5.5.3** L2-normalize output để phù hợp cosine similarity search

---

## PHASE 6 — MODULE 3, LAYER 2: XGBOOST MODELS
### Mục tiêu: Huấn luyện và deploy Model A (P_BSL/P_SSL) và Model B (P_hold)
### Ước tính: 10–14 ngày
### Dependency: Phase 4 (features), Phase 5 (LSTM z vector)

---

### 6.1 Feature Vector Builders 🔴

- [ ] **6.1.1** Viết `core/ai_engine/feature_engineering/feature_builder.py`:
  - `build_x_a(usv, symbolic_map, macro_context, z)` → `np.ndarray[648]`
    - Ghép nối 6 nhóm: z[512] + F_liq[24] + F_flow[20] + F_macro[12] + F_struct[64] + F_agg[16]
  - `build_x_b(usv, zone, contact_candle, macro_context, z)` → `np.ndarray[560]`
    - Ghép nối 4 nhóm: z[512] + F_zone[16] + F_contact[20] + F_macro[12]
- [ ] **6.1.2** Feature scaling:
  - StandardScaler cho các chiều liên tục
  - Không scale chiều boolean và categorical
  - Lưu scaler vào `models/model_a/feature_scaler.pkl` và `models/model_b/feature_scaler.pkl`
- [ ] **6.1.3** Unit test: verify output shape, range, no NaN/Inf values

### 6.2 Dataset Builder cho XGBoost 🔴

- [ ] **6.2.1** Viết `scripts/build_xgb_dataset.py`:
  - Chạy toàn bộ pipeline (Module 1 → 4 → LSTM inference) trên lịch sử
  - Tích hợp với Outcome Determination để label dữ liệu
  - Lưu: `X_A.parquet`, `y_A.parquet`, `X_B.parquet`, `y_B.parquet`
- [ ] **6.2.2** Stratified split theo thời gian VÀ regime:
  - `train_test_split_temporal` với không có data leakage
- [ ] **6.2.3** Class imbalance analysis:
  - In distribution report: tỷ lệ BSL_HIT/SSL_HIT/LATERAL cho Model A
  - Tỷ lệ hold=1/hold=0 cho Model B
- [ ] **6.2.4** Data quality checks: min 5,000 samples per class cho Model B

### 6.3 Model A Training 🔴

- [ ] **6.3.1** Viết `scripts/train_model_a.py`:
  - Custom objective với Gradient + Hessian tính theo công thức V4.5
  - Overconfidence Penalty (λ=0.5, θ=0.85) và News Regime Penalty (λ=0.3)
  - Hyperparameter tuning với Optuna (50 trials):
    - Search space: max_depth [4-8], learning_rate [0.01-0.1], subsample [0.6-1.0]
  - Cross-validation: 5-fold time-series split
- [ ] **6.3.2** Platt Scaling calibration trên validation set
- [ ] **6.3.3** Đánh giá: IC, Brier Score, ECE, accuracy by class
- [ ] **6.3.4** Lưu: `models/model_a/weights/model_a.ubj` (XGBoost binary format)
- [ ] **6.3.5** Lưu `model_config.yaml` với metadata training

### 6.4 Model B Training 🔴

- [ ] **6.4.1** Viết `scripts/train_model_b.py`:
  - Cost-sensitive learning: `scale_pos_weight = N_neg/N_pos`
  - Threshold optimization với ma trận chi phí (FP=2.5, FN=1.0)
  - `θ* = C_FP / (C_FP + C_FN) × p(y=0) / (p(y=0)+p(y=1))`
  - Stratified sampling theo regime
- [ ] **6.4.2** Isotonic Regression calibration
- [ ] **6.4.3** ECE per regime: verify không có calibration bias > 0.08
- [ ] **6.4.4** SHAP analysis: plot và lưu feature importance report
- [ ] **6.4.5** Lưu: `models/model_b/weights/model_b.ubj`

### 6.5 Stacked Ensemble Meta-Learner 🟡

- [ ] **6.5.1** Viết Meta-Learner (Logistic Regression):
  - Input: [p_bsl_xgb, p_ssl_xgb, p_lat_xgb, consensus_from_lstm, regime_code]
  - Train trên validation set predictions (out-of-fold)
  - Lưu: `models/ensemble/meta_learner.pkl`
- [ ] **6.5.2** A/B test framework: compare XGBoost-only vs Ensemble IC

### 6.6 Inference Engines 🔴

- [ ] **6.6.1** Viết `core/ai_engine/neural/model_a_liquidity_predictor.py`:
  - Load XGBoost model + scaler + calibration
  - Method `predict(x_a)` → `{p_bsl, p_ssl, p_lateral}`
  - Post-inference adjustment: Macro Guardrail + Session Weight
  - Target latency: < 5ms per inference
- [ ] **6.6.2** Viết `core/ai_engine/neural/model_b_holdzone_classifier.py`:
  - Loop qua tất cả UNMITIGATED zones
  - Method `predict_zone(x_b)` → `{p_hold, p_breach}`
  - Apply threshold θ* = 0.71 cho binary decision
- [ ] **6.6.3** Benchmark cả hai models: target < 10ms combined
- [ ] **6.6.4** Implement Model Rollback: giữ 3 phiên bản, rollback qua config

---

## PHASE 7 — MODULE 3, LAYER 3: MULTI-AGENT DEBATE
### Mục tiêu: Bull/Bear/Critic agents với RAG, consensus output
### Ước tính: 6–8 ngày
### Dependency: Phase 3 (RAG), Phase 4 (features), Phase 6 (predictions)

---

### 7.1 Technical Brief Builder 🔴

- [ ] **7.1.1** Viết `core/ai_engine/multi_agent/debate_orchestrator.py`:
  - `build_technical_brief(symbolic_map, predictions, macro_context, precedents)`:
    - Format markdown/text với tất cả thông tin định lượng
    - Cấu trúc: HTF structure → Liquidity targets → Active zones → Model outputs → Macro → Precedents
  - Trim brief nếu > 8,000 tokens (giữ phần quan trọng nhất)
- [ ] **7.1.2** Unit test: verify brief chứa đầy đủ thông tin cần thiết

### 7.2 Bull Agent 🔴

- [ ] **7.2.1** Viết `core/ai_engine/multi_agent/bull_agent.py`:
  - System Prompt theo spec V1.0 (bullish bias, cite ≥3 evidences, no hallucination)
  - Thêm News Regime caveat khi `active_guardrail = True`
  - API call đến Claude Sonnet (anthropic SDK)
  - Parse response thành `BullThesis` dataclass
  - Timeout: 3 giây
- [ ] **7.2.2** Fallback khi timeout: `BullThesis(confidence=0.5, evidence=["HTF_BIAS"])`
- [ ] **7.2.3** Token usage tracking: log tokens per debate

### 7.3 Bear Agent 🔴

- [ ] **7.3.1** Viết `core/ai_engine/multi_agent/bear_agent.py`:
  - Tương tự Bull Agent nhưng bearish bias
  - Chạy SONG SONG với Bull Agent (asyncio.gather)
- [ ] **7.3.2** Fallback tương tự

### 7.4 Critic Agent 🔴

- [ ] **7.4.1** Viết `core/ai_engine/multi_agent/critic_agent.py`:
  - Nhận Bull Thesis + Bear Thesis + Technical Brief + Precedents
  - Đánh giá tính logic của cả hai
  - Output: `ConsensusResult` (rating [-4,+4], direction, reasoning, agreement_score)
  - Timeout: 5 giây
- [ ] **7.4.2** Fallback khi timeout: Algorithmic Critic
  - `consensus_rating = round((bull_conf - bear_conf) × 4)`
  - `confidence_qualifier = "LOW"`

### 7.5 Debate Orchestrator 🔴

- [ ] **7.5.1** Orchestrate toàn bộ flow:
  - [1] Retrieve RAG precedents
  - [2] Build Technical Brief
  - [3] Bull + Bear concurrently (asyncio.gather với timeout)
  - [4] Critic (sequential, cần kết quả từ bước 3)
  - [5] Lưu vào Redis (debate log)
  - [6] Phát `CONSENSUS_READY` event
  - Total timeout: 8 giây
- [ ] **7.5.2** Điều kiện kích hoạt debate:
  - Mỗi khi có BOS/MSS mới trên H1 hoặc H4
  - Hoặc mỗi 5 phút (periodic refresh)
  - Hoặc khi nhận `MAJOR_SURPRISE_FLAG` từ Module 2
- [ ] **7.5.3** Rate limiting: không chạy debate nếu debate trước < 3 phút trước
- [ ] **7.5.4** Cost tracking: estimate USD cost per debate, log tổng chi phí hàng ngày

### 7.6 Rule-Based Fallback Engine 🟡

- [ ] **7.6.1** Viết `core/ai_engine/multi_agent/rule_based_fallback.py`:
  - HTF bias từ EMA50 + BOS detection (không cần ML)
  - Fallback P_BSL/P_SSL theo htf_bias
  - P_hold = w_zone × 0.55
- [ ] **7.6.2** Kích hoạt tự động khi `MODEL_DEGRADED = True`

---

## PHASE 8 — MODULE 5: BACKTESTING & DRIFT EVALUATION
### Mục tiêu: Event-driven backtest, IC tracking, drift detection
### Ước tính: 8–10 ngày
### Dependency: Phase 1, 2, 3, 4, 6

---

### 8.1 Event-Driven Simulator 🔴

- [ ] **8.1.1** Viết `core/backtesting/event_driven_simulator.py`:
  - Initialize state (backtest_mode = True, reset all modules)
  - Main loop: replay ticks theo thứ tự timestamp
  - Inject calendar events theo thời gian tương ứng
  - Collect predictions từ Event Bus
  - Look-ahead cho outcome determination (trong phạm vi MAX_HORIZON=4h)
- [ ] **8.1.2** Verify không có data leakage: LeakageGuard active trong toàn bộ run
- [ ] **8.1.3** Progress tracking: tqdm progress bar, ETA display
- [ ] **8.1.4** Speed target: backtest 1 năm tick data trong < 2 giờ

### 8.2 IC Calculator 🔴

- [ ] **8.2.1** Implement Spearman Rank Correlation:
  - `IC = spearmanr(ŷ_vector, y_vector)` với `ŷ = P_BSL - P_SSL`
  - Rolling IC với N=20 predictions
- [ ] **8.2.2** Ngưỡng và cảnh báo: IC < 0.05 → set `MODEL_DEGRADED = True`
- [ ] **8.2.3** Precision/Recall/F1 cho P_hold threshold = 0.70
- [ ] **8.2.4** Brier Score cho Model A
- [ ] **8.2.5** Ghi kết quả vào `model_performance` table (SQLite)

### 8.3 Feature Drift Detection (PSI) 🔴

- [ ] **8.3.1** Viết `core/backtesting/drift_tracker.py`:
  - PSI computation: 10 bins, compare current month vs training baseline
  - `FDS = (1/|F|) × Σ 𝟙[PSI_f >= 0.2]`
  - FDS > 0.4 → `MODEL_DEGRADED = True`
- [ ] **8.3.2** Log drifted features list vào `model_performance.top_drifted_features`
- [ ] **8.3.3** Baseline lưu trong `models/feature_baseline.parquet` (generated khi train)

### 8.4 Regime Shift Detector 🔴

- [ ] **8.4.1** Implement Regime Detection:
  - `vol_ratio = std(returns, 5) / std(returns, 60)`
  - ADX_14 trên D1
  - Phân loại 4 regime: TRENDING_LV, TRENDING_HV, CHOPPY_HV, NORMAL
- [ ] **8.4.2** Khi regime thay đổi: phát `REGIME_SHIFT_DETECTED` event
- [ ] **8.4.3** Tăng tần suất IC evaluation từ daily → mỗi 4 giờ khi regime shift

### 8.5 Backtest Report Generator 🟡

- [ ] **8.5.1** Viết `core/backtesting/backtest_report.py`:
  - Performance metrics tổng hợp
  - Breakdown theo regime, session, impact level
  - Overfitting detection: IC_backtest vs IC_forward
  - Data coverage analysis (SPARSE_DATA_ZONE detection)
- [ ] **8.5.2** Export report dạng JSON + HTML (optional)

### 8.6 Fine-Tuning Pipeline 🟡

- [ ] **8.6.1** Trigger conditions: `MODEL_DEGRADED AND len(new_samples) >= 200`
- [ ] **8.6.2** Fine-tune với `learning_rate = 1e-5`, 5 epochs
- [ ] **8.6.3** Deploy nếu `IC_new > IC_current × 1.1`
- [ ] **8.6.4** Rollback nếu IC không cải thiện

---

## PHASE 9 — MODULE 6: VISUALIZATION (REACT + TRADINGVIEW)
### Mục tiêu: Real-time chart với ghost zones, heatmap, macro timeline
### Ước tính: 10–14 ngày
### Dependency: Phase 0 (frontend setup), WebSocket (từ Module 7)

---

### 9.1 WebSocket Client và Redux Store 🔴

- [ ] **9.1.1** Viết `ui/src/hooks/useWebSocket.ts`:
  - Connect đến `ws://localhost:47290`
  - Reconnect với exponential backoff (1s, 2s, 4s, 8s)
  - Dispatch messages đến Redux store theo `message.type`
  - Track latency: `Date.now() - message.emit_time_ms`
- [ ] **9.1.2** Viết Redux slices:
  - `chartSlice.ts`: bars, zones, liquidity targets
  - `aiStateSlice.ts`: predictions, consensus, model status
  - `macroSlice.ts`: countdown state, news events
- [ ] **9.1.3** Viết `useChartState.ts` hook: memoized selectors cho chart data
- [ ] **9.1.4** Unit test: mock WebSocket → verify state updates

### 9.2 TradingView Chart Core 🔴

- [ ] **9.2.1** Viết `ui/src/components/ChartCanvas/ChartCanvas.tsx`:
  - Init `createChart()` với responsive layout
  - Subscribe đến `bar_update` và `new_bar_closed` WebSocket messages
  - CandlestickSeries: real-time update trên mỗi tick
  - Volume series (histogram)
  - Timezone handling: convert UTC → user local time
- [ ] **9.2.2** Viết Web Worker `chartRenderWorker.ts`:
  - Xử lý data transforms trong worker để không block main thread
- [ ] **9.2.3** Performance: verify 60fps với 1000+ bars displayed
- [ ] **9.2.4** Unit test: render với mock data

### 9.3 MTF Ghost Zones Overlay 🔴

- [ ] **9.3.1** Viết `ui/src/components/ChartCanvas/MTFGhostZones.tsx`:
  - Subscribe đến `zone_update` messages
  - Vẽ rectangles với TradingView Lightweight Charts ICustomSeriesPaneView
  - Tọa độ: `price_top`, `price_bottom` (Y), `formed_time` → current bar (X)
  - Border style theo TF: H4/D1 = dashed 2px, H1 = dashed 1.5px, LTF = dotted 1px
  - Status = PARTIALLY_MITIGATED: half opacity dưới midpoint
- [ ] **9.3.2** Giới hạn hiển thị: max 50 zones, lọc P_hold < 0.50 nếu vượt
- [ ] **9.3.3** Auto-hide LTF zones khi đang xem H1 trở lên (performance)

### 9.4 AI Heatmap Overlay 🔴

- [ ] **9.4.1** Viết `ui/src/components/ChartCanvas/AIHeatmapOverlay.tsx`:
  - Nhận `{zone_id, new_p_hold}` updates
  - Color mapping: `hsla(120/0, 80-0%, 50-60%, 0.4-0.12)` theo bảng spec
  - CSS transition: `fill 0.5s ease-in-out` cho smooth color change
  - Pulse animation khi `p_hold` thay đổi > 0.15
  - Chế độ tin tức: LTF zones opacity × 0.4

### 9.5 Liquidity Target Lines 🔴

- [ ] **9.5.1** Viết `ui/src/components/ChartCanvas/LiquidityTargetLines.tsx`:
  - BSL target: blue dashed line với label `BSL ↑ {P_BSL}%`
  - SSL target: red dashed line với label `SSL ↓ {P_SSL}%`
  - Pulse animation khi P > 0.70
  - Update khi nhận `prediction_update` từ WebSocket

### 9.6 Macro Timeline 🔴

- [ ] **9.6.1** Viết `ui/src/components/ChartCanvas/MacroTimeline.tsx`:
  - Vertical lines theo scheduled_time của events
  - Màu sắc: Low=gray, Medium=yellow, High=red
  - Style: dotted cho tương lai, solid cho đã xảy ra
  - Tooltip hover: tên event, currency, impact, forecast, actual, S
  - Pre-News Zone shading: `rgba(244,67,54,0.05)` trong [-15min, +5min]
- [ ] **9.6.2** Session dividers:
  - Background colors cho từng phiên (Asian/London/NY)
  - Kill Zone overlap (London-NY) với label "Kill Zone"

### 9.7 Sidebar Components 🟡

- [ ] **9.7.1** `AgentDebatePanel.tsx`:
  - Gauge từ -4 đến +4 với indicator position
  - Bull evidence list + Bear evidence list
  - Reasoning summary text (max 150 words)
  - "DEBATE TIMEOUT" badge nếu fallback được dùng
- [ ] **9.7.2** `NewsCountdownWidget.tsx`:
  - MM:SS countdown format
  - Màu đỏ nhấp nháy khi < 5 phút
  - Âm thanh cảnh báo (beep) khi còn 60 giây
  - Tên event và impact badge
- [ ] **9.7.3** `KillzoneIndicator.tsx`:
  - Indicator phiên hiện tại với màu sắc phiên
  - "KILL ZONE" label nếu đang trong London Open hoặc NY Open KZ
- [ ] **9.7.4** `ModelConfidenceGauge.tsx`:
  - P_BSL vs P_SSL vs P_lateral bar chart
  - confidence_qualifier badge: HIGH/MEDIUM/LOW
  - "MODEL DEGRADED" cảnh báo nếu applicable

### 9.8 Dashboard và Settings 🟢

- [ ] **9.8.1** Latency indicator tại góc UI:
  - Green < 30ms, Yellow < 50ms, Red > 50ms
- [ ] **9.8.2** System status banner:
  - "⚠ Mất kết nối" khi WebSocket down
  - "MODEL DEGRADED" khi model flag set
  - "⚠ Lịch kinh tế có thể không cập nhật" khi CALENDAR_STALE
- [ ] **9.8.3** Settings panel: toggle zone visibility, TF filter

---

## PHASE 10 — MODULE 7: DESKTOP SHELL & IPC
### Mục tiêu: Tauri wrapper, WebSocket server, < 50ms latency
### Ước tính: 5–7 ngày
### Dependency: Phase 0 (Tauri setup), Phase 9 (UI), All Python modules

---

### 10.1 Python WebSocket Server 🔴

- [ ] **10.1.1** Viết `core/ipc/websocket_server.py`:
  - `websockets` library, async server trên port 47290
  - Port fallback: thử 47291-47299 nếu bận
  - Ghi port vào `/tmp/aq_ws_port.txt`
  - Chạy trên thread riêng biệt (không share GIL với AI inference)
- [ ] **10.1.2** Viết `core/ipc/message_schema.py`:
  - Pydantic models cho tất cả message types (8 types Backend→Frontend)
  - Serialization: JSON (default) + MessagePack binary (cho `bar_update`)
  - `emit_time_ms` field trong mọi message
- [ ] **10.1.3** Viết `core/ipc/broadcast_dispatcher.py`:
  - Subscribe đến Event Bus cho tất cả events cần broadcast
  - Throttle `bar_update`: maximum 10 fps khi thị trường chậm
  - `countdown_update`: mỗi 1s khi active_guardrail=True, mỗi 10s khi không

### 10.2 Tauri IPC Bridge 🔴

- [ ] **10.2.1** Viết `desktop/src-tauri/src/ipc_bridge.rs`:
  - Spawn Python subprocess với đường dẫn cấu hình
  - Monitor stdout: detect "AGENTIQ_BACKEND_READY" signal
  - Đọc `/tmp/aq_ws_port.txt` → truyền cho React WebView
- [ ] **10.2.2** Pipe stderr từ Python → Tauri log file:
  - Windows: `%APPDATA%/agentic-quant/logs/`
  - macOS: `~/Library/Logs/agentic-quant/`
- [ ] **10.2.3** Python crash detection: `child.wait()` → auto-restart dialog

### 10.3 System Tray 🟡

- [ ] **10.3.1** Viết `desktop/src-tauri/src/system_tray.rs`:
  - Icon states: green (normal), yellow (warning), red (critical)
  - Context menu: Show/Hide, Restart Backend, Quit
  - Notification: popup khi có High Impact news trong 15 phút
- [ ] **10.3.2** Map system states → tray icon:
  - `MODEL_DEGRADED` → yellow
  - `CRITICAL_FEED_FAILURE` → red
  - `active_guardrail = True` → yellow pulse

### 10.4 Cold Start Sequence 🔴

- [ ] **10.4.1** Implement cold start theo spec (target < 8 giây):
  - T+0.0s: Tauri binary + WebView
  - T+0.2s: Splash screen
  - T+0.5s: Spawn Python
  - T+1.0s: Python init (config, SQLite, Redis, VectorDB, model load)
  - T+3.0s: Connect ZeroMQ MT5
  - T+3.5s: Fetch calendar
  - T+4.0s: Start WebSocket server
  - T+4.1s: Print "AGENTIQ_BACKEND_READY"
  - T+4.2s: React connect WebSocket
  - T+4.5s: Request full_state_snapshot
  - T+5.0s: Render chart
  - T+6.0s: Hide splash
- [ ] **10.4.2** Measure và optimize từng bước

### 10.5 Auto-Update 🟢

- [ ] **10.5.1** Configure Tauri updater với GitHub Releases endpoint
- [ ] **10.5.2** Version comparison logic
- [ ] **10.5.3** Schema migration khi update (SQLite version check)

---

## PHASE 11 — INTEGRATION TESTING & SYSTEM HARDENING
### Mục tiêu: Full system E2E test, performance validation
### Ước tính: 7–10 ngày
### Dependency: All phases above

---

### 11.1 Integration Test Suite 🔴

- [ ] **11.1.1** E2E test: từ tick replay → chart display
  - Mock MT5 EA gửi 1000 ticks
  - Verify chart cập nhật đúng
  - Verify predictions được tạo ra
  - Verify zones hiển thị trên UI
- [ ] **11.1.2** Module boundary tests:
  - Module 1 → Module 2 interface
  - Module 2 → Module 3 interface
  - Module 3 → Module 4 (persistence)
  - Module 4 → Module 3 (RAG retrieval)
  - Module 3 → Module 7 (IPC)
- [ ] **11.1.3** Backtest integration test:
  - Chạy backtest 1 tháng data
  - Verify no look-ahead bias (statistical test)
  - Verify IC > 0.05 trên test period

### 11.2 Performance Benchmarking 🔴

- [ ] **11.2.1** Đo latency end-to-end:
  - Tick received → USV built: < 5ms
  - USV → LSTM inference: < 20ms
  - LSTM z → XGBoost inference: < 10ms
  - Prediction → WebSocket broadcast: < 5ms
  - Total (Python): < 40ms
  - WebSocket → React render: < 15ms
  - **Total: < 50ms** ✓
- [ ] **11.2.2** Throughput test: handle 50 tick/second sustained
- [ ] **11.2.3** Memory usage: verify < 2GB RAM total

### 11.3 Stress Testing 🟡

- [ ] **11.3.1** Simulate NFP release:
  - Tăng đột biến tick rate 10× trong 30 giây
  - Verify ITQ queue không overflow
  - Verify guardrails kích hoạt đúng
- [ ] **11.3.2** Long-running stability test: 72 giờ continuous
- [ ] **11.3.3** Redis memory pressure test: fill đến 80% → verify LRU eviction

### 11.4 Security Audit 🟡

- [ ] **11.4.1** Verify TradingView HMAC signature không bị bypass
- [ ] **11.4.2** Redis chỉ bind localhost (không expose ra network)
- [ ] **11.4.3** SQLite file permissions (read-only cho Python processes không cần write)
- [ ] **11.4.4** API keys (Anthropic) không hardcode trong code, dùng environment variables

---

## PHASE 12 — OPERATIONAL TOOLING & DEPLOYMENT
### Mục tiêu: Runbooks, deployment scripts, monitoring
### Ước tính: 4–5 ngày

---

### 12.1 Deployment Scripts 🔴

- [ ] **12.1.1** Viết `scripts/bootstrap.sh`:
  - Cài đặt dependencies (Python, Node, Rust)
  - Tạo database, chạy migrations
  - Download model weights
  - Config environment variables
- [ ] **12.1.2** Viết `scripts/run_backtest.sh`:
  - Arguments: `--symbol`, `--start`, `--end`, `--output`
- [ ] **12.1.3** Viết `scripts/train_models.sh`:
  - Train LSTM → Train XGBoost A → Train XGBoost B → Evaluate → Deploy nếu pass

### 12.2 Runbooks 🟡

- [ ] **12.2.1** Viết `docs/runbooks/deployment_guide.md`:
  - Hướng dẫn cài đặt từ đầu trên Windows và macOS
- [ ] **12.2.2** Viết `docs/runbooks/incident_response.md`:
  - Xử lý khi: Model degraded / Feed failure / Redis down / VectorDB corrupt
- [ ] **12.2.3** Viết `docs/runbooks/model_retraining.md`

### 12.3 Tauri Build Pipeline 🔴

- [ ] **12.3.1** GitHub Actions CI/CD:
  - Build Python binary (PyInstaller) → embed trong Tauri
  - Build React UI
  - `tauri build` cho Windows + macOS
  - Sign với code signing certificate
  - Upload artifacts sang GitHub Releases

---

## TỔNG QUAN THỜI GIAN VÀ PHỤ THUỘC

```
GANTT CHART (ước tính tháng):

Tuần  →  1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16
Phase 0: [===]
Phase 1:      [========]
Phase 2:      [======]
Phase 3:           [========]
Phase 4:                [==========]
Phase 5:                     [=============]
Phase 6:                          [=============]
Phase 7:                               [=======]
Phase 8:                                    [========]
Phase 9:                [=======================]      (frontend, parallel với backend)
Phase 10:                               [======]
Phase 11:                                         [=========]
Phase 12:                                               [=====]

Critical Path: 0 → 1 → 3 → 4 → 5 → 6 → 7 → 11
Parallel tracks: Phase 2 (với 1), Phase 9 (với 5-7), Phase 8 (với 7)
```

---

## CHECKLIST TRƯỚC KHI GO-LIVE

### Technical Readiness
- [ ] Tất cả unit tests pass (coverage ≥ 70%)
- [ ] Tất cả integration tests pass
- [ ] Latency E2E < 50ms được xác nhận
- [ ] Backtest IC > 0.05 trên 6 tháng out-of-sample
- [ ] Memory usage < 2GB sustained
- [ ] Cold start < 8 giây

### Operational Readiness
- [ ] Model weights versioned và rollback tested
- [ ] SQLite migrations idempotent
- [ ] Redis eviction behavior tested
- [ ] News guardrails tested với mock NFP event
- [ ] Crash recovery tested (Python crash → auto restart)
- [ ] Port conflict handling tested

### Documentation
- [ ] API/interface documentation hoàn chỉnh
- [ ] Deployment guide tested trên clean machine
- [ ] Incident response runbook reviewed

---

## GHI CHÚ VỀ THỨ TỰ ƯU TIÊN

**Đề xuất thứ tự xây dựng MVP (Minimum Viable Product — 8 tuần đầu):**

```
MVP Scope:
  [✓] Phase 0 — Infrastructure (bắt buộc)
  [✓] Phase 1 — Module 1 không có backtesting
  [✓] Phase 2 — Module 2 countdown cơ bản
  [✓] Phase 3 — Module 4 chỉ Redis + SQLite (không cần VectorDB)
  [✓] Phase 4 — Feature Engineering cơ bản (FVG/OB/Swing Points)
  [✓] Phase 6 — XGBoost Models (không cần LSTM trong MVP)
  [✓] Phase 9 — UI chart với Ghost Zones và basic heatmap
  [✓] Phase 10 — IPC cơ bản

  Bỏ qua cho MVP:
  [ ] LSTM Autoencoder (Phase 5) — dùng hand-crafted features thay z
  [ ] Multi-Agent Debate (Phase 7) — dùng rule-based fallback
  [ ] Vector DB (Phase 3.5, 3.6) — skip RAG cho MVP
  [ ] Backtesting full suite (Phase 8) — chỉ basic IC tracking
```

---

*Tài liệu này là living document — cập nhật sau mỗi sprint khi task hoàn thành hoặc scope thay đổi.*

**Tổng số tasks:** ~220 tasks  
**Ước tính tổng thời gian (1 developer):** 16–20 tuần  
**Ước tính tổng thời gian (2 developers):** 10–12 tuần  
**MVP (core features):** 8–10 tuần với 1 developer