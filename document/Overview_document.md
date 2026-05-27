# TÀI LIỆU ĐẶC TẢ KỸ THUẬT & KIẾN TRÚC HỆ THỐNG
# HYBRID AGENTIC QUANTITATIVE TRADING FRAMEWORK
# AGENTIC-QUANT — Phiên bản 1.0 (Production-Grade)

---

> **Phạm vi tài liệu:** Đây là tài liệu đặc tả kiến trúc hệ thống dành cho đội ngũ kỹ thuật. Toàn bộ nội dung được viết bằng tiếng Việt theo yêu cầu. Tài liệu không chứa mã nguồn thực thi — tất cả logic được biểu diễn qua mã giả (pseudo-code), sơ đồ luồng ngôn ngữ tự nhiên và công thức toán học.

---

## PHẦN I — SƠ ĐỒ CÂY THƯ MỤC TỔNG QUAN DỰ ÁN

```
agentic-quant/
│
├── docs/                                  ← Tài liệu kỹ thuật (tài liệu này)
│   ├── architecture/
│   │   ├── system_overview.md
│   │   ├── module_contracts.md
│   │   └── data_flow_diagrams.md
│   └── runbooks/
│       ├── deployment_guide.md
│       └── incident_response.md
│
├── core/                                  ← Nhân xử lý Python (backend)
│   ├── ingestion/                         ← MODULE 1
│   │   ├── tick_receiver.py
│   │   ├── ohlcv_aggregator.py
│   │   ├── mtf_synchronizer.py
│   │   └── volumetrics_engine.py
│   │
│   ├── macro/                             ← MODULE 2
│   │   ├── calendar_scraper.py
│   │   ├── news_vectorizer.py
│   │   ├── volatility_countdown.py
│   │   └── regime_classifier.py
│   │
│   ├── ai_engine/                         ← MODULE 3
│   │   ├── feature_engineering/
│   │   │   ├── smc_detector.py
│   │   │   ├── ict_structure_mapper.py
│   │   │   ├── fvg_ob_scanner.py
│   │   │   └── liquidity_pool_indexer.py
│   │   ├── neural/
│   │   │   ├── hierarchical_lstm_ae.py
│   │   │   ├── model_a_liquidity_predictor.py
│   │   │   └── model_b_holdzone_classifier.py
│   │   └── multi_agent/
│   │       ├── bull_agent.py
│   │       ├── bear_agent.py
│   │       ├── critic_agent.py
│   │       └── debate_orchestrator.py
│   │
│   ├── memory/                            ← MODULE 4
│   │   ├── short_term/
│   │   │   ├── redis_cache_manager.py
│   │   │   └── active_zone_registry.py
│   │   └── long_term/
│   │       ├── sqlite_history_store.py
│   │       ├── vectordb_adapter.py
│   │       └── rag_retriever.py
│   │
│   ├── backtesting/                       ← MODULE 5
│   │   ├── event_driven_simulator.py
│   │   ├── historical_tick_loader.py
│   │   ├── drift_tracker.py
│   │   └── regime_shift_detector.py
│   │
│   └── ipc/                               ← MODULE 7 (backend side)
│       ├── websocket_server.py
│       ├── message_schema.py
│       └── broadcast_dispatcher.py
│
├── ui/                                    ← Frontend React + TradingView (MODULE 6)
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChartCanvas/
│   │   │   │   ├── MTFGhostZones.tsx
│   │   │   │   ├── AIHeatmapOverlay.tsx
│   │   │   │   └── MacroTimeline.tsx
│   │   │   ├── Sidebar/
│   │   │   │   ├── AgentDebatePanel.tsx
│   │   │   │   ├── NewsCountdownWidget.tsx
│   │   │   │   └── KillzoneIndicator.tsx
│   │   │   └── Dashboard/
│   │   │       ├── ModelConfidenceGauge.tsx
│   │   │       └── LiquidityTargetBar.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useChartState.ts
│   │   │   └── useMTFSync.ts
│   │   ├── store/
│   │   │   ├── chartSlice.ts
│   │   │   ├── aiStateSlice.ts
│   │   │   └── macroSlice.ts
│   │   └── workers/
│   │       └── chartRenderWorker.ts
│   └── package.json
│
├── desktop/                               ← MODULE 7 (Tauri shell)
│   ├── src-tauri/
│   │   ├── tauri.conf.json
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── ipc_bridge.rs
│   │   │   └── system_tray.rs
│   │   └── Cargo.toml
│   └── build_scripts/
│
├── models/                                ← Artifact lưu trữ mô hình học máy
│   ├── model_a_liquidity/
│   │   ├── weights/
│   │   ├── feature_scaler.pkl
│   │   └── model_config.yaml
│   └── model_b_holdzone/
│       ├── weights/
│       ├── feature_scaler.pkl
│       └── model_config.yaml
│
├── data/                                  ← Lưu trữ dữ liệu cục bộ
│   ├── historical_ticks/
│   │   └── XAUUSD/
│   │       ├── 2023/
│   │       └── 2024/
│   ├── economic_calendar/
│   │   ├── raw_cache/
│   │   └── vectorized/
│   └── sqlite/
│       └── agentic_quant.db
│
├── vectordb/                              ← Cơ sở dữ liệu vector (ChromaDB / Qdrant)
│   ├── debate_archive/
│   └── zone_embeddings/
│
├── config/
│   ├── system.yaml                        ← Tham số toàn hệ thống
│   ├── model_params.yaml
│   ├── killzones.yaml
│   └── news_weights.yaml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── backtest_scenarios/
│
└── scripts/
    ├── bootstrap.sh
    ├── train_models.sh
    └── run_backtest.sh
```

---

## PHẦN II — TRIẾT LÝ VÀ CÁC NGUYÊN TẮC THIẾT KẾ TOÀN HỆ THỐNG

### II.1 Nguyên tắc Phân tầng dữ liệu (Data Layering Principle)

Hệ thống AGENTIC-QUANT tuân thủ một nguyên tắc bất di bất dịch: **không có module nào ở tầng dưới được phép truy cập trực tiếp trạng thái nội tâm của module ở tầng trên**. Luồng dữ liệu chỉ đi theo một chiều xác định, được ký hiệu như sau:

```
[MODULE 1: Ingestion] → [MODULE 2: Macro] → [MODULE 3: AI Engine]
                                                      ↓
[MODULE 6: Visualization] ← [MODULE 7: IPC] ← [MODULE 4: Memory]
                                                      ↑
                              [MODULE 5: Backtesting] ↗
```

Mỗi lần truyền dữ liệu giữa các module đều thông qua một **Hợp đồng Giao diện (Interface Contract)** dạng bất biến (immutable), được định nghĩa rõ ràng tại `docs/architecture/module_contracts.md`.

### II.2 Bất biến Chống Nhìn Trước Tương Lai (Look-Ahead Bias Invariant)

Đây là ràng buộc kỹ thuật tuyệt đối: tại thời điểm xử lý thanh nến $t$, hệ thống **chỉ được phép** sử dụng thông tin thuộc tập $\{D_{t-n}, D_{t-n+1}, ..., D_{t-1}, D_t\}$. Không một giá trị nào thuộc $\{D_{t+1}, D_{t+2}, ...\}$ được phép tồn tại trong bất kỳ luồng tính toán nào. Cơ chế thực thi bất biến này được mô tả chi tiết trong Module 1.

### II.3 Kiến trúc Sự kiện (Event-Driven Architecture)

Toàn bộ hệ thống vận hành theo mô hình **Event Loop bất đồng bộ**. Mỗi tick giá mới đến là một sự kiện (Event) độc lập. Hệ thống không sử dụng vòng lặp polling định kỳ (polling loop) cho dữ liệu thời gian thực. Thay vào đó, một **Bộ phân phối sự kiện trung tâm (Central Event Bus)** điều phối việc truyền tải sự kiện đến các subscriber tương ứng với độ trễ nội bộ mục tiêu dưới 5ms.

---

## PHẦN III — ĐẶC TẢ CHI TIẾT TỪNG MODULE

---

# MODULE 1: BỘ ĐÓN NHẬN & ĐỒNG BỘ DỮ LIỆU ĐA KHUNG THỜI GIAN
## (Data Ingestion & Multi-Timeframe Synchronization Engine)

---

### A. TỔNG QUAN KỸ THUẬT & MỤC TIÊU

**Bài toán Module 1 giải quyết:**

Dữ liệu thị trường tài chính đến ở nhiều dạng và tần suất khác nhau — từ dữ liệu Tick (có thể 10–50 lần/giây với các cặp tiền tệ thanh khoản cao) đến nến OHLCV (Open-High-Low-Close-Volume) trên các khung thời gian từ M1 đến D1. Thách thức kỹ thuật cốt lõi là:

1. **Bài toán Đồng bộ bất đồng bộ (Asynchronous Alignment):** Nến H4 đang hình thành không thể bị "đóng" trước khi 240 nến M1 tương ứng hoàn thành. Hệ thống phải duy trì trạng thái "đang hình thành" của tất cả khung thời gian một cách nhất quán.

2. **Bài toán Chống rò rỉ dữ liệu (Data Leakage Prevention):** Trong backtesting, một sai lầm phổ biến là vô tình sử dụng giá đóng cửa của nến hiện tại để đưa ra quyết định tại điểm mở cửa của nến đó. Module này phải có cơ chế khóa cứng (hard lock) để ngăn chặn điều này.

3. **Bài toán Dấu chân tổ chức (Institutional Footprint Detection):** Dữ liệu Tick đơn thuần không đủ để phân biệt lệnh mua/bán. Module này phải kết hợp dữ liệu Bid/Ask và Depth of Market để suy luận ra áp lực mua/bán thực sự.

**Giao tiếp với các module khác:**

- **Đầu ra → Module 2 (Macro Engine):** Cung cấp Vector trạng thái thống nhất (Unified State Vector — USV) mỗi khi một nến M1 mới đóng cửa, hoặc khi nhận được Tick có biến động đủ lớn (vượt ngưỡng $\Delta P_{threshold}$).
- **Đầu ra → Module 3 (AI Engine):** Cung cấp chuỗi OHLCV đã được chuẩn hóa theo khung thời gian và mảng dữ liệu Volumetrics.
- **Đầu ra → Module 4 (Memory Engine):** Ghi nhật ký raw tick vào bộ nhớ đệm ngắn hạn để phục vụ tính toán nội tuyến (inline computation).
- **Nhận vào ← Module 2:** Nhận tín hiệu "Chế độ Tin tức Sắp xảy ra" để tăng tần suất sampling tick trong cửa sổ [-15 phút, +15 phút] quanh sự kiện.

---

### B. THUẬT TOÁN CỐT LÕI & LOGIC TOÁN HỌC

#### B.1 Cơ chế Thu thập Tick (Tick Ingestion Pipeline)

Dữ liệu đến từ hai nguồn song song:

**Nguồn 1 — MetaTrader 5 (MT5) qua ZeroMQ Bridge:**

```
LUỒNG LOGIC NHẬN TICK TỪ MT5:
─────────────────────────────────────────────────────────────
[1] Mở kết nối ZeroMQ PULL socket trên cổng 5556
[2] Trong vòng lặp sự kiện bất tận:
    [2.1] Chờ nhận frame dữ liệu nhị phân từ MT5 Expert Advisor
    [2.2] Giải mã (deserialize) frame theo cấu trúc TickFrame:
          TickFrame = {
              symbol     : string[8]     (ví dụ: "XAUUSD")
              timestamp  : int64         (Unix microseconds)
              bid        : float64
              ask        : float64
              last       : float64
              volume     : float64       (tick volume)
              flags      : uint8         (bitmask: bit0=buy, bit1=sell, bit2=dom_update)
          }
    [2.3] Kiểm tra tính hợp lệ:
          NẾU (ask - bid) > spread_threshold_pips × pip_size THÌ
              đánh dấu tick là ABNORMAL_SPREAD, ghi log, không đẩy vào pipeline
    [2.4] Đẩy tick hợp lệ vào hàng đợi nội bộ (Internal Tick Queue — ITQ)
─────────────────────────────────────────────────────────────
```

**Nguồn 2 — TradingView Webhook qua HTTP POST:**

```
LUỒNG LOGIC NHẬN ALERT TỪ TRADINGVIEW:
─────────────────────────────────────────────────────────────
[1] HTTP Server lắng nghe POST /webhook/tv tại cổng 8080
[2] Khi nhận request:
    [2.1] Xác thực HMAC-SHA256 signature từ header X-TV-Signature
    [2.2] Phân tích (parse) JSON payload theo cấu trúc TVAlert:
          TVAlert = {
              symbol      : string
              timeframe   : string       ("1", "5", "15", "60", "240", "D")
              bar_time    : int64        (Unix timestamp của nến vừa đóng)
              open        : float64
              high        : float64
              low         : float64
              close       : float64
              volume      : float64
              alert_type  : string       ("bar_close" | "condition_met")
          }
    [2.3] Chuyển đổi TVAlert → chuẩn hóa sang OHLCVRecord
    [2.4] Đẩy vào hàng đợi OHLCV tương ứng theo khung thời gian
─────────────────────────────────────────────────────────────
```

#### B.2 Bộ Tổng hợp OHLCV Nội bộ (Internal OHLCV Aggregator)

Từ dữ liệu Tick thô, hệ thống tự tổng hợp nến OHLCV cho tất cả khung thời gian bằng thuật toán sau:

**Định nghĩa khung thời gian (Timeframe Registry):**

```
TF_REGISTRY = {
    "M1"  : { seconds: 60,     dependency: None         },
    "M5"  : { seconds: 300,    dependency: "M1"         },
    "M15" : { seconds: 900,    dependency: "M5"         },
    "H1"  : { seconds: 3600,   dependency: "M15"        },
    "H4"  : { seconds: 14400,  dependency: "H1"         },
    "D1"  : { seconds: 86400,  dependency: "H4"         }
}
```

**Thuật toán Aggregation tuần tự theo phân cấp:**

```
THUẬT TOÁN TỔNG HỢP NẾN ĐA KHUNG THỜI GIAN:
─────────────────────────────────────────────────────────────
Đầu vào  : Tick mới (tick_t)
Đầu ra   : Danh sách các nến OHLCV đã cập nhật/đóng

[1] Tính bucket_time cho M1:
    m1_bucket = floor(tick_t.timestamp / 60) × 60

[2] Tra cứu nến M1 đang mở (nến "đang hình thành") trong bảng ActiveBars:
    NẾU m1_bucket không tồn tại trong ActiveBars THÌ
        Tạo nến M1 mới:
            open  = tick_t.last
            high  = tick_t.last
            low   = tick_t.last
            close = tick_t.last
            volume = tick_t.volume
            bar_open_time = m1_bucket
            status = "FORMING"
    NGƯỢC LẠI (nến M1 đã tồn tại):
        Cập nhật:
            high  = max(existing_high, tick_t.last)
            low   = min(existing_low, tick_t.last)
            close = tick_t.last
            volume += tick_t.volume

[3] Kiểm tra đóng cửa nến M1:
    current_time = now()
    NẾU current_time >= m1_bucket + 60 THÌ
        Đánh dấu nến M1 là CLOSED
        Phát sự kiện BAR_CLOSE(M1, bar_data)
        Kích hoạt Bước [4] cho M5, M15, H1, H4, D1

[4] Lan truyền đóng cửa lên khung thời gian cao hơn (Cascade Closure):
    VỚI MỖI tf trong ["M5", "M15", "H1", "H4", "D1"]:
        bucket = floor(tick_t.timestamp / TF_REGISTRY[tf].seconds)
                 × TF_REGISTRY[tf].seconds
        
        NẾU bucket chưa tồn tại trong ActiveBars[tf] THÌ
            Tạo nến mới với open = giá đóng nến M1 trước đó
        
        Cập nhật high, low, close, volume từ nến M1 mới đóng
        
        NẾU current_time >= bucket + TF_REGISTRY[tf].seconds THÌ
            Đánh dấu CLOSED
            Phát sự kiện BAR_CLOSE(tf, bar_data)
─────────────────────────────────────────────────────────────
```

#### B.3 Bộ Đồng Bộ Đa Khung Thời Gian — MTF Synchronizer

**Bài toán:** Tại thời điểm xử lý tick mới, hệ thống AI cần truy cập đồng thời trạng thái của tất cả 6 khung thời gian. Tuy nhiên, mỗi khung thời gian có chu kỳ cập nhật khác nhau. Đây là bài toán **đồng bộ hóa dị bộ (asynchronous synchronization)**.

**Cấu trúc dữ liệu: Unified State Vector (USV)**

```
UnifiedStateVector = {
    snapshot_time    : int64           ← Timestamp của tick kích hoạt
    bars             : {
        "M1"  : BarState,
        "M5"  : BarState,
        "M15" : BarState,
        "H1"  : BarState,
        "H4"  : BarState,
        "D1"  : BarState
    },
    tick_context     : TickContext,
    volumetrics      : VolumetricsState,
    leakage_guard    : LeakageGuardFlags
}

BarState = {
    bars_closed      : List[OHLCVRecord]   ← N nến đã đóng gần nhất (N = lookback)
    bar_forming      : OHLCVRecord         ← Nến đang hình thành (CHƯA đóng)
    last_close_time  : int64
}

LeakageGuardFlags = {
    backtest_mode    : bool
    forward_locked   : Set[string]         ← Tập khung TF bị khóa tương lai
}
```

**Cơ chế Chống Nhìn Trước Tương Lai (Look-Ahead Bias Guard):**

```
THUẬT TOÁN LEAKAGE GUARD:
─────────────────────────────────────────────────────────────
Trong chế độ BACKTEST (backtest_mode = True):

[1] Khi hệ thống xử lý bar tại timestamp T trên khung M1:

[2] Xác định thanh nến "cha" trên mỗi khung thời gian cao hơn:
    VỚI MỖI tf trong ["M5", "M15", "H1", "H4", "D1"]:
        parent_bar_close_time = floor(T / TF_REGISTRY[tf].seconds)
                                × TF_REGISTRY[tf].seconds
                                + TF_REGISTRY[tf].seconds
        
        NẾU T < parent_bar_close_time THÌ
            Đánh dấu tf vào forward_locked
            → Khi AI Engine truy vấn BarState của tf này,
              CHỈ được phép trả về bar_forming với close = None
              (chưa biết giá đóng cửa)

[3] Kết quả: Tại mọi thời điểm T trong backtest,
    AI chỉ có thể thấy giá đóng cửa của nến đã THỰC SỰ đóng trước T.
─────────────────────────────────────────────────────────────
```

#### B.4 Đo Lường Sổ Lệnh & Khối Lượng (Volumetrics Engine)

**Mục tiêu:** Trích xuất tín hiệu ẩn về hành vi của các tổ chức tài chính lớn (Smart Money) từ dòng chảy lệnh.

**Chỉ số 1 — Áp lực mua/bán Tick (Tick Delta):**

Mỗi tick được phân loại thành "lệnh mua chủ động" hoặc "lệnh bán chủ động" dựa trên quy tắc:

```
NẾU tick.last >= tick.ask THÌ
    delta = +tick.volume    ← Mua chủ động (Buy aggressor)
NGƯỢC LẠI NẾU tick.last <= tick.bid THÌ
    delta = -tick.volume    ← Bán chủ động (Sell aggressor)
NGƯỢC LẠI
    delta = 0               ← Giao dịch giữa spread (Mid-spread)
```

**Chỉ số 2 — Tích lũy Delta theo nến (Cumulative Volume Delta — CVD):**

$$CVD_t = \sum_{i=1}^{N_t} \delta_i$$

Trong đó $\delta_i$ là delta của tick thứ $i$ trong nến thứ $t$, và $N_t$ là tổng số tick trong nến đó.

**Chỉ số 3 — Sự mất cân bằng sổ lệnh (Order Book Imbalance — OBI):**

Khi có dữ liệu DOM (Depth of Market / Level 2), tính:

$$OBI = \frac{V_{bid,1} - V_{ask,1}}{V_{bid,1} + V_{ask,1}}$$

Trong đó $V_{bid,1}$ và $V_{ask,1}$ là khối lượng đang chờ ở mức giá bid tốt nhất và ask tốt nhất tương ứng. Giá trị $OBI \in [-1, +1]$. $OBI > +0.3$ báo hiệu áp lực mua mạnh từ phía tổ chức; $OBI < -0.3$ báo hiệu áp lực bán.

**Chỉ số 4 — Chỉ số Cường độ Tổ chức (Institutional Intensity Index — III):**

$$III_t = \frac{CVD_t}{\bar{V}_{30}} \times \frac{|\Delta P_t|}{\sigma_{ATR,14}}$$

Trong đó:
- $\bar{V}_{30}$ là khối lượng trung bình của 30 nến gần nhất (cùng khung thời gian)
- $|\Delta P_t|$ là biên độ giá của nến $t$ (tức là $|close_t - open_t|$)
- $\sigma_{ATR,14}$ là Average True Range 14 kỳ

$III > 1.5$ được coi là tín hiệu mạnh của sự tham gia tổ chức.

---

### C. PHÂN RÃ TÍNH NĂNG CHI TIẾT

#### Tính năng C.1 — Thu thập Tick thời gian thực

- **Đầu vào:** Kết nối ZeroMQ socket đến MT5 EA; hoặc WebSocket stream từ broker API.
- **Biến đổi:** Giải mã nhị phân → Kiểm tra spread → Phân loại buy/sell aggressor → Cập nhật CVD nội bộ.
- **Đầu ra:** Đối tượng `EnrichedTick` = TickFrame + delta + spread_flag, đẩy vào ITQ.

#### Tính năng C.2 — Tổng hợp nến OHLCV đa khung thời gian

- **Đầu vào:** Chuỗi `EnrichedTick` liên tục từ ITQ.
- **Biến đổi:** Phân bucket theo timestamp → Cập nhật OHLCV → Phát hiện đóng cửa nến → Lan truyền cascade lên khung cao.
- **Đầu ra:** Sự kiện `BAR_CLOSE(timeframe, OHLCVRecord)` được phát lên Event Bus.

#### Tính năng C.3 — Xây dựng Unified State Vector (USV)

- **Đầu vào:** Sự kiện `BAR_CLOSE(M1, ...)` mới nhất + toàn bộ `ActiveBars` hiện tại.
- **Biến đổi:** Thu thập BarState cho cả 6 khung thời gian → Áp dụng LeakageGuard → Đóng gói thành USV → Đính kèm VolumetricsState.
- **Đầu ra:** `UnifiedStateVector` hoàn chỉnh, đẩy vào hàng đợi xử lý AI.

#### Tính năng C.4 — Tính chỉ số Volumetrics

- **Đầu vào:** CVD của nến hiện tại; 30 nến OHLCV gần nhất; dữ liệu DOM nếu có.
- **Biến đổi:** Tính $CVD_t$, $OBI$, $III_t$ theo công thức đã định nghĩa.
- **Đầu ra:** Đối tượng `VolumetricsState = {cvd, obi, iii, buy_volume, sell_volume}`.

#### Tính năng C.5 — Tích hợp lịch sử dữ liệu Parquet (Backtest Mode)

- **Đầu vào:** File tick lịch sử định dạng Parquet (được tổ chức theo symbol/năm/tháng).
- **Biến đổi:** Đọc tuần tự từng hàng → Giả lập tick theo timestamp → Chạy toàn bộ pipeline tổng hợp OHLCV như chế độ live.
- **Đầu ra:** Chuỗi `UnifiedStateVector` lịch sử có đầy đủ LeakageGuard kích hoạt.

---

### D. CƠ CHẾ PHÒNG VỆ & XỬ LÝ CA BIÊN

#### D.1 — Mất kết nối ZeroMQ / WebSocket (Connection Loss)

```
TRẠNG THÁI: Không nhận được tick trong N giây (N có thể cấu hình, mặc định = 30 giây)

HÀNH ĐỘNG:
[1] Nâng cờ STALENESS_ALERT trong USV → Module 3 hạ thấp độ tin cậy xuống 50%
[2] Thử kết nối lại (reconnect) với back-off hàm mũ:
    Lần 1: chờ 1s → Lần 2: chờ 2s → Lần 3: chờ 4s → ... → tối đa 60s
[3] NẾU tái kết nối thành công:
    a) Yêu cầu broker gửi lại snapshot OHLCV đầy đủ của nến đang hình thành
    b) Tính lại CVD từ điểm tick cuối cùng được ghi nhận
    c) Đặt lại cờ STALENESS_ALERT
[4] NẾU không tái kết nối được sau 5 lần:
    Phát sự kiện CRITICAL_FEED_FAILURE → Kích hoạt chế độ "Chỉ quan sát"
    (Hệ thống hiển thị cảnh báo đỏ trên UI, ngừng phát tín hiệu mới)
```

#### D.2 — Giá giật mạnh bất thường khi ra tin (News Spike / Price Gap)

```
PHÁT HIỆN: |tick_t.last - tick_{t-1}.last| > 3 × ATR_5min_current

HÀNH ĐỘNG:
[1] Đánh dấu tick là NEWS_SPIKE
[2] Không cập nhật CVD (spike volume thường méo mó, không đại diện)
[3] Ghi log spike với biên độ và timestamp
[4] Đặt cờ SPIKE_REGIME = True trong USV trong cửa sổ 5 phút tiếp theo
[5] Module 3 sẽ đọc cờ này để vô hiệu hóa tín hiệu FVG/OB hình thành ngay 
    trước và sau spike (vì các vùng này thường bị "fill" rất nhanh trong spike)
```

#### D.3 — Bất đối xứng dữ liệu giữa khung thời gian (Timeframe Desync)

```
PHÁT HIỆN: Nến H1 đóng cửa nhưng trong cửa sổ [-60min, 0] không tồn tại 
           đủ 60 nến M1 hợp lệ (do khoảng trống dữ liệu)

HÀNH ĐỘNG:
[1] Đánh dấu BarState(H1) là INCOMPLETE_AGGREGATION
[2] Tính số nến M1 bị thiếu: gap_count = 60 - actual_m1_count
[3] NẾU gap_count <= 5 (ít hơn 5 nến M1 thiếu):
    Nội suy tuyến tính bằng last known price cho các khoảng trống
    Đánh dấu nội suy = True trong metadata
[4] NẾU gap_count > 5:
    Đánh dấu H1 bar là UNRELIABLE
    Module 3 sẽ loại bỏ H1 khỏi phân tích, chỉ dùng HTF từ H4 trở lên
```

#### D.4 — Hàng đợi ITQ bị đầy (Queue Overflow)

```
PHÁT HIỆN: len(ITQ) > ITQ_MAX_SIZE (mặc định = 10,000 tick)

HÀNH ĐỘNG:
[1] Kích hoạt chế độ Tick Sampling — chỉ xử lý 1 trong mỗi K tick
    K được tính động: K = ceil(len(ITQ) / ITQ_TARGET_SIZE)
    ITQ_TARGET_SIZE mặc định = 1,000
[2] Ghi metric TICK_SAMPLING_ACTIVE = True vào Prometheus endpoint
[3] Phát cảnh báo LOW_LATENCY_DEGRADATION lên Module 7 (IPC) để 
    hiển thị indicator chậm trên UI
[4] Khi len(ITQ) < ITQ_TARGET_SIZE: hủy bỏ sampling, quay lại xử lý đầy đủ
```

---

# MODULE 2: BỘ XỬ LÝ LỊCH KINH TẾ VĨ MÔ
## (Macro Calendar & News Volatility Engine)

---

### A. TỔNG QUAN KỸ THUẬT & MỤC TIÊU

**Bài toán Module 2 giải quyết:**

Các sự kiện tin tức kinh tế vĩ mô (CPI, FOMC, NFP, GDP, v.v.) là những chất xúc tác tạo ra sự dịch chuyển thanh khoản khổng lồ và có tính định hướng cao. Hệ thống AI thuần túy dựa trên phân tích giá sẽ thất bại hoàn toàn nếu không tích hợp được chiều thông tin này. Module 2 đảm nhận:

1. **Thu thập và chuẩn hóa lịch kinh tế** từ nhiều nguồn công khai.
2. **Vector hóa tác động tiềm năng** của từng sự kiện tin tức thành giá trị số dùng được bởi Module 3.
3. **Duy trì đồng hồ đếm ngược thời gian thực** đến từng sự kiện sắp xảy ra và phát tín hiệu thay đổi trạng thái sang toàn hệ thống.
4. **Phân loại chế độ thị trường hậu tin tức** — xác định xem tin tức đã tạo ra sự dịch chuyển một chiều hay chỉ là sự kiện "mua tin bán sự kiện" (Buy the rumor, sell the news).

**Giao tiếp với các module khác:**

- **Nhận vào ← Module 1:** Không có luồng dữ liệu trực tiếp từ Module 1; nhưng Module 2 sử dụng clock thời gian thực chung.
- **Đầu ra → Module 3 (AI Engine):** Đối tượng `MacroContext` chứa vector hóa tác động tin tức, trạng thái đếm ngược, và cờ chế độ thị trường.
- **Đầu ra → Module 6 (Visualization):** Dữ liệu sự kiện tin tức để vẽ đường dọc (vertical bar) trên đồ thị.
- **Đầu ra → Module 4 (Memory):** Lưu lịch sử kết quả tin tức thực tế so với dự báo.

---

### B. THUẬT TOÁN CỐT LÕI & LOGIC TOÁN HỌC

#### B.1 Thu Thập Lịch Kinh Tế (Calendar Scraping)

**Nguồn dữ liệu:** ForexFactory Calendar API (JSON endpoint không chính thức) và Investing.com Economic Calendar.

**Chu kỳ làm mới:** Mỗi 6 giờ một lần để cập nhật lịch tuần; nhưng riêng 30 phút trước mỗi sự kiện thì cập nhật mỗi 5 phút để bắt các thay đổi giờ chót.

**Cấu trúc dữ liệu sự kiện tin tức thô:**

```
RawNewsEvent = {
    event_id        : string       ← ID định danh duy nhất
    currency        : string       ← "USD", "EUR", "GBP", ...
    event_name      : string       ← "Core CPI m/m", "FOMC Statement", ...
    scheduled_time  : datetime     ← Thời gian dự kiến (UTC)
    impact          : string       ← "Low" | "Medium" | "High"
    forecast        : float | None ← Dự báo thị trường
    previous        : float | None ← Kết quả kỳ trước
    actual          : float | None ← Kết quả thực tế (None nếu chưa công bố)
    unit            : string       ← "%", "K", "B", ...
}
```

#### B.2 Vector Hóa Tác Động Tin Tức (News Impact Vectorization)

**Bước 1 — Mã hóa mức độ tác động:**

```
impact_base_score = {
    "Low"    : 0.2,
    "Medium" : 0.5,
    "High"   : 1.0
}
```

**Bước 2 — Điều chỉnh theo lịch sử biến động hậu sự kiện:**

Với mỗi loại sự kiện $e$ và đồng tiền $c$, hệ thống duy trì một bảng lịch sử $H_{e,c}$ lưu biên độ giá trung bình trong 15 phút sau khi sự kiện đó xảy ra (tính bằng pip):

$$\bar{M}_{e,c} = \frac{1}{|H_{e,c}|} \sum_{h \in H_{e,c}} M_h$$

**Bước 3 — Tính Vector tác động tin tức $I_{news}$:**

$$I_{news} = impact\_base\_score \times \left(1 + \alpha \cdot \frac{\bar{M}_{e,c}}{ATR_{D1}}\right)$$

Trong đó:
- $\alpha = 0.4$ là hệ số pha trộn lịch sử (có thể điều chỉnh trong `config/news_weights.yaml`)
- $ATR_{D1}$ là ATR 14 kỳ trên khung D1 (chuẩn hóa biên độ theo ngữ cảnh thị trường hiện tại)

**Bước 4 — Tính Sai lệch Bất ngờ (Surprise Factor) khi có kết quả thực tế:**

Ngay khi `actual` được công bố:

$$S = \frac{actual - forecast}{\sigma_{surprise,e,c}}$$

Trong đó $\sigma_{surprise,e,c}$ là độ lệch chuẩn lịch sử của (actual - forecast) cho sự kiện loại $e$, đồng tiền $c$.

Nếu $|S| > 2.0$, đây là một "Bất ngờ Lớn" (Major Surprise) và toàn hệ thống nhận tín hiệu `MAJOR_SURPRISE_FLAG`.

Hướng bất ngờ: $S > 0$ nghĩa là kết quả tốt hơn dự báo (bullish đối với đồng tiền đó); $S < 0$ ngược lại.

#### B.3 Bộ Đếm Ngược Biến Động Vĩ Mô (Volatility Countdown Timer)

**Cấu trúc trạng thái đếm ngược:**

```
CountdownState = {
    events_upcoming   : List[TimedEvent]   ← Sắp xếp theo scheduled_time tăng dần
    next_event        : TimedEvent | None
    seconds_to_next   : int
    regime_phase      : string             ← "NORMAL" | "PRE_NEWS" | "NEWS_WINDOW" | "POST_NEWS"
    active_guardrail  : bool
}

TimedEvent = {
    event_ref         : RawNewsEvent
    impact_vector     : float              ← I_news đã tính
    countdown_started : bool
    guardrail_trigger_time : datetime      ← scheduled_time - 15 phút
}
```

**Vòng lặp Đếm ngược Chính (Main Countdown Loop):**

```
THUẬT TOÁN VÒNG LẶP ĐẾM NGƯỢC TIN TỨC:
─────────────────────────────────────────────────────────────
Chạy mỗi 1 giây:

[1] Lấy current_time = now()
[2] Cập nhật seconds_to_next = next_event.scheduled_time - current_time

[3] PHÂN LOẠI GIAI ĐOẠN (regime_phase):
    NẾU seconds_to_next > 900 (15 phút):
        regime_phase = "NORMAL"
        active_guardrail = False
    
    NẾU 0 < seconds_to_next <= 900:
        regime_phase = "PRE_NEWS"
        active_guardrail = True
        Phát tín hiệu PRE_NEWS_ALERT đến Module 3 với:
            - impact_vector = I_news của next_event
            - seconds_remaining = seconds_to_next
    
    NẾU -300 <= seconds_to_next <= 0 (trong 5 phút sau khi tin ra):
        regime_phase = "NEWS_WINDOW"
        active_guardrail = True
        NẾU actual vừa được công bố:
            Tính S (Surprise Factor)
            Phát NEWS_RELEASE_EVENT với {I_news, S, direction}
    
    NẾU -1800 < seconds_to_next < -300 (5-30 phút sau tin):
        regime_phase = "POST_NEWS"
        active_guardrail = False
        Chạy POST_NEWS_CLASSIFICATION (xem B.4)

[4] Cập nhật CountdownState và đẩy lên Event Bus
─────────────────────────────────────────────────────────────
```

#### B.4 Phân Loại Chế Độ Hậu Tin Tức (Post-News Regime Classifier)

Sau 5 phút kể từ khi tin được công bố, hệ thống phân tích hành vi giá để phân loại chế độ:

```
POST_NEWS_CLASSIFICATION:
─────────────────────────────────────────────────────────────
Đầu vào: 
    - price_at_news_release = P0
    - price_current = P_now (5 phút sau)
    - surprise_direction = sign(S)
    - 5 nến M1 sau khi tin ra

Tính:
    directional_move = (P_now - P0) / ATR_H1

Phân loại:
    NẾU sign(directional_move) == surprise_direction
       VÀ |directional_move| > 0.5:
        → POST_REGIME = "IMPULSIVE_FOLLOW_THROUGH"
          (Tin tức mạnh, giá đi đúng hướng bất ngờ — 
           bắt đầu tìm kiếm thanh khoản HTF)
    
    NẾU sign(directional_move) != surprise_direction:
        → POST_REGIME = "REVERSAL_AFTER_SPIKE"
          (Hiện tượng "buy the rumor sell the news" —
           thận trọng, chờ xác nhận)
    
    NGƯỢC LẠI:
        → POST_REGIME = "CHOPPY_CONSOLIDATION"
          (Tin tức không tạo ra xung lực rõ ràng —
           hệ thống duy trì phân tích kỹ thuật bình thường)
─────────────────────────────────────────────────────────────
```

---

### C. PHÂN RÃ TÍNH NĂNG CHI TIẾT

#### Tính năng C.1 — Thu thập và lưu trữ lịch kinh tế

- **Đầu vào:** HTTP response từ ForexFactory/Investing.com; tham số lọc đồng tiền từ `config/system.yaml`.
- **Biến đổi:** Parse HTML/JSON → chuẩn hóa thành `RawNewsEvent` → lưu vào bảng `economic_calendar` trong SQLite → xây dựng index thời gian.
- **Đầu ra:** Danh sách `RawNewsEvent` của 7 ngày tiếp theo, có thể truy vấn theo symbol giao dịch.

#### Tính năng C.2 — Vector hóa tác động và tính $I_{news}$

- **Đầu vào:** `RawNewsEvent` + bảng lịch sử biến động hậu sự kiện từ SQLite.
- **Biến đổi:** Tra cứu $\bar{M}_{e,c}$ → Tính $I_{news}$ theo công thức → Đính kèm vào `TimedEvent`.
- **Đầu ra:** `TimedEvent` hoàn chỉnh với `impact_vector` hợp lệ.

#### Tính năng C.3 — Bộ đếm ngược thời gian thực

- **Đầu vào:** Đồng hồ hệ thống (NTP-synced); danh sách `TimedEvent` đã sắp xếp.
- **Biến đổi:** Vòng lặp 1 giây → tính `seconds_to_next` → phân loại `regime_phase` → phát sự kiện lên Event Bus.
- **Đầu ra:** `CountdownState` cập nhật mỗi giây; sự kiện chuyển trạng thái phát đến Module 3 và Module 6.

#### Tính năng C.4 — Tính Surprise Factor và phân loại hậu tin tức

- **Đầu vào:** `actual` vừa được scrape; `forecast` đã lưu trước đó; $\sigma_{surprise}$ từ lịch sử.
- **Biến đổi:** Tính $S$ → Phân loại ngưỡng → Phân loại `POST_REGIME` từ dữ liệu giá M1 của Module 1.
- **Đầu ra:** `NewsOutcome = {surprise_factor, direction, post_regime, timestamp_release}` → lưu vào SQLite.

#### Tính năng C.5 — Cập nhật lịch sử biến động hậu sự kiện

- **Đầu vào:** `NewsOutcome` + dữ liệu OHLCV M1 trong 30 phút sau sự kiện từ Module 1.
- **Biến đổi:** Tính biên độ thực tế M15 → Cập nhật $\bar{M}_{e,c}$ trong bảng lịch sử → Hiệu chỉnh $\sigma_{surprise}$ theo thuật toán cập nhật trực tuyến (online update).
- **Đầu ra:** Bảng `news_historical_volatility` trong SQLite được cập nhật, cải thiện độ chính xác của $I_{news}$ theo thời gian.

---

### D. CƠ CHẾ PHÒNG VỆ & XỬ LÝ CA BIÊN

#### D.1 — Không scrape được lịch kinh tế (Scraping Failure)

```
PHÁT HIỆN: HTTP request đến ForexFactory hoặc Investing.com thất bại
           sau 3 lần thử lại (retry) với back-off 2s, 4s, 8s

HÀNH ĐỘNG:
[1] Chuyển sang nguồn dự phòng: thử Investing.com (nếu FF thất bại) 
    hoặc FF (nếu Investing thất bại)
[2] NẾU cả hai thất bại:
    a) Nạp lịch kinh tế từ cache SQLite của ngày hôm qua
    b) Đánh dấu CALENDAR_STALE = True
    c) Hiển thị cảnh báo màu vàng trên UI: "Lịch kinh tế có thể không cập nhật"
[3] Thử lại tự động mỗi 5 phút
```

#### D.2 — Sự kiện tin tức bị thay đổi giờ (Reschedule)

```
PHÁT HIỆN: Khi cập nhật lịch, thời gian của một sự kiện thay đổi hơn 30 phút
           so với lần đọc trước

HÀNH ĐỘNG:
[1] Hủy bỏ trạng thái PRE_NEWS đang kích hoạt (nếu có)
[2] Cập nhật scheduled_time mới
[3] Tính lại guardrail_trigger_time
[4] Phát sự kiện CALENDAR_RESCHEDULE đến Module 6 để cập nhật đường dọc trên đồ thị
[5] Log với mức WARNING: "Event [name] rescheduled from [old_time] to [new_time]"
```

#### D.3 — Sự kiện tin tức bị hủy (Cancellation)

```
PHÁT HIỆN: Sự kiện biến mất khỏi lịch hoặc được đánh dấu là "Cancelled"

HÀNH ĐỘNG:
[1] Xóa khỏi danh sách events_upcoming
[2] NẾU đang trong giai đoạn PRE_NEWS cho sự kiện đó:
    Reset về NORMAL, tắt active_guardrail
[3] Phát sự kiện CALENDAR_CANCELLED đến Module 6
[4] Ghi log với mức INFO
```

#### D.4 — Xung đột lịch (Multiple High-Impact Events trong 30 phút)

```
PHÁT HIỆN: Hai hoặc nhiều sự kiện "High" impact của cùng đồng tiền 
           có scheduled_time cách nhau dưới 30 phút

HÀNH ĐỘNG:
[1] Gộp lại thành một "Cluster Event" duy nhất
[2] I_news(cluster) = max(I_news(e1), I_news(e2)) × 1.25
    (Tác động kép được khuếch đại 25%)
[3] guardrail_trigger_time = min(guardrail_trigger_time(e1), guardrail_trigger_time(e2))
[4] Phát cảnh báo riêng CLUSTER_NEWS_WARNING đến UI
```

---

# MODULE 3: AI ENGINE & TIẾN TRÌNH XỬ LÝ THẦN KINH — KÝ HIỆU ĐA KHUNG THỜI GIAN
## (AI Engine: Feature Engineering, Neural Compression & Multi-Agent Debate)

---

### A. TỔNG QUAN KỸ THUẬT & MỤC TIÊU

Module 3 là trung tâm trí tuệ nhân tạo của toàn hệ thống — nơi mọi dòng dữ liệu hội tụ lại và được biến đổi thành các dự đoán có giá trị hành động (Actionable Predictions). Module này có ba tầng xử lý kế tiếp nhau:

**Tầng 1 — Ký hiệu hoá đặc trưng (Symbolic Feature Engineering):** Trích xuất các cấu trúc SMC/ICT có ý nghĩa ngữ nghĩa cao từ dữ liệu giá thô.

**Tầng 2 — Nén thần kinh (Neural Compression):** Sử dụng Hierarchical LSTM Autoencoder để nén chuỗi thời gian dài thành các vector tiềm ẩn (Latent Vectors), phục vụ hai mô hình dự đoán chuyên biệt.

**Tầng 3 — Tranh biện đa tác nhân GenAI (Multi-Agent Debate):** Triển khai ba AI Agent (Bull, Bear, Critic) được cung cấp ngữ cảnh từ Vector DB, tranh luận để ra Điểm số Đồng thuận (Consensus Rating) cuối cùng.

**Giao tiếp với các module khác:**

- **Nhận vào ← Module 1:** `UnifiedStateVector` (USV) mỗi khi nến M1 đóng
- **Nhận vào ← Module 2:** `MacroContext` (CountdownState, NewsOutcome, regime_phase)
- **Đầu ra → Module 4 (Memory):** Kết quả dự đoán (Predictions) để lưu trữ và đánh giá
- **Đầu ra → Module 6 (Visualization):** $P_{BSL}$, $P_{SSL}$, $P_{hold}$, Consensus Rating, danh sách vùng cấu trúc
- **Nhận vào ← Module 4:** Vector ngữ cảnh lịch sử từ RAG retriever

---

### B. THUẬT TOÁN CỐT LÕI & LOGIC TOÁN HỌC

#### B.1 Trích Xuất Đặc Trưng SMC/ICT (Symbolic Feature Engineering)

##### B.1.1 — Nhận Diện Đỉnh/Đáy Sóng (Swing High / Swing Low = BSL / SSL)

**Định nghĩa toán học của Swing High (SH) và Swing Low (SL) với lookback $k$:**

Nến $i$ được gọi là Swing High với lookback $k$ khi:
$$high_i = \max(high_{i-k}, ..., high_{i-1}, high_i, high_{i+1}, ..., high_{i+k})$$

Nến $i$ được gọi là Swing Low với lookback $k$ khi:
$$low_i = \min(low_{i-k}, ..., low_{i-1}, low_i, low_{i+1}, ..., low_{i+k})$$

Trong hệ thống này, $k$ được hiệu chỉnh theo khung thời gian:
- M1, M5: $k = 3$
- M15, H1: $k = 5$
- H4, D1: $k = 10$

**Phân loại BSL / SSL:**
- Tất cả Swing High = **Buyside Liquidity (BSL)** — vùng có Stop Loss của người bán khống tích tụ
- Tất cả Swing Low = **Sellside Liquidity (SSL)** — vùng có Stop Loss của người mua tích tụ

**Thuật toán quét Swing Points:**

```
THUẬT TOÁN PHÁT HIỆN SWING POINTS:
─────────────────────────────────────────────────────────────
Đầu vào: Mảng nến OHLCV đã đóng, lookback k

[1] VỚI MỖI nến i trong phạm vi [k, len(bars)-k]:
    is_SH = (high[i] == max(high[i-k:i+k+1]))
    is_SL = (low[i]  == min(low[i-k:i+k+1]))
    
    NẾU is_SH VÀ low[i] > low[i-1] VÀ low[i] > low[i+1]:
        Thêm vào danh sách BSL_raw: {price: high[i], time: time[i], tf: current_tf}
    
    NẾU is_SL VÀ high[i] < high[i-1] VÀ high[i] < high[i+1]:
        Thêm vào danh sách SSL_raw: {price: low[i], time: time[i], tf: current_tf}

[2] Lọc bỏ Swing Points quá gần nhau (consolidation noise):
    VỚI MỖI cặp (SP1, SP2) liền kề trong BSL_raw:
        NẾU |SP1.price - SP2.price| < ATR_current × 0.15:
            Giữ lại SP mạnh hơn (theo biên độ nến)

[3] Kết quả: danh sách BSL và SSL đã lọc cho khung thời gian current_tf
─────────────────────────────────────────────────────────────
```

##### B.1.2 — Phân Vùng Premium/Discount

Dựa trên cặp (Swing High lớn nhất gần nhất, Swing Low lớn nhất gần nhất) của một chuyển động sóng:

$$Equilibrium = \frac{SH_{major} + SL_{major}}{2}$$

$$Premium\_Zone = [Equilibrium, SH_{major}]$$
$$Discount\_Zone = [SL_{major}, Equilibrium]$$

Thêm chi tiết theo thuật toán Fibonacci:
$$Level_{0.618} = SL_{major} + 0.618 \times (SH_{major} - SL_{major})$$
$$Level_{0.50} = Equilibrium$$
$$Level_{0.382} = SL_{major} + 0.382 \times (SH_{major} - SL_{major})$$

- Giá trên $Level_{0.618}$: Vùng Premium Cao (tốt cho Sell setup)
- Giá dưới $Level_{0.382}$: Vùng Discount Sâu (tốt cho Buy setup)

##### B.1.3 — Phát Hiện Dịch Chuyển Cấu Trúc (Market Structure Shift — MSS)

```
ĐỊNH NGHĨA MSS (Bullish):
Điều kiện 1: Tồn tại một chuỗi Lower High (LH) và Lower Low (LL) trước đó
             (xác nhận downtrend hiện tại)
Điều kiện 2: Giá đóng cửa của một nến vượt qua (break above) mức SH gần nhất
             trong chuỗi LH-LL nói trên, bằng thân nến (không chỉ bóng)

→ Đánh dấu MSS_BULLISH tại nến đó

ĐỊNH NGHĨA MSS (Bearish): Ngược lại với trên
```

**Phân biệt MSS với BOS (Break of Structure):**

- **BOS (Break of Structure):** Khi giá phá vỡ một Swing High trong điều kiện uptrend đang tiếp diễn → Xác nhận tiếp diễn xu hướng.
- **MSS (Market Structure Shift):** Khi giá phá vỡ Swing High nhưng cấu trúc trước đó là downtrend → Tín hiệu đảo chiều xu hướng. MSS có trọng số quan trọng hơn BOS trong mô hình dự đoán.

##### B.1.4 — Quét Vùng FVG và OB (Fair Value Gap & Order Block)

**Thuật toán Phát hiện FVG (Fair Value Gap):**

```
PHÁT HIỆN FVG (Bullish Imbalance):
─────────────────────────────────────────────────────────────
VỚI MỖI chuỗi 3 nến liên tiếp (n-2, n-1, n):
    NẾU low[n] > high[n-2]:
        → Tồn tại khoảng trống giá (Price Gap)
        → FVG_Bullish = {
              top    : low[n],
              bottom : high[n-2],
              midpoint: (low[n] + high[n-2]) / 2,
              formed_time: time[n],
              timeframe: current_tf,
              status : "UNMITIGATED"
          }

PHÁT HIỆN FVG (Bearish Imbalance): Ngược lại
    NẾU high[n] < low[n-2]:
        → FVG_Bearish = {
              top    : low[n-2],
              bottom : high[n],
              midpoint: (low[n-2] + high[n]) / 2,
              ...
          }
─────────────────────────────────────────────────────────────
```

**Trạng thái FVG:** `UNMITIGATED` → khi giá quay lại chạm vùng FVG lần đầu → `PARTIALLY_MITIGATED` → khi giá lấp đầy hoàn toàn → `MITIGATED`.

Chỉ FVG ở trạng thái `UNMITIGATED` mới được Module AI sử dụng.

**Thuật toán Phát hiện Order Block (OB):**

```
PHÁT HIỆN ORDER BLOCK (Bullish OB):
─────────────────────────────────────────────────────────────
Điều kiện tiên quyết: Tồn tại BOS hoặc MSS Bullish tại nến n

[1] Quét ngược từ nến n về trước để tìm:
    Nến Bearish cuối cùng trước khi xu hướng xảy ra =  nến m
    (Đây là nến nơi các tổ chức đã tích lũy lệnh mua)

[2] OB_Bullish = {
        top     : max(open[m], close[m])
        bottom  : min(open[m], close[m])
        wick_low: low[m]         ← Đây là điểm tốt nhất để vào lệnh
        formed_time: time[m]
        associated_bos_time: time[n]
        timeframe: current_tf
        status  : "UNMITIGATED"
        strength: |high[n] - low[n]| / ATR_current  ← Sức mạnh của chuyển động sau OB
    }
─────────────────────────────────────────────────────────────
```

##### B.1.5 — Phân Loại Premium/Discount cho FVG và OB

Mỗi FVG và OB được kiểm tra xem có nằm trong Vùng Premium hay Discount của HTF không. Vùng OB Bearish trong Premium của HTF được đánh giá trọng số cao hơn 2 lần so với OB Bearish trong Discount:

$$w_{zone} = \begin{cases} 2.0 & \text{nếu OB/FVG phù hợp với Premium/Discount HTF} \\ 1.0 & \text{nếu trung lập} \\ 0.5 & \text{nếu ngược chiều với HTF} \end{cases}$$

#### B.2 Nén Đặc Trưng bằng Mạng Thần Kinh (Hierarchical LSTM Autoencoder)

**Kiến trúc tổng quan:**

```
KIẾN TRÚC HIERARCHICAL LSTM AUTOENCODER:
─────────────────────────────────────────────────────────────
ENCODER:
  Lớp 1 (Tick-Level Encoder):
    Đầu vào: Chuỗi 512 tick gần nhất, mỗi tick là vector 8 chiều:
             [bid, ask, last, volume, delta, spread, dom_imbalance, time_normalized]
    Kiến trúc: Bidirectional LSTM, hidden_size=128, num_layers=2
    Đầu ra: Latent vector h_tick ∈ ℝ^128

  Lớp 2 (Bar-Level Encoder):
    Đầu vào: Chuỗi 100 nến M1 gần nhất, mỗi nến là vector 12 chiều:
             [open, high, low, close, volume, cvd, iii, fvg_flag, ob_flag, 
              bsl_distance, ssl_distance, session_flag]
    Kiến trúc: Unidirectional LSTM, hidden_size=256, num_layers=3
    Đầu ra: Latent vector h_bar ∈ ℝ^256

  Lớp 3 (Multi-Timeframe Encoder):
    Đầu vào: Ghép nối 6 Bar-Level Encoders riêng biệt cho M1→D1
    Mỗi TF có LSTM riêng với lookback phù hợp:
             M1: 100 nến, M5: 96 nến, M15: 96 nến
             H1: 48 nến, H4: 30 nến, D1: 20 nến
    Cross-TF Attention Layer: Multi-Head Attention (8 heads)
             → học mối quan hệ chéo giữa các khung thời gian
    Đầu ra: Unified Latent Vector z ∈ ℝ^512

DECODER (chỉ dùng trong quá trình huấn luyện):
  Tái tạo chuỗi giá M1 từ z
  Loss = MSE(reconstructed, actual) + KL_divergence (nếu VAE variant)

KẾT QUẢ:
  z ∈ ℝ^512 = "Mã hóa trạng thái thị trường hiện tại" 
  → Được sử dụng làm đầu vào cho Model A và Model B
─────────────────────────────────────────────────────────────
```

#### B.3 Model A — Dự Đoán Vùng Hút Thanh Khoản ($P_{BSL}$ vs $P_{SSL}$)

**Đặc trưng đầu vào của Model A:**

```
Feature Vector Model A (dimensionality = 512 + 64 = 576):
  [0:511]    = Unified Latent Vector z từ LSTM Autoencoder
  [512]      = I_news (từ Module 2)
  [513]      = seconds_to_next_news (chuẩn hóa về [0,1])
  [514]      = III_current (Institutional Intensity Index)
  [515]      = OBI_current (Order Book Imbalance)
  [516:517]  = {BSL_count_H1, SSL_count_H1} (số vùng thanh khoản trên H1)
  [518:519]  = {BSL_nearest_distance_pips, SSL_nearest_distance_pips}
  [520:521]  = {BSL_strength_H4, SSL_strength_H4}  ← tổng khối lượng tại SH/SL
  [522]      = session_flag (0=Asian, 1=London, 2=NY)
  [523:527]  = MSS/BOS indicator vector (5 khung TF nhỏ: M1-H1)
  [528:575]  = 48 chiều đặc trưng SMC được lọc và chuẩn hóa
```

**Kiến trúc Model A:**

```
Model A — Architecture:
  Input: Feature Vector (576 chiều)
  Layer 1: Fully Connected (576 → 256), activation=GELU, BatchNorm
  Layer 2: Fully Connected (256 → 128), activation=GELU, Dropout(0.3)
  Layer 3: Fully Connected (128 → 64),  activation=GELU
  Output Layer: Fully Connected (64 → 3), activation=Softmax
    Output[0] = P_BSL  ← Xác suất giá đến BSL tiếp theo trước
    Output[1] = P_SSL  ← Xác suất giá đến SSL tiếp theo trước
    Output[2] = P_lateral ← Xác suất đi ngang (Chop/Range)
  
  Ràng buộc: P_BSL + P_SSL + P_lateral = 1.0

Hàm Loss: Categorical Cross-Entropy + Penalty cho Overconfidence
    L = CE_loss + λ × max(0, max_prob - 0.85)
    λ = 0.5 (ngăn mô hình tự tin quá mức)
```

#### B.4 Model B — Dự Đoán Hiệu Lực Giữ Giá của FVG/OB ($P_{hold}$)

**Đặc trưng đầu vào của Model B (tính cho từng vùng FVG/OB riêng lẻ):**

```
Feature Vector Model B (dimensionality = 512 + 48 = 560):
  [0:511]   = Unified Latent Vector z
  [512:523] = Zone Properties:
    zone_type        (0=FVG_bull, 1=FVG_bear, 2=OB_bull, 3=OB_bear)
    zone_size_pips   (kích thước vùng)
    zone_age_bars    (số nến M1 kể từ khi vùng hình thành)
    htf_alignment    (hệ số w_zone: 0.5, 1.0, hoặc 2.0)
    premium_discount_position (khoảng cách đến điểm cân bằng HTF)
    nearest_bos_age  (số nến từ BOS/MSS gần nhất)
    touch_count      (số lần giá đã chạm vùng này)
    volume_at_formation (III tại thời điểm hình thành vùng)
    macro_regime     (0=NORMAL, 1=PRE_NEWS, 2=NEWS_WINDOW, 3=POST_NEWS)
    surprise_factor  (S nếu POST_NEWS, else 0)
    session_at_formation (0,1,2)
    confluence_count (số tín hiệu khác hội tụ vào vùng này)
  [524:559] = 36 chiều đặc trưng bổ sung từ USV

Output:
  P_hold ∈ [0, 1]  ← Xác suất vùng này giữ được giá và tạo phản ứng
  P_breach = 1 - P_hold

Ngưỡng hành động:
  P_hold >= 0.70: Vùng ĐỦ TIN CẬY — hiển thị màu xanh đậm trên UI
  0.50 <= P_hold < 0.70: Vùng TRUNG BÌNH — hiển thị màu vàng
  P_hold < 0.50: Vùng YẾU — hiển thị màu xám mờ, không khuyến nghị
```

#### B.5 Lớp Tranh Biện Đa Tác Nhân GenAI (Multi-Agent Debate Layer)

**Kiến trúc tổng thể:**

```
LUỒNG TRANH BIỆN ĐA TÁC NHÂN:
─────────────────────────────────────────────────────────────
[1] GIAI ĐOẠN CHUẨN BỊ NGỮ CẢNH (Context Preparation):
    a) Truy vấn Vector DB (RAG) với embedding của USV hiện tại
       → Lấy 3 tình huống lịch sử tương tự nhất (Similarity > 0.80)
    b) Định dạng bản tóm tắt kỹ thuật (Technical Brief):
       - Cấu trúc HTF: trend D1, H4, H1 (Bullish/Bearish/Neutral)
       - Vùng hút thanh khoản gần nhất: BSL/SSL + khoảng cách
       - Vùng FVG/OB đang kích hoạt với P_hold tương ứng
       - P_BSL và P_SSL từ Model A
       - MacroContext: regime_phase, I_news, seconds_to_next
       - 3 precedents lịch sử từ RAG

[2] BULL AGENT (Thiên kiến Tăng):
    System Prompt: "Bạn là chuyên gia phân tích kỹ thuật với thiên kiến TĂNG.
    Nhiệm vụ: Xây dựng luận điểm MẠNH NHẤT có thể cho kịch bản giá đi LÊN,
    dựa HOÀN TOÀN vào dữ liệu định lượng được cung cấp. 
    Bạn phải trích dẫn cụ thể ít nhất 3 bằng chứng từ dữ liệu.
    Không được bịa đặt bất kỳ thông tin nào ngoài context được cung cấp."
    
    Input: Technical Brief
    Output: Bull Thesis = {
        direction: "BULLISH",
        confidence: float [0.0-1.0],
        primary_target: BSL level (giá),
        key_evidence: List[string] (danh sách bằng chứng),
        invalidation_level: price level
    }

[3] BEAR AGENT (Thiên kiến Giảm):
    System Prompt: "Bạn là chuyên gia phân tích kỹ thuật với thiên kiến GIẢM.
    Nhiệm vụ: Xây dựng luận điểm MẠNH NHẤT có thể cho kịch bản giá đi XUỐNG,
    dựa HOÀN TOÀN vào dữ liệu định lượng được cung cấp.
    Bạn phải trích dẫn cụ thể ít nhất 3 bằng chứng từ dữ liệu.
    Không được bịa đặt bất kỳ thông tin nào ngoài context được cung cấp."
    
    Input: Technical Brief
    Output: Bear Thesis = {
        direction: "BEARISH",
        confidence: float [0.0-1.0],
        primary_target: SSL level (giá),
        key_evidence: List[string],
        invalidation_level: price level
    }

[4] CRITIC AGENT (Trọng tài Tổng hợp):
    System Prompt: "Bạn là một trọng tài khách quan. Bạn nhận được hai luận điểm
    đối lập (Bull Thesis và Bear Thesis) và toàn bộ dữ liệu gốc.
    Nhiệm vụ: Đánh giá tính logic và độ bền vững của từng luận điểm.
    Chỉ ra điểm mạnh và điểm yếu của mỗi bên.
    Cuối cùng, đưa ra Consensus Rating theo thang đo 9 điểm:
    -4 (Strongly Bearish), -3, -2, -1, 0 (Neutral), +1, +2, +3, +4 (Strongly Bullish).
    Cung cấp giải thích bằng văn bản ngắn gọn cho điểm số."
    
    Input: Bull Thesis + Bear Thesis + Technical Brief + 3 Historical Precedents
    Output: ConsensusResult = {
        consensus_rating: int [-4, +4],
        preferred_direction: "BULLISH" | "BEARISH" | "NEUTRAL",
        high_conviction_zone: price_level | None,
        confidence_qualifier: "HIGH" | "MEDIUM" | "LOW",
        reasoning_summary: string (max 150 từ),
        agreement_score: float  ← Mức độ đồng ý Bull-Bear [0,1]
    }
─────────────────────────────────────────────────────────────
```

**Cơ chế truy vấn Vector DB (RAG Retrieval):**

Embedding của USV hiện tại được tạo bằng cách ghép nối và chiếu (projection):

$$e_{USV} = W_{proj} \cdot [z; f_{SMC}; f_{macro}]$$

Trong đó $W_{proj} \in \mathbb{R}^{256 \times 600}$ là ma trận chiếu được học. Sau đó tìm kiếm cosine similarity trong Vector DB:

$$sim(e_{USV}, e_{hist}) = \frac{e_{USV} \cdot e_{hist}}{||e_{USV}|| \cdot ||e_{hist}||}$$

Chỉ lấy các kết quả có $sim > 0.80$ để đảm bảo precedents thực sự tương đồng.

---

### C. PHÂN RÃ TÍNH NĂNG CHI TIẾT

#### Tính năng C.1 — Pipeline Trích xuất đặc trưng SMC/ICT

- **Đầu vào:** `UnifiedStateVector` với 6 BarState đầy đủ.
- **Biến đổi:** Quét BSL/SSL/FVG/OB cho cả 6 khung thời gian → Phân loại Premium/Discount → Tính $w_{zone}$ → Xây dựng `SymbolicFeatureMap`.
- **Đầu ra:** `SymbolicFeatureMap = {bsl_list, ssl_list, fvg_list, ob_list, mss_events, bos_events, premium_zones, discount_zones}`.

#### Tính năng C.2 — Inference LSTM Autoencoder

- **Đầu vào:** Chuỗi tick và OHLCV từ USV; Encoder weights đã được nạp.
- **Biến đổi:** Forward pass qua 3 lớp encoder → Multi-Head Attention cross-TF.
- **Đầu ra:** `z ∈ ℝ^512`, được cache trong Redis với key `latent_vector:SYMBOL:TIMESTAMP`.

#### Tính năng C.3 — Inference Model A (Liquidity Draw Prediction)

- **Đầu vào:** `z`, `SymbolicFeatureMap`, `MacroContext`.
- **Biến đổi:** Xây dựng feature vector 576 chiều → Forward pass Model A → Softmax → Điều chỉnh theo macro guardrail nếu `active_guardrail = True`.
- **Đầu ra:** `{P_BSL, P_SSL, P_lateral}` kèm BSL/SSL level cụ thể được dự đoán là mục tiêu.

**Điều chỉnh Macro Guardrail cho Model A:**

```
NẾU active_guardrail = True:
    P_BSL = P_BSL × (1 - guardrail_dampening_factor × I_news)
    P_SSL = P_SSL × (1 - guardrail_dampening_factor × I_news)
    P_lateral += (P_BSL_delta + P_SSL_delta) / 2
    Tái chuẩn hóa sao cho tổng = 1.0
    guardrail_dampening_factor = 0.3 (mặc định, tăng tuyến tính khi seconds_to_next giảm)
```

#### Tính năng C.4 — Inference Model B (Hold Zone Classification)

- **Đầu vào:** `z`, thuộc tính từng vùng FVG/OB cụ thể, `MacroContext`.
- **Biến đổi:** Lặp qua danh sách tất cả UNMITIGATED FVG/OB → Xây dựng feature vector 560 chiều cho từng vùng → Forward pass → Gán `P_hold`.
- **Đầu ra:** Danh sách vùng FVG/OB với `P_hold` cập nhật, sắp xếp giảm dần theo `P_hold × w_zone`.

#### Tính năng C.5 — Chu Trình Tranh Biện Đa Tác Nhân

- **Đầu vào:** `SymbolicFeatureMap`, `{P_BSL, P_SSL}`, `MacroContext`, 3 precedents từ Vector DB.
- **Biến đổi:** Gọi Bull Agent → Gọi Bear Agent (song song) → Gọi Critic Agent → Tổng hợp `ConsensusResult`.
- **Đầu ra:** `ConsensusResult` với `consensus_rating ∈ [-4, +4]`, được lưu vào Module 4 và đẩy lên Module 6.

#### Tính năng C.6 — Embedding và lưu trữ vào Vector DB

- **Đầu vào:** USV hiện tại + `ConsensusResult` sau khi kết quả được xác nhận (30–60 phút sau).
- **Biến đổi:** Tính `e_USV` theo công thức chiếu → Lưu `(e_USV, metadata)` vào ChromaDB/Qdrant.
- **Đầu ra:** Precedent mới được lưu trong Vector DB để cải thiện RAG retrieval trong tương lai.

---

### D. CƠ CHẾ PHÒNG VỆ & XỬ LÝ CA BIÊN

#### D.1 — Mô hình AI bị suy giảm độ chính xác (Model Drift)

```
PHÁT HIỆN (xem Module 5 chi tiết):
    IC_rolling_20 < IC_threshold (mặc định IC_threshold = 0.05)
    hoặc Feature Drift Score > 0.4

HÀNH ĐỘNG:
[1] Nâng cờ MODEL_DEGRADED trong AIEngineState
[2] Giảm confidence_multiplier từ 1.0 xuống 0.6
    → Tất cả P_hold, P_BSL, P_SSL được nhân với 0.6
    → Hiển thị badge "Cảnh báo: Mô hình cần hiệu chỉnh" trên UI
[3] Ưu tiên sử dụng Rule-Based Fallback (xem D.2) thay vì Model A/B output
[4] Tự động kích hoạt fine-tuning nếu đủ dữ liệu mới (>= 200 mẫu)
```

#### D.2 — Fallback Rule-Based khi Model không khả dụng

```
RULE-BASED FALLBACK ENGINE:
─────────────────────────────────────────────────────────────
NẾU model không khả dụng hoặc MODEL_DEGRADED = True:

[1] Tính hướng HTF bằng quy tắc cứng:
    htf_bias = "BULLISH" NẾU close_H4[-1] > EMA_50_H4 VÀ BOS_H4_bullish trong 10 nến gần nhất
             = "BEARISH" NẾU close_H4[-1] < EMA_50_H4 VÀ BOS_H4_bearish trong 10 nến gần nhất
             = "NEUTRAL" ngược lại

[2] Tính P_BSL, P_SSL bằng quy tắc đơn giản:
    NẾU htf_bias = "BULLISH":
        P_BSL = 0.60, P_SSL = 0.25, P_lateral = 0.15
    NẾU htf_bias = "BEARISH":
        P_BSL = 0.25, P_SSL = 0.60, P_lateral = 0.15
    NẾU htf_bias = "NEUTRAL":
        P_BSL = 0.35, P_SSL = 0.35, P_lateral = 0.30

[3] P_hold cho FVG/OB: sử dụng w_zone × 0.55 (heuristic cơ bản)

[4] Multi-Agent Debate vẫn chạy bình thường (không phụ thuộc ML model)
─────────────────────────────────────────────────────────────
```

#### D.3 — Vòng lặp Tranh biện vượt quá thời gian (Debate Timeout)

```
GIỚI HẠN THỜI GIAN: Toàn bộ chu kỳ tranh biện phải hoàn thành trong 8 giây

NẾU Bull Agent hoặc Bear Agent mất > 3s:
    Timeout Agent đó, sử dụng Default Thesis:
    Bull Default: {direction: "BULLISH", confidence: 0.5, evidence: ["HTF_BIAS"]}
    Bear Default: {direction: "BEARISH", confidence: 0.5, evidence: ["HTF_BIAS"]}

NẾU Critic Agent mất > 5s:
    Timeout, sử dụng Algorithmic Critic:
    consensus_rating = round((bull_confidence - bear_confidence) × 4)
    confidence_qualifier = "LOW"

Phát DEBATE_TIMEOUT_WARNING đến Module 6
```

#### D.4 — USV không đầy đủ (Thiếu dữ liệu một số khung thời gian)

```
PHÁT HIỆN: Một hoặc nhiều BarState trong USV có status = UNRELIABLE

HÀNH ĐỘNG:
[1] Loại bỏ khung TF thiếu dữ liệu khỏi Multi-TF Encoder
[2] Điền vào chiều tương ứng trong z bằng giá trị 0 (zero-imputation)
[3] Giảm confidence_multiplier thêm 0.1 cho mỗi khung TF bị loại
[4] Đánh dấu INCOMPLETE_MTF_FLAG trong kết quả đầu ra
```

---

# MODULE 4: BỘ QUẢN LÝ TRẠNG THÁI & LƯU TRỮ HỆ THỐNG
## (Memory & Persistence Engine)

---

### A. TỔNG QUAN KỸ THUẬT & MỤC TIÊU

Module 4 là "bộ nhớ sống" của hệ thống — tương tự như vùng đồi hải mã (hippocampus) và vỏ não trước trán trong não người. Nó quản lý hai tầng lưu trữ riêng biệt với đặc tính bổ sung cho nhau:

**Tầng 1 — Bộ nhớ Ngắn hạn (Short-Term Memory):** Lưu trữ trạng thái vận hành thời gian thực trong RAM — các vùng cấu trúc đang kích hoạt, trạng thái đếm ngược, kết quả tranh biện gần nhất, và luồng dữ liệu xử lý hiện tại. Đây là bộ nhớ cần truy cập với độ trễ dưới 1ms.

**Tầng 2 — Bộ nhớ Dài hạn (Long-Term Memory):** Lưu trữ lịch sử toàn bộ trên đĩa — lịch sử dự đoán, độ chính xác theo thời gian, và cơ sở dữ liệu ngữ nghĩa để phục vụ tìm kiếm ngữ cảnh (RAG). Đây là nền tảng để hệ thống "học hỏi" và cải thiện theo thời gian.

**Giao tiếp với các module khác:**

- **Nhận vào ← Module 3:** Predictions để lưu; kết quả tranh biện để vector hóa và lưu.
- **Đầu ra → Module 3:** Precedents từ Vector DB (RAG); lịch sử độ chính xác mô hình.
- **Nhận vào ← Module 2:** NewsOutcome để lưu vào SQLite.
- **Đầu ra → Module 5:** Truy cập lịch sử dự đoán để tính metrics đánh giá.

---

### B. THUẬT TOÁN CỐT LÕI & LOGIC TOÁN HỌC

#### B.1 Bộ Nhớ Ngắn Hạn — Thiết Kế Redis Schema

**Tất cả key Redis đều được đặt TTL (Time-to-Live) phù hợp để tự động dọn dẹp:**

**Nhóm 1 — Active Zone Registry (Các vùng cấu trúc đang kích hoạt):**

```
Key Pattern: "zone:{symbol}:{timeframe}:{zone_type}:{zone_id}"
Value: Serialized JSON của FVGRecord hoặc OBRecord
TTL: 7 × 24 × 3600 = 604800 giây (7 ngày) — vùng tự hết hạn sau 7 ngày nếu không bị mitigated

Ví dụ:
Key  = "zone:XAUUSD:H1:FVG_BULL:1703123456"
Value = {
    "top": 2045.50,
    "bottom": 2043.20,
    "midpoint": 2044.35,
    "formed_time": 1703123456,
    "status": "UNMITIGATED",
    "p_hold": 0.72,
    "w_zone": 2.0
}
```

**Nhóm 2 — Model Outputs (Kết quả dự đoán gần nhất):**

```
Key = "ai:output:{symbol}:latest"
Value = {
    "timestamp": int64,
    "p_bsl": float,
    "p_ssl": float,
    "p_lateral": float,
    "bsl_target": float,
    "ssl_target": float,
    "consensus_rating": int,
    "confidence_qualifier": string
}
TTL: 120 giây (tự vô hiệu nếu không có update sau 2 phút)
```

**Nhóm 3 — Countdown State:**

```
Key = "macro:countdown:{currency}"
Value = Serialized CountdownState
TTL: 60 giây (luôn fresh)
```

**Nhóm 4 — Debate Log (Nhật ký tranh biện):**

```
Key = "debate:{symbol}:{timestamp}"
Value = Serialized DebateRecord (Bull+Bear+Critic theses)
TTL: 3600 giây (1 giờ trong RAM; sau đó archive sang Vector DB)
```

**Nhóm 5 — Latent Vector Cache:**

```
Key = "latent:{symbol}:{bar_close_timestamp}"
Value = Binary blob của z ∈ ℝ^512 (float32, 2048 bytes)
TTL: 300 giây (5 phút)
```

#### B.2 Bộ Nhớ Dài Hạn — Schema SQLite

**Bảng `predictions` — Lịch sử dự đoán Model A/B:**

```
TABLE predictions:
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    symbol          TEXT NOT NULL
    prediction_time INTEGER NOT NULL         ← Unix timestamp
    bar_close_time  INTEGER NOT NULL         ← Timestamp nến M1 đóng kích hoạt
    timeframe       TEXT NOT NULL
    model_name      TEXT NOT NULL            ← "model_a" hoặc "model_b"
    
    -- Model A specific
    p_bsl           REAL
    p_ssl           REAL
    p_lateral       REAL
    predicted_bsl_level  REAL
    predicted_ssl_level  REAL
    
    -- Model B specific
    zone_id         TEXT
    p_hold          REAL
    
    -- Consensus
    consensus_rating  INTEGER
    confidence_qualifier TEXT
    
    -- Outcome (điền sau khi kết quả xác nhận)
    outcome_determined  INTEGER DEFAULT 0   ← 0: chưa xác định, 1: đã xác định
    outcome_time        INTEGER
    actual_direction    TEXT                 ← "BSL_HIT", "SSL_HIT", "LATERAL"
    actual_hold         INTEGER             ← 0 hoặc 1 (Model B)
    pips_to_target      REAL
    
    -- Metadata
    macro_regime        TEXT
    i_news              REAL
    iii_current         REAL
    
    INDEX: (symbol, prediction_time)
    INDEX: (outcome_determined, symbol)
```

**Bảng `news_outcomes` — Lịch sử kết quả tin tức:**

```
TABLE news_outcomes:
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    event_id        TEXT NOT NULL
    currency        TEXT NOT NULL
    event_name      TEXT NOT NULL
    scheduled_time  INTEGER NOT NULL
    actual_release_time  INTEGER
    forecast        REAL
    actual          REAL
    surprise_factor REAL
    post_regime     TEXT                     ← "IMPULSIVE_FOLLOW_THROUGH", v.v.
    max_move_pips_15min REAL                 ← Biên độ tối đa trong 15 phút sau
    impact_vector_used  REAL                 ← I_news đã sử dụng
    i_news_predicted_accuracy INTEGER        ← 1 nếu I_news dự đoán đúng biên độ ±20%
```

**Bảng `model_performance` — Theo dõi hiệu suất mô hình theo thời gian:**

```
TABLE model_performance:
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    evaluation_time INTEGER NOT NULL
    model_name      TEXT NOT NULL
    window_size     INTEGER NOT NULL         ← N mẫu được đánh giá
    ic              REAL                     ← Information Coefficient
    precision_hold  REAL                     ← Precision của P_hold >= 0.70
    recall_hold     REAL
    f1_hold         REAL
    brier_score_a   REAL                     ← Brier Score của Model A
    feature_drift_score REAL
    regime          TEXT                     ← Chế độ thị trường trong cửa sổ này
```

#### B.3 Bộ Nhớ Dài Hạn — Schema Vector DB (ChromaDB)

**Collection 1: `debate_archive`**

```
Collection: debate_archive
Embedding dimensions: 256 (e_USV projection)
Metadata per document:
    {
        symbol          : string,
        timestamp       : int64,
        consensus_rating: int,
        actual_outcome  : string,       ← được điền sau khi outcome xác nhận
        i_news          : float,
        session         : string,
        macro_regime    : string,
        p_bsl           : float,
        p_ssl           : float,
        full_debate_text: string        ← text đầy đủ để hiển thị RAG result cho Critic Agent
    }
Distance metric: cosine
```

**Collection 2: `zone_embeddings`**

```
Collection: zone_embeddings
Embedding dimensions: 64 (zone-specific embedding)
Mục đích: Tìm kiếm các vùng FVG/OB lịch sử có đặc tính tương tự vùng hiện tại,
          phục vụ ước lượng P_hold offline
```

#### B.4 Cơ Chế Xác Nhận Kết Quả (Outcome Determination)

```
THUẬT TOÁN XÁC NHẬN KẾT QUẢ DỰ ĐOÁN:
─────────────────────────────────────────────────────────────
Chạy mỗi khi nhận BAR_CLOSE(M1):

[1] Truy vấn SQLite: Lấy tất cả predictions có:
    outcome_determined = 0
    VÀ prediction_time > current_time - MAX_HORIZON (mặc định = 4 giờ)

[2] VỚI MỖI prediction p chưa xác nhận:
    a) Kiểm tra p_bsl target:
       NẾU high_since_prediction >= p.predicted_bsl_level:
           p.actual_direction = "BSL_HIT"
           p.pips_to_target = predicted_bsl_level - close_at_prediction_time
           p.outcome_determined = 1
    
    b) Kiểm tra p_ssl target:
       NẾU low_since_prediction <= p.predicted_ssl_level:
           p.actual_direction = "SSL_HIT"
           p.outcome_determined = 1
    
    c) NẾU (current_time - prediction_time) > MAX_HORIZON:
           p.actual_direction = "LATERAL"
           p.outcome_determined = 1
    
    d) Cho Model B (p_hold):
       NẾU giá chạm vào zone của p VÀ zone giữ được giá (không bị break):
           p.actual_hold = 1
       NẾU giá chạm vào zone và phá qua:
           p.actual_hold = 0

[3] Cập nhật bảng predictions trong SQLite

[4] NẾU có prediction vừa được xác nhận:
    Phát sự kiện OUTCOME_CONFIRMED đến Module 5 (Drift Tracker)
─────────────────────────────────────────────────────────────
```

---

### C. PHÂN RÃ TÍNH NĂNG CHI TIẾT

#### Tính năng C.1 — Ghi và cập nhật Zone Registry

- **Đầu vào:** `SymbolicFeatureMap` mới từ Module 3 kèm `P_hold` từ Model B.
- **Biến đổi:** Đối với mỗi FVG/OB: kiểm tra xem đã tồn tại trong Redis chưa → Cập nhật `P_hold` nếu có; tạo mới nếu chưa có → Cập nhật `status` (UNMITIGATED/PARTIALLY_MITIGATED/MITIGATED) dựa trên giá hiện tại.
- **Đầu ra:** Zone Registry cập nhật trong Redis; phát sự kiện `ZONE_STATUS_CHANGED` nếu có thay đổi trạng thái.

#### Tính năng C.2 — Lưu và truy xuất kết quả tranh biện (Debate Persistence)

- **Đầu vào:** `ConsensusResult` và full debate transcript từ Module 3.
- **Biến đổi:** Lưu vào Redis (TTL 1 giờ) → Sau 1 giờ, tự động archive sang ChromaDB `debate_archive` với embedding và metadata.
- **Đầu ra:** Document trong Vector DB có thể tìm kiếm bằng cosine similarity.

#### Tính năng C.3 — Ghi và truy vấn lịch sử dự đoán

- **Đầu vào:** Prediction output từ Module 3.
- **Biến đổi:** INSERT vào bảng `predictions` → Theo dõi asynchronously để xác nhận outcome.
- **Đầu ra:** Row trong `predictions` với `outcome_determined = 0` ban đầu; cập nhật về 1 khi xác nhận.

#### Tính năng C.4 — Truy vấn RAG cho Module 3

- **Đầu vào:** `e_USV` embedding từ Module 3 (256 chiều).
- **Biến đổi:** Truy vấn ChromaDB với cosine similarity → Lọc kết quả `sim > 0.80` → Lấy `full_debate_text` và metadata.
- **Đầu ra:** Danh sách tối đa 3 precedents tốt nhất kèm outcome thực tế.

---

### D. CƠ CHẾ PHÒNG VỆ & XỬ LÝ CA BIÊN

#### D.1 — Redis bị đầy bộ nhớ (Memory Overflow)

```
CẤU HÌNH Redis: maxmemory = 512MB, maxmemory-policy = allkeys-lru

NẾU Redis gần đầy (>80% capacity):
[1] Redis tự động evict key ít được dùng nhất (LRU)
[2] Hệ thống phát REDIS_MEMORY_WARNING qua Prometheus
[3] Module 4 ngay lập tức chạy emergency archival:
    - Chuyển tất cả debate logs từ Redis sang SQLite/ChromaDB
    - Xóa latent vector cache (có thể tái tính)
```

#### D.2 — SQLite locked (Write contention)

```
PHÁT HIỆN: SQLite WRITE operation bị SQLITE_BUSY exception

HÀNH ĐỘNG:
[1] Sử dụng chế độ WAL (Write-Ahead Logging) — cấu hình ngay khi khởi động
    → WAL cho phép read và write đồng thời
[2] Retry với exponential backoff: 10ms, 20ms, 40ms (tối đa 5 lần)
[3] NẾU vẫn thất bại: buffer write vào in-memory queue
    Flush lại khi lock được giải phóng
```

#### D.3 — ChromaDB corruption (Vector DB lỗi)

```
PHÁT HIỆN: ChromaDB trả về lỗi hoặc query cho kết quả không hợp lệ

HÀNH ĐỘNG:
[1] Đặt VECTOR_DB_UNAVAILABLE = True
[2] Multi-Agent Debate vẫn chạy nhưng không có RAG precedents
    Critic Agent nhận được thông báo: "Historical context unavailable"
[3] Chạy ChromaDB integrity check dưới nền
[4] NẾU integrity check thất bại: rebuild từ debate transcripts trong SQLite
```

---

# MODULE 5: BỘ KIỂM THỬ DỰA TRÊN SỰ KIỆN & ĐÁNH GIÁ SAI LỆCH
## (Event-Driven Backtesting & Model Drift Evaluation Engine)

---

### A. TỔNG QUAN KỸ THUẬT & MỤC TIÊU

Module 5 phục vụ hai mục đích không thể tách rời:

**Mục đích 1 — Kiểm thử Lịch sử (Historical Backtesting):** Đánh giá hiệu suất của toàn bộ pipeline AI — từ Feature Engineering đến Model A/B và Multi-Agent Debate — trên dữ liệu lịch sử, với đầy đủ bảo vệ chống look-ahead bias.

**Mục đích 2 — Theo dõi Sai lệch Mô hình (Model Drift Tracking):** Liên tục giám sát chất lượng dự đoán trong thực tế (production), phát hiện sớm khi mô hình bắt đầu xuống cấp do thay đổi điều kiện thị trường.

**Giao tiếp với các module khác:**

- **Nhận vào ← Module 1 (trong backtest mode):** Dữ liệu tick và OHLCV lịch sử
- **Nhận vào ← Module 4:** Lịch sử dự đoán và outcomes đã xác nhận
- **Đầu ra → Module 3:** Cảnh báo Model Drift (MODEL_DEGRADED flag)
- **Đầu ra → Module 6:** Kết quả backtest để hiển thị trên UI

---

### B. THUẬT TOÁN CỐT LÕI & LOGIC TOÁN HỌC

#### B.1 Kiến Trúc Vòng Lặp Backtest Dựa Trên Sự Kiện

```
VÒNG LẶP BACKTEST CHÍNH:
─────────────────────────────────────────────────────────────
Đầu vào: 
    - tick_data: List[TickFrame] (đã sắp xếp theo timestamp)
    - calendar_data: List[RawNewsEvent] (lịch kinh tế lịch sử)
    - start_time, end_time
    - config: backtest parameters

Khởi tạo:
    - Reset toàn bộ state của Module 1, 3, 4 về trạng thái ban đầu
    - Đặt backtest_mode = True trên LeakageGuard
    - Nạp calendar_data vào Module 2 với chế độ "historical playback"

Vòng lặp chính:
    Pointer i = 0
    TRONG KHI i < len(tick_data):
        tick = tick_data[i]
        
        [1] Đẩy tick vào Module 1 pipeline (giả lập real-time)
        [2] NẾU tick.timestamp trùng với event trong calendar_data:
            Đẩy event vào Module 2 pipeline
        [3] Thu thập tất cả events từ Event Bus:
            - BAR_CLOSE events → Module 3 inference
            - Kết quả Model A/B, Consensus → Lưu vào BacktestResultBuffer
        [4] Cập nhật outcome determination (không phải chờ 4 giờ — 
            trong backtest có thể look ahead TRONG PHẠM VI 4 giờ)
        
        QUAN TRỌNG: Trong bước [4], chỉ được phép look-ahead để xác nhận
        outcome (không ảnh hưởng đến quyết định tại i)
        
        i += 1

Sau khi vòng lặp kết thúc:
    Tính toán toàn bộ Performance Metrics
    Xuất BacktestReport
─────────────────────────────────────────────────────────────
```

#### B.2 Đo Lường Hệ Số Thông Tin (Information Coefficient — IC)

IC đo lường mức độ tương quan giữa dự đoán của mô hình và kết quả thực tế:

$$IC = \text{Spearman Rank Correlation}(\hat{y}, y)$$

Trong đó:
- $\hat{y}$ = vector dự đoán (ví dụ: $P_{BSL} - P_{SSL}$ cho Model A)
- $y$ = vector kết quả thực tế (ví dụ: +1 nếu BSL hit, -1 nếu SSL hit, 0 nếu lateral)

Sử dụng **Spearman** thay vì Pearson vì phân phối $\hat{y}$ có thể không chuẩn.

$$IC_{rolling,N} = \text{Spearman}(\hat{y}_{t-N:t}, y_{t-N:t})$$

Mặc định $N = 20$ dự đoán gần nhất.

Ngưỡng đánh giá:
- $IC > 0.10$: Mô hình đang hoạt động tốt
- $0.05 \leq IC \leq 0.10$: Cảnh báo suy giảm
- $IC < 0.05$: Kích hoạt MODEL_DEGRADED

#### B.3 Đo Lường Precision/Recall cho $P_{hold}$

$$Precision_{hold} = \frac{TP_{hold}}{TP_{hold} + FP_{hold}}$$

$$Recall_{hold} = \frac{TP_{hold}}{TP_{hold} + FN_{hold}}$$

Trong đó:
- $TP_{hold}$: Số vùng có $P_{hold} \geq 0.70$ và thực tế giữ được giá
- $FP_{hold}$: Số vùng có $P_{hold} \geq 0.70$ nhưng thực tế bị phá vỡ
- $FN_{hold}$: Số vùng có $P_{hold} < 0.70$ nhưng thực tế giữ được giá

$$F1_{hold} = 2 \times \frac{Precision_{hold} \times Recall_{hold}}{Precision_{hold} + Recall_{hold}}$$

**Brier Score cho Model A (xác suất có hiệu chỉnh):**

$$BS_A = \frac{1}{N} \sum_{i=1}^{N} (P_{BSL,i} - \mathbb{1}[outcome_i = BSL\_HIT])^2$$

Brier Score $\in [0, 1]$; giá trị thấp hơn là tốt hơn. Dưới 0.20 là chấp nhận được.

#### B.4 Phát Hiện Dịch Chuyển Đặc Trưng (Feature Drift Detection)

**Thuật toán Population Stability Index (PSI):**

Với mỗi đặc trưng $f$ trong feature vector đầu vào của Model A/B:

$$PSI_f = \sum_{b=1}^{B} (A_b - E_b) \times \ln\left(\frac{A_b}{E_b}\right)$$

Trong đó:
- $B = 10$: Số bins phân vị
- $A_b$: Tỷ lệ dữ liệu của tháng hiện tại rơi vào bin thứ $b$
- $E_b$: Tỷ lệ dữ liệu của tháng huấn luyện rơi vào bin thứ $b$ (baseline)

**Ngưỡng PSI:**
- $PSI < 0.1$: Phân phối ổn định (No Drift)
- $0.1 \leq PSI < 0.2$: Có thay đổi nhỏ (Minor Drift)
- $PSI \geq 0.2$: Thay đổi đáng kể (Significant Drift — kích hoạt cảnh báo)

**Feature Drift Score tổng hợp:**

$$FDS = \frac{1}{|F|} \sum_{f \in F} \mathbb{1}[PSI_f \geq 0.2]$$

Trong đó $|F|$ là tổng số đặc trưng. $FDS > 0.4$ kích hoạt MODEL_DEGRADED.

#### B.5 Phát Hiện Dịch Chuyển Chế Độ Thị Trường (Regime Shift Detection)

```
THUẬT TOÁN PHÁT HIỆN REGIME SHIFT:
─────────────────────────────────────────────────────────────
[1] Tính chỉ số biến động ngắn hạn và dài hạn:
    vol_short = std(daily_returns, window=5)  ← 5 ngày
    vol_long  = std(daily_returns, window=60) ← 60 ngày

[2] Tỷ số biến động:
    vol_ratio = vol_short / vol_long

[3] Tính chỉ số hướng (Trend Strength):
    ADX_14 = Average Directional Index trên D1

[4] Phân loại chế độ:
    NẾU vol_ratio < 0.7 VÀ ADX_14 > 25:
        regime = "TRENDING_LOW_VOL"
    NẾU vol_ratio > 1.5 VÀ ADX_14 > 25:
        regime = "TRENDING_HIGH_VOL"
    NẾU vol_ratio > 1.5 VÀ ADX_14 < 20:
        regime = "CHOPPY_HIGH_VOL"
    NGƯỢC LẠI:
        regime = "NORMAL"

[5] NẾU regime khác với regime của 7 ngày trước:
    Phát sự kiện REGIME_SHIFT_DETECTED
    → Tăng tốc chu kỳ đánh giá IC từ hàng ngày xuống mỗi 4 giờ
    → Ghi log với timestamp để phân tích hậu kỳ
─────────────────────────────────────────────────────────────
```

---

### C. PHÂN RÃ TÍNH NĂNG CHI TIẾT

#### Tính năng C.1 — Vòng lặp Backtest Event-Driven

- **Đầu vào:** Tick data Parquet + Calendar data + Config (start/end time, lookback).
- **Biến đổi:** Replay tick theo chronological order → Gọi toàn bộ pipeline Module 1→2→3 → Collect predictions → Xác nhận outcomes bằng look-ahead có kiểm soát.
- **Đầu ra:** `BacktestResultBuffer` chứa toàn bộ prediction-outcome pairs; `BacktestReport` tổng hợp.

#### Tính năng C.2 — Tính IC rolling và lưu vào `model_performance`

- **Đầu vào:** 20 dự đoán gần nhất với outcome đã xác nhận từ SQLite.
- **Biến đổi:** Tính Spearman correlation → So sánh với ngưỡng → Cập nhật bảng `model_performance`.
- **Đầu ra:** `IC_rolling_20` được cập nhật; cờ MODEL_DEGRADED nếu cần.

#### Tính năng C.3 — PSI Feature Drift Analysis

- **Đầu vào:** Feature vectors của 30 ngày gần nhất (từ Redis logs hoặc SQLite) vs. baseline của tháng huấn luyện.
- **Biến đổi:** Tính PSI cho mỗi đặc trưng → Tính FDS tổng hợp → So sánh ngưỡng.
- **Đầu ra:** `DriftReport = {fds, psi_by_feature, drifted_features_list}`.

#### Tính năng C.4 — Phân tích hiệu suất theo chế độ thị trường

- **Đầu vào:** `BacktestResultBuffer` + nhãn regime cho từng thời điểm.
- **Biến đổi:** Phân tách kết quả theo regime → Tính IC, Precision/Recall riêng biệt cho từng regime.
- **Đầu ra:** Bảng phân tích `Performance_by_Regime` — giúp xác định mô hình hoạt động tốt nhất trong chế độ nào.

---

### D. CƠ CHẾ PHÒNG VỆ & XỬ LÝ CA BIÊN

#### D.1 — Thiếu dữ liệu tick lịch sử (Sparse Historical Data)

```
PHÁT HIỆN: Tỷ lệ tick trên mỗi giây < 1.0 trong một khoảng thời gian > 30 phút

HÀNH ĐỘNG:
[1] Đánh dấu khoảng thời gian đó là SPARSE_DATA_ZONE
[2] Tự động loại trừ các dự đoán hình thành trong SPARSE_DATA_ZONE
    khỏi tính toán IC và Drift (vì tick thưa dẫn đến CVD không chính xác)
[3] Ghi tỷ lệ coverage vào BacktestReport:
    data_coverage = actual_tick_count / expected_tick_count
[4] NẾU data_coverage < 0.7 trong toàn backtest:
    WARN: "Backtest results may be unreliable due to sparse tick data"
```

#### D.2 — Quá khớp (Overfitting) phát hiện trong backtest

```
PHÁT HIỆN: IC_backtest > IC_forward × 2.0
    (hiệu suất trên backtest gấp đôi live trading)

HÀNH ĐỘNG:
[1] Ghi cảnh báo POTENTIAL_OVERFITTING vào BacktestReport
[2] Đề xuất các biện pháp kiểm tra:
    - Walk-Forward Analysis (không triển khai tự động)
    - Out-of-Sample test trên dữ liệu 3 tháng gần nhất
    - Giảm độ phức tạp mô hình (model complexity reduction)
[3] KHÔNG tự động thay đổi mô hình
```

---

# MODULE 6: GIAO DIỆN TRỰC QUAN HÓA ĐỒ THỊ
## (TradingView Lightweight Charts Integration & Real-Time Visualization)

---

### A. TỔNG QUAN KỸ THUẬT & MỤC TIÊU

Module 6 là mặt tiền thị giác của hệ thống — nơi tất cả dữ liệu tính toán phức tạp được chuyển hóa thành thông tin trực quan có thể nắm bắt ngay lập tức. Được xây dựng trên nền tảng React với TradingView Lightweight Charts (thư viện chart hiệu năng cao, WebGL-accelerated), Module 6 đảm bảo:

- Hiển thị vùng cấu trúc đa khung thời gian "bóng ma" (Ghost Zones) chồng lên đồ thị khung nhỏ
- Cập nhật heatmap AI theo thời gian thực dựa trên $P_{hold}$
- Hiển thị đường mục tiêu thanh khoản $P_{BSL}$/$P_{SSL}$
- Tích hợp timeline tin tức kinh tế vĩ mô

**Giao tiếp với các module khác:**

- **Nhận vào ← Module 7 (IPC):** JSON messages qua WebSocket local
- **Đầu ra → Module 7:** User actions (thay đổi khung TF, cấu hình hiển thị)

---

### B. THUẬT TOÁN CỐT LÕI & LOGIC TOÁN HỌC

#### B.1 Tính Toán Tọa Độ Ghost Zones (MTF Ghost Zones)

**Bài toán:** Các vùng FVG/OB của H4 cần được vẽ lên đồ thị M5 với tọa độ không gian - thời gian chính xác.

**Quy tắc chuyển đổi tọa độ:**

```
CHO MỖI GhostZone (FVG/OB từ khung HTF như H4/H1):
    zone.price_top    → tọa độ Y trên đồ thị (giá trực tiếp, không cần đổi)
    zone.price_bottom → tọa độ Y dưới
    zone.formed_time  → tọa độ X bắt đầu (timestamp)
    
    Tọa độ X kết thúc: hiển thị đến cạnh phải của màn hình (open-ended)
    
    NẾU zone.status = "PARTIALLY_MITIGATED":
        Vẽ phần bên dưới zone.midpoint với opacity thấp hơn (×0.5)
    NẾU zone.status = "UNMITIGATED":
        Vẽ toàn bộ zone

CHIỀU RỘNG ĐƯỜNG VIỀN:
    HTF = H4 hoặc D1: border_width = 2px, dash_pattern = [4, 2]
    HTF = H1:          border_width = 1.5px, dash_pattern = [3, 2]
    LTF = M15 hoặc M5: border_width = 1px, dash_pattern = [2, 2]
```

#### B.2 Bản Đồ Nhiệt AI — Quy Tắc Màu Sắc và Opacity

**Quy tắc ánh xạ $P_{hold}$ → Màu sắc:**

```
BẢNG ÁNH XẠ P_hold → MÀU SẮC:

VỚI FVG/OB Bullish (mong đợi giá tăng):
    P_hold >= 0.80: fill_color = hsla(120, 80%, 50%, 0.40)  ← Xanh lá đậm
    P_hold >= 0.70: fill_color = hsla(120, 60%, 55%, 0.30)  ← Xanh lá vừa
    P_hold >= 0.60: fill_color = hsla(60, 70%, 55%, 0.25)   ← Vàng-xanh
    P_hold >= 0.50: fill_color = hsla(30, 60%, 55%, 0.20)   ← Cam-vàng
    P_hold <  0.50: fill_color = hsla(0, 0%, 60%, 0.12)     ← Xám mờ

VỚI FVG/OB Bearish (mong đợi giá giảm):
    P_hold >= 0.80: fill_color = hsla(0, 80%, 50%, 0.40)    ← Đỏ đậm
    P_hold >= 0.70: fill_color = hsla(0, 60%, 55%, 0.30)    ← Đỏ vừa
    P_hold >= 0.60: fill_color = hsla(15, 70%, 55%, 0.25)   ← Cam-đỏ
    P_hold <  0.50: fill_color = hsla(0, 0%, 60%, 0.12)     ← Xám mờ

Trong "Chế độ Tin tức" (active_guardrail = True):
    TẤT CẢ LTF zones (M1, M5, M15):
        opacity × 0.4  (mờ đi đáng kể)
    HTF zones (H1, H4, D1):
        opacity không thay đổi (hoặc × 1.1 để nổi bật hơn)
```

**Cập nhật màu sắc theo thời gian thực:**

```
KHI NHẬN ĐƯỢC P_hold MỚI TỪ WEBSOCKET:
[1] Tra cứu zone_id tương ứng trong React component tree
[2] Tính màu mới theo bảng ánh xạ
[3] Áp dụng transition animation: transition: fill 0.5s ease-in-out
    (tránh flash màu đột ngột)
[4] Nếu P_hold thay đổi > 0.15 (nhảy lớn), thêm brief pulse animation 
    để thu hút sự chú ý của trader
```

#### B.3 Đường Mục Tiêu Thanh Khoản (Liquidity Target Lines)

```
HIỂN THỊ BSL/SSL TARGET LINES:

VỚI MỖI BSL target:
    y_coordinate   = P.predicted_bsl_level (giá)
    line_style     = DASHED, color = BLUE (#2196F3)
    line_width     = 1.5px
    
    Nhãn bên phải:
        "BSL ↑  {P_BSL × 100:.0f}%"
        font_size = 11px, color = BLUE
        background = semi-transparent white (rgba(255,255,255,0.8))
    
    NẾU P_BSL > 0.70: thêm animated pulse ở đầu dòng (visual affordance)

VỚI MỖI SSL target:
    Tương tự nhưng color = RED (#F44336), direction indicator = "↓"

Cập nhật: Xóa target cũ, vẽ target mới mỗi khi nhận WebSocket update
```

#### B.4 Trục Thời Gian Sự Kiện Vĩ Mô (Macro Event Timeline)

```
HIỂN THỊ VERTICAL BAR TIN TỨC:

VỚI MỖI upcoming event trong 24 giờ tới:
    x_coordinate = event.scheduled_time (timestamp)
    
    Màu sắc theo impact:
        "Low"    → GRAY   (#9E9E9E), opacity = 0.5
        "Medium" → YELLOW (#FFC107), opacity = 0.7
        "High"   → RED    (#F44336), opacity = 0.9
    
    Line width = 1px (thin, không che khuất giá)
    Style = DOTTED cho tương lai; SOLID cho đã xảy ra
    
    TOOLTIP khi hover (mouseover/touchstart):
        Nội dung:
            Tên sự kiện: "{event_name}"
            Đồng tiền: "{currency}"
            Tác động: "{impact}"
            Dự báo: "{forecast}{unit}" (nếu có)
            Thực tế: "{actual}{unit}" (nếu đã công bố)
            Surprise: "{S:+.2f}σ" (nếu đã có actual)
        Style: Dark tooltip, rounded corners
        Vị trí: Trên cùng của chart, không che nến giá

VÙNG TÔ MÀU GUARDRAIL:
    Khu vực [-15min, +5min] quanh event có impact = "High":
        Tô màu nền nhạt: rgba(244, 67, 54, 0.05)
        Hiển thị label "Pre-News Zone" ở trên cùng
```

#### B.5 Bộ Phân Chia Phiên Giao Dịch (Session Dividers)

```
CẤU HÌNH PHIÊN GIAO DỊCH (từ config/killzones.yaml):
    Asian:  00:00 - 09:00 UTC, màu nền = rgba(33, 150, 243, 0.03)
    London: 07:00 - 16:00 UTC, màu nền = rgba(76, 175, 80, 0.03)
    NY:     12:00 - 21:00 UTC, màu nền = rgba(255, 152, 0, 0.03)
    
    Overlap London-NY (12:00-16:00 UTC):
        màu nền đậm hơn = rgba(156, 39, 176, 0.05)
        Label "Kill Zone" ở đỉnh
    
ĐƯỜNG PHÂN CÁCH PHIÊN:
    Mỗi ngày giao dịch mới (00:00 UTC):
        Vertical line: SOLID, DARK GRAY (#424242), width=1px, opacity=0.4
    Mỗi điểm chuyển phiên:
        Vertical line: DASHED, phù hợp màu phiên, width=0.5px, opacity=0.3
```

---

### C. PHÂN RÃ TÍNH NĂNG CHI TIẾT

#### Tính năng C.1 — Khởi tạo và cập nhật TradingView Chart

- **Đầu vào:** OHLCV series ban đầu (1000 nến lịch sử), timezone của user, khung TF hiển thị mặc định.
- **Biến đổi:** Khởi tạo `createChart()` với thiết lập responsive → Đăng ký CandlestickSeries → Nạp dữ liệu lịch sử.
- **Đầu ra:** Chart được render trên canvas, sẵn sàng nhận update.

#### Tính năng C.2 — Hiển thị Ghost Zones

- **Đầu vào:** `zone_list` từ WebSocket message, danh sách `{zone_id, tf, type, top, bottom, formed_time, p_hold, status}`.
- **Biến đổi:** Với mỗi zone: tính màu theo B.2 → Tạo `RectanglePriceLine` (hoặc custom series primitive) → Đăng ký vào chart.
- **Đầu ra:** Các hình chữ nhật bán trong suốt hiển thị trên chart.

#### Tính năng C.3 — Hiển thị và animate Heatmap

- **Đầu vào:** WebSocket update `{zone_id, new_p_hold}`.
- **Biến đổi:** Tra cứu zone element → Tính màu mới → Áp dụng CSS transition → Update chart primitive.
- **Đầu ra:** Zone đổi màu mượt mà trên chart.

#### Tính năng C.4 — Panel Tranh biện Agent (Sidebar Component)

- **Đầu vào:** `ConsensusResult` từ WebSocket.
- **Biến đổi:** Parse `consensus_rating` → Hiển thị thanh gauge từ -4 đến +4 → Render `reasoning_summary` dưới dạng text → Hiển thị key evidence bullets của Bull/Bear.
- **Đầu ra:** Sidebar component hiển thị kết quả tranh biện có thể đọc được trong < 10 giây.

#### Tính năng C.5 — Widget đếm ngược tin tức

- **Đầu vào:** `CountdownState` cập nhật mỗi giây từ WebSocket.
- **Biến đổi:** Định dạng `seconds_to_next` thành `MM:SS` → Đổi màu khi < 5 phút (đỏ nhấp nháy) → Hiển thị tên sự kiện và mức tác động.
- **Đầu ra:** Widget đếm ngược nhỏ ở góc trên phải, phát âm thanh cảnh báo khi còn 60 giây.

---

### D. CƠ CHẾ PHÒNG VỆ & XỬ LÝ CA BIÊN

#### D.1 — Quá nhiều zones làm chậm render (Performance Degradation)

```
PHÁT HIỆN: Số zone đang hiển thị > MAX_VISIBLE_ZONES (mặc định = 50)
           HOẶC FPS < 30

HÀNH ĐỘNG:
[1] Lọc theo P_hold: chỉ hiển thị zones có P_hold >= 0.50
[2] Nếu vẫn > MAX_VISIBLE_ZONES: ẩn hoàn toàn các LTF zones (M1, M5)
    khi đang nhìn khung H1 trở lên
[3] Sử dụng virtual scrolling cho zone list trong sidebar
[4] Ghi log RENDER_THROTTLE_ACTIVE
```

#### D.2 — Mất kết nối WebSocket (Frontend)

```
PHÁT HIỆN: WebSocket onclose event HOẶC không nhận message trong 10 giây

HÀNH ĐỘNG:
[1] Hiển thị banner đỏ: "⚠ Mất kết nối với engine — Dữ liệu có thể lỗi thời"
[2] Tất cả P_hold values fade to grayscale (opacity giảm về 0.3)
[3] Đồng hồ đếm ngược dừng lại, hiển thị "---"
[4] Tự động reconnect với exponential backoff (1s, 2s, 4s, 8s)
[5] Khi reconnect thành công: request full state refresh từ backend,
    sau đó xóa banner cảnh báo
```

---

# MODULE 7: VỎ BỌC ỨNG DỤNG DESKTOP & GIAO TIẾP SIÊU TỐC
## (Desktop Shell Architecture & Inter-Process Communication)

---

### A. TỔNG QUAN KỸ THUẬT & MỤC TIÊU

Module 7 là "vỏ bọc" bên ngoài biến toàn bộ hệ thống AGENTIC-QUANT — vốn là một tập hợp các tiến trình Python và ứng dụng web React — thành một ứng dụng desktop tích hợp, mượt mà, có thể cài đặt trên Windows và macOS.

**Lý do chọn Tauri thay vì Electron:**

| Tiêu chí | Tauri (Rust) | Electron (Node.js) |
|---|---|---|
| Kích thước bundle | ~5-15 MB | ~100-300 MB |
| RAM sử dụng khi idle | ~30-80 MB | ~200-400 MB |
| Tốc độ khởi động | < 1s | 2-5s |
| Khả năng IPC low-latency | WebSocket / Tauri commands | IPC via Node.js |
| Bảo mật | Allowlist-based permissions | Rộng hơn |

Với yêu cầu độ trễ IPC dưới 50ms, Tauri (Rust backend) là lựa chọn tối ưu.

**Mục tiêu của Module 7:**
1. Đóng gói Python backend và React UI trong một binary có thể phân phối.
2. Quản lý lifecycle của Python subprocess.
3. Cung cấp kênh IPC WebSocket cục bộ với độ trễ < 50ms.

**Giao tiếp với các module khác:**

- **Nhận vào ← Module 3, 4:** Kết quả tính toán (predictions, zones, consensus) để đẩy sang UI
- **Đầu ra → Module 6:** JSON messages qua WebSocket

---

### B. THUẬT TOÁN CỐT LÕI & LOGIC TOÁN HỌC

#### B.1 Kiến Trúc Phân Tầng Tauri

```
SƠ ĐỒ PHÂN TẦNG KIẾN TRÚC TAURI:
─────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────┐
│                  HỆ ĐIỀU HÀNH (Windows / macOS)          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              TAURI PROCESS (Rust Binary)           │  │
│  │  ┌─────────────────────┐  ┌──────────────────────┐│  │
│  │  │  Tauri Core         │  │  Python Subprocess   ││  │
│  │  │  - Window Manager   │  │  (core/ package)     ││  │
│  │  │  - System Tray      │  │  - All Modules 1-5   ││  │
│  │  │  - File System API  │  │  - WebSocket Server  ││  │
│  │  │  - IPC Bridge       │◄─►  Port: 47290         ││  │
│  │  └────────┬────────────┘  └──────────────────────┘│  │
│  │           │ WebView2 / WKWebView                   │  │
│  │  ┌────────▼────────────────────────────────────┐  │  │
│  │  │         REACT UI (Module 6)                  │  │  │
│  │  │  - Rendered in native WebView                │  │  │
│  │  │  - WebSocket Client → ws://localhost:47290   │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────
```

#### B.2 Giao Thức IPC WebSocket — Cấu Trúc Thông Điệp

**Nguyên tắc thiết kế message schema:**
- Mọi message đều có `type` và `payload`
- Tất cả timestamps là Unix milliseconds (int64)
- Số thực được làm tròn đến 4 chữ số thập phân trước khi serialize
- Message size không vượt quá 64KB (nếu lớn hơn: chia nhỏ thành chunks)

**Message Types từ Backend → Frontend:**

```
MESSAGE TYPE: "zone_update"
Tần suất: Mỗi khi một zone thay đổi P_hold hoặc status
Payload: {
    symbol: string,
    zones: [
        {
            zone_id: string,
            tf: string,
            type: "FVG_BULL" | "FVG_BEAR" | "OB_BULL" | "OB_BEAR",
            top: float,
            bottom: float,
            formed_time: int64,
            status: "UNMITIGATED" | "PARTIALLY_MITIGATED" | "MITIGATED",
            p_hold: float,
            w_zone: float
        },
        ...  (tối đa 100 zones mỗi message)
    ]
}

──────────────────────────────────────────────

MESSAGE TYPE: "prediction_update"
Tần suất: Mỗi khi Model A/B hoàn thành inference (≈ mỗi nến M1)
Payload: {
    symbol: string,
    timestamp: int64,
    p_bsl: float,
    p_ssl: float,
    p_lateral: float,
    bsl_target_price: float,
    ssl_target_price: float,
    bsl_target_tf: string,     ← Khung TF của BSL target
    ssl_target_tf: string,
    confidence_qualifier: string
}

──────────────────────────────────────────────

MESSAGE TYPE: "consensus_update"
Tần suất: Sau mỗi vòng tranh biện hoàn thành (≈ mỗi 3-5 phút hoặc khi có BOS/MSS mới)
Payload: {
    symbol: string,
    timestamp: int64,
    consensus_rating: int,       ← [-4, +4]
    preferred_direction: string,
    high_conviction_zone_price: float | null,
    confidence_qualifier: string,
    reasoning_summary: string,
    bull_confidence: float,
    bear_confidence: float,
    key_bull_evidence: [string, string, string],
    key_bear_evidence: [string, string, string],
    precedents_used: int         ← Số lượng RAG precedents đã dùng
}

──────────────────────────────────────────────

MESSAGE TYPE: "countdown_update"
Tần suất: Mỗi giây (nếu active_guardrail = False: mỗi 10 giây)
Payload: {
    next_events: [
        {
            event_name: string,
            currency: string,
            impact: string,
            scheduled_time: int64,
            seconds_to_event: int,
            i_news: float,
            forecast: float | null,
            actual: float | null
        }
    ],
    regime_phase: string,
    active_guardrail: bool
}

──────────────────────────────────────────────

MESSAGE TYPE: "bar_update"
Tần suất: Mỗi tick (hoặc throttle đến 10 fps khi thị trường chậm)
Payload: {
    symbol: string,
    timestamp: int64,
    bid: float,
    ask: float,
    last: float,
    cvd_m1: float,        ← CVD của nến M1 đang hình thành
    iii_m1: float,        ← III của nến M1 đang hình thành
    spread_pips: float
}

──────────────────────────────────────────────

MESSAGE TYPE: "new_bar_closed"
Tần suất: Chỉ khi nến đóng cửa
Payload: {
    symbol: string,
    timeframe: string,
    bar: { open, high, low, close, volume, time: int64 },
    session: string
}

──────────────────────────────────────────────

MESSAGE TYPE: "system_alert"
Tần suất: Khi có sự kiện hệ thống bất thường
Payload: {
    severity: "INFO" | "WARNING" | "CRITICAL",
    code: string,       ← Ví dụ: "FEED_RECONNECTING", "MODEL_DEGRADED"
    message: string,
    timestamp: int64
}
```

**Message Types từ Frontend → Backend:**

```
MESSAGE TYPE: "user_action"
Payload: {
    action: "change_symbol" | "change_timeframe" | "toggle_zone_visibility"
            | "request_full_state" | "toggle_guardrail_override",
    params: { ... }     ← phụ thuộc action
}
```

#### B.3 Vòng Lặp IPC và Đảm Bảo Độ Trễ < 50ms

```
ĐẶC TẢ PIPELINE ĐỘ TRỄ:
─────────────────────────────────────────────────────────────
Mục tiêu: Từ lúc Python tính toán xong → React render xong ≤ 50ms

Phân tích nguồn độ trễ:
    T1: Python serialize JSON:                  ~0.5ms
    T2: Python ghi vào WebSocket send buffer:   ~0.2ms
    T3: OS TCP stack (loopback):                ~0.1ms
    T4: JavaScript WebSocket onmessage:         ~0.5ms
    T5: React state update (useState/Redux):    ~2-5ms
    T6: TradingView chart re-render:            ~5-15ms
    T7: GPU/CPU frame rendering:                ~8-16ms
    ─────────────────────────────────────────────────────────
    TỔNG (best case):                           ~17ms  ✓
    TỔNG (expected):                            ~25-35ms ✓
    TỔNG (worst case, old CPU):                 ~45-48ms ✓ (cần kiểm tra)

Biện pháp đảm bảo:
[1] WebSocket server chạy trên thread riêng biệt trong Python,
    không chia sẻ GIL với tính toán AI
[2] Sử dụng binary MessagePack thay vì JSON cho bar_update messages
    (giảm 40% kích thước message)
[3] React sử dụng useMemo và React.memo để tránh re-render không cần thiết
[4] TradingView chart updates được batch trong requestAnimationFrame
[5] Chart primitive updates sử dụng canvas 2D trực tiếp
    (không qua React reconciliation)
─────────────────────────────────────────────────────────────
```

**Cơ chế đo lường độ trễ thực tế (Latency Telemetry):**

```
LUỒNG ĐO ĐỘ TRỄ:
[1] Backend thêm trường "emit_time_ms" vào mỗi message (Unix ms)
[2] Frontend đo: receive_time_ms = Date.now()
[3] latency = receive_time_ms - emit_time_ms
[4] Frontend giữ moving average của 100 samples
[5] Hiển thị ở góc UI: "Latency: Xms" (màu xanh nếu <30ms, vàng <50ms, đỏ >50ms)
[6] Nếu average latency > 50ms trong 30 giây liên tiếp:
    Phát LATENCY_DEGRADATION alert
    Tự động chuyển bar_update sang chế độ throttle 5fps
```

---

### C. PHÂN RÃ TÍNH NĂNG CHI TIẾT

#### Tính năng C.1 — Khởi động và quản lý Python subprocess

- **Đầu vào:** Cấu hình từ `tauri.conf.json` (đường dẫn đến Python interpreter, entry point script, port number).
- **Biến đổi:** Tauri Rust spawns Python subprocess → Gắn stdout/stderr pipes → Chờ signal "READY" từ Python (Python in ra "AGENTIQ_BACKEND_READY" khi WebSocket server sẵn sàng).
- **Đầu ra:** Python process chạy ngầm; UI hiển thị loading spinner đến khi nhận tín hiệu READY.

#### Tính năng C.2 — WebSocket Server trong Python

- **Đầu vào:** Module outputs (predictions, zones, consensus, countdown, bars).
- **Biến đổi:** `BroadcastDispatcher` đăng ký handlers cho từng event type → Khi nhận event, serialize → gửi đến tất cả connected WebSocket clients.
- **Đầu ra:** JSON messages trên ws://localhost:47290.

#### Tính năng C.3 — WebSocket Client trong React

- **Đầu vào:** WebSocket messages từ backend.
- **Biến đổi:** Custom hook `useWebSocket` → Parse message type → Dispatch đến Redux store slice tương ứng → Components tự động re-render.
- **Đầu ra:** UI cập nhật theo real-time.

#### Tính năng C.4 — System Tray Integration

- **Đầu vào:** Trạng thái hệ thống (connected, degraded, news alert).
- **Biến đổi:** Tauri system tray icon thay đổi theo trạng thái → Click vào tray: hiện/ẩn cửa sổ.
- **Đầu ra:** Tray icon màu xanh (normal), vàng (warning), đỏ (critical); notification popup khi có sự kiện tin tức High Impact sắp xảy ra.

#### Tính năng C.5 — Auto-update Mechanism

- **Đầu vào:** Kiểm tra update server (hoặc GitHub Releases) mỗi khi khởi động.
- **Biến đổi:** So sánh version số → Nếu có bản mới: hiển thị dialog cho phép user chọn update.
- **Đầu ra:** Download và install tự động (không cần uninstall thủ công).

---

### D. CƠ CHẾ PHÒNG VỆ & XỬ LÝ CA BIÊN

#### D.1 — Python subprocess crash (Backend crash)

```
PHÁT HIỆN: Tauri phát hiện Python subprocess exit với code != 0

HÀNH ĐỘNG:
[1] Ghi crash log vào %APPDATA%/agentic-quant/logs/ (Windows)
    hoặc ~/Library/Logs/agentic-quant/ (macOS)
[2] Hiển thị dialog: "Backend gặp lỗi. Khởi động lại?"
[3] NẾU user chọn Yes:
    a) Chờ 2 giây
    b) Spawn lại Python subprocess
    c) Restore UI state từ last known good state (lưu trong localStorage)
[4] NẾU crash xảy ra > 3 lần trong 5 phút:
    Hiển thị dialog với nút "Gửi báo cáo lỗi" (gửi crash log đến developer endpoint)
    Ngừng auto-restart để tránh crash loop
```

#### D.2 — Cổng WebSocket đã bị chiếm (Port Conflict)

```
PHÁT HIỆN: Python WebSocket server không bind được port 47290
           (OSError: Address already in use)

HÀNH ĐỘNG:
[1] Thử lần lượt các ports: 47291, 47292, ..., 47299
[2] NẾU tìm được port khả dụng:
    Python ghi port thực tế vào file temp: /tmp/aq_ws_port.txt
    Tauri đọc file này → Truyền cho React hook useWebSocket
[3] NẾU không có port nào trong range 47290-47299:
    Hiển thị lỗi: "Không thể khởi động server. Vui lòng tắt ứng dụng khác 
    đang sử dụng các cổng 47290-47299 và thử lại."
```

#### D.3 — Xung đột version dữ liệu (Schema Migration)

```
PHÁT HIỆN: SQLite schema version trong agentic_quant.db khác với 
           schema version trong code (SCHEMA_VERSION constant)

HÀNH ĐỘNG:
[1] NẾU db_version < code_version:
    Chạy migration scripts tuần tự:
    migration_001.sql, migration_002.sql, ...
    đến khi db_version = code_version
[2] NẾU db_version > code_version:
    Cảnh báo: "Database được tạo từ phiên bản mới hơn.
    Có thể mất dữ liệu nếu tiếp tục. Khuyến nghị cập nhật ứng dụng."
[3] Luôn tạo backup trước khi migration: agentic_quant.db.backup.YYYYMMDD
```

---

## PHẦN IV — CÁC TÍNH NĂNG VẬN HÀNH ĐẶC BIỆT TRÊN TOÀN HỆ THỐNG

---

### IV.1 PHÂN ĐỊNH RANH GIỚI PHIÊN GIAO DỊCH (KILLZONES)

**Mục tiêu:** Thị trường ngoại hối và vàng có hành vi phân kỳ rõ rệt theo phiên giao dịch. Phiên Á thường có thanh khoản thấp và giá đi ngang (tích lũy); phiên London và NY Open thường có biến động lớn nhất và xác suất quét thanh khoản cao nhất.

**Cấu hình Killzone (từ `config/killzones.yaml`):**

```
KILLZONE_DEFINITIONS:
    Asian_Session:
        utc_start: "22:00"   ← ngày hôm trước
        utc_end:   "07:00"
        behavior_profile: "ACCUMULATION"
        ltf_signal_weight: 0.6     ← Giảm trọng số tín hiệu LTF
        htf_signal_weight: 1.0

    London_Open_Killzone:
        utc_start: "07:00"
        utc_end:   "09:00"
        behavior_profile: "EXPANSION"
        ltf_signal_weight: 1.2     ← Tăng nhẹ
        htf_signal_weight: 1.5     ← Tăng mạnh
        
    London_Session:
        utc_start: "09:00"
        utc_end:   "12:00"
        behavior_profile: "TRENDING"
        ltf_signal_weight: 1.0
        htf_signal_weight: 1.2

    NY_Open_Killzone:
        utc_start: "12:00"
        utc_end:   "14:00"
        behavior_profile: "EXPANSION"
        ltf_signal_weight: 1.3
        htf_signal_weight: 1.6     ← Killzone quan trọng nhất

    NY_AM_Session:
        utc_start: "14:00"
        utc_end:   "17:00"
        behavior_profile: "TRENDING"

    NY_PM_Session:
        utc_start: "17:00"
        utc_end:   "20:00"
        behavior_profile: "REVERSAL_RISK"
        ltf_signal_weight: 0.8
        htf_signal_weight: 1.0
```

**Logic thay đổi hành vi phân tích theo phiên:**

```
THUẬT TOÁN ÁP DỤNG SESSION WEIGHTS VÀO MODEL A/B:
─────────────────────────────────────────────────────────────
[1] Xác định phiên hiện tại dựa trên UTC time:
    current_session = lookup_session(utc_now())

[2] Lấy behavioral weights từ KILLZONE_DEFINITIONS

[3] Áp dụng vào P_BSL / P_SSL output của Model A:
    
    NẾU current_session.behavior_profile = "ACCUMULATION":
        P_lateral += (P_BSL + P_SSL) × 0.2
        P_BSL    -= P_BSL × 0.2
        P_SSL    -= P_SSL × 0.2
        Tái chuẩn hóa
    
    NẾU current_session.behavior_profile = "EXPANSION":
        Không điều chỉnh P_BSL/P_SSL (tin tưởng mô hình trong Kill Zone)
        Nhưng TĂNG độ tin cậy hiển thị (confidence_qualifier = "HIGH" nếu >= MEDIUM)
    
    NẾU current_session.behavior_profile = "REVERSAL_RISK":
        Tăng trọng số của Bear Evidence trong Critic Agent thêm 10%
        (phiên chiều NY thường đảo chiều xu hướng buổi sáng)

[4] Áp dụng session_weight vào feature engineering:
    Tất cả P_hold của FVG/OB LTF được nhân với ltf_signal_weight
    Tất cả P_hold của FVG/OB HTF được nhân với htf_signal_weight
    → Khi trong London Open KZ: tín hiệu HTF H4/H1 được ưu tiên tuyệt đối
─────────────────────────────────────────────────────────────
```

---

### IV.2 CƠ CHẾ PHÒNG VỆ KHI RA TIN (NEWS-REGIME GUARDRAILS)

**Mục tiêu:** Trong cửa sổ tin tức, mọi mô hình kỹ thuật truyền thống đều trở nên kém tin cậy vì giá có thể di chuyển hàng chục pip chỉ trong vài giây. Hệ thống phải tự động chuyển sang chế độ "hỗ trợ tin tức" thay vì "phân tích kỹ thuật thuần túy".

**Bản đồ chuyển trạng thái (State Transition Map):**

```
TRẠNG THÁI HỆ THỐNG KHI VÀO CHẾ ĐỘ TIN TỨC:
─────────────────────────────────────────────────────────────
TRIGGER: seconds_to_next <= 900 (15 phút) VÀ impact = "High"

THAY ĐỔI TRONG MODULE 3 (AI ENGINE):
    [a] Tắt tính năng tìm kiếm FVG/OB mới trên LTF (M1, M5)
        Lý do: FVG/OB hình thành ngay trước tin tức thường bị "blown out"
    [b] Tăng ngưỡng P_hold tối thiểu để hiển thị từ 0.50 lên 0.65
        Lý do: Chỉ hiển thị những vùng đủ mạnh để chịu được volatility
    [c] Áp dụng guardrail_dampening lên P_BSL / P_SSL (xem Module 3 B.3.C.3)
    [d] Multi-Agent Debate: Thêm paragraph vào System Prompt của cả 3 agents:
        "QUAN TRỌNG: Hiện đang trong cửa sổ Pre-News ({event_name}, 
        impact: High, I_news={I_news:.2f}). Đây là yếu tố bất định CỰC CAO.
        Bạn PHẢI đề cập đến rủi ro tin tức trong luận điểm của mình và
        KHÔNG được đưa ra mức tự tin cao hơn 0.65."

THAY ĐỔI TRONG MODULE 6 (VISUALIZATION):
    [a] Tất cả LTF zones (M1, M5): giảm opacity × 0.4, thêm dấu hỏi (?)
    [b] Hiển thị banner màu đỏ: "⚠ PRE-NEWS: {event_name} in {MM}:{SS}"
    [c] Phóng to widget đếm ngược (chiếm 30% sidebar width)
    [d] Vẽ shaded zone [-15min, +5min] trên chart

THAY ĐỔI TRONG MODULE 2 (MACRO ENGINE):
    [a] Tăng tần suất scraping kết quả actual lên mỗi 15 giây 
        (từ lúc T-0 đến T+5 phút)
    [b] Ngay khi nhận actual: Tính Surprise Factor S và phát NEWS_RELEASE_EVENT

TRIGGER ĐẶT LẠI: 15 phút sau khi actual được công bố
    HOẶC khi seconds_to_next > 900 (sự kiện đã qua, không có sự kiện mới gần)
─────────────────────────────────────────────────────────────
```

**Phản ứng hệ thống khi nhận NEWS_RELEASE_EVENT:**

```
PHẢN ỨNG TỨC THÌ KHI CÓ KẾT QUẢ TIN TỨC:
─────────────────────────────────────────────────────────────
Đầu vào: NewsReleaseEvent {S, direction, I_news}

[1] NẾU |S| > 2.0 (Major Surprise):
    a) Phát MAJOR_SURPRISE_FLAG đến Module 3
    b) Module 3 tức thì khởi động vòng Debate mới (emergency debate)
       với thêm thông tin surprise vào Brief:
       "MAJOR SURPRISE: Kết quả [{actual}{unit}] lệch {S:.1f}σ so với dự báo
        [{forecast}{unit}]. Hướng surprise: {direction}"
    c) Model A recalibrate P_BSL/P_SSL:
       NẾU direction = "BULLISH_SURPRISE":
           P_BSL_new = P_BSL_old × (1 + |S| × 0.15)  (cập nhật Bayesian)
           Tái chuẩn hóa
       NẾU direction = "BEARISH_SURPRISE": Ngược lại

[2] Module 3 tìm kiếm HTF BSL/SSL gần nhất phù hợp với hướng surprise
    → Đây là "Liquidity Target" chính sau tin tức

[3] Module 6 cập nhật ngay:
    - Hiển thị nhãn "SURPRISE" bên cạnh đường dọc tin tức
    - Thêm S vào tooltip
    - Tô màu đường dọc đậm hơn (phản ánh mức độ bất ngờ)
─────────────────────────────────────────────────────────────
```

---

## PHẦN V — PHỤ LỤC: CÁC CÔNG THỨC TOÁN HỌC TỔNG HỢP

| Ký hiệu | Định nghĩa | Module sử dụng |
|---|---|---|
| $I_{news}$ | Chỉ số tác động biến động tin tức | 2, 3 |
| $S$ | Surprise Factor (Z-score kết quả actual vs forecast) | 2, 3 |
| $P_{BSL}$ | Xác suất giá đến vùng Buyside Liquidity tiếp theo | 3 |
| $P_{SSL}$ | Xác suất giá đến vùng Sellside Liquidity tiếp theo | 3 |
| $P_{hold}$ | Xác suất vùng FVG/OB giữ được giá khi bị test | 3 |
| $P_{lateral}$ | Xác suất giá đi ngang (không đến BSL hay SSL) | 3 |
| $CVD_t$ | Cumulative Volume Delta của nến $t$ | 1 |
| $OBI$ | Order Book Imbalance tại bid/ask tốt nhất | 1 |
| $III_t$ | Institutional Intensity Index của nến $t$ | 1 |
| $w_{zone}$ | Trọng số điều chỉnh FVG/OB theo alignment HTF | 3 |
| $IC$ | Information Coefficient (Spearman rank correlation) | 5 |
| $BS_A$ | Brier Score của Model A | 5 |
| $PSI_f$ | Population Stability Index của đặc trưng $f$ | 5 |
| $FDS$ | Feature Drift Score tổng hợp | 5 |
| $z$ | Unified Latent Vector từ LSTM Autoencoder ($\in \mathbb{R}^{512}$) | 3 |
| $e_{USV}$ | Embedding của Unified State Vector để tìm kiếm trong Vector DB | 3, 4 |

---

## PHẦN VI — PHỤ LỤC: MÔ HÌNH TRIỂN KHAI VÀ CẤU HÌNH HỆ THỐNG

### VI.1 Cấu hình Phần Cứng Tối Thiểu Khuyến Nghị

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| CPU | Intel i5-8th gen / Ryzen 5 3600 | Intel i7-12th gen / Ryzen 7 5800X |
| RAM | 16 GB | 32 GB |
| GPU | Không bắt buộc | NVIDIA GTX 1660+ (cho inference nhanh hơn) |
| SSD | 256 GB NVMe | 512 GB NVMe |
| Mạng | 50 Mbps | 100 Mbps, độ trễ < 20ms đến broker |
| OS | Windows 10 (64-bit) / macOS 12+ | Windows 11 / macOS 14+ |

### VI.2 Quy Trình Khởi Động Hệ Thống (Cold Start Sequence)

```
TRÌNH TỰ KHỞI ĐỘNG AGENTIC-QUANT:
─────────────────────────────────────────────────────────────
T+0.0s: Tauri binary khởi động, load WebView
T+0.2s: Hiển thị splash screen với logo
T+0.5s: Spawn Python subprocess
T+1.0s: Python khởi tạo:
    - Nạp config từ YAML files
    - Kết nối SQLite, chạy migration nếu cần
    - Kết nối Redis (nếu không có: dùng in-memory dict)
    - Nạp ChromaDB
    - Load Model A và Model B weights vào bộ nhớ
    - Warm up LSTM Autoencoder với dummy data
T+3.0s: Python kết nối ZeroMQ → MT5 EA
T+3.5s: Python thu thập lịch kinh tế từ web (parallel)
T+4.0s: Python khởi động WebSocket server tại port 47290
T+4.1s: Python in ra "AGENTIQ_BACKEND_READY" vào stdout
T+4.2s: Tauri nhận READY signal → React UI kết nối WebSocket
T+4.5s: React request full_state_snapshot từ backend
T+5.0s: Backend gửi toàn bộ zones, predictions, macro context
T+5.5s: React render chart và tất cả overlays
T+6.0s: Splash screen ẩn đi → UI hiển thị đầy đủ
─────────────────────────────────────────────────────────────
TỔNG THỜI GIAN COLD START MỤC TIÊU: < 8 giây
```

### VI.3 Vòng Đời Mô Hình (Model Lifecycle)

```
HUẤN LUYỆN BAN ĐẦU (Offline):
[1] Thu thập 2+ năm dữ liệu tick lịch sử
[2] Chạy toàn bộ pipeline feature engineering trong backtest mode
[3] Xây dựng dataset (X, y) cho Model A và Model B
[4] Huấn luyện LSTM Autoencoder (unsupervised) → tạo z vectors
[5] Huấn luyện Model A: predict P_BSL/P_SSL sử dụng feature vectors
[6] Huấn luyện Model B: classify P_hold cho từng zone
[7] Đánh giá trên out-of-sample (3 tháng cuối)
[8] Export weights và feature scalers

TINH CHỈNH ĐỊNH KỲ (Online Fine-tuning):
Điều kiện kích hoạt: MODEL_DEGRADED = True VÀ có >= 200 mẫu mới
[1] Thu thập 200 prediction-outcome pairs mới nhất
[2] Chạy fine-tuning với learning rate nhỏ (1e-5) trong 5 epochs
[3] Đánh giá IC_new trên 50 mẫu hold-out gần nhất
[4] NẾU IC_new > IC_current × 1.1:
    Deploy model mới, reset MODEL_DEGRADED = False
[5] NẾU IC_new <= IC_current:
    Giữ nguyên model cũ, tăng cường guardrails

ROLLBACK:
    Giữ 3 phiên bản model gần nhất trong thư mục models/
    Có thể rollback thủ công qua UI settings
```

---

*Tài liệu này được hoàn thành theo yêu cầu đặc tả toàn diện cho hệ thống AGENTIC-QUANT phiên bản 1.0. Mọi thay đổi kiến trúc phải được phản ánh vào tài liệu này và được đánh phiên bản tương ứng. Người phê duyệt kiến trúc cần ký xác nhận trước khi triển khai lên môi trường production.*

---
**Phiên bản tài liệu:** 1.0  
**Ngày tạo:** [Điền ngày tạo]  
**Phạm vi áp dụng:** Toàn đội ngũ kỹ thuật AGENTIC-QUANT  
**Ngôn ngữ:** Tiếng Việt (theo yêu cầu)  
**Tổng số module:** 7  
**Tổng số trang ước tính khi in:** ~85 trang A4