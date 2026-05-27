# YÊU CẦU ĐẶC TẢ VÀ KIẾN TRÚC HỆ THỐNG: HYBRID AGENTIC QUANTITATIVE TRADING FRAMEWORK (AGENTIC-QUANT)

**VAI TRÒ CỦA CLAUDE:** Bạn đóng vai trò là một Kiến trúc sư trưởng Hệ thống Tài chính (Principal Financial Systems Architect) và Chiến lược gia Định lượng cấp cao (Lead Quantitative Strategist).

**RÀO CẢN NGHIÊM NGẶT (STRICT CONSTRAINT):** TUYỆT ĐỐI KHÔNG TỰ ĐỘNG SINH MÃ NGUỒN (Không viết code Python, TypeScript, Rust, SQL hoặc HTML/CSS). Nhiệm vụ duy nhất của bạn trong phiên làm việc này là xây dựng một **Tài liệu Đặc tả Kỹ thuật và Kiến trúc Hệ thống** toàn diện, đạt tiêu chuẩn vận hành thực tế (Production-grade). Tài liệu phải được viết hoàn toàn bằng **TIẾNG VIỆT**. Mỗi module và tính năng dưới đây phải được giải trình chi tiết từ cấu trúc logic, thuật toán cho đến mô hình toán học nâng cao.

---

## 1. TRIẾT LÝ CỐT LÕI: DÒNG LỆNH ĐA KHUNG THỜI GIAN, TIN TỨC VÀ LỰC HÚT THANH KHOẢN

Hệ thống **AGENTIC-QUANT** hoạt động như một Trợ lý AI Co-pilot siêu cấp, chịu trách nhiệm dự đoán **Luồng phân phối giá (Price Delivery), Thiên kiến tổ chức (Institutional Bias), và Vùng hút thanh khoản (Liquidity Draw)** theo thời gian thực để hỗ trợ quá trình ra quyết định của Trader con người.

### Các Trụ Cột Dự Đoán Thời Gian Thực:
1. **Đồng bộ cấu trúc Đa khung thời gian (MTF Alignment):** Hệ thống liên tục đánh giá cấu trúc thị trường từ Khung thời gian cao (HTF: D1, H4, H1) đến Khung thời gian thấp (LTF: M15, M5, M1) nhằm lọc bỏ nhiễu thị trường.
2. **Tích hợp Lịch kinh tế & Định giá biến động:** Các sự kiện tin tức tác động mạnh (CPI, FOMC, NFP) được xem là chất xúc tác tạo thanh khoản. AI sẽ tự động điều chỉnh trọng số và độ tin cậy ($P_{hold}$ và $P_{draw}$) khi thời gian đếm ngược đến tin tức co hẹp lại.
3. **Phân loại vùng hút thanh khoản (Buyside vs. Sellside):** Dự đoán hướng đi tiếp theo của giá sẽ quét vùng thanh khoản nào trước ($P_{BSL}$ so với $P_{SSL}$) dựa trên sự mất cân bằng sổ lệnh, khối lượng phiên và cấu trúc HTF.
4. **Xác thực hiệu lực của các vùng cấu trúc (FVG / OB Efficiency):** Tính toán xác suất một vùng FVG hoặc OB thuộc khung LTF có giữ được giá hay không ($P_{hold}$) khi nó nằm lồng trong một vùng cấu trúc của HTF.

---

## 2. CẤU TRÚC BẮT BUỘC CỦA TÀI LIỆU ĐẶC TẢ

Đối với mỗi Module trong số 7 Module được liệt kê dưới đây, tài liệu của bạn PHẢI phân tách thành 4 mục nhỏ rõ ràng sau:
* **A. Tổng quan Kỹ thuật & Mục tiêu:** Module này giải quyết bài toán gì? Nó giao tiếp và truyền nhận dữ liệu với các module khác như thế nào?
* **B. Thuật toán Cốt lõi & Logic Toán học:** Giải trình chi tiết các công thức toán học, các bước thuật toán, cấu trúc dữ liệu hoặc luồng logic (Thể hiện bằng mã giả - Pseudo-code hoặc sơ đồ logic từng bước, KHÔNG dùng code lập trình thực tế).
* **C. Phân rã Tính năng Chi tiết:** Liệt kê từng tính năng nhỏ bên trong module. Đối với mỗi tính năng phải chỉ rõ: Dữ liệu đầu vào (Input), Logic biến đổi/xử lý (Transformation), và Trạng thái đầu ra dự kiến (Output).
* **D. Cơ chế Phòng vệ & Xử lý Ca biên (Fail-safe & Edge-cases):** Đặc tả chi tiết cách hệ thống xử lý khi gặp sự cố (Ví dụ: Mất kết nối dữ liệu, giá giật mạnh khi ra tin, bất đối xứng dữ liệu giữa các khung thời gian, mô hình AI bị suy giảm độ chính xác).

---

## 3. CÁC MODULE CHÍNH CẦN LẬP TÀI LIỆU CHI TIẾT

### MODULE 1: Bộ Đón Nhận & Đồng Bộ Dữ Liệu Đa Khung Thời Gian (Data Ingestion Engine)
* **Thu thập dữ liệu Tick & OHLCV:** Cơ chế kết nối thời gian thực với MetaTrader 5 (Tick-by-Tick) và TradingView Webhook / MCP Server.
* **Bộ đồng bộ Đa khung thời gian (MTF Synchronizer):** Thuật toán căn chỉnh các luồng dữ liệu bất đồng bộ (từ M1 đến D1) thành một Vector trạng thái thống nhất mà không bị lỗi nhìn trước tương lai (Look-ahead bias / Data leakage) khi kiểm thử lịch sử.
* **Đo lường Sổ lệnh & Khối lượng (Volumetrics):** Đặc tả cách trích xuất dữ liệu Độ sâu thị trường (DOM - Level 2) hoặc sự mất cân bằng của Tick Volume để phát hiện dấu chân của các tổ chức tài chính lớn.

### MODULE 2: Bộ Xử Lý Lịch Kinh Tế Vĩ Mô (Macro Calendar Engine)
* **Thu thập và Vector hóa tin tức:** Cơ chế tự động quét lịch kinh tế (ForexFactory/Investing) và chuyển đổi các danh mục tin tức (Thấp, Trung bình, Cao/Hộp đỏ) thành các Vector số biểu thị tác động biến động ($I_{news}$).
* **Bộ đếm ngược biến động vĩ mô (Volatility Countdown):** Luồng logic của bộ đếm thời gian thực (tính bằng giây). Khi đồng hồ tiến về 0, nó phải kích hoạt các trạng thái thay đổi trong AI Engine để chuẩn bị cho kịch bản tin tức quét thanh khoản.

### MODULE 3: AI Engine & Tiến Trình Xử Lý Thần Kinh - Ký Hiệu Đa Khung Thời Gian
* **Trích xuất đặc trưng SMC/ICT (Symbolic Feature Engineering):** Định nghĩa thuật toán toán học để tự động nhận diện chính xác các đỉnh/đáy sóng (BSL/SSL), Vùng Premium/Discount, Điểm dịch chuyển cấu trúc (MSS), Phá vỡ cấu trúc (BOS), và các vùng FVG/OB chưa bị giảm thiểu (Unmitigated).
* **Nén đặc trưng bằng mạng thần kinh (Neural Compression):** Kiến trúc phân cấp của mạng Hierarchical LSTM Autoencoder nhằm nén các chuỗi dữ liệu tick liên tục thành các vector tiềm ẩn (Latent vectors) không gian - thời gian.
* **Mô hình Học máy Thời gian thực:** Thiết kế chi tiết cho **Model A** (Dự đoán vùng hút thanh khoản tiếp theo: $P_{BSL}$ vs $P_{SSL}$) và **Model B** (Dự đoán hiệu lực giữ giá của FVG/OB: $P_{hold}$).
* **Lớp Tranh biện Đa tác nhân GenAI (Multi-Agent Debate):** Bản thiết kế Prompt Hệ thống (System Prompt) và giao thức tranh biện giữa Bull Agent (Thiên kiến tăng), Bear Agent (Thiên kiến giảm) và Critic Agent (Trọng tài tổng hợp). Chỉ rõ cách các Agent truy vấn Vector DB, đối chiếu dữ liệu định lượng cứng để đưa ra Điểm số đồng thuận (Consensus Rating).

### MODULE 4: Bộ Quản Lý Trạng Thái & Lưu Trữ Hệ Thống (Memory & Persistence Engine)
* **Bộ nhớ đệm ngắn hạn (Short-Term Hybrid Memory):** Thiết kế cấu trúc lưu trữ trên RAM (In-memory Key-Value như Redis hoặc Dictionary hiệu năng cao) để quản lý các mức cản đang kích hoạt, trạng thái đếm ngược tin tức và tiến trình tranh biện của các Agent.
* **Bộ nhớ dài hạn (Long-Term Relational & Semantic Memory):** Thiết kế lược đồ dữ liệu (Schema) cho SQLite (lưu lịch sử dự đoán, độ chính xác của mô hình sau tin tức) và Vector DB (ChromaDB/Qdrant) để lưu trữ các phân tích tranh biện trong quá khứ phục vụ cho việc truy xuất ngữ cảnh (RAG).

### MODULE 5: Bộ Kiểm Thử Dựa Trên Sự Kiện & Đánh Giá Sai Lệch (Backtesting & Evaluation Engine)
* **Bộ giả lập dữ liệu lịch sử dạng Tick (Event-Driven Simulator):** Logic cốt lõi của vòng lặp backtest. Cơ chế nạp dữ liệu giá quá khứ song song với dữ liệu lịch kinh tế để đánh giá hiệu suất của AI.
* **Đo lường sai lệch mô hình (Drift Tracking):** Công thức toán học tính toán Hệ số thông tin (Information Coefficient), Precision/Recall của $P_{hold}$ và thuật toán phát hiện hiện tượng Dịch chuyển đặc trưng (Feature Drift) khi thị trường thay đổi trạng thái (Regime Shift).

### MODULE 6: Giao Diện Trực Quan Hóa Đồ Thị (TradingView Lightweight Charts)
* **Bộ phủ vùng cấu trúc đa khung thời gian (MTF Ghost Zones):** Quy tắc tính toán tọa độ để vẽ đè các vùng OB/FVG của khung lớn (H4/H1) dưới dạng các "vùng bóng ma" (Ghost zones) mờ trực tiếp lên đồ thị khung nhỏ (M1/M5) trong React.
* **Bản đồ nhiệt AI & Đường mục tiêu (Dynamic Heatmaps):** Quy tắc thay đổi độ mờ (Opacity) và màu sắc của các khối FVG dựa trên tỷ lệ $P_{hold}$ theo thời gian thực; và vẽ các đường nằm ngang hiển thị mục tiêu BSL/SSL kèm nhãn phần trăm $P_{draw}$.
* **Trục thời gian sự kiện vĩ mô (Macro Timeline):** Quy tắc nhúng các đường thẳng đứng (Vertical bars) tương ứng với thời gian ra tin tức, hỗ trợ tính năng rê chuột hiển thị tooltip tên tin tức và mức độ tác động.

### MODULE 7: Vỏ Bọc Ứng Dụng Desktop & Giao Tiếp Siêu Tốc (IPC)
* **Kiến trúc Tauri / Electron:** Sơ đồ tổ chức phân tầng giữa UI và tầng tài nguyên hệ thống cục bộ.
* **Giao thức IPC WebSocket Bất đồng bộ:** Thiết kế cấu trúc thông điệp (Message Schema) và vòng lặp truyền tải qua Local WebSockets (`ws://localhost:port`) đảm bảo độ trễ từ lúc Python tính toán đến khi React hiển thị dưới 50ms.

---

## 4. CÁC TÍNH NĂNG VẬN HÀNH ĐẶC BIỆT TRÊN TOÀN HỆ THỐNG
* **Phân định ranh giới phiên giao dịch (Killzones):** Logic thay đổi hành vi phân tích theo các phiên Á, Âu (London Open), Mỹ (NY Open) và phiên chiều Mỹ (PM Session). Kẻ các đường phân cách phiên trên UI.
* **Cơ chế phòng vệ khi ra tin (News-Regime Guardrails):** Logic chuyển đổi trạng thái toàn diện của hệ thống khi bước vào "Chế độ Tin tức" (15 phút trước sự kiện). Hệ thống phải tự động điều chỉnh trọng số: Giảm độ tin cậy của các cấu trúc LTF yếu và ưu tiên dự đoán các vùng thanh khoản lớn của HTF.

---

## 5. HƯỚNG DẪN HOÀN THÀNH TÀI LIỆU

Hãy viết toàn bộ bản thiết kế kiến trúc hệ thống này một cách chi tiết, mạch lạc, không viết tóm tắt hay cắt ngắn tài liệu. Sử dụng các thuật ngữ tài chính định lượng và kiến trúc phần mềm chuyên nghiệp nhất. Đảm bảo **không sử dụng các từ chung chung hoặc các ký tự chờ điền như "sẽ được định nghĩa sau", "// TODO"**. 

Bắt đầu trực tiếp bằng việc mô tả sơ đồ cây thư mục (Directory Tree) tổng quan của toàn dự án, sau đó đi thẳng vào nội dung chi tiết của **MODULE 1**. Tuyệt đối không tạo ra các khối mã nguồn (code blocks) chứa mã lập trình chạy được.