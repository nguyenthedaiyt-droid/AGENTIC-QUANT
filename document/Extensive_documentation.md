# TÀI LIỆU MỞ RỘNG ĐẶC TẢ KỸ THUẬT NÂNG CAO
# AGENTIC-QUANT ANALYTICS EXTENSION — Phiên bản 4.5
### Toán học hóa Chỉ báo · Kiến trúc XGBoost · Hệ thống Cơ sở dữ liệu Hỗn hợp

---

> **Phạm vi tài liệu:** Đây là tài liệu bổ sung và làm sâu thêm cho Phiên bản 1.0 cơ sở. Tài liệu tập trung vào ba cấu phần kỹ thuật chưa được triển khai đầy đủ trong V1.0: (1) Toán học hóa chính xác các chỉ báo từ logic Pine Script, (2) Thiết kế chi tiết hai mô hình XGBoost, và (3) Kiến trúc hệ thống cơ sở dữ liệu hỗn hợp ba tầng. Mọi biểu diễn logic đều sử dụng ký hiệu toán học, mã giả cấu trúc, và sơ đồ tuần tự — không chứa mã nguồn thực thi.

---

## PHẦN I — TOÁN HỌC HÓA CHỈ BÁO & THIẾT KẾ ĐẶC TRƯNG AI
### (AI Feature Engineering: Mathematical Formalization of Institutional Order Flow)

---

### I.1 PHÂN LOẠI SWING POINTS VÀ THANH KHOẢN ĐA TẦNG (ST / IT / LT LIQUIDITY)

#### I.1.1 Nền Tảng Lý Thuyết: Cấu Trúc Pivot Phân Cấp

Hệ thống phân loại Swing Points trong Pine Script gốc (`ICT Institutional Order Flow`) hoạt động theo nguyên tắc đệ quy phân cấp: mỗi Pivot ở tầng cao hơn (IT, LT) phải được xác nhận bởi ba Pivot tầng thấp hơn liên tiếp. Tài liệu này toán học hóa toàn bộ cơ chế này.

**Định nghĩa tập hợp Pivot thô (Raw Pivot Set):**

Tại thời điểm nến $t$ (đã đóng), định nghĩa hai vị từ nhận diện đỉnh/đáy sóng:

$$\Pi^H_t = \mathbb{1}\left[high_t > high_{t-1}^{EQ^+} \;\wedge\; high_t > high_{t+1}\right]$$

$$\Pi^L_t = \mathbb{1}\left[low_t < low_{t-1}^{EQ^-} \;\wedge\; low_t < low_{t+1}\right]$$

Trong đó toán tử `EQ` (Equal Skip) xử lý chuỗi nến có giá trị bằng nhau liên tiếp:

$$high_{t-1}^{EQ^+} = high_{t - \delta^+}, \quad \delta^+ = \min\left\{k \geq 1 : high_{t-k} \neq high_{t-k-1}\right\}$$

$$low_{t-1}^{EQ^-} = low_{t - \delta^-}, \quad \delta^- = \min\left\{k \geq 1 : low_{t-k} \neq low_{t-k-1}\right\}$$

*Ý nghĩa:* Bộ lọc EQ bỏ qua các chuỗi giá "bằng phẳng" (equal highs/lows liên tiếp) để không nhận diện nhầm đỉnh/đáy ảo trong vùng tích lũy ngang.

**Tập hợp Pivot thô:**

$$\mathcal{P}^H = \{t \in \mathbb{Z}_+ : \Pi^H_t = 1\}, \quad \mathcal{P}^L = \{t \in \mathbb{Z}_+ : \Pi^L_t = 1\}$$

#### I.1.2 Thuật Toán Phân Tầng ST → IT → LT

Ký hiệu ba mảng lưu trữ Pivot Highs đã phân loại:
- $\mathcal{STH}$: Short-Term Highs (hàng đợi theo thứ tự thời gian giảm dần)
- $\mathcal{ITH}$: Intermediate-Term Highs
- $\mathcal{LTH}$: Long-Term Highs

Tương tự cho Lows: $\mathcal{STL}$, $\mathcal{ITL}$, $\mathcal{LTL}$.

**Quy tắc thăng cấp ST → IT (FindIT):**

Cho ba Pivot Highs liên tiếp gần nhất trong $\mathcal{STH}$: $h_1$ (mới nhất), $h_2$, $h_3$ (cũ nhất):

$$h_2 \in \mathcal{ITH} \iff h_2.\text{price} > h_3^{EQ}.\text{price} \;\wedge\; h_2.\text{price} > h_1.\text{price} \;\wedge\; h_2 \notin \mathcal{ITH}$$

Trong đó $h_3^{EQ}$ là Pivot thứ ba sau khi bỏ qua các Pivot có giá bằng nhau liên tiếp (SkipEQPivot).

Quy tắc tương tự cho Lows với chiều ngược lại:
$$l_2 \in \mathcal{ITL} \iff l_2.\text{price} < l_3^{EQ}.\text{price} \;\wedge\; l_2.\text{price} < l_1.\text{price} \;\wedge\; l_2 \notin \mathcal{ITL}$$

**Quy tắc thăng cấp IT → LT (FindLT):**

Cho ba Pivot Highs liên tiếp gần nhất trong $\mathcal{ITH}$: $h_1^{IT}$, $h_2^{IT}$, $h_3^{IT}$:

$$h_2^{IT} \in \mathcal{LTH} \iff h_2^{IT}.\text{price} > h_3^{IT}.\text{price} \;\wedge\; h_2^{IT}.\text{price} > h_1^{IT}.\text{price} \;\wedge\; h_2^{IT} \notin \mathcal{LTH}$$

**Phân loại cấu trúc nội bộ (Higher High / Lower Low):**

Khi thêm một Pivot mới $p$ vào $\mathcal{STH}$, phân loại như sau:

$$p.\text{isHigherHigh}^{ST} = \mathbb{1}\left[p.\text{price} \geq \mathcal{STH}[0].\text{price}\right]$$
$$p.\text{isHigherHigh}^{IT} = \mathbb{1}\left[p.\text{price} \geq \mathcal{ITH}[0].\text{price}\right] \quad \text{(khi } p \text{ được thăng cấp lên IT)}$$

*Ý nghĩa kỹ thuật:* Phân loại HH/LH/HL/LL này trực tiếp mã hóa xu hướng thị trường tại từng tầng — đầu vào ngữ nghĩa quan trọng cho XGBoost.

#### I.1.3 Cơ Chế Phát Hiện Thanh Khoản Bị Quét (Claimed Liquidity)

Một Pivot $p$ bị coi là "đã bị quét" (claimed) theo quy tắc:

$$p.\text{claimed} = \begin{cases} \mathbb{1}\left[high_t > p.\text{price}\right] & \text{nếu } p \in \mathcal{STH} \cup \mathcal{ITH} \cup \mathcal{LTH} \\ \mathbb{1}\left[low_t < p.\text{price}\right] & \text{nếu } p \in \mathcal{STL} \cup \mathcal{ITL} \cup \mathcal{LTL} \end{cases}$$

Thời điểm quét: $p.\text{time\_claimed} = t : p.\text{claimed}_t = 1 \;\wedge\; p.\text{claimed}_{t-1} = 0$.

#### I.1.4 Vector Hóa Đặc Trưng Thanh Khoản cho AI

Từ các tập hợp $\mathcal{STH}$, $\mathcal{ITH}$, $\mathcal{LTH}$, $\mathcal{STL}$, $\mathcal{ITL}$, $\mathcal{LTL}$, xây dựng vector đặc trưng thanh khoản $F_{liq} \in \mathbb{R}^{24}$:

$$F_{liq} = \left[\begin{array}{c}
\Delta P_{BSL}^{ST} \\ \Delta P_{BSL}^{IT} \\ \Delta P_{BSL}^{LT} \\
\Delta P_{SSL}^{ST} \\ \Delta P_{SSL}^{IT} \\ \Delta P_{SSL}^{LT} \\
\Delta T_{BSL}^{ST} \\ \Delta T_{BSL}^{IT} \\ \Delta T_{BSL}^{LT} \\
\Delta T_{SSL}^{ST} \\ \Delta T_{SSL}^{IT} \\ \Delta T_{SSL}^{LT} \\
V_{acc,BSL}^{IT} \\ V_{acc,BSL}^{LT} \\
V_{acc,SSL}^{IT} \\ V_{acc,SSL}^{LT} \\
N_{BSL}^{ST} \\ N_{BSL}^{IT} \\ N_{BSL}^{LT} \\
N_{SSL}^{ST} \\ N_{SSL}^{IT} \\ N_{SSL}^{LT} \\
r_{claimed}^{IT} \\ r_{claimed}^{LT}
\end{array}\right]$$

Trong đó:
- $\Delta P_{BSL}^{tier} = \frac{P_{BSL,nearest}^{tier} - P_{current}}{ATR_{H4}}$ — khoảng cách tương đối (pips) từ giá hiện tại đến BSL gần nhất tại tầng tương ứng, chuẩn hóa theo ATR H4
- $\Delta T_{BSL}^{tier}$ — khoảng cách thời gian (tính bằng số nến M1) từ hiện tại đến BSL gần nhất
- $V_{acc,BSL}^{tier}$ — khối lượng tích lũy CVD trong vùng $[P_{BSL} - \epsilon, P_{BSL} + \epsilon]$ ($\epsilon = 0.5 \times ATR_{M1}$), đo lường "độ dày" của thanh khoản
- $N_{BSL}^{tier}$ — số lượng mức BSL chưa bị quét tại tầng đó (chuẩn hóa về $[0,1]$ bằng $\min$-$\max$ trên cửa sổ 200 nến)
- $r_{claimed}^{tier}$ — tỷ lệ Pivot đã bị quét trong 50 Pivot gần nhất tại tầng đó, đo lường "tốc độ tiêu thụ thanh khoản"

---

### I.2 ĐỊNH LƯỢNG HÓA KHOẢNG TRỐNG GIÁ VÀ SỰ MẤT CÂN BẰNG (FVG / iFVG / VI / GAP)

#### I.2.1 Điều Kiện Hình Thành và Hệ Phương Trình Trạng Thái

**Fair Value Gap (FVG) — Định nghĩa chính thức:**

Cho chuỗi ba nến liên tiếp $(n-2, n-1, n)$ đã đóng cửa:

$$FVG\_Bull_n = \mathbb{1}\left[low_n > high_{n-2} \;\wedge\; \neg Gap_{n} \;\wedge\; \neg Gap_{n-1}\right]$$
$$FVG\_Bear_n = \mathbb{1}\left[high_n < low_{n-2} \;\wedge\; \neg Gap_{n} \;\wedge\; \neg Gap_{n-1}\right]$$

Điều kiện bổ sung loại trừ Open Gap thực sự ($Gap_n = \mathbb{1}[low_n > high_{n-1}] \vee \mathbb{1}[high_n < low_{n-1}]$) để FVG không bao gồm khoảng trống mở cửa phiên.

Khi $FVG\_Bull_n = 1$, cấu trúc FVG Bullish được định nghĩa bởi bộ ba:

$$\text{FVG}^+ = \left(o_n = high_{n-2},\; c_n = low_n,\; ce_n = \frac{high_{n-2} + low_n}{2},\; t_{open} = t_{n-1}\right)$$

**Bộ lọc FVG theo cường độ Displacement:**

Trong trường hợp cài đặt không phải "Always Display", FVG chỉ hợp lệ khi:

$$|body_{n-1}| > \sigma_{body,N} \times D_{level}$$

Trong đó:
- $body_{n-1} = |open_{n-1} - close_{n-1}|$
- $\sigma_{body,N} = \text{std}\left(\{|open_t - close_t|\}_{t=t-N}^{t}\right)$, $N = 100$ (mặc định)
- $D_{level} \in \{1, 2, 3, 4\}$ — hệ số độ mạnh Displacement theo cài đặt

**Hệ thống trạng thái FVG (State Machine):**

Mỗi FVG tồn tại trong một trong năm trạng thái chuyển đổi một chiều:

```
TRẠNG THÁI FVG — MÁY TRẠNG THÁI (STATE MACHINE):
──────────────────────────────────────────────────────────────────
Trạng thái 1: UNMITIGATED (Chưa bị giảm thiểu)
    → Chuyển sang WICK_TOUCHED:
       NẾU FVG_Bull VÀ low_t ≤ FVG.close (= low_n)
       NẾU FVG_Bear VÀ high_t ≥ FVG.close (= high_n)

Trạng thái 2: WICK_TOUCHED (Bóng nến chạm vào)
    → Chuyển sang WICK_FILLED:
       NẾU FVG_Bull VÀ low_t ≤ FVG.open (= high_{n-2})
       NẾU FVG_Bear VÀ high_t ≥ FVG.open (= low_{n-2})

Trạng thái 3: WICK_FILLED_HALF (Bóng nến lấp đầy nửa)
    Kích hoạt bởi: low_t ≤ FVG.ce (= Consequent Encroachment)
    
Trạng thái 4: BODY_FILLED (Thân nến lấp đầy)
    Kích hoạt bởi: min(open_t, close_t) ≤ FVG.open (FVG_Bull)
                   max(open_t, close_t) ≥ FVG.open (FVG_Bear)

Trạng thái 5: MITIGATED (Đã bị giảm thiểu hoàn toàn)
    Trạng thái cuối — FVG không còn tác dụng phân tích
──────────────────────────────────────────────────────────────────
```

**Chuyển đổi iFVG (Inverted Fair Value Gap):**

FVG chuyển sang iFVG khi giá đóng cửa xuyên thủng hoàn toàn qua mức mở của FVG ban đầu:

$$\text{iFVG}^+_{from\;FVG^+} = \mathbb{1}\left[close_t < FVG^+.o_n \;\wedge\; FVG^+.\text{state} \neq MITIGATED\right]$$

Khi điều kiện này thỏa mãn, iFVG mới được tạo ra với cấu trúc đảo chiều:

$$\text{iFVG}^- = \left(o = FVG^+.c_n,\; c = FVG^+.o_n,\; t_{open} = FVG^+.t_{open}\right)$$

Ý nghĩa: FVG Bullish gốc đã bị "đảo chiều" — vùng mà trước đây kỳ vọng mua nay trở thành kháng cự.

#### I.2.2 Volume Imbalance (VI) — Mất Cân Bằng Thân Nến

VI phát sinh khi hai thân nến liên tiếp không chồng lấp nhau (không có khoảng trống mở cửa thực sự):

$$VI\_Bull_n = \mathbb{1}\left[\min(o_n, c_n) > \max(o_{n-1}, c_{n-1}) \;\wedge\; \neg Gap_n\right]$$
$$VI\_Bear_n = \mathbb{1}\left[\max(o_n, c_n) < \min(o_{n-1}, c_{n-1}) \;\wedge\; \neg Gap_n\right]$$

Cấu trúc VI:
$$VI^+ = \left(o = \max(o_{n-1}, c_{n-1}),\; c = \min(o_n, c_n),\; ce = \frac{o + c}{2}\right)$$

**Điều kiện gộp FVG + VI (MergeVI):**

Khi `mergeVI = True`, nếu FVG và VI chồng lấp, FVG được mở rộng để bao gồm cả VI:

$$\text{Điều kiện gộp:} \quad FVG_{n-1} = 1 \;\wedge\; VI\_Bull_{n-1} = 1$$
Khi đó: $FVG.c_n \leftarrow \min(o_n, c_n)$ thay vì $low_n$.

#### I.2.3 Open Gap — Khoảng Trống Mở Cửa Phiên

$$Gap\_Bull_n = \mathbb{1}\left[low_n > high_{n-1}\right]$$
$$Gap\_Bear_n = \mathbb{1}\left[high_n < low_{n-1}\right]$$

Open Gap có tính chất "nam châm" mạnh hơn FVG vì nó đại diện cho khoảng giá chưa bao giờ được giao dịch trong phiên trước đó.

#### I.2.4 Consequent Encroachment (C.E.) — Điểm Giữa Khoảng Trống

$$CE = \frac{o_{imbalance} + c_{imbalance}}{2}$$

Đây là mức giá thống kê quan trọng: 70-80% FVG bị giảm thiểu tối thiểu đến mức C.E. trước khi có phản ứng kỹ thuật. Vector đặc trưng luôn bao gồm khoảng cách tương đối từ giá hiện tại đến C.E.:

$$\Delta CE = \frac{CE - P_{current}}{ATR_{current\_tf}}$$

#### I.2.5 Vector Hóa Đặc Trưng FVG/iFVG/VI cho AI

Với mỗi vùng imbalance $k$ (FVG, iFVG, hoặc VI), xây dựng vector đặc trưng vùng $F_{zone}^{(k)} \in \mathbb{R}^{16}$:

$$F_{zone}^{(k)} = \left[\begin{array}{c}
z_{type} \\
z_{tf} \\
\Delta P_{to\_open} \\
\Delta P_{to\_CE} \\
\Delta P_{to\_close} \\
z_{size} \\
z_{age} \\
z_{state} \\
z_{invertible} \\
z_{inverted} \\
V_{formation} \\
V_{mitigation} \\
\Delta T_{mitigation} \\
z_{htf\_alignment} \\
z_{session} \\
z_{displacement\_strength}
\end{array}\right]$$

Chi tiết từng chiều:
- $z_{type}$: mã hóa one-hot — $[FVG^+, FVG^-, iFVG^+, iFVG^-, VI^+, VI^-, GAP^+, GAP^-]$
- $z_{tf}$: mã hóa thứ tự — $[M1=1, M5=2, M15=3, H1=4, H4=5, D1=6]$ (chuẩn hóa về $[0,1]$)
- $\Delta P_{to\_open/CE/close}$: khoảng cách giá tương đối, chuẩn hóa theo ATR của khung TF tương ứng
- $z_{size} = \frac{|o_{imb} - c_{imb}|}{ATR_{tf}}$: kích thước tương đối của vùng
- $z_{age}$: số nến M1 kể từ khi vùng hình thành, chuẩn hóa logarithm: $\log_2(1 + age)$
- $z_{state}$: mã số trạng thái $\{0=UNM, 1=WICK\_T, 2=WICK\_F\_H, 3=BODY\_F, 4=MITIG\}$
- $V_{formation}$: giá trị $III_t$ tại thời điểm vùng hình thành (đã định nghĩa trong V1.0)
- $V_{mitigation}$: giá trị $III_t$ tại thời điểm vùng bị test lần đầu (0 nếu chưa bị test)
- $\Delta T_{mitigation}$: thời gian (số nến M1) từ khi hình thành đến khi bị test lần đầu (0 nếu chưa)
- $z_{htf\_alignment}$: trọng số $w_{zone} \in \{0.5, 1.0, 2.0\}$ theo alignment HTF (xem V1.0 Module 3)
- $z_{session}$: $\{0=Asian, 1=London\_Open\_KZ, 2=London, 3=NY\_Open\_KZ, 4=NY\_AM, 5=NY\_PM\}$
- $z_{displacement\_strength}$: cường độ Displacement tại nến trung tâm ($n-1$), chuẩn hóa theo $\sigma_{body}$

---

### I.3 CHỈ SỐ CƯỜNG ĐỘ TỔ CHỨC VÀ TRƯỢT GIÁ (DISPLACEMENT / CVD / III)

#### I.3.1 Toán Học Hóa Displacement Strength

Hàm `f_highlightDisplacement()` trong Pine Script gốc dựa trên hai điều kiện:

**Điều kiện cơ bản:**

$$Displaced_t = \mathbb{1}\left[|body_{t-1}| > \sigma_{body,N} \times D_{factor}\right] \quad \text{(khi không yêu cầu FVG)}$$

**Điều kiện kết hợp FVG (chế độ mặc định — require FVG = True):**

$$Displaced_t = \mathbb{1}\left[|body_{t-1}| > \sigma_{body,N} \times D_{factor} \;\wedge\; FVG_{t}\right]$$

Trong đó $\sigma_{body,N} = \text{std}\left(\{|o_i - c_i|\}_{i=t-N}^{t}\right)$, $N=100$.

Chỉ số cường độ liên tục (thay vì nhị phân) cho AI:

$$D_{strength,t} = \frac{|body_{t-1}|}{\sigma_{body,N} \times D_{factor}}$$

$D_{strength} > 1$: Displacement đủ điều kiện. Giá trị càng cao → xung lực cú đẩy càng mạnh.

**Phân rã thêm:**

$$D_{strength,t}^{expanded} = \left[\frac{|body_{t-1}|}{\sigma_{body,N}},\; \frac{|body_{t-1}|}{ATR_{14}},\; \mathbb{1}[FVG_t],\; \mathbb{1}[FVG_{t-1}],\; D_{strength,t}\right]$$

Vector 5 chiều này cho phép mô hình tự học trọng số tốt hơn so với scalar đơn lẻ.

#### I.3.2 Cumulative Volume Delta (CVD) — Định Nghĩa Đầy Đủ

Như đã nêu trong V1.0, nhưng mở rộng thêm ba biến thể phục vụ AI:

**CVD tuyến tính (đã định nghĩa trong V1.0):**
$$CVD_t^{bar} = \sum_{i=1}^{N_t} \delta_i$$

**CVD chuẩn hóa theo khối lượng (Normalized CVD):**
$$CVD_t^{norm} = \frac{CVD_t^{bar}}{V_{total,t}}, \quad V_{total,t} = \sum_{i=1}^{N_t} |\delta_i|$$

Giá trị $CVD_t^{norm} \in [-1, +1]$. Đây là chỉ số tinh tế hơn: loại bỏ ảnh hưởng của khối lượng tuyệt đối để so sánh giữa các phiên khác nhau.

**CVD Rolling Window (Cửa sổ trượt):**
$$CVD_{tw,k}^{roll} = \sum_{j=0}^{k-1} \delta_{t-j}, \quad k \in \{5, 10, 20, 50\}$$

Tập hợp $\{CVD_{tw,5}, CVD_{tw,10}, CVD_{tw,20}, CVD_{tw,50}\}$ mã hóa xung lực dòng lệnh ở nhiều độ dài thời gian khác nhau — đặc biệt hữu ích để phát hiện phân kỳ (divergence) giữa CVD ngắn hạn và dài hạn.

**Phân kỳ CVD — Divergence Score:**

$$DIV_{CVD,t} = \text{sign}\left(\Delta P_t^{bar}\right) \times \text{sign}\left(CVD_t^{bar}\right)$$

$DIV_{CVD} = -1$: Phân kỳ âm (giá tăng nhưng CVD âm → dấu hiệu yếu kém ẩn của bull). $DIV_{CVD} = +1$: Hội tụ (giá và CVD đồng hướng → xác nhận).

#### I.3.3 Institutional Intensity Index (III) — Toán Học Hóa Đầy Đủ

Như V1.0, với bổ sung thêm:

$$III_t = \frac{CVD_t^{bar}}{\bar{V}_{30}} \times \frac{|\Delta P_t|}{\sigma_{ATR,14}}$$

**Phân rã hướng (Directional III):**
$$III_t^{bull} = \max(III_t, 0), \quad III_t^{bear} = \min(III_t, 0)$$

**Tích lũy III theo vùng (Zone III Accumulation):**

Với mỗi vùng FVG/OB có khoảng giá $[P_{low}, P_{high}]$:

$$III_{zone} = \sum_{t: P_{low} \leq close_t \leq P_{high}} III_t$$

$III_{zone}$ lớn dương = nhiều tổ chức đã mua trong vùng này → $P_{hold}$ cao cho FVG Bull.

#### I.3.4 Vector Đặc Trưng Tổng Hợp Dòng Lệnh $F_{flow} \in \mathbb{R}^{20}$

$$F_{flow} = \left[\begin{array}{c}
CVD_t^{bar} / ATR_{M1} \\
CVD_t^{norm} \\
CVD_{tw,5}^{roll} / ATR_{M1} \\
CVD_{tw,10}^{roll} / ATR_{M1} \\
CVD_{tw,20}^{roll} / ATR_{M1} \\
CVD_{tw,50}^{roll} / ATR_{M1} \\
DIV_{CVD,t} \\
III_t \\
III_t^{bull} \\
III_t^{bear} \\
OBI_t \\
D_{strength,t} \\
D_{strength,t}^{norm,body} \\
D_{strength,t}^{norm,ATR} \\
\mathbb{1}[FVG_t] \\
\mathbb{1}[FVG_{t-1}] \\
\sigma_{body,N} / ATR_{M1} \\
V_{total,t} / \bar{V}_{30} \\
\text{spread}_{t} / ATR_{M1} \\
\mathbb{1}[SPIKE\_REGIME_t]
\end{array}\right]$$

---

## PHẦN II — THIẾT KẾ CHI TIẾT THUẬT TOÁN XGBOOST
### (XGBoost Model A & Model B: Architecture, Features, Objectives)

> **Lý do chọn XGBoost thay thế (bổ sung) cho Neural Network:**
> Trong khi LSTM Autoencoder (đặc tả trong V1.0) cung cấp vector tiềm ẩn $z \in \mathbb{R}^{512}$ từ chuỗi thời gian dài, XGBoost vượt trội ở việc xử lý **đặc trưng bảng (tabular features)** có ngữ nghĩa cao — tức là các vector $F_{liq}$, $F_{flow}$, $F_{zone}$ đã được toán học hóa ở Phần I. Kiến trúc cuối cùng sử dụng cả hai: $z$ từ LSTM làm một nhóm trong feature vector đầu vào của XGBoost.

---

### II.1 MODEL A — DỰ ĐOÁN VÙNG HÚT THANH KHOẢN ($P_{BSL}$ vs $P_{SSL}$)

#### II.1.1 Định Nghĩa Bài Toán Chính Thức

**Bài toán:** Phân loại đa lớp tại mỗi thời điểm nến M1 đóng cửa $t$:

$$\hat{y}_t = \arg\max_{c \in C} P(y_t = c \mid X_A^{(t)})$$

$$C = \{BSL\_HIT,\; SSL\_HIT,\; LATERAL\}$$

Trong đó $y_t$ là nhãn thực tế được xác nhận trong cửa sổ tối đa 4 giờ sau $t$ (xem Module 4 V1.0 về Outcome Determination).

#### II.1.2 Cấu Trúc Vector Đầu Vào $X_A$

Tổng số chiều: $|X_A| = 512 + 24 + 20 + 64 + 16 + 12 = 648$

```
THIẾT KẾ FEATURE VECTOR X_A — CẤU TRÚC ĐẦY ĐỦ:
──────────────────────────────────────────────────────────────────
NHÓM 1: Latent Vector từ LSTM Autoencoder        [512 chiều, index 0:511]
    z ∈ ℝ^512: Mã hóa trạng thái thị trường MTF
    (được tính trước và cache trong Redis, key: "latent:{symbol}:{ts}")

NHÓM 2: Đặc trưng Thanh khoản F_liq             [24 chiều, index 512:535]
    Bao gồm: khoảng cách đến BSL/SSL tại ST/IT/LT, khối lượng tích lũy,
    số lượng mức chưa bị quét, tỷ lệ tiêu thụ thanh khoản
    (xem định nghĩa F_liq tại Phần I.1.4)

NHÓM 3: Đặc trưng Dòng lệnh F_flow              [20 chiều, index 536:555]
    Bao gồm: CVD đa cửa sổ, III, OBI, Displacement
    (xem định nghĩa F_flow tại Phần I.3.4)

NHÓM 4: Đặc trưng Macro-Context                 [12 chiều, index 556:567]
    [0]: I_news — tác động biến động tin tức chuẩn hóa [0,1]
    [1]: seconds_to_next_news / 3600 — chuẩn hóa về đơn vị giờ
    [2]: regime_phase — {0=NORMAL, 1=PRE_NEWS, 2=NEWS_WINDOW, 3=POST_NEWS}
    [3]: surprise_factor S nếu POST_NEWS, else 0
    [4]: surprise_direction — {-1, 0, +1}
    [5]: post_regime_code — {0=N/A, 1=IMPULSIVE, 2=REVERSAL, 3=CHOPPY}
    [6]: session_code — {0,...,5} (6 phiên giao dịch theo Killzone)
    [7]: session_weight_ltf — trọng số tín hiệu LTF theo phiên
    [8]: session_weight_htf — trọng số tín hiệu HTF theo phiên
    [9]: is_london_open_kz — boolean
    [10]: is_ny_open_kz — boolean
    [11]: day_of_week — {1,...,5} chuẩn hóa về [0,1]

NHÓM 5: Đặc trưng Cấu trúc Tổng hợp            [64 chiều, index 568:631]
    [0:5]: MSS_bullish_flag cho 6 khung TF {M1,...,D1}
    [6:11]: MSS_bearish_flag cho 6 khung TF
    [12:17]: BOS_bullish_flag cho 6 khung TF
    [18:23]: BOS_bearish_flag cho 6 khung TF
    [24:29]: HH_flag cho 6 khung TF (đang trong chuỗi Higher High)
    [30:35]: LL_flag cho 6 khung TF (đang trong chuỗi Lower Low)
    [36:41]: premium_zone_flag cho 6 khung TF (giá trong Premium)
    [42:47]: discount_zone_flag cho 6 khung TF (giá trong Discount)
    [48:53]: EMA_50_bias cho 6 khung TF (close vs EMA50, chuẩn hóa)
    [54:59]: fibonacci_level cho 6 khung TF (0=dưới 0.382, 1=trung, 2=trên 0.618)
    [60:63]: 4 chiều bổ sung từ phân tích cấu trúc Equal Levels (EQ)

NHÓM 6: Đặc trưng Tổng hợp FVG/OB Đang Kích Hoạt [16 chiều, index 632:647]
    [0]: Số FVG_Bull UNMITIGATED trong vùng ±5×ATR_M1 từ giá hiện tại
    [1]: Số FVG_Bear UNMITIGATED
    [2]: Số OB_Bull UNMITIGATED
    [3]: Số OB_Bear UNMITIGATED
    [4]: P_hold trung bình của FVG_Bull gần nhất (0 nếu không có)
    [5]: P_hold trung bình của FVG_Bear gần nhất
    [6]: P_hold của OB gần nhất phía dưới
    [7]: P_hold của OB gần nhất phía trên
    [8]: Tổng III_zone của tất cả FVG_Bull đang kích hoạt
    [9]: Tổng III_zone của tất cả FVG_Bear đang kích hoạt
    [10]: Khoảng cách đến FVG_Bull gần nhất / ATR_H1
    [11]: Khoảng cách đến FVG_Bear gần nhất / ATR_H1
    [12]: w_zone tổng hợp cao nhất của vùng Bull đang hoạt động
    [13]: w_zone tổng hợp cao nhất của vùng Bear đang hoạt động
    [14]: iFVG_count gần đây (số iFVG hình thành trong 20 nến M1 gần nhất)
    [15]: EQ_active_count (số Equal Levels chưa bị quét, tổng cả Bull/Bear)
──────────────────────────────────────────────────────────────────
```

#### II.1.3 Kiến Trúc XGBoost và Siêu Tham Số

**Cấu hình XGBoost Multi-Class:**

```
CẤU HÌNH XGBOOST MODEL A — SIÊU THAM SỐ:
──────────────────────────────────────────────────────────────────
objective           : "multi:softprob"     ← Đầu ra xác suất mềm
num_class           : 3                    ← BSL_HIT, SSL_HIT, LATERAL
eval_metric         : ["mlogloss", "merror"]
n_estimators        : 500                  ← Số cây
max_depth           : 6                    ← Độ sâu tối đa mỗi cây
learning_rate       : 0.05                 ← Tốc độ học (eta)
subsample           : 0.8                  ← Tỷ lệ mẫu mỗi vòng lặp
colsample_bytree    : 0.7                  ← Tỷ lệ đặc trưng mỗi cây
colsample_bylevel   : 0.8
min_child_weight    : 5                    ← Ngăn overfit trên mẫu ít
reg_alpha           : 0.1                  ← L1 regularization
reg_lambda          : 1.5                  ← L2 regularization
scale_pos_weight    : [w_BSL, w_SSL, w_LAT] ← Điều chỉnh mất cân bằng lớp

Trọng số lớp (tính từ dữ liệu huấn luyện):
    w_c = N_total / (3 × N_c),   c ∈ {BSL_HIT, SSL_HIT, LATERAL}

early_stopping_rounds : 50
tree_method           : "hist"             ← Hiệu năng cao cho dataset lớn
device                : "cuda"             ← GPU acceleration nếu có
──────────────────────────────────────────────────────────────────
```

#### II.1.4 Hàm Mục Tiêu Tùy Chỉnh (Custom Objective)

XGBoost cho phép định nghĩa Gradient và Hessian tùy chỉnh. Hàm Loss của Model A kết hợp Cross-Entropy tiêu chuẩn với hai hình phạt bổ sung:

**Hàm Loss tổng hợp:**

$$\mathcal{L}_A = \underbrace{-\sum_{i=1}^{N} \sum_{c=1}^{3} y_{i,c} \log(\hat{p}_{i,c})}_{\text{Cross-Entropy}} + \lambda_{OC} \underbrace{\sum_{i=1}^{N} \max\left(0, \max_{c}(\hat{p}_{i,c}) - \theta_{OC}\right)}_{\text{Overconfidence Penalty}} + \lambda_{NR} \underbrace{\sum_{i \in \mathcal{I}_{news}} \left|\hat{p}_{i,BSL} - \hat{p}_{i,SSL}\right|}_{\text{News Regime Penalty}}$$

Trong đó:
- $\theta_{OC} = 0.85$: Ngưỡng quá tự tin (mô hình không được dự đoán xác suất > 85% cho bất kỳ lớp nào)
- $\lambda_{OC} = 0.5$: Hệ số hình phạt quá tự tin
- $\mathcal{I}_{news}$: Tập hợp các mẫu huấn luyện có `macro_regime ∈ {PRE_NEWS, NEWS_WINDOW}`
- $\lambda_{NR} = 0.3$: Trong cửa sổ tin tức, mô hình bị phạt khi dự đoán BSL và SSL quá chênh lệch (vì bất định cao)

**Gradient và Hessian của phần Cross-Entropy (standard softmax):**

$$g_{i,c} = \frac{\partial \mathcal{L}}{\partial \hat{f}_{i,c}} = \hat{p}_{i,c} - y_{i,c}$$

$$h_{i,c} = \frac{\partial^2 \mathcal{L}}{\partial \hat{f}_{i,c}^2} = \hat{p}_{i,c}(1 - \hat{p}_{i,c})$$

Gradient và Hessian của phần Overconfidence Penalty được tính bổ sung vào $g_{i,c^*}$ và $h_{i,c^*}$ (với $c^* = \arg\max_c \hat{p}_{i,c}$) khi $\hat{p}_{i,c^*} > \theta_{OC}$:

$$g_{i,c^*}^{OC} = \lambda_{OC} \cdot \hat{p}_{i,c^*}(1 - \hat{p}_{i,c^*}), \quad h_{i,c^*}^{OC} = \lambda_{OC} \cdot \hat{p}_{i,c^*}(1 - \hat{p}_{i,c^*})(1 - 2\hat{p}_{i,c^*})$$

#### II.1.5 Cơ Chế Calibration Xác Suất

Đầu ra thô của XGBoost thường bị lệch (miscalibrated). Áp dụng Platt Scaling:

$$\hat{p}_{calibrated,c} = \text{Softmax}\left(W_{cal} \cdot \hat{f}_{c} + b_{cal}\right)$$

$W_{cal} \in \mathbb{R}^{3\times3}$, $b_{cal} \in \mathbb{R}^3$ được học trên tập validation riêng biệt bằng maximum likelihood.

**Kiểm tra Calibration bằng Expected Calibration Error (ECE):**

$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

Trong đó $M=10$ bins, $B_m$ là tập dự đoán có xác suất rơi vào $[(m-1)/10, m/10]$. Mục tiêu: $ECE < 0.05$.

#### II.1.6 Điều Chỉnh Sau Suy Luận (Post-Inference Adjustment)

Sau khi có xác suất từ XGBoost, áp dụng hai lớp điều chỉnh theo ngữ cảnh vận hành:

**Lớp 1 — Macro Guardrail (như V1.0):**

$$\hat{p}_{BSL}^{adj} = \hat{p}_{BSL} \times (1 - \gamma \cdot I_{news}), \quad \hat{p}_{SSL}^{adj} = \hat{p}_{SSL} \times (1 - \gamma \cdot I_{news})$$

$$\hat{p}_{LAT}^{adj} = 1 - \hat{p}_{BSL}^{adj} - \hat{p}_{SSL}^{adj}$$

Với $\gamma = \gamma_0 \times \left(1 - \frac{\max(0, t_{to\_news})}{900}\right)$, $\gamma_0 = 0.3$ (tăng tuyến tính khi gần tin).

**Lớp 2 — Session Weight Adjustment:**

$$\hat{p}_{BSL}^{final} = \text{Normalize}\left(\hat{p}_{BSL}^{adj} \times w_{session,htf}\right)$$

$$\hat{p}_{SSL}^{final} = \text{Normalize}\left(\hat{p}_{SSL}^{adj} \times w_{session,htf}\right)$$

---

### II.2 MODEL B — DỰ ĐOÁN HIỆU LỰC GIỮ GIÁ CỦA VÙNG CẤU TRÚC ($P_{hold}$)

#### II.2.1 Định Nghĩa Bài Toán Chính Thức

**Bài toán:** Phân loại nhị phân trên mỗi vùng FVG/OB khi giá lần đầu chạm vào (Mitigation Event):

$$\hat{y}_{zone} = P\left(hold=1 \mid X_B^{(zone)}\right)$$

Nhãn thực tế: $y_{zone} = 1$ nếu vùng giữ được giá (giá phản ứng và đảo chiều rời xa vùng ít nhất $R_{min}$ pip trong vòng $H_{max}$ nến M1 sau khi chạm). Ngược lại $y_{zone} = 0$.

**Chuẩn hóa nhãn:**

$$R_{min} = 0.5 \times ATR_{H1,\;tại\;thời\;điểm\;chạm}, \quad H_{max} = 60 \text{ nến M1}$$

#### II.2.2 Cấu Trúc Vector Đầu Vào $X_B$

Tổng số chiều: $|X_B| = 512 + 16 + 20 + 12 = 560$ (giống V1.0 nhưng chi tiết hóa)

```
THIẾT KẾ FEATURE VECTOR X_B — CẤU TRÚC ĐẦY ĐỦ:
──────────────────────────────────────────────────────────────────
NHÓM 1: Latent Vector z                          [512 chiều, index 0:511]
    (Giống Model A — ngữ cảnh thị trường tổng thể)

NHÓM 2: Đặc trưng Vùng cụ thể F_zone^(k)        [16 chiều, index 512:527]
    (Xem định nghĩa F_zone tại Phần I.2.5)
    Đây là đặc trưng đặc trưng cho từng vùng FVG/OB cụ thể đang được đánh giá

NHÓM 3: Đặc trưng Nến Chạm Vùng (Contact Candle) [20 chiều, index 528:547]
    [0]: Displacement_strength của nến đang chạm vào vùng
    [1]: CVD_current / ATR_M1
    [2]: OBI_current
    [3]: close_vs_zone_open — (close_touch - zone.open) / zone_size
         (0 = chạm chính xác mở, 1 = chạm chính xác đóng, >1 = đã vượt qua)
    [4]: close_vs_zone_CE — (close_touch - zone.CE) / zone_size
    [5]: wick_ratio — (high - max(o,c)) / body — tỷ lệ bóng nến
    [6]: body_direction — sign(close - open) — {-1, +1}
    [7]: close_vs_open_of_prevbar — (close - open[-1]) / ATR_M1
    [8]: III_contact — III_t tại nến chạm vùng
    [9]: volume_vs_avg30 — V_current / V_bar_30 — volume bất thường
    [10]: consecutive_bearish_count — số nến đỏ liên tiếp trước khi chạm (max=10)
    [11]: consecutive_bullish_count — số nến xanh liên tiếp trước khi chạm
    [12]: RSI_14_M1 / 100 — momentum chuẩn hóa
    [13]: is_inside_bar — boolean (thân nến nằm bên trong thân nến trước)
    [14]: gap_to_zone — khoảng cách từ close nến trước đến cạnh vùng / ATR
    [15]: speed_of_approach — tốc độ tiến đến vùng (pip per bar, trung bình 5 nến)
    [16]: approach_vol_trend — hồi quy tuyến tính khối lượng 10 nến gần nhất
    [17]: htf_trend_alignment — sign(trend_H4) × sign(zone_direction) {-1,+1}
    [18]: nearest_bos_age — số nến M1 từ BOS/MSS gần nhất / 100 (chuẩn hóa)
    [19]: touch_count — số lần vùng này đã bị chạm trước đó (0-5, saturate)

NHÓM 4: Ngữ cảnh Macro tại thời điểm chạm       [12 chiều, index 548:559]
    (Giống Nhóm 4 của Model A — ngữ cảnh tin tức và phiên giao dịch)
──────────────────────────────────────────────────────────────────
```

#### II.2.3 Kiến Trúc XGBoost và Siêu Tham Số

```
CẤU HÌNH XGBOOST MODEL B — SIÊU THAM SỐ:
──────────────────────────────────────────────────────────────────
objective           : "binary:logistic"    ← Phân loại nhị phân
eval_metric         : ["logloss", "auc", "aucpr"]
n_estimators        : 600
max_depth           : 5                    ← Thấp hơn Model A: tránh overfit
learning_rate       : 0.03                 ← Học chậm hơn (dataset nhỏ hơn)
subsample           : 0.75
colsample_bytree    : 0.65
colsample_bylevel   : 0.75
min_child_weight    : 8                    ← Cao hơn: đặc trưng tổng hợp ổn hơn
reg_alpha           : 0.2
reg_lambda          : 2.0
scale_pos_weight    : N_neg / N_pos        ← Cân bằng lớp (vùng bị phá >> giữ)

early_stopping_rounds : 60
tree_method         : "hist"
──────────────────────────────────────────────────────────────────
```

#### II.2.4 Ma Trận Chi Phí và Xử Lý Mất Cân Bằng Dữ Liệu

Trong thực tế, số lượng FVG/OB bị phá vỡ (hold=0) thường gấp 2-3 lần số bị giữ (hold=1). Chiến lược xử lý đa tầng:

**Tầng 1 — Class Weight trong XGBoost:**
$$\text{scale\_pos\_weight} = \frac{N_{hold=0}}{N_{hold=1}}$$

**Tầng 2 — Ma Trận Chi Phí Bất Đối Xứng:**

Trong bối cảnh giao dịch, False Positive (dự đoán giữ nhưng thực ra bị phá) nguy hiểm hơn False Negative (bỏ lỡ vùng giữ tốt):

$$\text{Cost Matrix} = \begin{pmatrix} 0 & C_{FP} \\ C_{FN} & 0 \end{pmatrix} = \begin{pmatrix} 0 & 2.5 \\ 1.0 & 0 \end{pmatrix}$$

Ngưỡng quyết định tối ưu:

$$\theta^* = \arg\min_\theta \mathbb{E}\left[C_{FP} \cdot \mathbb{1}[\hat{y}=1, y=0] + C_{FN} \cdot \mathbb{1}[\hat{y}=0, y=1]\right]$$

Giải tích: $\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}} \cdot \frac{p(y=0)}{p(y=0) + p(y=1)} \approx 0.71$ (với tỷ lệ lớp 2:1 và Cost Matrix trên).

**Tầng 3 — Stratified Sampling theo Regime:**

Đảm bảo mỗi chế độ thị trường đóng góp đều nhau vào tập huấn luyện:

$$N_{train,regime\_r} = \min\left(N_{available,r}, \frac{N_{total\_target}}{|R|}\right)$$

Trong đó $R = \{TRENDING\_LV, TRENDING\_HV, CHOPPY\_HV, NORMAL\}$.

#### II.2.5 Hiệu Chuẩn Xác Suất Nâng Cao (Isotonic Regression Calibration)

Đối với Model B, sử dụng Isotonic Regression (thay vì Platt Scaling đơn giản) vì phân phối xác suất thô thường không đơn điệu:

**Mục tiêu:** Tìm hàm $f^*$ đơn điệu không giảm sao cho:

$$f^* = \arg\min_{f: f \text{ isotonic}} \sum_{i=1}^{N_{val}} \left(y_i - f(\hat{p}_i^{raw})\right)^2$$

Thuật toán Pool Adjacent Violators (PAV) giải bài toán này trong $O(N)$.

**Kiểm tra calibration theo chế độ thị trường:**

Tính ECE riêng biệt cho từng regime $r$ để phát hiện calibration bias phụ thuộc vào trạng thái thị trường:

$$ECE_r = \sum_{m=1}^{M} \frac{|B_m \cap D_r|}{|D_r|} \left|\text{acc}(B_m \cap D_r) - \text{conf}(B_m \cap D_r)\right|$$

NẾU $\max_r ECE_r > 0.08$: cần calibration riêng theo regime.

#### II.2.6 Importance Analysis — Feature Importance và SHAP Values

Sau khi huấn luyện, tính SHAP values để hiểu đóng góp của từng đặc trưng:

$$\hat{y}_i = \phi_0 + \sum_{j=1}^{p} \phi_{ij}$$

Trong đó $\phi_0$ là giá trị cơ sở (base rate), $\phi_{ij}$ là SHAP value của đặc trưng $j$ cho mẫu $i$.

**Nhóm đặc trưng theo SHAP importance (thứ tự kỳ vọng từ cao đến thấp):**

```
BẢNG PHÂN TÍCH ĐẶC TRƯNG QUAN TRỌNG DỰ KIẾN — MODEL B:
──────────────────────────────────────────────────────────────────
Nhóm 1 (Critical): close_vs_zone_CE, III_contact, htf_trend_alignment
                   → Nến chạm vùng càng không vào sâu và thuận chiều HTF → hold cao

Nhóm 2 (High): volume_vs_avg30, touch_count, w_zone (htf_alignment)
               → Volume lớn tại điểm chạm và vùng được HTF xác nhận → hold cao

Nhóm 3 (Medium): approach_speed, Displacement_strength, nearest_bos_age
                 → Tiếp cận chậm hơn thường có hold cao hơn

Nhóm 4 (Context): session_code, I_news, macro_regime
                  → Trong Kill Zone hoặc Pre-News: cần thận trọng hơn
──────────────────────────────────────────────────────────────────
```

---

### II.3 QUYẾT ĐỊNH KIẾN TRÚC CUỐI CÙNG: XGBOOST vs LSTM

Trong phiên bản V4.5 này, hai kiến trúc được sử dụng **song song và bổ sung** theo mô hình Stacked Ensemble:

```
STACKED ENSEMBLE ARCHITECTURE:
──────────────────────────────────────────────────────────────────
Tầng 1 — Base Models:
    [A] LSTM Autoencoder → z ∈ ℝ^512 (ngữ cảnh chuỗi thời gian dài)
    [B] XGBoost Model A trên X_A^{full} = [z, F_liq, F_flow, F_macro, F_struct]
    [C] XGBoost Model B trên X_B^{full} = [z, F_zone, F_contact, F_macro]

Tầng 2 — Meta-Learner (Logistic Regression đơn giản):
    Đầu vào: Xác suất từ mô hình XGBoost + confidence_qualifier từ LSTM
    Đầu ra: Xác suất tổng hợp cuối cùng
    
    Ưu điểm: 
    - LSTM giỏi nhận diện pattern thời gian phức tạp (Wave, Fractal)
    - XGBoost giỏi tận dụng đặc trưng tabular có ngữ nghĩa (FVG, CVD, Macro)
    - Meta-Learner học trọng số phù hợp theo từng chế độ thị trường
──────────────────────────────────────────────────────────────────
```

---

## PHẦN III — KIẾN TRÚC HỆ THỐNG CƠ SỞ DỮ LIỆU HỖN HỢP
### (Hybrid Database Architecture: In-Memory · Relational · Semantic)

---

### III.1 TỔNG QUAN KIẾN TRÚC BA TẦNG

```
SƠ ĐỒ KIẾN TRÚC HYBRID DATABASE:
══════════════════════════════════════════════════════════════════
                    ┌──────────────────────────────────┐
                    │        APPLICATION LAYER          │
                    │  Module 1,2,3,4,5,6,7            │
                    └──────────┬───────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼──────┐         ┌────▼──────┐        ┌────▼──────┐
    │  TẦNG 1   │         │  TẦNG 2   │        │  TẦNG 3   │
    │  REDIS    │         │  SQLITE   │        │  QDRANT/  │
    │ In-Memory │         │Relational │        │ CHROMADB  │
    │  Cache    │         │ Persistent│        │ Semantic  │
    │           │         │ Storage   │        │ Vector DB │
    │ ≤ 1ms     │         │ ≤ 10ms    │        │ ≤ 50ms    │
    └───────────┘         └───────────┘        └───────────┘
    RAM/SSD cache         Full ACID             Approximate
    Volatile state        Historical log        Nearest Neighbor
══════════════════════════════════════════════════════════════════
```

**Nguyên tắc phân loại dữ liệu:**

| Loại dữ liệu | Tầng | Lý do |
|---|---|---|
| Trạng thái vận hành thời gian thực | Redis | Độ trễ < 1ms là bắt buộc |
| Zone Registry đang kích hoạt | Redis | Cập nhật mỗi tick |
| Lịch sử dự đoán | SQLite | ACID, cần truy vấn phức tạp |
| Kết quả tin tức | SQLite | Quan hệ với dự đoán |
| Hiệu suất mô hình | SQLite | Báo cáo định kỳ |
| Precedents tranh biện | Vector DB | Tìm kiếm semantic similarity |
| Embedding vùng FVG/OB | Vector DB | k-NN query theo pattern |

---

### III.2 TẦNG NHỚ ĐỆM THỜI GIAN THỰC (IN-MEMORY LAYER — REDIS)

#### III.2.1 Thiết Kế Schema Chi Tiết

**Namespace tổng quan:**

```
REDIS KEY NAMESPACE DESIGN:
──────────────────────────────────────────────────────────────────
Format chuẩn: "{domain}:{entity}:{identifier}:{sub_key}"
Ví dụ: "zone:XAUUSD:H1:FVG_BULL:1703123456"
──────────────────────────────────────────────────────────────────
```

**Bảng 1 — Zone Registry (Sổ Vùng Cấu Trúc Đang Kích Hoạt):**

```
KEY SCHEMA: zone:{symbol}:{timeframe}:{zone_type}:{formed_unix_ms}
DATA TYPE:  Redis Hash (HSET / HGETALL)
TTL:        604800 giây (7 ngày)

Fields:
    top             FLOAT    ← Cạnh trên của vùng
    bottom          FLOAT    ← Cạnh dưới của vùng
    ce              FLOAT    ← Consequent Encroachment (điểm giữa)
    formed_time     INT64    ← Unix timestamp khi vùng hình thành
    status          STRING   ← {UNMITIGATED, WICK_TOUCHED, WICK_FILLED_HALF, BODY_FILLED, MITIGATED}
    p_hold          FLOAT    ← Xác suất giữ giá từ Model B [0,1]
    p_hold_updated  INT64    ← Timestamp lần cuối cập nhật p_hold
    w_zone          FLOAT    ← Trọng số alignment HTF {0.5, 1.0, 2.0}
    iii_formation   FLOAT    ← III_t khi vùng hình thành
    touch_count     INT      ← Số lần giá chạm vùng
    last_touch_time INT64    ← Timestamp lần chạm gần nhất
    htf_tf          STRING   ← Khung HTF xác nhận vùng này

Secondary Index: Sorted Set theo P_hold để lấy top-K nhanh
    Key: zone_rank:{symbol}:{zone_type}
    Value: zone_key, Score: p_hold × w_zone
```

**Bảng 2 — AI Output Cache (Kết Quả Suy Luận Mới Nhất):**

```
KEY SCHEMA: ai:output:{symbol}:latest
DATA TYPE:  Redis Hash
TTL:        120 giây

Fields:
    timestamp           INT64
    p_bsl               FLOAT
    p_ssl               FLOAT
    p_lateral           FLOAT
    bsl_target_price    FLOAT
    ssl_target_price    FLOAT
    bsl_target_tf       STRING
    ssl_target_tf       STRING
    consensus_rating    INT     ← [-4, +4]
    confidence_qual     STRING  ← {HIGH, MEDIUM, LOW}
    model_version       STRING  ← "xgb_v4.5" | "lstm_v1.0" | "ensemble_v4.5"
    inference_latency_ms FLOAT  ← Thời gian suy luận thực tế
```

**Bảng 3 — Countdown và Macro State:**

```
KEY SCHEMA: macro:state:{currency}
DATA TYPE:  Redis Hash
TTL:        60 giây

Fields:
    next_event_name     STRING
    next_event_time     INT64
    seconds_to_next     INT
    impact              STRING   ← {Low, Medium, High}
    i_news              FLOAT
    regime_phase        STRING
    active_guardrail    BOOL
    forecast            FLOAT    ← Có thể NULL
    actual              FLOAT    ← Có thể NULL
    surprise_factor     FLOAT    ← 0 nếu chưa có actual

KEY SCHEMA: macro:events:{currency}:upcoming
DATA TYPE:  Redis Sorted Set (score = scheduled_unix_time)
TTL:        86400 giây (1 ngày)
Members:    event_id strings
```

**Bảng 4 — Debate Log (Nhật Ký Tranh Biện):**

```
KEY SCHEMA: debate:{symbol}:{bar_close_timestamp}
DATA TYPE:  Redis Hash
TTL:        3600 giây (1 giờ → sau đó archive sang Vector DB)

Fields:
    bull_direction      STRING
    bull_confidence     FLOAT
    bull_evidence       STRING  ← JSON-serialized list[string]
    bull_target         FLOAT
    bull_invalidation   FLOAT
    bear_direction      STRING
    bear_confidence     FLOAT
    bear_evidence       STRING
    bear_target         FLOAT
    bear_invalidation   FLOAT
    consensus_rating    INT
    preferred_dir       STRING
    conviction_zone     FLOAT
    confidence_qual     STRING
    reasoning           STRING  ← Văn bản tóm tắt của Critic Agent
    precedents_count    INT
    debate_latency_ms   FLOAT
    archived            BOOL    ← False ban đầu, True sau khi chuyển sang VectorDB
```

**Bảng 5 — Feature Cache cho Model Inference:**

```
KEY SCHEMA: features:{symbol}:{bar_close_timestamp}:{model_name}
DATA TYPE:  Redis String (Binary-packed Float32 array)
TTL:        300 giây (5 phút)

Kích thước:
    Model A: 648 × 4 bytes = 2592 bytes
    Model B: 560 × 4 bytes = 2240 bytes (per zone; không cache vì zone-specific)

Format lưu trữ: MessagePack binary với numpy float32 array
```

**Bảng 6 — Latent Vector Cache:**

```
KEY SCHEMA: latent:{symbol}:{bar_close_timestamp}
DATA TYPE:  Redis String (Binary blob)
TTL:        300 giây

Kích thước: 512 × 4 bytes = 2048 bytes (float32)
Encoding:   Raw binary (không JSON vì quá lớn)
```

**Bảng 7 — System Metrics (Real-time Telemetry):**

```
KEY SCHEMA: metrics:{component}:latest
DATA TYPE:  Redis Hash
TTL:        60 giây

Ví dụ: metrics:ipc:latest
Fields:
    ws_latency_avg_ms   FLOAT   ← Moving average 100 samples
    ws_latency_p95_ms   FLOAT
    ws_latency_p99_ms   FLOAT
    messages_per_sec    FLOAT
    client_count        INT

Ví dụ: metrics:model:latest
Fields:
    ic_rolling_20       FLOAT
    brier_score_50      FLOAT
    model_degraded      BOOL
    last_eval_time      INT64
    feature_drift_score FLOAT
```

#### III.2.2 Cấu Hình Redis Nâng Cao

```
REDIS CONFIGURATION:
──────────────────────────────────────────────────────────────────
# Memory Management
maxmemory            512mb
maxmemory-policy     allkeys-lru
maxmemory-samples    5

# Persistence (chỉ AOF, không RDB — ưu tiên consistency)
appendonly           yes
appendfsync          everysec
no-appendfsync-on-rewrite yes
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size   64mb

# Performance
hz                   20          ← Tần suất background tasks (mặc định 10)
dynamic-hz           yes
lazyfree-lazy-eviction yes      ← Eviction bất đồng bộ
lazyfree-lazy-expire  yes

# Networking
tcp-keepalive        60
timeout              0           ← Không timeout kết nối nội bộ
bind                 127.0.0.1   ← Chỉ local connections
protected-mode       yes

# Keyspace Notifications (cho Module 4 monitoring zone status changes)
notify-keyspace-events "Ex"     ← Notify khi key expire
──────────────────────────────────────────────────────────────────
```

---

### III.3 TẦNG LƯU TRỮ QUAN HỆ DÀI HẠN (RELATIONAL LAYER — SQLITE)

#### III.3.1 Cấu Hình Khởi Tạo SQLite

```
PRAGMA STATEMENTS (chạy ngay khi mở connection):
──────────────────────────────────────────────────────────────────
PRAGMA journal_mode = WAL;          ← Write-Ahead Logging: đọc/ghi đồng thời
PRAGMA synchronous = NORMAL;        ← Cân bằng tốc độ/độ bền
PRAGMA cache_size = -65536;         ← 64MB page cache (âm = KB)
PRAGMA temp_store = MEMORY;         ← Bảng tạm trong RAM
PRAGMA mmap_size = 268435456;       ← 256MB memory-mapped I/O
PRAGMA foreign_keys = ON;           ← Ràng buộc khóa ngoại
PRAGMA auto_vacuum = INCREMENTAL;   ← Tránh database bloat
PRAGMA page_size = 4096;            ← Tối ưu cho SSD NVMe
PRAGMA busy_timeout = 5000;         ← 5 giây timeout cho lock contention
──────────────────────────────────────────────────────────────────
```

#### III.3.2 Schema Chi Tiết Bảng `predictions`

```
TABLE predictions:
──────────────────────────────────────────────────────────────────
Trường                Kiểu            Ràng buộc           Mô tả
──────────────────────────────────────────────────────────────────
id                    INTEGER         PK AUTOINCREMENT
symbol                TEXT            NOT NULL            "XAUUSD"
prediction_time       INTEGER         NOT NULL            Unix ms
bar_close_time        INTEGER         NOT NULL            Timestamp nến M1 kích hoạt
timeframe             TEXT            NOT NULL            "M1"
model_name            TEXT            NOT NULL            "xgb_a_v4.5" | "ensemble_v4.5"
model_version         TEXT            NOT NULL            Phiên bản cụ thể

-- Model A Outputs
p_bsl                 REAL            CHECK(0<=p_bsl<=1)
p_ssl                 REAL            CHECK(0<=p_ssl<=1)
p_lateral             REAL            CHECK(0<=p_lateral<=1)
p_sum_check           REAL AS (p_bsl + p_ssl + p_lateral) VIRTUAL
predicted_bsl_level   REAL                                Mức giá BSL mục tiêu
predicted_ssl_level   REAL                                Mức giá SSL mục tiêu
bsl_tf                TEXT                                Khung TF của BSL target
ssl_tf                TEXT                                Khung TF của SSL target

-- Model B Outputs (per zone — denormalized cho hiệu suất query)
zone_id               TEXT                                NULL nếu đây là Model A record
zone_type             TEXT                                {FVG_BULL, FVG_BEAR, OB_BULL, OB_BEAR}
zone_timeframe        TEXT
zone_top              REAL
zone_bottom           REAL
zone_ce               REAL
p_hold                REAL            CHECK(0<=p_hold<=1)
p_hold_pre_adj        REAL                                Trước khi áp dụng session weight

-- Consensus (Multi-Agent Debate)
consensus_rating      INTEGER         CHECK(-4<=r<=4)
preferred_direction   TEXT            CHECK(dir IN ('BULLISH','BEARISH','NEUTRAL'))
conviction_zone_price REAL
confidence_qualifier  TEXT            CHECK(q IN ('HIGH','MEDIUM','LOW'))

-- Outcome (điền sau khi xác nhận — Module 4 Outcome Determination)
outcome_determined    INTEGER         DEFAULT 0           {0=pending, 1=confirmed, 2=expired}
outcome_time          INTEGER                             Unix ms khi xác nhận
actual_direction      TEXT                                {BSL_HIT, SSL_HIT, LATERAL}
actual_hold           INTEGER                             {0, 1} — cho Model B
pips_to_bsl           REAL                                pip đến BSL khi hit
pips_to_ssl           REAL
max_adverse_excursion REAL                                Pip di chuyển ngược chiều tối đa
time_to_outcome_min   REAL                                Phút từ dự đoán đến kết quả

-- Input Context (lưu để audit và tái huấn luyện)
macro_regime          TEXT
i_news                REAL
iii_current           REAL
obi_current           REAL
session_code          INTEGER
is_london_kz          INTEGER
is_ny_kz              INTEGER
mss_bullish_h1        INTEGER         {0,1}
mss_bearish_h1        INTEGER         {0,1}
bos_recent_tf         TEXT                                Khung TF của BOS gần nhất

-- Model Diagnostics
inference_latency_ms  REAL
feature_drift_flag    INTEGER         DEFAULT 0           {0=stable, 1=minor, 2=significant}
model_degraded_at_pred INTEGER        DEFAULT 0           {0,1}

CONSTRAINTS:
    CHECK(p_bsl IS NULL OR p_ssl IS NULL OR ABS(p_bsl+p_ssl+p_lateral - 1.0) < 0.001)

INDEXES:
    CREATE INDEX idx_pred_symbol_time  ON predictions(symbol, prediction_time DESC);
    CREATE INDEX idx_pred_outcome      ON predictions(outcome_determined, symbol, prediction_time);
    CREATE INDEX idx_pred_bar_close    ON predictions(symbol, bar_close_time DESC);
    CREATE INDEX idx_pred_model        ON predictions(model_name, prediction_time DESC);
    CREATE INDEX idx_zone_id           ON predictions(zone_id) WHERE zone_id IS NOT NULL;
──────────────────────────────────────────────────────────────────
```

#### III.3.3 Schema Chi Tiết Bảng `news_outcomes`

```
TABLE news_outcomes:
──────────────────────────────────────────────────────────────────
Trường                    Kiểu        Ràng buộc       Mô tả
──────────────────────────────────────────────────────────────────
id                        INTEGER     PK AUTOINCREMENT
event_id                  TEXT        NOT NULL UNIQUE  ID từ ForexFactory/Investing
source                    TEXT        NOT NULL         {forex_factory, investing_com}
currency                  TEXT        NOT NULL
event_name                TEXT        NOT NULL
event_category            TEXT                         {CPI, FOMC, NFP, GDP, PMI, ...}
scheduled_time            INTEGER     NOT NULL         Unix ms (UTC)
actual_release_time       INTEGER                      NULL trước khi có actual
timezone_offset           INTEGER     DEFAULT 0        Offset phút từ UTC

-- Dữ liệu dự báo và thực tế
forecast                  REAL
previous                  REAL
actual                    REAL
unit                      TEXT                         "%", "K", "B", v.v.
revision_of_previous      REAL                         NULL nếu không có revision

-- Phân tích tác động
impact_rated              TEXT        NOT NULL         {Low, Medium, High}
i_news_computed           REAL                         Giá trị I_news đã tính
surprise_factor_s         REAL                         Z-score (actual - forecast) / sigma
surprise_direction        INTEGER                      {-1, 0, +1}
is_major_surprise         INTEGER     DEFAULT 0        {0,1}: |S| > 2.0

-- Chế độ hậu tin tức
post_regime               TEXT                         {IMPULSIVE, REVERSAL, CHOPPY, NULL}
post_regime_classified_at INTEGER                      Timestamp phân loại
directional_move_h1_norm  REAL                         (P5min - P0) / ATR_H1

-- Biến động thực tế
max_move_pips_5min        REAL
max_move_pips_15min       REAL
max_move_pips_30min       REAL
move_direction            INTEGER                      {-1=down, +1=up}

-- Đánh giá độ chính xác của I_news
i_news_accuracy_flag      INTEGER                      {0=no, 1=yes}: dự đoán đúng ±20%
sigma_surprise_used       REAL                         Giá trị sigma khi tính S
sigma_surprise_updated    REAL                         Sigma sau khi cập nhật online

-- Tham chiếu đến predictions bị ảnh hưởng
affected_prediction_count INTEGER     DEFAULT 0        Số dự đoán trong cửa sổ ±15 phút

INDEXES:
    CREATE INDEX idx_news_currency_time ON news_outcomes(currency, scheduled_time DESC);
    CREATE INDEX idx_news_category      ON news_outcomes(event_category, scheduled_time DESC);
    CREATE INDEX idx_news_surprise      ON news_outcomes(is_major_surprise, scheduled_time DESC);
──────────────────────────────────────────────────────────────────
```

#### III.3.4 Schema Chi Tiết Bảng `model_performance`

```
TABLE model_performance:
──────────────────────────────────────────────────────────────────
Trường                    Kiểu        Ràng buộc       Mô tả
──────────────────────────────────────────────────────────────────
id                        INTEGER     PK AUTOINCREMENT
evaluation_time           INTEGER     NOT NULL         Unix ms
model_name                TEXT        NOT NULL
model_version             TEXT        NOT NULL
symbol                    TEXT        NOT NULL
window_size               INTEGER     NOT NULL         N mẫu được đánh giá
window_start_time         INTEGER     NOT NULL         Unix ms bắt đầu cửa sổ
window_end_time           INTEGER     NOT NULL

-- Metrics Model A
ic_spearman               REAL                         Information Coefficient
ic_pearson                REAL                         Pearson IC (tham khảo)
brier_score_bsl           REAL                         Brier Score cho lớp BSL
brier_score_ssl           REAL
ece_overall               REAL                         Expected Calibration Error
accuracy_top1             REAL                         Accuracy dự đoán lớp 1
precision_bsl             REAL
recall_bsl                REAL
precision_ssl             REAL
recall_ssl                REAL

-- Metrics Model B
precision_hold_70         REAL                         Precision với ngưỡng p_hold >= 0.70
recall_hold_70            REAL
f1_hold_70                REAL
auroc                     REAL                         Area Under ROC Curve
auprc                     REAL                         Area Under PR Curve
brier_score_hold          REAL

-- Drift Analysis
feature_drift_score_fds   REAL                         FDS (tỷ lệ đặc trưng PSI >= 0.2)
top_drifted_features      TEXT                         JSON: list[(feature_name, psi)]
regime_current            TEXT                         {TRENDING_LV, TRENDING_HV, CHOPPY_HV, NORMAL}
model_degraded            INTEGER     DEFAULT 0        {0,1}

-- Breakdown by Regime
ic_trending_lv            REAL
ic_trending_hv            REAL
ic_choppy_hv              REAL
ic_normal                 REAL
f1_hold_trending_lv       REAL
f1_hold_trending_hv       REAL
f1_hold_choppy_hv         REAL

-- Breakdown by Session
ic_asian_session          REAL
ic_london_kz              REAL
ic_ny_kz                  REAL
ic_ny_pm                  REAL

-- Fine-tuning History
fine_tuned                INTEGER     DEFAULT 0        {0,1}: đã được fine-tune
fine_tune_samples         INTEGER                      Số mẫu fine-tuning
fine_tune_time            INTEGER                      Unix ms khi fine-tune
ic_before_finetune        REAL
ic_after_finetune         REAL

INDEXES:
    CREATE INDEX idx_perf_model_time ON model_performance(model_name, evaluation_time DESC);
    CREATE INDEX idx_perf_degraded   ON model_performance(model_degraded, evaluation_time DESC);
──────────────────────────────────────────────────────────────────
```

#### III.3.5 Schema Chi Tiết Bảng `system_metrics`

```
TABLE system_metrics:
──────────────────────────────────────────────────────────────────
Trường                    Kiểu        Mô tả
──────────────────────────────────────────────────────────────────
id                        INTEGER     PK AUTOINCREMENT
recorded_time             INTEGER     Unix ms
component                 TEXT        {ipc, redis, sqlite, xgboost_a, xgboost_b, lstm, debate}

-- Độ trễ
latency_p50_ms            REAL
latency_p95_ms            REAL
latency_p99_ms            REAL
latency_max_ms            REAL
sample_count              INTEGER

-- Tài nguyên
memory_used_mb            REAL
cpu_percent               REAL

-- Component-specific
redis_memory_mb           REAL        ← NULL nếu không phải Redis
redis_hit_rate            REAL
sqlite_wal_size_mb        REAL        ← NULL nếu không phải SQLite
ws_client_count           INTEGER     ← NULL nếu không phải IPC
inference_queue_depth     INTEGER     ← NULL nếu không phải model

INDEXES:
    CREATE INDEX idx_metrics_component_time ON system_metrics(component, recorded_time DESC);
──────────────────────────────────────────────────────────────────
```

#### III.3.6 Schema Bảng `zone_history` (Lịch Sử Vùng Cấu Trúc)

```
TABLE zone_history:
──────────────────────────────────────────────────────────────────
Trường                    Kiểu        Mô tả
──────────────────────────────────────────────────────────────────
id                        INTEGER     PK AUTOINCREMENT
zone_id                   TEXT        NOT NULL UNIQUE  Format: {symbol}:{tf}:{type}:{formed_ts}
symbol                    TEXT        NOT NULL
timeframe                 TEXT        NOT NULL
zone_type                 TEXT        NOT NULL         {FVG_BULL, FVG_BEAR, OB_BULL, OB_BEAR, VI_BULL,...}
formed_time               INTEGER     NOT NULL
top                       REAL        NOT NULL
bottom                    REAL        NOT NULL
ce                        REAL        NOT NULL
size_pips                 REAL
iii_at_formation          REAL
session_at_formation      TEXT
htf_alignment             REAL                         w_zone: 0.5, 1.0, 2.0

-- Lịch sử tương tác giá
first_touch_time          INTEGER
first_touch_candle_type   TEXT
touch_count_total         INTEGER     DEFAULT 0
final_status              TEXT                         {MITIGATED, ACTIVE, EXPIRED}
mitigated_time            INTEGER
mitigation_type           TEXT                         {WICK_TOUCHED,...,BODY_FILLED}

-- Kết quả Model B tại mỗi lần chạm (denormalized)
p_hold_at_touch_1         REAL
p_hold_at_touch_2         REAL
p_hold_at_touch_3         REAL
actual_held_touch_1       INTEGER                      {0,1}
actual_held_touch_2       INTEGER
actual_held_touch_3       INTEGER

-- Embedding tham chiếu
vector_db_doc_id          TEXT                         ID trong Collection zone_embeddings

INDEXES:
    CREATE INDEX idx_zone_symbol_tf   ON zone_history(symbol, timeframe, formed_time DESC);
    CREATE INDEX idx_zone_status      ON zone_history(final_status, symbol);
    CREATE INDEX idx_zone_type_result ON zone_history(zone_type, actual_held_touch_1);
──────────────────────────────────────────────────────────────────
```

---

### III.4 TẦNG LƯU TRỮ NGỮ NGHĨA (SEMANTIC LAYER — QDRANT / CHROMADB)

#### III.4.1 Thiết Kế Phương Pháp Chiếu Ma Trận (Matrix Projection)

**Bài toán:** Chuyển đổi trạng thái đa chiều phức tạp (USV + MacroContext + SymbolicFeatureMap) thành một Embedding Vector duy nhất có số chiều cố định $d = 256$ phục vụ tìm kiếm ANN (Approximate Nearest Neighbor).

**Input cho Projection:**

$$\mathbf{v}_{raw} = \left[z \in \mathbb{R}^{512} \;;\; f_{SMC} \in \mathbb{R}^{64} \;;\; f_{macro} \in \mathbb{R}^{12} \;;\; f_{flow} \in \mathbb{R}^{20}\right] \in \mathbb{R}^{608}$$

**Ma trận chiếu:**

$$e_{USV} = \text{LayerNorm}\left(\tanh\left(W_{proj} \cdot \mathbf{v}_{raw} + b_{proj}\right)\right)$$

Trong đó:
- $W_{proj} \in \mathbb{R}^{256 \times 608}$: Ma trận chiếu được học (Projection Matrix)
- $b_{proj} \in \mathbb{R}^{256}$: Bias
- $\text{LayerNorm}$: Layer Normalization đảm bảo unit variance
- $e_{USV} \in \mathbb{R}^{256}$: Embedding vector cuối cùng, chuẩn hóa về $\ell_2$-unit sphere

**Huấn luyện ma trận chiếu:**

$W_{proj}$ được học với mục tiêu Metric Learning: các trạng thái thị trường dẫn đến cùng outcome (đều là BSL_HIT chẳng hạn) phải có cosine similarity cao trong không gian embedding:

$$\mathcal{L}_{metric} = \sum_{(i,j) \in \text{pos}} \max(0, \delta - \cos(e_i, e_j)) + \sum_{(i,k) \in \text{neg}} \max(0, \cos(e_i, e_k) - \delta_{neg})$$

Với $\delta = 0.85$ (ngưỡng similarity), $\delta_{neg} = 0.50$.

#### III.4.2 Collection `debate_archive`

**Cấu trúc document:**

```
COLLECTION: debate_archive
──────────────────────────────────────────────────────────────────
Vector Dimension : 256
Distance Metric  : Cosine
Index Type       : HNSW (Hierarchical Navigable Small World)
    m = 16            ← Số cạnh tối đa mỗi node
    ef_construction = 200  ← Độ chính xác khi xây dựng
    ef             = 128   ← Độ chính xác khi tìm kiếm

Document Structure:
    id          : string (format: "{symbol}_{bar_close_timestamp}")
    vector      : float32[256] = e_USV tại thời điểm tranh biện
    payload:
        symbol              : string
        bar_close_time      : int64
        consensus_rating    : int8      ← [-4, +4]
        preferred_direction : string
        actual_outcome      : string    ← Điền sau khi outcome xác nhận
        actual_outcome_time : int64     ← NULL trước khi điền
        i_news              : float32
        session             : string
        macro_regime        : string
        p_bsl               : float32
        p_ssl               : float32
        bull_confidence     : float32
        bear_confidence     : float32
        bull_evidence       : string[]  ← List bằng chứng Bull Agent
        bear_evidence       : string[]
        full_debate_text    : string    ← Full transcript (cho Critic Agent RAG)
        key_fvg_zones       : string    ← JSON list of active FVG zones
        key_ob_zones        : string    ← JSON list of active OB zones

Payload Index (cho filter trước khi search):
    CREATE INDEX ON payload.symbol
    CREATE INDEX ON payload.macro_regime
    CREATE INDEX ON payload.actual_outcome
    CREATE INDEX ON payload.consensus_rating
──────────────────────────────────────────────────────────────────
```

**Luồng tìm kiếm RAG cho Critic Agent:**

```
THUẬT TOÁN TÌM KIẾM RAG — DEBATE ARCHIVE:
──────────────────────────────────────────────────────────────────
Đầu vào: e_USV_current (256-dim), symbol, macro_regime

[1] PRE-FILTER (Payload Filter):
    Chỉ xem xét documents có:
        symbol = current_symbol
        VÀ actual_outcome IS NOT NULL    ← Chỉ lấy precedents đã có kết quả
        VÀ macro_regime IN {same_regime, "NORMAL"}   ← Ưu tiên cùng chế độ

[2] ANN SEARCH:
    Tìm top-10 nearest neighbors bằng cosine similarity
    với ef = 128 (chế độ tìm kiếm chính xác cao)

[3] POST-FILTER và Threshold:
    Giữ lại kết quả có cosine_sim >= 0.80
    Nếu ít hơn 3 kết quả: giảm ngưỡng xuống 0.75

[4] RE-RANKING:
    Score cuối = 0.7 × cosine_sim + 0.3 × recency_weight
    recency_weight = exp(-λ × days_ago), λ = 0.01

[5] Lấy top-3 sau re-ranking
    Trả về: [(full_debate_text, actual_outcome, consensus_rating, cosine_sim), ...]
──────────────────────────────────────────────────────────────────
```

#### III.4.3 Collection `zone_embeddings`

**Mục đích:** Lưu trữ embedding của các vùng FVG/OB lịch sử để tìm kiếm vùng tương tự, phục vụ ước lượng $P_{hold}$ offline và phân tích pattern.

**Cấu trúc embedding vùng:**

Vector đặc trưng vùng $e_{zone} \in \mathbb{R}^{64}$ được tính như sau:

$$e_{zone} = W_{zone\_proj} \cdot F_{zone}^{(k)} \in \mathbb{R}^{64}$$

$W_{zone\_proj} \in \mathbb{R}^{64 \times 16}$ — ma trận chiếu nhỏ, học bởi Metric Learning với mục tiêu: các vùng có $P_{hold}$ tương tự phải gần nhau trong không gian embedding.

```
COLLECTION: zone_embeddings
──────────────────────────────────────────────────────────────────
Vector Dimension : 64
Distance Metric  : Cosine
Index Type       : HNSW (m=8, ef_construction=100)

Document Structure:
    id          : string (= zone_id từ zone_history table)
    vector      : float32[64] = e_zone
    payload:
        symbol              : string
        timeframe           : string
        zone_type           : string
        actual_held         : bool      ← Ground truth khi có
        p_hold_model_b      : float32   ← Dự đoán của Model B
        iii_formation       : float32
        w_zone              : float32
        session             : string
        touch_count         : int8
        formed_time         : int64
        size_pips           : float32
        mitigation_type     : string    ← NULL nếu chưa mitigated
──────────────────────────────────────────────────────────────────
```

**Ứng dụng k-NN cho ước tính P_hold offline:**

Với một vùng mới $z_{new}$, ước tính $P_{hold}$ bằng k-NN weighted average:

$$\hat{P}_{hold,kNN} = \frac{\sum_{i=1}^{k} w_i \cdot y_i}{\sum_{i=1}^{k} w_i}, \quad w_i = \cos(e_{z_{new}}, e_{z_i})$$

Đây là ước tính tham khảo bổ sung cho Model B XGBoost (không thay thế). Hữu ích nhất khi Model B gặp vùng với đặc trưng nằm ngoài phân phối huấn luyện.

#### III.4.4 Vòng Đời Dữ Liệu và Chính Sách Lưu Trữ

```
CHÍNH SÁCH LƯU TRỮ VÀ QUẢN LÝ DỮ LIỆU:
──────────────────────────────────────────────────────────────────
REDIS (Volatile Cache):
    → Tự động expire theo TTL
    → Emergency eviction: LRU policy (allkeys-lru)
    → Tải lại từ SQLite khi cần: Zone Registry và Macro State

SQLITE (Persistent Store):
    → Lưu vĩnh viễn (không tự xóa)
    → Archival policy:
        predictions có outcome_determined=1 AND prediction_time < NOW()-90days
        → Nén thành Parquet trong thư mục data/archive/
    → Vacuum định kỳ: mỗi Chủ Nhật 02:00 UTC (PRAGMA incremental_vacuum)
    → WAL checkpoint: tự động khi WAL > 64MB

VECTOR DB (Semantic Store):
    → debate_archive: Giữ 12 tháng gần nhất
    → Mỗi tháng: chạy compaction để giảm index size
    → Xóa documents có actual_outcome IS NULL VÀ tuổi > 30 ngày
      (chưa xác nhận outcome = precedent không tin cậy)
    → zone_embeddings: Giữ 24 tháng, limit 500,000 zones
──────────────────────────────────────────────────────────────────
```

---

### III.5 LUỒNG DỮ LIỆU TỔNG HỢP (END-TO-END DATA FLOW)

```
LUỒNG DỮ LIỆU ĐẦY ĐỦ — TỪ TICK ĐẾN PERSISTENCE:
══════════════════════════════════════════════════════════════════

[1] TICK MỚI ĐẾN (Module 1):
    → Redis SET "latent:{sym}:{ts}" ← Sau khi LSTM inference
    → Redis HSET "zone:{sym}:{tf}:{type}:{ts}" ← Zone Registry update
    → Redis HSET "metrics:ipc:latest" ← Telemetry

[2] NẾN M1 ĐÓNG CỬA (Module 1 → Module 3):
    → XGBoost A/B inference ← đọc từ Redis: "latent:", "zone:"
    → Redis HSET "ai:output:{sym}:latest" ← Kết quả mới nhất
    → SQLite INSERT INTO predictions ← Lưu vĩnh viễn (async)

[3] VÒNG TRANH BIỆN HOÀN THÀNH (Module 3):
    → Redis HSET "debate:{sym}:{ts}" ← Debate log (TTL 1h)
    → SQLite UPDATE predictions SET consensus_rating ← Cập nhật record

[4] SAU 1 GIỜ — ARCHIVAL (Module 4 Background Worker):
    → Đọc Redis "debate:{sym}:{ts}" (trước khi expire)
    → Tính e_USV projection
    → Qdrant INSERT INTO debate_archive ← Vector + payload
    → Redis DEL "debate:{sym}:{ts}" (hoặc để expire tự nhiên)

[5] OUTCOME XÁC NHẬN (Module 4 Outcome Determination):
    → SQLite UPDATE predictions SET outcome_determined=1, actual_direction=...
    → SQLite UPDATE zone_history SET actual_held_touch_1=...
    → Qdrant UPDATE debate_archive.payload.actual_outcome ← Điền kết quả
    → Redis PUBLISH "outcome_confirmed:{sym}" ← Notify Module 5

[6] ĐÁNH GIÁ DRIFT (Module 5):
    → SQLite SELECT FROM predictions WHERE outcome_determined=1 ORDER BY prediction_time DESC LIMIT 20
    → Tính IC_rolling_20, Brier Score
    → SQLite INSERT INTO model_performance ← Lưu metrics
    → Redis HSET "metrics:model:latest" ← Update real-time flag

[7] MODULE 3 RAG QUERY (trước mỗi vòng tranh biện):
    → Tính e_USV_current
    → Qdrant ANN SEARCH debate_archive (cosine >= 0.80)
    → Trả về top-3 precedents → Bull/Bear/Critic Agent context

══════════════════════════════════════════════════════════════════
```

---

## PHẦN IV — PHỤ LỤC: BẢNG ĐẶC TRƯNG TỔNG HỢP

### IV.1 Master Feature Registry — Toàn Bộ Đặc Trưng Đầu Vào AI

| Nhóm | Ký hiệu | Dim | Nguồn | Model A | Model B |
|---|---|---|---|---|---|
| Latent LSTM | $z$ | 512 | LSTM Autoencoder | ✓ | ✓ |
| Thanh khoản | $F_{liq}$ | 24 | Module 3 SMC Scanner | ✓ | — |
| Dòng lệnh | $F_{flow}$ | 20 | Module 1 Volumetrics | ✓ | ✓ (trong contact) |
| Macro | $F_{macro}$ | 12 | Module 2 Calendar | ✓ | ✓ |
| Cấu trúc | $F_{struct}$ | 64 | Module 3 Feature Map | ✓ | — |
| FVG/OB tổng hợp | $F_{agg}$ | 16 | Module 3 Zone Scan | ✓ | — |
| Vùng cụ thể | $F_{zone}^{(k)}$ | 16 | Module 3 per Zone | — | ✓ |
| Nến chạm vùng | $F_{contact}$ | 20 | Module 1 + 3 | — | ✓ |
| **Tổng** | | **648** | | **Model A** | **560** |

### IV.2 Bảng Ngưỡng Vận Hành Hệ Thống

| Chỉ số | Ngưỡng Tốt | Ngưỡng Cảnh Báo | Ngưỡng Kích Hoạt MODEL_DEGRADED |
|---|---|---|---|
| $IC_{rolling,20}$ | > 0.10 | 0.05–0.10 | < 0.05 |
| $BS_A$ (Brier Score) | < 0.20 | 0.20–0.28 | > 0.28 |
| $F1_{hold,70}$ | > 0.65 | 0.50–0.65 | < 0.50 |
| $ECE$ (Model A) | < 0.05 | 0.05–0.08 | > 0.08 |
| $FDS$ (Feature Drift) | < 0.20 | 0.20–0.40 | > 0.40 |
| IPC Latency avg | < 25ms | 25–50ms | > 50ms |
| Redis Memory | < 60% | 60–80% | > 80% |

---

*Tài liệu này là phần mở rộng chính thức của AGENTIC-QUANT V1.0, tạo thành phiên bản tổng hợp V4.5. Mọi thay đổi về kiến trúc XGBoost, schema cơ sở dữ liệu, hoặc thiết kế đặc trưng phải được cập nhật đồng thời vào cả hai tài liệu và được đánh phiên bản theo nguyên tắc Semantic Versioning (MAJOR.MINOR.PATCH).*

---
**Phiên bản tài liệu:** 4.5  
**Phạm vi:** Mở rộng từ V1.0 — Toán học hóa đặc trưng + XGBoost + Hybrid DB  
**Ngôn ngữ:** Tiếng Việt (theo yêu cầu)  
**Tổng số đặc trưng Model A:** 648 chiều  
**Tổng số đặc trưng Model B:** 560 chiều  
**Tầng DB:** Redis (< 1ms) + SQLite (< 10ms) + Qdrant (< 50ms)