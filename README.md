# Walmart Demand Forecasting — Dự báo nhu cầu với định lượng rủi ro

**Dự báo doanh số hàng tuần cho 3,331 phòng ban trên toàn hệ thống, kèm khoảng tin cậy cụ thể — giúp đội ngũ Inventory Planning biết chính xác nên nhập bao nhiêu và tự tin đến mức nào, thay vì chỉ một con số dự báo đơn lẻ.**

> ⚠️ **Lưu ý quan trọng — đọc trước khi xem kết quả:**
> - Dữ liệu công khai từ *Walmart Recruiting - Store Sales Forecasting* (Kaggle, 2010–2012), không phải dữ liệu vận hành thật.
> - Dataset không có cột tồn kho — toàn bộ phân tích giả định `Sales ≈ Demand`, **không kiểm chứng được** hiện tượng stockout (hết hàng làm demand thật bị che khuất).
> - Cột `MarkDown` chỉ có dữ liệu từ 11/2011 trở đi (63% dòng trước đó là thiếu thật, không phải lỗi).
> - Đây là dự án portfolio mô phỏng quy trình phân tích chuyên nghiệp: kiểm định giả định thống kê đầy đủ, so sánh nhiều phương pháp bằng bằng chứng thực nghiệm, và báo cáo trung thực cả những gì không hiệu quả — không chỉ chọn con số đẹp nhất để trình bày.

---

## 1. Câu hỏi kinh doanh

> "Tuần tới mỗi phòng ban cần dự báo doanh số bao nhiêu — và khoảng tin cậy bao nhiêu để không tồn kho dư thừa hay thiếu hàng?"

## 2. Kết quả chính

![Xu hướng dự báo](../reports/sarimax_bootstrap_final.png)

Trên **2,847 phòng ban** đủ lịch sử dữ liệu liên tục, mô hình SARIMAX cải thiện độ chính xác có ý nghĩa so với phương pháp đơn giản nhất (dự báo bằng doanh số cùng kỳ năm trước):

| Thước đo | Giá trị |
|---|---|
| MAPE trung vị | **12.61%** |
| Tỷ lệ thắng seasonal naive | **58.8%** |
| Phương pháp định lượng bất định | Bootstrap Prediction Interval (không giả định phân phối chuẩn) |

Ở mức tổng hợp toàn công ty, model đạt **MAPE 1.23%**, thắng rõ rệt so với naive (2.28%).

## 3. Phương pháp luận — kiến trúc 3 nhánh dựa trên bằng chứng thực nghiệm

Không phải mọi phòng ban đều dự báo được bằng cùng một phương pháp. Sau khi kiểm chứng bằng backtest trên dữ liệu thật, kiến trúc cuối cùng được chốt như sau:

| Nhóm | Tiêu chí | Số phòng ban | Phương pháp | Kết quả |
|---|---|---|---|---|
| **A** | ≥104 tuần dữ liệu liên tục | 2,847 | SARIMAX(2,0,2)(1,0,0,52) + Bootstrap PI | MAPE trung vị 12.61%, thắng naive 58.8% |
| **B** | Quy mô lớn nhưng theo mùa/gián đoạn | 121 | LightGBM (pooled, cross-series) | Thắng naive ở nhóm doanh số lớn (25.7% vs 27.2%) |
| **C** | Quy mô nhỏ, theo mùa/gián đoạn | 363 | Seasonal Naive trực tiếp | Không model phức tạp nào thắng nổi baseline |

**Vì sao chia 3 nhánh, không dùng 1 model cho tất cả:** Thử nghiệm ban đầu cho thấy SARIMAX với order cố định thắng naive ở 80% mẫu thử nhỏ, nhưng khi chạy hàng loạt trên toàn bộ 2,847 phòng ban, tỷ lệ thắng thực tế là 58.8% — và với nhóm phòng ban theo mùa (68% các trường hợp thiếu dữ liệu), LightGBM không những không cải thiện mà còn học nhiễu do dữ liệu quá thưa. Quyết định dùng seasonal naive cho nhóm này không phải "bỏ cuộc", mà là lựa chọn thực dụng: khi baseline đơn giản đã đủ tốt, thêm độ phức tạp chỉ tăng chi phí vận hành mà không tăng giá trị.

## 4. Insight kinh doanh nổi bật

**a. Độ chính xác tỷ lệ thuận với quy mô doanh số**

| Quy mô phòng ban | MAPE trung vị | Độ rộng PI 95% |
|---|---|---|
| Q4 (lớn nhất) | 6.6% | 28% giá trị dự báo |
| Q1 (nhỏ nhất) | 25.3% | 120% giá trị dự báo |

→ Khuyến nghị: dùng sai số tuyệt đối (USD) thay vì phần trăm khi đánh giá phòng ban nhỏ; ưu tiên đầu tư độ chính xác cho nhóm lớn, nơi 1% sai số tương đương giá trị USD lớn nhất.

**b. Cờ `IsHoliday` gốc của Walmart bị lệch nhãn cho Christmas**

Tuần được đánh dấu lễ (chứa 31/12) thực chất là tuần *sau* cao điểm mua sắm — tuần thật sự tăng vọt là tuần chứa 24/12 (**+70.3%** so với tuần thường, vượt cả Thanksgiving +42.8%). Phát hiện qua kiểm tra thủ công 3 tuần lân cận mỗi năm, xác nhận nhất quán ở cả 2010 và 2011.

**c. 68% phòng ban "thiếu dữ liệu" thực chất là đặc tính kinh doanh, không phải lỗi**

Phân tích độ dài chuỗi gián đoạn (streak) cho thấy phần lớn missing weeks đến từ các phòng ban dừng hoạt động theo mùa (VD: sản phẩm theo lễ hội), lặp lại đúng chu kỳ mỗi năm — không phải dữ liệu bị mất ngẫu nhiên.

## 5. Kiểm định giả định thống kê — không dừng ở accuracy

- **Stationarity:** ADF và KPSS đồng thuận stationary trên chuỗi tổng hợp (p<0.001 và p>0.1).
- **Seasonal unit root:** Canova-Hansen và OCSB bất đồng (D=1 vs D=0) — quyết định D=0 dựa trên rolling-origin backtest (MAPE thấp hơn, ổn định hơn), không dựa vào AIC vì AIC không so sánh được giữa các bậc differencing khác nhau.
- **Residual diagnostics:** Ljung-Box xác nhận residual là white noise sau khi loại bỏ 55 quan sát đầu (artifact khởi tạo Kalman filter, xác nhận qua thực nghiệm kurtosis giảm từ 38.4 xuống 1.29).
- **Normality:** Shapiro-Wilk ban đầu bác bỏ mạnh → chọn Bootstrap Prediction Interval thay vì CI dựa trên phân phối chuẩn.

## 6. Những gì đã thử và KHÔNG hiệu quả — báo cáo trung thực

Đây là phần thường bị bỏ qua trong các báo cáo phân tích, nhưng quan trọng để đánh giá đúng độ tin cậy của phương pháp:

- **Đưa holiday dummies (Super Bowl, Thanksgiving...) làm exogenous variable trong SARIMAX** → gây collinearity nghiêm trọng với seasonal AR term (condition number >10⁴⁰), model overfit vào từng ngày lễ hiếm gặp, MAPE tệ hơn cả naive (17.94% vs 2.28%). Quyết định: bỏ hẳn exog, giữ định lượng hiệu ứng lễ ở mức EDA (group-mean), tách bạch khỏi model dự báo chính.
- **LightGBM cho toàn bộ nhóm theo mùa (B+C)** → chỉ thắng naive ở nhóm doanh số lớn (Q4); thua rõ rệt ở Q1-Q3 do dữ liệu quá thưa để học tốt hơn baseline.
- **MAPE làm thước đo tổng hợp duy nhất** → phân phối lệch cực mạnh do một số phòng ban có doanh số gần 0 (mẫu số MAPE không ổn định về mặt toán học). Chuyển sang dùng median và WAPE, báo cáo tách riêng nhóm doanh số cực nhỏ.

## 7. Giới hạn (Limitations)

- **Không kiểm chứng được stockout:** dataset không có cột tồn kho, giả định sales ≈ demand có thể không đúng ở các phòng ban thường xuyên hết hàng.
- **Prediction Interval của nhóm B+C dùng residual gộp (pooled), không riêng từng phòng ban** — do mỗi phòng ban trong nhóm này có quá ít quan sát test để bootstrap ổn định riêng lẻ. PI phản ánh mức độ bất định trung bình của nhóm, không phải đặc thù từng phòng ban cụ thể.
- **Order SARIMAX (2,0,2)(1,0,0,52) được chốt từ chuỗi tổng hợp, áp dụng chung cho 2,847 chuỗi riêng lẻ** — chưa tối ưu riêng từng chuỗi bằng auto_arima do giới hạn thời gian tính toán (~100 phút cho 1 lần chạy hàng loạt).
- **Dữ liệu chỉ đến 2012**, khoảng cách thời gian với hiện tại lớn — mô hình phản ánh hành vi tiêu dùng giai đoạn đó, cần huấn luyện lại với dữ liệu mới nếu áp dụng thực tế.
- **Coverage 95% CI được đánh giá trên tập test 8 tuần** — cỡ mẫu nhỏ, cần diễn giải thận trọng, không khẳng định tuyệt đối tỷ lệ bao phủ dài hạn.

## 8. Công nghệ sử dụng

`DuckDB (SQL)` → `Python (statsmodels, pmdarima, LightGBM)` → `Streamlit`

## 9. Cấu trúc repo


```text
walmart-demand-forecast/
│
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   └── processed/            # Dữ liệu đã làm sạch và feature engineering
│
├── notebooks/               # EDA, feature engineering, modeling
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── sql/                     # SQL scripts (DuckDB)
│   ├── create_features.sql
│   └── analysis.sql
│
├── streamlit_app/           # Dashboard tương tác
│   ├── app.py
│   ├── pages/
│   └── assets/
│
├── reports/
│   ├── figures/             # Biểu đồ
│   └── report.md            # Báo cáo kết quả
│
├── models/                  # Mô hình đã huấn luyện
│
├── requirements.txt         # Danh sách thư viện
├── README.md
└── .gitignore
```

## 10. Demo trực tiếp

🔗 [Link Streamlit App] *(cập nhật sau khi deploy, kiểm tra link còn hoạt động trước khi public)*

📓 [Notebook đầy đủ] *(link GitHub)*

