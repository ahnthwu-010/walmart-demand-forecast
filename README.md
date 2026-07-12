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

### 4.1. Phân bổ nguồn lực theo quy mô — nơi đáng đầu tư nhất

| Nhóm quy mô | % số phòng ban | % tổng doanh thu | MAPE trung vị | Độ rộng CI 95% |
|---|---|---|---|---|
| Q4 (lớn nhất) | 25% | **69.0%** | 6.6% | 28% |
| Q3 | 25% | 21.2% | 9.8% | 44% |
| Q2 | 25% | 8.1% | 14.3% | 60% |
| Q1 (nhỏ nhất) | 25% | **1.7%** | 25.3% | 120% |

**Hành động cụ thể:** 25% phòng ban lớn nhất (Q4) chiếm tới **69% tổng doanh thu** của toàn bộ Nhánh A, đồng thời có độ chính xác dự báo cao nhất (MAPE 6.6%, CI hẹp nhất 28%) — đây là nơi tập trung nhân sự Inventory Planning cấp cao mang lại ROI rõ ràng nhất, vì sai số 1% ở nhóm này tương đương giá trị USD lớn hơn nhiều so với sai số 1% ở nhóm nhỏ.

Ngược lại, 25% phòng ban nhỏ nhất (Q1) chỉ chiếm **1.7% doanh thu** nhưng có CI rộng gấp hơn 4 lần (120% so với 28%) — đầu tư công sức tinh chỉnh model cho nhóm này mang lại lợi ích không tương xứng với công sức. Khuyến nghị: quản lý nhóm Q1 bằng quy tắc đơn giản (% dự phòng cố định, VD +30% so với dự báo), dành nguồn lực kỹ thuật cho Q3-Q4 — nơi chiếm 90% doanh thu.

### 4.2. Phát hiện vận hành: nhãn ngày lễ hiện tại đang gây quyết định sai

Hệ thống dữ liệu Walmart gắn cờ "tuần lễ Giáng Sinh" vào tuần chứa 31/12 — nhưng phân tích cho thấy **tuần chứa 24/12 mới là tuần cao điểm thật** (+70.3% so với tuần thường, vượt cả Thanksgiving +42.8%), còn tuần được gắn cờ chính thức thực chất **thấp hơn cả tuần bình thường** (-7.7%).

**Hành động cụ thể:** Nếu team Merchandising/Supply Chain hiện đang dùng cờ `IsHoliday` gốc để lên kế hoạch nhập hàng trước Giáng Sinh, họ đang chuẩn bị hàng **sai thời điểm 1 tuần** — nhập hàng dồn vào đúng tuần nhu cầu đã giảm, trong khi tuần thực sự cần hàng (chứa 24/12) lại không được đánh dấu ưu tiên. Khuyến nghị: cập nhật lại quy tắc gắn cờ ngày lễ trong hệ thống vận hành, dùng `is_Christmas_true_peak` (tuần chứa 24/12) thay vì nhãn gốc.

### 4.3. 68% "dữ liệu thiếu" thực chất là tín hiệu kinh doanh, không phải lỗi hệ thống

457/671 phòng ban có khoảng trống dữ liệu dài (>10 tuần liên tục) — không phải do lỗi thu thập, mà do các phòng ban này **chủ động ngừng bán theo mùa**, lặp lại đúng khoảng thời gian mỗi năm (VD: gián đoạn nhất quán vào giai đoạn tháng 4-9 hàng năm ở 1 số phòng ban cụ thể).

**Hành động cụ thể:** đây là cơ hội bị bỏ lỡ cho Merchandising — nếu 457 phòng ban này đang được lên kế hoạch tồn kho như thể chúng hoạt động quanh năm, ngân sách mua hàng đang bị phân bổ sai mùa. Khuyến nghị: rà soát lại chu kỳ mở/đóng của các phòng ban này, đối chiếu với category sản phẩm thực tế (nhiều khả năng là sản phẩm theo mùa/lễ hội) để tối ưu lịch nhập hàng theo đúng cửa sổ hoạt động, thay vì áp dụng chu kỳ tồn kho tiêu chuẩn.

### 4.4. Không phải mọi phòng ban đều đáng đầu tư công nghệ dự báo phức tạp

Với 363 phòng ban quy mô nhỏ/theo mùa, không model nào (kể cả LightGBM) thắng nổi phương pháp đơn giản nhất (seasonal naive — dự báo bằng đúng doanh số cùng kỳ năm trước).

**Hành động cụ thể:** đây là khuyến nghị tiết kiệm chi phí vận hành rõ ràng — với nhóm 363 phòng ban này, không cần đầu tư hạ tầng tính toán, không cần retrain model định kỳ. Chỉ cần 1 quy tắc đơn giản (tham chiếu cùng kỳ năm trước) đã đủ tốt, giải phóng nguồn lực kỹ thuật để tập trung vào nhóm 2,968 phòng ban (A+B) nơi model phức tạp thực sự tạo ra giá trị.

### 4.5. Cảnh báo về độ tin cậy khi báo cáo theo phần trăm

Với các phòng ban doanh số cực nhỏ, sai số phần trăm (MAPE) có thể tăng vọt phi thực tế (lên đến hàng nghìn %) chỉ do bản chất toán học khi mẫu số gần 0 — không phản ánh model tệ.

**Hành động cụ thể:** nếu ban lãnh đạo yêu cầu báo cáo "độ chính xác dự báo" theo phần trăm trung bình toàn công ty, con số đó **dễ bị hiểu sai** nếu tính bằng mean thay vì median/WAPE. Khuyến nghị chuẩn hóa cách báo cáo nội bộ: dùng WAPE (weighted) cho báo cáo tổng hợp, dùng sai số tuyệt đối USD cho các phòng ban dưới ngưỡng doanh số nhất định (đề xuất <$100/tuần), tránh quyết định sai vì tin vào con số phần trăm bị outlier chi phối.

## 5. Kiểm định giả định thống kê — không dừng ở accuracy

- **Stationarity:** ADF và KPSS đồng thuận stationary trên chuỗi tổng hợp (p<0.001 và p>0.1).
- **Seasonal unit root:** Canova-Hansen và OCSB bất đồng (D=1 vs D=0) — quyết định D=0 dựa trên rolling-origin backtest (MAPE thấp hơn, ổn định hơn), không dựa vào AIC vì AIC không so sánh được giữa các bậc differencing khác nhau.
- **Residual diagnostics:** Ljung-Box xác nhận residual là white noise sau khi loại bỏ 55 quan sát đầu (artifact khởi tạo Kalman filter, xác nhận qua thực nghiệm kurtosis giảm từ 38.4 xuống 1.29).
- **Normality:** Shapiro-Wilk ban đầu bác bỏ mạnh → chọn Bootstrap Prediction Interval thay vì CI dựa trên phân phối chuẩn.

## 6. Những gì đã thử và KHÔNG hiệu quả 

Đây là phần thường bị bỏ qua trong các báo cáo phân tích, nhưng quan trọng để đánh giá đúng độ tin cậy của phương pháp:

- **Đưa holiday dummies (Super Bowl, Thanksgiving...) làm exogenous variable trong SARIMAX** → gây collinearity nghiêm trọng với seasonal AR term (condition number >10⁴⁰), model overfit vào từng ngày lễ hiếm gặp, MAPE tệ hơn cả naive (17.94% vs 2.28%). Quyết định: bỏ hẳn exog, giữ định lượng hiệu ứng lễ ở mức EDA (group-mean), tách bạch khỏi model dự báo chính.
- **LightGBM cho toàn bộ nhóm theo mùa (B+C)** → chỉ thắng naive ở nhóm doanh số lớn (Q4); thua rõ rệt ở Q1-Q3 do dữ liệu quá thưa để học tốt hơn baseline.
- **MAPE làm thước đo tổng hợp duy nhất** → phân phối lệch cực mạnh do một số phòng ban có doanh số gần 0 (mẫu số MAPE không ổn định về mặt toán học). Chuyển sang dùng median và WAPE, báo cáo tách riêng nhóm doanh số cực nhỏ.

## 7. Limitations

- **Không kiểm chứng được stockout:** dataset không có cột tồn kho, giả định sales ≈ demand có thể không đúng ở các phòng ban thường xuyên hết hàng.
- **Prediction Interval của nhóm B+C dùng residual gộp (pooled), không riêng từng phòng ban** — do mỗi phòng ban trong nhóm này có quá ít quan sát test để bootstrap ổn định riêng lẻ. PI phản ánh mức độ bất định trung bình của nhóm, không phải đặc thù từng phòng ban cụ thể.
- **Order SARIMAX (2,0,2)(1,0,0,52) được chốt từ chuỗi tổng hợp, áp dụng chung cho 2,847 chuỗi riêng lẻ** — chưa tối ưu riêng từng chuỗi bằng auto_arima do giới hạn thời gian tính toán (~100 phút cho 1 lần chạy hàng loạt).
- **Dữ liệu chỉ đến 2012**, khoảng cách thời gian với hiện tại lớn — mô hình phản ánh hành vi tiêu dùng giai đoạn đó, cần huấn luyện lại với dữ liệu mới nếu áp dụng thực tế.
- **Coverage 95% CI được đánh giá trên tập test 8 tuần** — cỡ mẫu nhỏ, cần diễn giải thận trọng, không khẳng định tuyệt đối tỷ lệ bao phủ dài hạn.

## 8. Công nghệ sử dụng

`DuckDB (SQL)` → `Python (statsmodels, pmdarima, LightGBM)` → `Streamlit`

## 9. Cấu trúc repo


# Walmart Demand Forecast

## Project Structure

```text
walmart-demand-forecast/
├── data/
│   ├── raw/                              # Dữ liệu gốc Kaggle (không public do dung lượng/license)
│   └── processed/                        # Dữ liệu đã xử lý, forecast production
│       └── production_forecast_ALL_final_v2.csv   # File forecast chính thức, dashboard đọc từ đây
│
├── notebooks/                            # Xem notebooks/README.md để biết thứ tự chạy chi tiết
│   ├── step1_join_and_check.ipynb        # Join dữ liệu, kiểm tra toàn vẹn
│   ├── step1b_investigate.ipynb          # Điều tra Missing Not At Random (MNAR)
│   ├── step2_eda_stationarity.ipynb      # ADF/KPSS, decomposition, phát hiện lệch nhãn Christmas
│   ├── step4_batch_sarimax.py            # Backtest SARIMAX hàng loạt (Nhánh A)
│   ├── step4_investigate_bias.py         # Điều tra phân phối MAPE bất thường
│   ├── step5_analyze_nhanh_a_results.ipynb   # Phân tích kết quả Nhánh A
│   ├── step5_lightgbm_nhanh_bc.ipynb     # Backtest LightGBM (Nhánh B+C)
│   ├── step6_production_forecast.py      # Forecast production Nhánh A
│   ├── step7_production_forecast_bc.py   # Forecast production Nhánh B+C
│   ├── step8_add_pi_bc.py                # Bootstrap Prediction Interval cho Nhánh B+C
│   └── step9_final_sanity_check.ipynb    # Kiểm tra toàn vẹn file forecast cuối cùng
│
├── sql/
│   └── 02_features_nhanh_bc.sql          # Feature engineering (lag, rolling, seasonal flags) bằng DuckDB
│
├── streamlit_app/
│   └── app.py                            # Dashboard tương tác
│
├── reports/                              # Biểu đồ và log
├── requirements.txt
└── README.md
```

## 10. Demo trực tiếp

🔗 [Link Streamlit App] *[ https://walmart-demand-forecast-jhkhxzynmevejab2hb9r5f.streamlit.app/]*

📓 [Notebook đầy đủ] *[https://github.com/ahnthwu-010/walmart-demand-forecast ]*

