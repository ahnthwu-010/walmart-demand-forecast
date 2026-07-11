import pandas as pd
import duckdb

PROCESSED = f"{BASE}/data/processed"
results_df = pd.read_csv(f"{PROCESSED}/nhanh_a_full_results.csv")

# 1. Phân phối MAPE - có bị lệch bởi 1 nhóm nhỏ MAPE cực cao không?
print("=== PHÂN PHỐI MAPE (percentile) ===")
print(results_df['mape'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

# 2. Bao nhiêu % có MAPE cực đoan (>100%, tức sai hơn cả giá trị thật)
extreme = results_df[results_df['mape'] > 100]
print(f"\nSố cặp có MAPE > 100%: {len(extreme)} ({len(extreme)/len(results_df)*100:.1f}%)")
print(f"Số cặp có MAPE > 50%: {(results_df['mape']>50).sum()} ({(results_df['mape']>50).mean()*100:.1f}%)")

# 3. Join với avg_sales để kiểm tra giả thuyết: MAPE cao có tương quan với
#    doanh số nhỏ không (giống pattern đã thấy ở Case 2)
con = duckdb.connect()
avg_sales = con.execute(f"""
    SELECT Store, Dept, AVG(Weekly_Sales) AS avg_sales, STDDEV(Weekly_Sales) AS std_sales
    FROM read_parquet('{PROCESSED}/walmart_clean.parquet')
    GROUP BY Store, Dept
""").df()

merged = results_df.merge(avg_sales, on=['Store','Dept'])
merged['cv'] = merged['std_sales'] / merged['avg_sales']  # coefficient of variation - đo độ biến động tương đối

# Chia theo quartile doanh số, xem MAPE trung bình mỗi nhóm
merged['sales_quartile'] = pd.qcut(merged['avg_sales'], 4, labels=['Q1_nhỏ_nhất','Q2','Q3','Q4_lớn_nhất'])
print("\n=== MAPE TRUNG BÌNH THEO QUARTILE DOANH SỐ ===")
print(merged.groupby('sales_quartile', observed=True).agg(
    mape_mean=('mape','mean'), mape_median=('mape','median'),
    beats_naive_rate=('beats_naive','mean'), n=('mape','count')
))

# Tương quan CV (độ biến động) với MAPE - giả thuyết: chuỗi biến động mạnh khó dự báo hơn
print(f"\nTương quan giữa CV (độ biến động) và MAPE: {merged['cv'].corr(merged['mape']):.3f}")

merged.to_csv(f"{PROCESSED}/nhanh_a_full_results_enriched.csv", index=False)