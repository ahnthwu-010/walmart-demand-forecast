import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

BASE = "C:/Users/HP/Downloads/DS/da9/walmart-demand-forecast"
PROCESSED = f"{BASE}/data/processed"
FORECAST_HORIZON = 4

con = duckdb.connect()

# -1. Xác định lại đúng nhóm Nhánh B+C, và phân chia Q4 vs Q1-Q3 
streak_df = pd.read_csv(f"{PROCESSED}/gap_classification.csv")
weeks_count = con.execute(f"""
    SELECT Store, Dept, COUNT(DISTINCT Date) AS n_weeks
    FROM read_parquet('{PROCESSED}/walmart_clean.parquet') GROUP BY Store, Dept
""").df()
merged = weeks_count.merge(streak_df[['Store','Dept','max_streak']], on=['Store','Dept'], how='left')
merged['max_streak'] = merged['max_streak'].fillna(0)
nhanh_bc = merged[~((merged['n_weeks'] >= 104) & (merged['max_streak'] <= 10))][['Store','Dept']]

avg_sales = con.execute(f"""
    SELECT Store, Dept, AVG(Weekly_Sales) AS avg_sales
    FROM read_parquet('{PROCESSED}/walmart_clean.parquet') GROUP BY Store, Dept
""").df()
nhanh_bc = nhanh_bc.merge(avg_sales, on=['Store','Dept'])
nhanh_bc['quartile'] = pd.qcut(nhanh_bc['avg_sales'], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')

q4_pairs = nhanh_bc[nhanh_bc['quartile']=='Q4'][['Store','Dept']]
q123_pairs = nhanh_bc[nhanh_bc['quartile']!='Q4'][['Store','Dept']]
print(f"Q4 (LightGBM): {len(q4_pairs)} cặp")
print(f"Q1-Q3 (seasonal naive): {len(q123_pairs)} cặp")

# --- 2. Chuẩn bị feature đầy đủ (giống bước backtest, nhưng fit trên TOÀN BỘ dữ liệu) ---
sql_query = open(f"{BASE}/sql/02_features_nhanh_bc.sql", encoding='utf-8').read().format(PROCESSED=PROCESSED)
con.execute(sql_query)
features_df = con.execute("SELECT * FROM nhanh_bc_features").df()
features_df['Date'] = pd.to_datetime(features_df['Date'])
features_df['Store'] = features_df['Store'].astype('category')
features_df['Dept'] = features_df['Dept'].astype('category')
features_df['Store_Type'] = features_df['Store_Type'].astype('category')

feature_cols = ['Store', 'Dept', 'is_active_season', 'lag_1', 'lag_52', 'rolling_mean_4',
                 'pct_active_recent_8w', 'IsHoliday', 'Temperature', 'Fuel_Price',
                 'CPI', 'Unemployment', 'Store_Type', 'Store_Size']

# --- 3. Q4: train LightGBM trên TOÀN BỘ lịch sử, forecast 4 tuần tới ---
q4_features = features_df.merge(q4_pairs, on=['Store','Dept'])

q4_features['Store'] = q4_features['Store'].astype('category')
q4_features['Dept'] = q4_features['Dept'].astype('category')
q4_features['Store_Type'] = q4_features['Store_Type'].astype('category')

train_q4 = q4_features.dropna(subset=['lag_1','lag_52'])

model_q4 = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                               num_leaves=31, random_state=42, verbose=-1)
model_q4.fit(train_q4[feature_cols], train_q4['Weekly_Sales'],
             categorical_feature=['Store','Dept','Store_Type'])

last_date = features_df['Date'].max()
store_categories = train_q4['Store'].cat.categories
dept_categories = train_q4['Dept'].cat.categories
store_type_categories = train_q4['Store_Type'].cat.categories

q4_forecasts = []
for s, d in zip(q4_pairs['Store'], q4_pairs['Dept']):
    hist = features_df[(features_df['Store']==s) & (features_df['Dept']==d)].sort_values('Date').copy()
    hist['Store'] = pd.Categorical(hist['Store'], categories=store_categories)
    hist['Dept'] = pd.Categorical(hist['Dept'], categories=dept_categories)
    hist['Store_Type'] = pd.Categorical(hist['Store_Type'], categories=store_type_categories)

    for h in range(1, FORECAST_HORIZON+1):
        next_date = last_date + pd.Timedelta(weeks=h)
        last_row = hist.iloc[-1:].copy()
        last_row['Date'] = next_date
        last_row['lag_1'] = hist['Weekly_Sales'].iloc[-1] if h==1 else q4_forecasts[-1]['forecast']
        lag52_val = hist[hist['Date'] == next_date - pd.Timedelta(weeks=52)]['Weekly_Sales']
        last_row['lag_52'] = lag52_val.values[0] if len(lag52_val) > 0 else hist['Weekly_Sales'].mean()

        pred = model_q4.predict(last_row[feature_cols])[0]
        pred = max(0, pred)
        q4_forecasts.append({'Store': s, 'Dept': d, 'Date': next_date, 'forecast': pred, 'method': 'LightGBM'})

        new_row = last_row.assign(Weekly_Sales=pred)
        hist = pd.concat([hist, new_row], ignore_index=True)
        hist['Store'] = pd.Categorical(hist['Store'], categories=store_categories)
        hist['Dept'] = pd.Categorical(hist['Dept'], categories=dept_categories)
        hist['Store_Type'] = pd.Categorical(hist['Store_Type'], categories=store_type_categories)

q4_forecast_df = pd.DataFrame(q4_forecasts)
print(f"Xong Q4: {len(q4_forecast_df)} dòng forecast")

# --- 4. Q1-Q3: seasonal naive trực tiếp (lag_52)
q123_forecasts = []
for s, d in zip(q123_pairs['Store'], q123_pairs['Dept']):
    hist = features_df[(features_df['Store']==s) & (features_df['Dept']==d)].sort_values('Date')
    for h in range(1, FORECAST_HORIZON+1):
        next_date = last_date + pd.Timedelta(weeks=h)
        lag52_val = hist[hist['Date'] == next_date - pd.Timedelta(weeks=52)]['Weekly_Sales']
        pred = max(0, lag52_val.values[0]) if len(lag52_val) > 0 else 0
        q123_forecasts.append({'Store': s, 'Dept': d, 'Date': next_date, 'forecast': pred, 'method': 'Seasonal_Naive'})

q123_forecast_df = pd.DataFrame(q123_forecasts)
print(f"Xong Q1-Q3: {len(q123_forecast_df)} dòng forecast")

# --- 5. Gộp và lưu ---
final_bc = pd.concat([q4_forecast_df, q123_forecast_df], ignore_index=True)
final_bc.to_csv(f"{PROCESSED}/production_forecast_nhanh_bc.csv", index=False)
print(f"\n=== Đã lưu production_forecast_nhanh_bc.csv ({len(final_bc)} dòng) ===")