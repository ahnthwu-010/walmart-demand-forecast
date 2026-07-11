import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

BASE = "C:/Users/HP/Downloads/DS/da9/walmart-demand-forecast"
PROCESSED = f"{BASE}/data/processed"
FORECAST_HORIZON = 4
N_BOOTSTRAP = 2000

con = duckdb.connect()

# 1. Tái tạo lại đúng train/test split (8 tuần cuối) để có residual

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

cutoff_date = features_df['Date'].max() - pd.Timedelta(weeks=8)


# 2. Q4 - LightGBM: fit trên train (8 tuần cuối giữ lại làm test),

q4_features = features_df.merge(q4_pairs, on=['Store','Dept'])
q4_features['Store'] = q4_features['Store'].astype('category')
q4_features['Dept'] = q4_features['Dept'].astype('category')
q4_features['Store_Type'] = q4_features['Store_Type'].astype('category')

train_q4_holdout = q4_features[q4_features['Date'] <= cutoff_date].dropna(subset=['lag_1','lag_52'])
test_q4_holdout = q4_features[q4_features['Date'] > cutoff_date].dropna(subset=['lag_1','lag_52']).copy()

model_q4_holdout = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                       num_leaves=31, random_state=42, verbose=-1)
model_q4_holdout.fit(train_q4_holdout[feature_cols], train_q4_holdout['Weekly_Sales'],
                      categorical_feature=['Store','Dept','Store_Type'])

test_q4_holdout['predicted'] = model_q4_holdout.predict(test_q4_holdout[feature_cols])
# CHỈ lấy residual từ tuần đang active 
active_q4 = test_q4_holdout[test_q4_holdout['is_active_season']==1]
residuals_q4 = (active_q4['Weekly_Sales'] - active_q4['predicted']).values

print(f"Q4: {len(residuals_q4)} residual ngoài mẫu, std={residuals_q4.std():,.1f}")


# 3. Q1-Q3 - Seasonal naive: residual = actual - lag_52, đo trên tuần active

q123_features = features_df.merge(q123_pairs, on=['Store','Dept'])
q123_active = q123_features[
    (q123_features['is_active_season']==1) & (q123_features['lag_52'].notna())
].copy()
residuals_q123 = (q123_active['Weekly_Sales'] - q123_active['lag_52']).values

print(f"Q1-Q3: {len(residuals_q123)} residual ngoài mẫu (naive), std={residuals_q123.std():,.1f}")


# 4. Bootstrap PI cho forecast production đã có sẵn

forecast_bc = pd.read_csv(f"{PROCESSED}/production_forecast_nhanh_bc.csv")
np.random.seed(42)

def add_bootstrap_pi(df, residual_pool, method_name):
    subset = df[df['method']==method_name].copy()
    lower_list, upper_list = [], []
    for _, row in subset.iterrows():
        sampled = np.random.choice(residual_pool, size=N_BOOTSTRAP, replace=True)
        boot_vals = row['forecast'] + sampled
        lower_list.append(max(0, np.percentile(boot_vals, 2.5)))
        upper_list.append(max(0, np.percentile(boot_vals, 97.5)))
    subset['lower_95'] = lower_list
    subset['upper_95'] = upper_list
    return subset

q4_with_pi = add_bootstrap_pi(forecast_bc, residuals_q4, 'LightGBM')
q123_with_pi = add_bootstrap_pi(forecast_bc, residuals_q123, 'Seasonal_Naive')

final_bc_with_pi = pd.concat([q4_with_pi, q123_with_pi], ignore_index=True)
final_bc_with_pi.to_csv(f"{PROCESSED}/production_forecast_nhanh_bc_with_pi.csv", index=False)
print(f"\n=== Đã lưu production_forecast_nhanh_bc_with_pi.csv ({len(final_bc_with_pi)} dòng) ===")


# 5. Gộp với Nhánh A thành file cuối cùng thống nhất

forecast_a = pd.read_csv(f"{PROCESSED}/production_forecast_nhanh_a_final.csv")
forecast_a_slim = forecast_a[['Store','Dept','Date','forecast','lower_95','upper_95','method']]
final_bc_slim = final_bc_with_pi[['Store','Dept','Date','forecast','lower_95','upper_95','method']]

final_all = pd.concat([forecast_a_slim, final_bc_slim], ignore_index=True)
final_all.to_csv(f"{PROCESSED}/production_forecast_ALL_final.csv", index=False)
print(f"\n=== HOÀN TẤT: production_forecast_ALL_final.csv - {len(final_all)} dòng, {final_all[['Store','Dept']].drop_duplicates().shape[0]} cặp Store-Dept ===")
print(final_all['method'].value_counts())