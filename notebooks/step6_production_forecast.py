import duckdb
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from multiprocessing import Pool, cpu_count
import lightgbm as lgb
import warnings, time

warnings.filterwarnings('ignore')

BASE = "C:/Users/HP/Downloads/DS/da9/walmart-demand-forecast"
PROCESSED = f"{BASE}/data/processed"
FORECAST_HORIZON = 4 


def forecast_sarimax(args):
    """Nhánh A: fit trên TOÀN BỘ dữ liệu (không giữ test), dự báo thật ra tương lai"""
    s, d = args
    con_local = duckdb.connect()
    try:
        series_df = con_local.execute(f"""
            SELECT Date, Weekly_Sales FROM read_parquet('{PROCESSED}/walmart_clean.parquet')
            WHERE Store={s} AND Dept={d} ORDER BY Date
        """).df()
        ts_s = series_df.set_index('Date')['Weekly_Sales'].asfreq('W-FRI')
        ts_s = ts_s.interpolate(limit=10, limit_direction='both')
        model = SARIMAX(ts_s, order=(2,0,2), seasonal_order=(1,0,0,52),
                         enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=50)

        residuals = fitted.resid[55:] if len(fitted.resid) > 55+20 else fitted.resid[-20:]
        forecast_result = fitted.get_forecast(steps=FORECAST_HORIZON)
        point_forecast = forecast_result.predicted_mean

        # Bootstrap PI cho forecast thật
        np.random.seed(42)
        n_boot = 2000
        boot_forecasts = np.array([
            point_forecast.values + np.random.choice(residuals, size=FORECAST_HORIZON, replace=True)
            for _ in range(n_boot)
        ])
        lower_95 = np.percentile(boot_forecasts, 2.5, axis=0)
        upper_95 = np.percentile(boot_forecasts, 97.5, axis=0)

        return pd.DataFrame({
            'Store': s, 'Dept': d, 'Date': point_forecast.index,
            'forecast': point_forecast.values, 'lower_95': lower_95, 'upper_95': upper_95,
            'method': 'SARIMAX'
        })
    except Exception as e:
        return pd.DataFrame({'Store': [s], 'Dept': [d], 'Date': [None],
                              'forecast': [None], 'lower_95': [None], 'upper_95': [None],
                              'method': [f'ERROR: {str(e)[:50]}']})


if __name__ == '__main__':
    con = duckdb.connect()
    streak_df = pd.read_csv(f"{PROCESSED}/gap_classification.csv")
    weeks_count = con.execute(f"""
        SELECT Store, Dept, COUNT(DISTINCT Date) AS n_weeks
        FROM read_parquet('{PROCESSED}/walmart_clean.parquet') GROUP BY Store, Dept
    """).df()
    merged = weeks_count.merge(streak_df[['Store','Dept','max_streak']], on=['Store','Dept'], how='left')
    merged['max_streak'] = merged['max_streak'].fillna(0)
    nhanh_a = merged[(merged['n_weeks'] >= 104) & (merged['max_streak'] <= 10)][['Store','Dept']]

    tasks = list(zip(nhanh_a['Store'].astype(int), nhanh_a['Dept'].astype(int)))
    n_workers = max(1, cpu_count() - 1)
    print(f"[SARIMAX] Dự báo {FORECAST_HORIZON} tuần tới cho {len(tasks)} cặp Nhánh A...")

    t0 = time.time()
    all_forecasts = []
    with Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(forecast_sarimax, tasks), 1):
            all_forecasts.append(res)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} ({(time.time()-t0)/60:.1f} phút)")

    final_forecast_a = pd.concat(all_forecasts, ignore_index=True)
    final_forecast_a.to_csv(f"{PROCESSED}/production_forecast_nhanh_a.csv", index=False)
    print(f"Xong Nhánh A: {(time.time()-t0)/60:.1f} phút -> production_forecast_nhanh_a.csv")