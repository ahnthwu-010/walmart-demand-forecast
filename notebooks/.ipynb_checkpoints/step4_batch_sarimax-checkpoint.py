import duckdb
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from multiprocessing import Pool, cpu_count
import warnings, time

warnings.filterwarnings('ignore')

BASE = "C:/Users/HP/Downloads/DS/da9/walmart-demand-forecast"
PROCESSED = f"{BASE}/data/processed"


def fit_one_series(args):
    s, d = args
    con_local = duckdb.connect()
    try:
        series_df = con_local.execute(f"""
            SELECT Date, Weekly_Sales FROM read_parquet('{PROCESSED}/walmart_clean.parquet')
            WHERE Store={s} AND Dept={d} ORDER BY Date
        """).df()
        series_df['Date'] = pd.to_datetime(series_df['Date'])
        ts_s = series_df.set_index('Date')['Weekly_Sales'].asfreq('W-FRI')
        ts_s = ts_s.interpolate(limit=10)

        test_h = 8
        train, test = ts_s.iloc[:-test_h], ts_s.iloc[-test_h:]

        model = SARIMAX(train, order=(2, 0, 2), seasonal_order=(1, 0, 0, 52),
                         enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=50)
        pred = fitted.get_forecast(steps=test_h).predicted_mean

        mape = mean_absolute_percentage_error(test, pred) * 100
        naive_pred = ts_s.iloc[-test_h - 52:-52].values if len(ts_s) >= test_h + 52 else [train.mean()] * test_h
        mape_naive = mean_absolute_percentage_error(test, naive_pred) * 100

        return {'Store': s, 'Dept': d, 'mape': mape, 'mape_naive': mape_naive,
                'beats_naive': mape < mape_naive, 'status': 'OK'}
    except Exception as e:
        return {'Store': s, 'Dept': d, 'status': 'ERROR', 'error_msg': str(e)[:100]}


def main():
    con = duckdb.connect()
    streak_df = pd.read_csv(f"{PROCESSED}/gap_classification.csv")
    weeks_count = con.execute(f"""
        SELECT Store, Dept, COUNT(DISTINCT Date) AS n_weeks
        FROM read_parquet('{PROCESSED}/walmart_clean.parquet') GROUP BY Store, Dept
    """).df()
    merged = weeks_count.merge(streak_df[['Store', 'Dept', 'max_streak']], on=['Store', 'Dept'], how='left')
    merged['max_streak'] = merged['max_streak'].fillna(0)
    nhanh_a = merged[(merged['n_weeks'] >= 104) & (merged['max_streak'] <= 10)][['Store', 'Dept']]

    tasks = list(zip(nhanh_a['Store'].astype(int), nhanh_a['Dept'].astype(int)))
    n_workers = max(1, cpu_count() - 1)
    print(f"Chạy {len(tasks)} chuỗi trên {n_workers} core song song...", flush=True)

    t0 = time.time()
    results = []
    with Pool(n_workers) as pool:
        # dùng imap thay vì map để in tiến trình theo thời gian thực
        for i, res in enumerate(pool.imap_unordered(fit_one_series, tasks), 1):
            results.append(res)
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"  Đã xong {i}/{len(tasks)} ({elapsed/60:.1f} phút)", flush=True)

    print(f"Hoàn thành trong {(time.time()-t0)/60:.1f} phút")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{PROCESSED}/nhanh_a_full_results.csv", index=False)

    ok = results_df[results_df['status'] == 'OK']
    print(f"\nOK: {len(ok)}/{len(results_df)}")
    print(f"Thắng naive: {ok['beats_naive'].sum()}/{len(ok)} ({ok['beats_naive'].mean()*100:.1f}%)")
    print(f"MAPE TB model: {ok['mape'].mean():.2f}%, MAPE TB naive: {ok['mape_naive'].mean():.2f}%")


if __name__ == '__main__':
    main()