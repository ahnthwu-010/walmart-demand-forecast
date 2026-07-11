-- notebooks/sql/02_features_nhanh_bc.sql
-- Chuẩn bị dữ liệu cho LightGBM: lag, rolling, và cờ theo mùa

CREATE OR REPLACE TABLE nhanh_bc_features AS
WITH full_grid AS (
    -- Grid đầy đủ Store-Dept x Date, để phát hiện đúng tuần nào "không active"
    SELECT p.Store, p.Dept, a.Date
    FROM (SELECT DISTINCT Store, Dept FROM read_parquet('{PROCESSED}/walmart_clean.parquet')) p
    CROSS JOIN (SELECT DISTINCT Date FROM read_parquet('{PROCESSED}/walmart_clean.parquet')) a
),
joined AS (
    SELECT
        g.Store, g.Dept, g.Date,
        d.Weekly_Sales,
        CASE WHEN d.Date IS NULL THEN 0 ELSE 1 END AS is_active_season,
        d.IsHoliday, d.Temperature, d.Fuel_Price, d.CPI, d.Unemployment,
        d.Store_Type, d.Store_Size
    FROM full_grid g
    LEFT JOIN read_parquet('{PROCESSED}/walmart_clean.parquet') d
        ON g.Store = d.Store AND g.Dept = d.Dept AND g.Date = d.Date
)
SELECT
    Store, Dept, Date,
    COALESCE(Weekly_Sales, 0) AS Weekly_Sales,  -- off-season = 0 thật, không NaN
    is_active_season,
    -- Lag features - dùng window function, tính TRÊN chuỗi đã fill 0
    LAG(COALESCE(Weekly_Sales,0), 1) OVER (PARTITION BY Store, Dept ORDER BY Date) AS lag_1,
    LAG(COALESCE(Weekly_Sales,0), 52) OVER (PARTITION BY Store, Dept ORDER BY Date) AS lag_52,
    AVG(COALESCE(Weekly_Sales,0)) OVER (
        PARTITION BY Store, Dept ORDER BY Date ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
    ) AS rolling_mean_4,
    -- Cờ mùa: đang active bao nhiêu % trong 8 tuần gần nhất - báo hiệu sắp vào/ra mùa
    AVG(is_active_season) OVER (
        PARTITION BY Store, Dept ORDER BY Date ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
    ) AS pct_active_recent_8w,
    IsHoliday, Temperature, Fuel_Price, CPI, Unemployment, Store_Type, Store_Size
FROM joined
ORDER BY Store, Dept, Date;