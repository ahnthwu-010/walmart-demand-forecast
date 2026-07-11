import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Demand Forecast Ops", layout="wide", page_icon="📦")

BASE = Path(__file__).resolve().parent.parent

FORECAST_PATH = BASE / "data" / "processed" / "production_forecast_ALL_final_v2.csv"
HIST_PATH = BASE / "data" / "processed" / "walmart_clean.parquet"
LOG_PATH = BASE / "reports" / "refresh_log.txt"

LOG_PATH = BASE / "reports" / "refresh_log.txt"

st.write("DEBUG BASE:", BASE)
st.write("DEBUG BASE EXISTS:", BASE.exists())

st.write("DEBUG FORECAST:", FORECAST_PATH)
st.write("DEBUG FORECAST EXISTS:", FORECAST_PATH.exists())

st.write("DEBUG HIST:", HIST_PATH)
st.write("DEBUG HIST EXISTS:", HIST_PATH.exists())

# ============================================================
# CUSTOM CSS 
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #0F1419;
    color: #F5F1E8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #12181F;
    border-right: 1px solid #2A333D;
}
section[data-testid="stSidebar"] * { color: #F5F1E8 !important; }

/* Tiêu đề trang - kiểu biển báo kho hàng */
h1 {
    font-family: 'Oswald', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #F5F1E8 !important;
    border-bottom: 3px solid #E8A33D;
    padding-bottom: 12px;
    font-size: 2.1rem !important;
}
h2, h3 {
    font-family: 'Oswald', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #E8A33D !important;
    font-size: 1.15rem !important;
}

/* Metric cards - kiểu phiếu vận đơn (ticket stub) */
div[data-testid="stMetric"] {
    background-color: #1A2129;
    border: 1px dashed #3A4550;
    border-left: 4px solid #E8A33D;
    border-radius: 4px;
    padding: 16px 18px;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: 0.72rem !important;
    color: #8B95A1 !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: #F5F1E8 !important;
}
div[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Divider màu hổ phách mảnh */
hr { border-color: #2A333D !important; }

/* Caption / info box */
.stCaption, div[data-testid="stCaptionContainer"] { color: #8B95A1 !important; }
div[data-testid="stAlert"] {
    background-color: #1A2129;
    border-left: 4px solid #E8A33D;
}

/* Radio buttons trong sidebar */
div[role="radiogroup"] label {
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.3px;
}

/* Selectbox */
div[data-baseweb="select"] { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    plot_bgcolor='#0F1419', paper_bgcolor='#0F1419',
    font=dict(family='Inter, sans-serif', color='#F5F1E8'),
    xaxis=dict(gridcolor='#2A333D', linecolor='#3A4550'),
    yaxis=dict(gridcolor='#2A333D', linecolor='#3A4550'),
)

@st.cache_data(ttl=300)
def load_forecast():
    return pd.read_csv(FORECAST_PATH, parse_dates=['Date'])

@st.cache_data(ttl=300)
def load_historical():
    import duckdb
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{HIST_PATH}')").df()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data(ttl=300)
def get_last_updated():
    return datetime.fromtimestamp(FORECAST_PATH.stat().st_mtime)

forecast_df = load_forecast()
hist_df = load_historical()
last_updated = get_last_updated()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### 📦 DEMAND FORECAST")
st.sidebar.caption(f"⏱ CẬP NHẬT: {last_updated.strftime('%d/%m/%Y — %H:%M')}")

if st.sidebar.button("🔄 LÀM MỚI DỮ LIỆU"):
    st.cache_data.clear()
    st.rerun()

page = st.sidebar.radio("ĐIỀU HƯỚNG", ["TỔNG QUAN CÔNG TY", "CHI TIẾT STORE-DEPT"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**PHƯƠNG PHÁP DỰ BÁO**

🔵 **SARIMAX** — 2,847 phòng ban, chuỗi dài ổn định

🟢 **LightGBM** — 121 phòng ban, quy mô lớn/theo mùa

⚪ **Seasonal Naive** — 363 phòng ban, quy mô nhỏ/theo mùa

**GIỚI HẠN**

- Giả định sales ≈ demand (không kiểm chứng được stockout)
- Nhóm quy mô nhỏ có CI rộng hơn nhiều
""")

method_colors = {'SARIMAX': '#4A90D9', 'LightGBM': '#52B788', 'Seasonal_Naive': '#6C757D'}

# ============================================================
# TRANG 1: TỔNG QUAN CÔNG TY
# ============================================================
if page == "TỔNG QUAN CÔNG TY":
    st.title("TỔNG QUAN DỰ BÁO NHU CẦU")

    total_forecast = forecast_df['forecast'].sum()
    n_weeks = forecast_df['Date'].nunique()
    avg_weekly = total_forecast / n_weeks
    hist_avg_weekly = hist_df.groupby('Date')['Weekly_Sales'].sum().mean()
    pct_change = (avg_weekly / hist_avg_weekly - 1) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TỔNG DỰ BÁO", f"${total_forecast:,.0f}")
    col2.metric("TB / TUẦN", f"${avg_weekly:,.0f}", f"{pct_change:+.1f}% vs LS")
    col3.metric("SỐ PHÒNG BAN", f"{forecast_df[['Store','Dept']].drop_duplicates().shape[0]:,}")
    col4.metric("KỲ DỰ BÁO", f"{forecast_df['Date'].min().strftime('%d/%m')} → {forecast_df['Date'].max().strftime('%d/%m/%y')}")

    st.markdown("---")
    st.subheader("XU HƯỚNG: LỊCH SỬ → DỰ BÁO")

    hist_weekly = hist_df.groupby('Date')['Weekly_Sales'].sum().reset_index().tail(20)
    fc_weekly = forecast_df.groupby('Date').agg(
        forecast=('forecast', 'sum'), lower_95=('lower_95', 'sum'), upper_95=('upper_95', 'sum')
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_weekly['Date'], y=hist_weekly['Weekly_Sales'],
                              mode='lines+markers', name='Lịch sử thực tế',
                              line=dict(color='#F5F1E8', width=2)))
    fig.add_trace(go.Scatter(x=fc_weekly['Date'], y=fc_weekly['forecast'],
                              mode='lines+markers', name='Dự báo',
                              line=dict(color='#E8A33D', width=2, dash='dash')))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_weekly['Date'], fc_weekly['Date'][::-1]]),
        y=pd.concat([fc_weekly['upper_95'], fc_weekly['lower_95'][::-1]]),
        fill='toself', fillcolor='rgba(232,163,61,0.15)', line=dict(width=0),
        name='95% khoảng tin cậy', showlegend=True
    ))
    fig.update_layout(height=450, hovermode='x unified',
                       yaxis_title="Doanh số (USD)", xaxis_title="Tuần", **PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("ĐÓNG GÓP THEO PHƯƠNG PHÁP")

    method_summary = forecast_df.groupby('method').agg(
        n_pairs=('Store', lambda x: x.nunique()), total_forecast=('forecast', 'sum')
    ).reset_index()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(method_summary.style.format({'total_forecast': '${:,.0f}'}), use_container_width=True)
    with col_b:
        fig2 = go.Figure(go.Pie(
            labels=method_summary['method'], values=method_summary['n_pairs'],
            marker=dict(colors=[method_colors[m] for m in method_summary['method']]),
            hole=0.5,
            textinfo='label+percent'
        ))
        fig2.update_layout(height=300, margin=dict(t=10, b=10), **PLOTLY_TEMPLATE)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("⚠ CẢNH BÁO — TOP 10 PHÒNG BAN BẤT ĐỊNH CAO NHẤT")

    forecast_df['ci_width_pct'] = (
        (forecast_df['upper_95'] - forecast_df['lower_95']) / forecast_df['forecast'].replace(0, np.nan) * 100
    )
    risky = forecast_df.nlargest(10, 'ci_width_pct')[
        ['Store', 'Dept', 'Date', 'forecast', 'lower_95', 'upper_95', 'ci_width_pct', 'method']
    ]
    st.dataframe(risky.style.format({
        'forecast': '${:,.0f}', 'lower_95': '${:,.0f}', 'upper_95': '${:,.0f}', 'ci_width_pct': '{:.0f}%'
    }), use_container_width=True)
    st.caption("Các phòng ban này có khoảng tin cậy rất rộng — nên áp dụng safety stock theo tỷ lệ %, không theo số tuyệt đối.")

# ============================================================
# TRANG 2: CHI TIẾT STORE-DEPT
# ============================================================
else:
    st.title("CHI TIẾT DỰ BÁO STORE-DEPT")

    col1, col2 = st.columns(2)
    with col1:
        store_list = sorted(forecast_df['Store'].unique())
        selected_store = st.selectbox("STORE", store_list)
    with col2:
        dept_list = sorted(forecast_df[forecast_df['Store'] == selected_store]['Dept'].unique())
        selected_dept = st.selectbox("DEPT", dept_list)

    pair_forecast = forecast_df[
        (forecast_df['Store'] == selected_store) & (forecast_df['Dept'] == selected_dept)
    ].sort_values('Date')
    pair_hist = hist_df[
        (hist_df['Store'] == selected_store) & (hist_df['Dept'] == selected_dept)
    ].sort_values('Date').tail(30)

    if len(pair_forecast) == 0:
        st.warning("Không tìm thấy dự báo cho cặp Store-Dept này.")
    else:
        method_used = pair_forecast['method'].iloc[0]
        st.info(f"**PHƯƠNG PHÁP:** {method_used}")

        col1, col2, col3 = st.columns(3)
        col1.metric("DỰ BÁO TUẦN TỚI", f"${pair_forecast['forecast'].iloc[0]:,.0f}")
        col2.metric("CẬN DƯỚI 95%", f"${pair_forecast['lower_95'].iloc[0]:,.0f}")
        col3.metric("CẬN TRÊN 95%", f"${pair_forecast['upper_95'].iloc[0]:,.0f}")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=pair_hist['Date'], y=pair_hist['Weekly_Sales'],
                                    mode='lines+markers', name='Lịch sử',
                                    line=dict(color='#F5F1E8', width=2)))
        fig3.add_trace(go.Scatter(x=pair_forecast['Date'], y=pair_forecast['forecast'],
                                    mode='lines+markers', name='Dự báo',
                                    line=dict(color=method_colors.get(method_used, '#E8A33D'), width=2, dash='dash')))
        fig3.add_trace(go.Scatter(
            x=pd.concat([pair_forecast['Date'], pair_forecast['Date'][::-1]]),
            y=pd.concat([pair_forecast['upper_95'], pair_forecast['lower_95'][::-1]]),
            fill='toself', fillcolor='rgba(232,163,61,0.15)', line=dict(width=0), name='95% CI'
        ))
        fig3.update_layout(height=400, hovermode='x unified', yaxis_title="Doanh số (USD)", **PLOTLY_TEMPLATE)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.subheader("DỮ LIỆU CHI TIẾT")
        display_cols = ['Date', 'forecast', 'lower_95', 'upper_95', 'method']
        st.dataframe(pair_forecast[display_cols].style.format({
            'forecast': '${:,.0f}', 'lower_95': '${:,.0f}', 'upper_95': '${:,.0f}'
        }), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
if LOG_PATH.exists():
    last_lines = LOG_PATH.read_text(encoding='utf-8').strip().split("\n")[-3:]
    st.sidebar.caption("**TRẠNG THÁI TỰ ĐỘNG CẬP NHẬT**")
    for line in last_lines:
        display_line = line[:60] + "..." if len(line) > 60 else line
        st.sidebar.caption(f"`{display_line}`")
else:
    st.sidebar.caption("Chưa có lịch sử tự động cập nhật")