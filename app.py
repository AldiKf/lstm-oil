"""
Dashboard Prediksi Harga Minyak Mentah (Brent) — Multimodal LSTM
Berbasis Data Historis & Sentimen Berita Global
==================================================================
Jalankan dengan: streamlit run app.py

File yang harus berada di folder yang sama:
- Model_lstm_oil.keras (atau model_lstm_oil.h5 sebagai fallback)
- scaler.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import os

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Brent Oil Price Forecast | LSTM Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS — tema energi (gelap, aksen emas/oranye)
# =========================================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(212, 160, 23, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f2f2f2;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #d4a017;
        font-weight: 500;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(212, 160, 23, 0.2);
        padding: 1rem 1.2rem;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] { color: #b0b0b0; }
    div[data-testid="stMetricValue"] { color: #f2f2f2; }

    .info-card {
        background: rgba(212, 160, 23, 0.08);
        border-left: 4px solid #d4a017;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-size: 0.92rem;
        color: #e0e0e0;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f2f2f2;
        margin: 1.2rem 0 0.6rem 0;
        border-bottom: 2px solid rgba(212, 160, 23, 0.3);
        padding-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛢️ Prediksi Harga Minyak Mentah Brent</div>
    <div class="hero-subtitle">Multimodal LSTM — Data Historis & Fitur Sentimen/Geopolitik Global</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# KONSTANTA — HARUS SAMA PERSIS DENGAN SAAT TRAINING
# =========================================================
FEATURES = [
    'brent_price', 'wti_price', 'dxy_index', 'vix', 'gpr_index',
    'brent_return', 'wti_return',
    'brent_lag_1', 'brent_lag_3', 'brent_lag_7',
    'wti_lag_1', 'wti_lag_3', 'wti_lag_7',
    'brent_volatility_7d', 'brent_volatility_30d',
    'wti_volatility_7d', 'wti_volatility_30d',
    'brent_wti_spread', 'event_severity', 'event_flag'
]
TARGET_COL_INDEX = 0  # brent_price
SEQ_LEN = 30

MODEL_PATH_KERAS = "Model_lstm_oil.keras"
MODEL_PATH_H5 = "model_lstm_oil.h5"
SCALER_PATH = "scaler.pkl"


# =========================================================
# LOADER (cached)
# =========================================================
@st.cache_resource(show_spinner="Memuat model LSTM...")
def load_lstm_model():
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH_KERAS):
        return load_model(MODEL_PATH_KERAS), MODEL_PATH_KERAS
    elif os.path.exists(MODEL_PATH_H5):
        return load_model(MODEL_PATH_H5, compile=False), MODEL_PATH_H5
    else:
        return None, None


@st.cache_resource(show_spinner="Memuat scaler...")
def load_scaler_obj():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def run_inference(df_feat, model, scaler):
    """Sliding-window t+1 inference di seluruh dataset yang diunggah."""
    scaled = scaler.transform(df_feat[FEATURES])

    X = []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
    X = np.array(X)

    if len(X) == 0:
        return None

    preds_scaled = model.predict(X, verbose=0).flatten()

    dummy = np.zeros((len(preds_scaled), len(FEATURES)))
    dummy[:, TARGET_COL_INDEX] = preds_scaled
    preds_inverse = scaler.inverse_transform(dummy)[:, TARGET_COL_INDEX]

    # prediksi[i] adalah ramalan untuk baris (i + SEQ_LEN), yaitu t+1 setelah window
    pred_dates = df_feat.index[SEQ_LEN:]
    actual_vals = df_feat['brent_price'].values[SEQ_LEN:]

    result = pd.DataFrame({
        "date": pred_dates,
        "actual": actual_vals,
        "predicted": preds_inverse
    }).set_index("date")

    return result


def forecast_next_step(df_feat, model, scaler):
    """Ramalan t+1 tunggal menggunakan 30 baris terakhir (di luar data historis)."""
    last_window = df_feat[FEATURES].values[-SEQ_LEN:]
    scaled_window = scaler.transform(last_window)
    X_next = np.expand_dims(scaled_window, axis=0)

    pred_scaled = model.predict(X_next, verbose=0).flatten()[0]
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, TARGET_COL_INDEX] = pred_scaled
    pred_price = scaler.inverse_transform(dummy)[0, TARGET_COL_INDEX]

    next_date = df_feat.index[-1] + timedelta(days=1)
    return next_date, pred_price


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Panel Kontrol")

    uploaded_file = st.file_uploader(
        "Unggah dataset (CSV)",
        type=["csv"],
        help="CSV harus memiliki kolom 'date' dan seluruh 20 fitur yang digunakan saat training."
    )

    st.markdown("---")
    st.markdown("**📋 Kolom fitur yang dibutuhkan:**")
    with st.expander("Lihat daftar 20 fitur"):
        for f in FEATURES:
            st.markdown(f"- `{f}`")

    st.markdown("---")
    st.markdown("""
    <div class="info-card">
    ⚠️ <b>Catatan model:</b> Model ini melakukan forecasting <b>single-step (t+1)</b>.
    Ramalan multi-hari ke depan tidak didukung langsung karena nilai fitur
    eksogen (DXY, VIX, GPR, sentimen berita, dll.) untuk hari-hari mendatang
    belum tersedia.
    </div>
    """, unsafe_allow_html=True)

    model, model_used = load_lstm_model()
    scaler = load_scaler_obj()

    st.markdown("---")
    st.markdown("**🧠 Status Model**")
    if model is not None:
        st.success(f"Model dimuat: `{model_used}`")
    else:
        st.error("Model tidak ditemukan (Model_lstm_oil.keras / model_lstm_oil.h5)")
    if scaler is not None:
        st.success("Scaler dimuat: `scaler.pkl`")
    else:
        st.error("scaler.pkl tidak ditemukan")


# =========================================================
# MAIN CONTENT
# =========================================================
if uploaded_file is None:
    st.info("👈 Unggah file CSV di sidebar untuk memulai analisis dan prediksi.")
    st.markdown("""
    <div class="info-card">
    Dashboard ini menerima data historis harga minyak (Brent & WTI) beserta fitur
    makro/geopolitik/sentimen yang telah direkayasa (lag, volatilitas, spread, event flag),
    lalu menghasilkan prediksi harga Brent hari berikutnya menggunakan model LSTM
    yang telah dilatih.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if model is None or scaler is None:
    st.error("Model atau scaler belum tersedia. Pastikan file model (.keras/.h5) dan scaler.pkl berada di direktori yang sama dengan app.py.")
    st.stop()

# ---- Baca & validasi data ----
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Gagal membaca CSV: {e}")
    st.stop()

if "date" not in df_raw.columns:
    st.error("Kolom 'date' tidak ditemukan dalam CSV.")
    st.stop()

df_raw["date"] = pd.to_datetime(df_raw["date"])
df_raw = df_raw.sort_values("date").set_index("date")

missing_cols = [c for c in FEATURES if c not in df_raw.columns]
if missing_cols:
    st.error(f"Kolom fitur berikut tidak ditemukan di CSV: {missing_cols}")
    st.stop()

if len(df_raw) <= SEQ_LEN:
    st.error(f"Data terlalu pendek. Dibutuhkan minimal {SEQ_LEN + 1} baris, ditemukan {len(df_raw)} baris.")
    st.stop()

df_feat = df_raw[FEATURES].dropna()

# ---- Jalankan inferensi ----
with st.spinner("Menjalankan prediksi..."):
    results = run_inference(df_feat, model, scaler)
    next_date, next_pred = forecast_next_step(df_feat, model, scaler)

if results is None or results.empty:
    st.error("Tidak cukup data untuk menghasilkan prediksi.")
    st.stop()

# =========================================================
# KPI ROW
# =========================================================
last_actual = results['actual'].iloc[-1]
last_pred = results['predicted'].iloc[-1]
mae = np.mean(np.abs(results['actual'] - results['predicted']))
rmse = np.sqrt(np.mean((results['actual'] - results['predicted']) ** 2))
delta_forecast = next_pred - last_actual

c1, c2, c3, c4 = st.columns(4)
c1.metric("Harga Aktual Terakhir", f"${last_actual:,.2f}")
c2.metric(
    f"Ramalan {next_date.date()}",
    f"${next_pred:,.2f}",
    delta=f"{delta_forecast:+.2f}"
)
c3.metric("MAE (in-sample)", f"${mae:,.2f}")
c4.metric("RMSE (in-sample)", f"${rmse:,.2f}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Prediksi vs Aktual", "🌍 Event Geopolitik", "🔗 Korelasi Fitur", "🗂️ Data & Unduhan"
])

with tab1:
    st.markdown('<div class="section-title">Aktual vs Prediksi Harga Brent</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results['actual'],
        name="Aktual", line=dict(color="#4dabf7", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results['predicted'],
        name="Prediksi", line=dict(color="#d4a017", width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=[next_date], y=[next_pred],
        name="Ramalan t+1", mode="markers",
        marker=dict(color="#ff4757", size=12, symbol="star")
    ))
    fig.update_layout(
        template="plotly_dark",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Residual Error</div>', unsafe_allow_html=True)
    resid = results['actual'] - results['predicted']
    fig_resid = px.line(x=results.index, y=resid, template="plotly_dark")
    fig_resid.add_hline(y=0, line_dash="dash", line_color="#ff4757")
    fig_resid.update_layout(height=300, xaxis_title="Tanggal", yaxis_title="Residual (Aktual - Prediksi)")
    st.plotly_chart(fig_resid, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Harga Brent vs Kejadian Geopolitik</div>', unsafe_allow_html=True)
    if "event_flag" in df_raw.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_raw.index, y=df_raw['brent_price'],
            name="Brent Price", line=dict(color="#4dabf7", width=1.5)
        ))
        events = df_raw[df_raw['event_flag'] == 1]
        fig2.add_trace(go.Scatter(
            x=events.index, y=events['brent_price'],
            name="Event Geopolitik", mode="markers",
            marker=dict(color="#ff4757", size=7)
        ))
        fig2.update_layout(template="plotly_dark", height=450, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

        if "event_type" in df_raw.columns:
            st.markdown('<div class="section-title">Distribusi Tipe Kejadian</div>', unsafe_allow_html=True)
            counts = df_raw.loc[df_raw['event_flag'] == 1, 'event_type'].value_counts().reset_index()
            counts.columns = ["event_type", "count"]
            fig3 = px.bar(counts, x="event_type", y="count", template="plotly_dark",
                          color="count", color_continuous_scale="oranges")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Kolom 'event_flag' tidak tersedia di data yang diunggah.")

with tab3:
    st.markdown('<div class="section-title">Heatmap Korelasi Fitur Utama</div>', unsafe_allow_html=True)
    corr_candidates = [c for c in
                        ['brent_price', 'wti_price', 'dxy_index', 'vix', 'gpr_index',
                         'event_severity', 'brent_volatility_30d', 'brent_wti_spread']
                        if c in df_raw.columns]
    corr_matrix = df_raw[corr_candidates].corr()
    fig4 = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
        template="plotly_dark", aspect="auto"
    )
    fig4.update_layout(height=500)
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Tabel Hasil Prediksi</div>', unsafe_allow_html=True)
    display_df = results.copy()
    display_df["residual"] = display_df["actual"] - display_df["predicted"]
    st.dataframe(display_df.style.format("{:.2f}"), use_container_width=True, height=350)

    csv_out = display_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Unduh Hasil Prediksi (CSV)",
        data=csv_out,
        file_name="brent_predictions.csv",
        mime="text/csv"
    )

    st.markdown('<div class="section-title">Data Mentah (5 baris pertama)</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(), use_container_width=True)

st.markdown("---")
st.caption("Dashboard prediksi harga minyak Brent — Multimodal LSTM (Data Historis & Sentimen Berita Global). Untuk keperluan riset/akademik.")
