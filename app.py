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
MODEL_WEIGHTS_PATH = "model_weights.weights.h5"  # cara paling aman lintas versi Keras
SCALER_PATH = "scaler.pkl"


def build_model_architecture(n_timesteps=SEQ_LEN, n_features=len(FEATURES)):
    """
    Arsitektur HARUS identik dengan skrip training, supaya bobot yang
    dimuat cocok bentuk (shape) tiap layer-nya.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# =========================================================
# LOADER (cached)
# =========================================================
@st.cache_resource(show_spinner="Memuat model LSTM...")
def load_lstm_model():
    from tensorflow.keras.models import load_model

    # 1) Cara paling robust lintas-versi: bangun ulang arsitektur, muat bobot saja
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            model = build_model_architecture()
            model.load_weights(MODEL_WEIGHTS_PATH)
            return model, MODEL_WEIGHTS_PATH
        except Exception as e:
            st.warning(f"Gagal memuat model_weights.weights.h5: {e}")

    # 2) Fallback: coba load_model utuh (rawan gagal jika versi Keras beda)
    if os.path.exists(MODEL_PATH_KERAS):
        try:
            return load_model(MODEL_PATH_KERAS), MODEL_PATH_KERAS
        except Exception as e:
            st.warning(f"Gagal memuat {MODEL_PATH_KERAS}: {e}")

    if os.path.exists(MODEL_PATH_H5):
        try:
            return load_model(MODEL_PATH_H5, compile=False), MODEL_PATH_H5
        except Exception as e:
            st.warning(f"Gagal memuat {MODEL_PATH_H5}: {e}")

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


def build_flat_seed(latest_values, seq_len=SEQ_LEN):
    """
    Membangun jendela sintetis 30 hari dari satu snapshot nilai manual
    (skenario 'flat' / tidak ada pergerakan historis). Dipakai untuk mode
    Input Manual, ketika pengguna tidak mengunggah data historis riil.
    """
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=seq_len, freq="D")
    rows = []
    for _ in range(seq_len):
        rows.append({
            'brent_price': latest_values['brent_price'],
            'wti_price': latest_values['wti_price'],
            'dxy_index': latest_values['dxy_index'],
            'vix': latest_values['vix'],
            'gpr_index': latest_values['gpr_index'],
            'brent_return': 0.0,
            'wti_return': 0.0,
            'brent_lag_1': latest_values['brent_price'],
            'brent_lag_3': latest_values['brent_price'],
            'brent_lag_7': latest_values['brent_price'],
            'wti_lag_1': latest_values['wti_price'],
            'wti_lag_3': latest_values['wti_price'],
            'wti_lag_7': latest_values['wti_price'],
            'brent_volatility_7d': 0.0,
            'brent_volatility_30d': 0.0,
            'wti_volatility_7d': 0.0,
            'wti_volatility_30d': 0.0,
            'brent_wti_spread': latest_values['brent_price'] - latest_values['wti_price'],
            'event_severity': latest_values['event_severity'],
            'event_flag': latest_values['event_flag'],
        })
    seed_df = pd.DataFrame(rows, index=dates)[FEATURES]
    return seed_df


def recursive_forecast(seed_df_raw, days_ahead, model, scaler,
                        wti_trend_pct=0.0, dxy_trend_pct=0.0,
                        vix_trend_pct=0.0, gpr_trend_pct=0.0,
                        assumed_event_severity=0.0, assumed_event_flag=0):
    """
    Ramalan multi-hari ke depan secara rekursif ("bulan depan").
    Prediksi brent_price hari ke-t dipakai sebagai bagian input untuk
    memprediksi hari ke-(t+1), dst. Fitur eksogen (WTI, DXY, VIX, GPR)
    diproyeksikan mengikuti tren % harian yang diasumsikan (default 0% = flat).
    event_severity/event_flag diasumsikan konstan sesuai input pengguna.

    PERINGATAN: akurasi menurun signifikan untuk horizon panjang karena error
    prediksi menumpuk (compounding error), dan asumsi tren di atas belum tentu
    merefleksikan kondisi pasar riil di masa depan.
    """
    working = seed_df_raw.copy()
    forecasts = []

    for _ in range(days_ahead):
        window = working[FEATURES].values[-SEQ_LEN:]
        scaled_window = scaler.transform(window)
        X_next = np.expand_dims(scaled_window, axis=0)

        pred_scaled = model.predict(X_next, verbose=0).flatten()[0]
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, TARGET_COL_INDEX] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0, TARGET_COL_INDEX]

        prev_row = working.iloc[-1]
        next_date = working.index[-1] + timedelta(days=1)

        new_wti = prev_row['wti_price'] * (1 + wti_trend_pct / 100)
        new_dxy = prev_row['dxy_index'] * (1 + dxy_trend_pct / 100)
        new_vix = max(prev_row['vix'] * (1 + vix_trend_pct / 100), 0.0)
        new_gpr = max(prev_row['gpr_index'] * (1 + gpr_trend_pct / 100), 0.0)

        brent_hist = list(working['brent_price'].values) + [pred_price]
        wti_hist = list(working['wti_price'].values) + [new_wti]

        new_row = {
            'brent_price': pred_price,
            'wti_price': new_wti,
            'dxy_index': new_dxy,
            'vix': new_vix,
            'gpr_index': new_gpr,
            'brent_return': (pred_price - prev_row['brent_price']) / prev_row['brent_price'],
            'wti_return': (new_wti - prev_row['wti_price']) / prev_row['wti_price'] if prev_row['wti_price'] != 0 else 0.0,
            'brent_lag_1': brent_hist[-2],
            'brent_lag_3': brent_hist[-4] if len(brent_hist) >= 4 else brent_hist[0],
            'brent_lag_7': brent_hist[-8] if len(brent_hist) >= 8 else brent_hist[0],
            'wti_lag_1': wti_hist[-2],
            'wti_lag_3': wti_hist[-4] if len(wti_hist) >= 4 else wti_hist[0],
            'wti_lag_7': wti_hist[-8] if len(wti_hist) >= 8 else wti_hist[0],
            'brent_volatility_7d': float(np.std(brent_hist[-7:])),
            'brent_volatility_30d': float(np.std(brent_hist[-30:])),
            'wti_volatility_7d': float(np.std(wti_hist[-7:])),
            'wti_volatility_30d': float(np.std(wti_hist[-30:])),
            'brent_wti_spread': pred_price - new_wti,
            'event_severity': assumed_event_severity,
            'event_flag': assumed_event_flag,
        }

        new_row_df = pd.DataFrame([new_row], index=[next_date])[FEATURES]
        working = pd.concat([working, new_row_df])
        forecasts.append((next_date, pred_price))

    forecast_df = pd.DataFrame(forecasts, columns=["date", "predicted"]).set_index("date")
    return forecast_df


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Panel Kontrol")

    mode = st.radio(
        "Sumber data",
        ["📤 Upload CSV", "✍️ Input Manual"],
        help="Upload CSV memakai data historis riil (lebih akurat). Input Manual memakai skenario yang kamu masukkan sendiri (lebih kasar/cepat)."
    )

    uploaded_file = None
    if mode == "📤 Upload CSV":
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
    ⚠️ <b>Catatan model:</b> Model ini pada dasarnya forecasting <b>single-step (t+1)</b>.
    Ramalan lebih dari 1 hari ke depan (mis. "bulan depan") dilakukan secara
    <b>rekursif</b> — prediksi hari ini dipakai sebagai input prediksi hari
    berikutnya. Akurasi menurun untuk horizon panjang karena error menumpuk,
    dan fitur eksogen (DXY, VIX, GPR, event) masa depan hanya asumsi, bukan data riil.
    </div>
    """, unsafe_allow_html=True)

    model, model_used = load_lstm_model()
    scaler = load_scaler_obj()

    st.markdown("---")
    st.markdown("**🧠 Status Model**")
    if model is not None:
        st.success(f"Model dimuat: `{model_used}`")
    else:
        st.error("Model tidak ditemukan (model_weights.weights.h5 / .keras / .h5)")
    if scaler is not None:
        st.success("Scaler dimuat: `scaler.pkl`")
    else:
        st.error("scaler.pkl tidak ditemukan")


# =========================================================
# MAIN CONTENT
# =========================================================
if model is None or scaler is None:
    st.error("Model atau scaler belum tersedia. Pastikan model_weights.weights.h5 (atau .keras/.h5) dan scaler.pkl berada di direktori yang sama dengan app.py.")
    st.stop()

df_raw = None
df_feat = None
results = None
next_date = next_pred = None
manual_seed_df = None

# =========================================================
# MODE: UPLOAD CSV
# =========================================================
if mode == "📤 Upload CSV":
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

    with st.spinner("Menjalankan prediksi..."):
        results = run_inference(df_feat, model, scaler)
        next_date, next_pred = forecast_next_step(df_feat, model, scaler)

    if results is None or results.empty:
        st.error("Tidak cukup data untuk menghasilkan prediksi.")
        st.stop()

# =========================================================
# MODE: INPUT MANUAL
# =========================================================
else:
    st.markdown('<div class="section-title">✍️ Masukkan Kondisi Pasar Saat Ini</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    Mode ini <b>tidak</b> memakai data historis riil — hanya snapshot kondisi hari ini
    yang diulang menjadi jendela 30 hari "flat" sebagai titik awal. Cocok untuk simulasi
    cepat "bagaimana jika", tapi jauh lebih kasar dibanding mode Upload CSV yang memakai
    pola historis asli.
    </div>
    """, unsafe_allow_html=True)

    with st.form("manual_input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            brent_price_in = st.number_input("Harga Brent saat ini (USD)", min_value=0.0, value=80.0, step=0.5)
            wti_price_in = st.number_input("Harga WTI saat ini (USD)", min_value=0.0, value=76.0, step=0.5)
        with c2:
            dxy_in = st.number_input("DXY Index", min_value=0.0, value=100.0, step=0.5)
            vix_in = st.number_input("VIX", min_value=0.0, value=15.0, step=0.5)
        with c3:
            gpr_in = st.number_input("GPR Index (risiko geopolitik)", min_value=0.0, value=100.0, step=1.0)
            event_severity_in = st.slider("Asumsi Severity Event", 0.0, 10.0, 0.0, 0.5)
            event_flag_in = st.checkbox("Asumsikan ada event geopolitik aktif")

        submitted = st.form_submit_button("🔮 Bangun Skenario")

    # Simpan ke session_state supaya skenario TETAP ada walau ada rerun lain
    # (misalnya saat tombol "Jalankan Ramalan ke Depan" di bawah diklik).
    if submitted:
        manual_values = {
            'brent_price': brent_price_in,
            'wti_price': wti_price_in,
            'dxy_index': dxy_in,
            'vix': vix_in,
            'gpr_index': gpr_in,
            'event_severity': event_severity_in,
            'event_flag': 1 if event_flag_in else 0,
        }
        st.session_state['manual_seed_df'] = build_flat_seed(manual_values)

    if st.session_state.get('manual_seed_df') is None:
        st.info("Isi form di atas lalu klik **Bangun Skenario** untuk melanjutkan.")
        st.stop()

    manual_seed_df = st.session_state['manual_seed_df']

    if st.button("🔄 Reset Skenario"):
        st.session_state['manual_seed_df'] = None
        st.rerun()

    next_date, next_pred = forecast_next_step(manual_seed_df, model, scaler)

# =========================================================
# KPI ROW
# =========================================================
if mode == "📤 Upload CSV":
    last_actual = results['actual'].iloc[-1]
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
else:
    last_actual = manual_seed_df['brent_price'].iloc[-1]
    delta_forecast = next_pred - last_actual
    c1, c2 = st.columns(2)
    c1.metric("Harga Brent Input", f"${last_actual:,.2f}")
    c2.metric(
        f"Ramalan {next_date.date()}",
        f"${next_pred:,.2f}",
        delta=f"{delta_forecast:+.2f}"
    )

# =========================================================
# RAMALAN BULAN DEPAN (rekursif, tersedia di kedua mode)
# =========================================================
st.markdown('<div class="section-title">🔮 Ramalan ke Depan (Rekursif)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-card">
Ramalan di bawah ini bersifat <b>rekursif</b>: prediksi hari-t dipakai lagi sebagai
input untuk memprediksi hari-(t+1), dan seterusnya. Semakin jauh horizonnya, semakin
besar potensi errornya menumpuk — anggap ini sebagai <b>skenario</b>, bukan angka pasti.
</div>
""", unsafe_allow_html=True)

fc1, fc2, fc3, fc4, fc5 = st.columns(5)
with fc1:
    days_ahead = st.number_input("Horizon (hari)", min_value=1, max_value=90, value=30, step=1)
with fc2:
    wti_trend = st.slider("Tren WTI (%/hari)", -2.0, 2.0, 0.0, 0.1)
with fc3:
    dxy_trend = st.slider("Tren DXY (%/hari)", -2.0, 2.0, 0.0, 0.1)
with fc4:
    vix_trend = st.slider("Tren VIX (%/hari)", -5.0, 5.0, 0.0, 0.1)
with fc5:
    gpr_trend = st.slider("Tren GPR (%/hari)", -5.0, 5.0, 0.0, 0.1)

fc6, fc7 = st.columns(2)
with fc6:
    assumed_severity = st.slider("Asumsi severity event ke depan", 0.0, 10.0, 0.0, 0.5)
with fc7:
    assumed_flag = st.checkbox("Asumsikan event geopolitik terus aktif selama horizon")

seed_for_forecast = df_feat if mode == "📤 Upload CSV" else manual_seed_df

if st.button("▶️ Jalankan Ramalan ke Depan"):
    with st.spinner(f"Menjalankan ramalan rekursif {days_ahead} hari..."):
        forecast_df = recursive_forecast(
            seed_for_forecast, days_ahead, model, scaler,
            wti_trend_pct=wti_trend, dxy_trend_pct=dxy_trend,
            vix_trend_pct=vix_trend, gpr_trend_pct=gpr_trend,
            assumed_event_severity=assumed_severity,
            assumed_event_flag=1 if assumed_flag else 0
        )
        st.session_state['forecast_df'] = forecast_df
        st.session_state['forecast_days'] = days_ahead
        st.session_state['forecast_mode'] = mode

if st.session_state.get('forecast_df') is not None:
    forecast_df = st.session_state['forecast_df']
    days_ahead_shown = st.session_state.get('forecast_days', days_ahead)

    fig_fc = go.Figure()
    if st.session_state.get('forecast_mode') == "📤 Upload CSV" and df_feat is not None:
        recent_hist = df_feat['brent_price'].iloc[-60:]
        fig_fc.add_trace(go.Scatter(
            x=recent_hist.index, y=recent_hist.values,
            name="Historis", line=dict(color="#4dabf7", width=2)
        ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['predicted'],
        name=f"Ramalan {days_ahead_shown} hari", line=dict(color="#d4a017", width=2, dash="dot"),
        mode="lines+markers"
    ))
    fig_fc.update_layout(template="plotly_dark", height=460, hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)

    st.dataframe(forecast_df.style.format("{:.2f}"), use_container_width=True, height=300)
    csv_fc = forecast_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Unduh Ramalan (CSV)",
        data=csv_fc,
        file_name=f"brent_forecast_{days_ahead_shown}d.csv",
        mime="text/csv"
    )

# =========================================================
# TABS (analisis historis — hanya tersedia di mode Upload CSV)
# =========================================================
if mode != "📤 Upload CSV":
    st.stop()

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
