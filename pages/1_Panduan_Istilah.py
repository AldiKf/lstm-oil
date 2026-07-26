import streamlit as st
import pandas as pd

st.set_page_config(page_title="Panduan Istilah | Brent Oil Forecast", page_icon="📖", layout="wide")

st.markdown("""
<style>
    .glossary-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(212, 160, 23, 0.25);
    }
    .term-card {
        background: rgba(212, 160, 23, 0.06);
        border-left: 4px solid #d4a017;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.9rem 0;
    }
    .term-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f2f2f2;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glossary-hero">
    <h2>📖 Panduan Penggunaan Dashboard</h2>
    <p style="color:#d4a017;">
        Halaman ini menjelaskan arti setiap parameter yang digunakan pada dashboard prediksi harga minyak mentah Brent agar lebih mudah dipahami.
    </p>
</div>
""", unsafe_allow_html=True)

st.info("💡 **Tips:** Jika baru pertama kali menggunakan dashboard, cobalah Horizon **7 hari** dengan seluruh parameter tren bernilai **0%**. Setelah itu ubah satu parameter secara bertahap untuk melihat pengaruhnya terhadap hasil prediksi.")

# =========================================================
# PARAMETER
# =========================================================
st.markdown("## 🔮 Parameter Ramalan")

st.markdown("""
<div class="term-card">
<div class="term-title">📅 Horizon (Hari)</div>

Menentukan berapa hari ke depan yang ingin diprediksi.

<b>Contoh:</b>

<ul>
<li>Data terakhir: <b>10 Juli</b></li>
<li>Horizon: <b>30 hari</b></li>
<li>Hasil prediksi: <b>11 Juli – 9 Agustus</b></li>
</ul>

Model hanya memprediksi <b>satu hari ke depan</b>. Untuk menghasilkan prediksi beberapa hari sekaligus, hasil prediksi hari sebelumnya akan digunakan sebagai input untuk hari berikutnya (<i>recursive forecasting</i>).

<br><br>

<b>Catatan:</b> Semakin panjang horizon, semakin besar kemungkinan akumulasi error. Prediksi 7 hari umumnya lebih akurat dibandingkan prediksi 30 atau 90 hari.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">🛢️ Tren WTI (% per Hari)</div>

WTI (<i>West Texas Intermediate</i>) merupakan salah satu harga acuan minyak mentah dunia dan umumnya bergerak searah dengan harga Brent.

Karena harga WTI pada masa depan belum diketahui, Anda dapat menentukan asumsi perubahan hariannya.

<ul>
<li><b>0%</b> → Harga tetap.</li>
<li><b>+0.5%</b> → Harga naik 0,5% setiap hari.</li>
<li><b>-0.5%</b> → Harga turun 0,5% setiap hari.</li>
</ul>

<b>Saran:</b> Gunakan nilai kecil (sekitar <b>-0.5% hingga +0.5%</b>) agar skenario tetap realistis.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">💵 Tren DXY (% per Hari)</div>

DXY (<i>US Dollar Index</i>) menunjukkan kekuatan Dolar AS terhadap mata uang utama dunia.

Secara umum:

<ul>
<li>DXY naik → Harga minyak cenderung turun.</li>
<li>DXY turun → Harga minyak cenderung naik.</li>
</ul>

Masukkan asumsi perubahan DXY setiap hari selama periode prediksi.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">📉 Tren VIX (% per Hari)</div>

VIX atau <i>Fear Index</i> menggambarkan tingkat ketidakpastian pasar.

<ul>
<li>VIX naik → Pasar lebih tidak stabil.</li>
<li>VIX turun → Kondisi pasar lebih tenang.</li>
</ul>

Gunakan slider untuk menentukan asumsi perubahan VIX selama periode prediksi.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">🌍 Tren GPR (% per Hari)</div>

GPR (<i>Geopolitical Risk Index</i>) mengukur tingkat risiko geopolitik global, seperti perang, konflik, maupun sanksi ekonomi.

Secara umum:

<ul>
<li>GPR naik → Risiko geopolitik meningkat sehingga harga minyak berpotensi naik.</li>
<li>GPR turun → Kondisi geopolitik lebih stabil.</li>
</ul>

Masukkan asumsi perubahan GPR setiap hari selama periode prediksi.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">⚠️ Status & Tingkat Keparahan Kejadian Geopolitik</div>

Digunakan untuk mensimulasikan adanya kejadian geopolitik selama periode prediksi.

<ul>
<li><b>Status Event</b> → Menentukan apakah terjadi kejadian geopolitik atau tidak.</li>
<li><b>Severity (1–10)</b> → Menunjukkan tingkat keparahan dampak kejadian.</li>
</ul>

Semakin tinggi nilai <b>Severity</b>, semakin besar pengaruhnya terhadap hasil prediksi.

</div>
""", unsafe_allow_html=True)

# =========================================================
# SEVERITY
# =========================================================
st.markdown("## ⚠️ Skala Severity Kejadian Geopolitik")

st.markdown("""
Skala berikut digunakan untuk menggambarkan tingkat keparahan suatu kejadian geopolitik.
Semakin tinggi nilainya, semakin besar potensi dampaknya terhadap pasar minyak.
""")

SEVERITY_LEGEND = {
    1: ("Sangat Rendah","Berita rutin pasar tanpa dampak signifikan"),
    2: ("Sangat Rendah","Rumor atau ketegangan kecil"),
    3: ("Rendah","Friksi diplomatik lokal"),
    4: ("Rendah","Sanksi ringan atau pertemuan OPEC rutin"),
    5: ("Sedang","Krisis diplomatik regional"),
    6: ("Sedang","Krisis Qatar, pencabutan sanksi Iran"),
    7: ("Tinggi","Deepwater Horizon, konflik Yaman"),
    8: ("Tinggi","Embargo minyak, blokade Terusan Suez"),
    9: ("Sangat Tinggi","Perang Libya, Israel–Hamas"),
    10:("Ekstrem","Invasi Rusia–Ukraina, serangan Aramco")
}

legend_df = pd.DataFrame(
    [{"Severity":k,"Kategori":v[0],"Contoh":v[1]}
     for k,v in SEVERITY_LEGEND.items()]
)

st.dataframe(legend_df,use_container_width=True,hide_index=True)

# =========================================================
# INPUT MODE
# =========================================================
st.markdown("## 📥 Mode Input Data")

col1,col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="term-card">
    <div class="term-title">📤 Upload CSV</div>

    Gunakan data historis yang Anda miliki sebagai dasar prediksi.

    Model akan membaca <b>30 hari terakhir</b> dari data tersebut sehingga pola harga, lag, dan volatilitas dapat dimanfaatkan secara optimal.

    <br><br>

    <b>Cocok digunakan jika memiliki data historis terbaru.</b>

    </div>
    """,unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="term-card">
    <div class="term-title">✍️ Input Manual</div>

    Masukkan nilai Brent, WTI, DXY, VIX, GPR, dan variabel lainnya secara manual untuk membuat skenario prediksi.

    Mode ini cocok digunakan untuk melakukan simulasi <i>"bagaimana jika"</i> tanpa perlu mengunggah data historis.

    </div>
    """,unsafe_allow_html=True)

# =========================================================
# PENUTUP
# =========================================================
st.markdown("---")

st.markdown("""
<div class="term-card">
<div class="term-title">💡 Ringkasan</div>

Dashboard ini dirancang sebagai <b>alat simulasi dan analisis skenario</b>, bukan untuk memberikan kepastian harga minyak di masa depan.

Hasil prediksi sangat dipengaruhi oleh asumsi parameter yang dimasukkan. Semakin panjang horizon dan semakin ekstrem nilai parameter yang digunakan, semakin besar kemungkinan prediksi menyimpang dari kondisi sebenarnya.

Gunakan dashboard ini sebagai pendukung analisis dan eksplorasi skenario, bukan sebagai satu-satunya dasar dalam pengambilan keputusan.

</div>
""",unsafe_allow_html=True)

st.page_link("app.py", label="⬅️ Kembali ke Dashboard", icon="🛢️")
